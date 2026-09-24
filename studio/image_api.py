"""Provider image-edit clients. No Torch, browser secrets, or request-body logs."""

from __future__ import annotations

import base64
import binascii
import email.utils
import hashlib
import http.client
import io
import ipaddress
import json
import os
import re
import socket
import ssl
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from PIL import Image, ImageOps

from studio.errors import ImageAPIError

ENDPOINTS = {"doubao-seedream-5-0-260128": ("ark.cn-beijing.volces.com", "/api/v3/images/generations"), "gpt-image-2": ("grsai.dakka.com.cn", "/v1/api/generate")}
IMAGE_LIMIT = 32 * 1024 * 1024
# The gateway may return base64 despite response_format=url. Bound that JSON too.
JSON_LIMIT = 48 * 1024 * 1024
PIXEL_LIMIT = 32_000_000
MODELS = {
    "doubao-seedream-5-0-260128": {
        "label": "Doubao · Seedream 5.0 (Volcengine)",
        "input_transport": "data_url",
        "response_format": "url",
        "sizes": [(2848, 1600), (2304, 1728), (2048, 2048), (1728, 2304), (1600, 2848)],
    },
    "gpt-image-2": {
        "label": "GPT Image 2 (GrsAI)",
        "input_transport": "data_url",
        "response_format": "url",
        "sizes": [(1672, 941), (1443, 1090), (1024, 1024), (1090, 1443), (941, 1672)],
    },
}


class ImageDownloadError(ImageAPIError):
    """Generation finished; retrying the download never creates another image."""


def read_key(root: Path, model: str) -> str:
    names = {"doubao-seedream-5-0-260128": ("ark-key.txt", "ARK_API_KEY"),
             "gpt-image-2": ("grs-key.txt", "GRSAI_API_KEY")}
    if model not in names:
        raise ImageAPIError("Choose a supported image model.")
    filename, variable = names[model]
    path = root / filename
    if path.is_file():
        if path.stat().st_size > 4096:
            raise ImageAPIError(f"{filename} is unexpectedly large.")
        key = path.read_text(encoding="utf-8-sig").strip()
    else:
        key = os.environ.get(variable, "").strip()
    if not key:
        raise ImageAPIError(f"Add the API key to {filename} or {variable}.")
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ImageAPIError("The API key must be a single ASCII token without whitespace.")
    return key


def validate_prompt(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 16000:
        raise ImageAPIError("Prompt must contain 1–16000 characters.")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ImageAPIError("Prompt contains control characters.")
    return value.strip()


def model_size(model: str, width: int, height: int) -> str:
    if model not in MODELS:
        raise ImageAPIError("Choose a supported image model; no automatic fallback is used.")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ImageAPIError("Input image dimensions must be positive.")
    ratio = width / height
    w, h = min(MODELS[model]["sizes"], key=lambda size: abs(size[0] / size[1] - ratio))
    if abs(w * height - width * h) * 100 > width * h * 2:
        raise ImageAPIError("This view ratio is not supported by the selected model.")
    if model == "gpt-image-2":
        return {(1672, 941): "16:9", (1443, 1090): "4:3", (1024, 1024): "1:1",
                (1090, 1443): "3:4", (941, 1672): "9:16"}[(w, h)]
    return f"{w}x{h}"


def model_preflight(model: str) -> None:
    if model not in MODELS:
        raise ImageAPIError("Choose a supported image model; no automatic fallback is used.")


def public_models() -> list[dict]:
    # This is the UI's single source of truth; do not duplicate model sizes in JS.
    return [{"id": model, **settings} for model, settings in MODELS.items()]


def _bounded_read(response, limit: int) -> bytes:
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise ImageAPIError("Remote server returned an invalid content length.") from None
        if length < 0 or length > limit:
            raise ImageAPIError("Remote response exceeds the size limit.")
    blocks, count = [], 0
    started = time.monotonic()
    while True:
        if time.monotonic() - started > 240:
            raise ImageAPIError("Remote response exceeded the download time limit.")
        block = response.read(min(65536, limit + 1 - count))
        if not block:
            break
        count += len(block)
        if count > limit:
            raise ImageAPIError("Remote response exceeds the size limit.")
        blocks.append(block)
    if declared is not None and count != int(declared):
        raise ImageAPIError("Remote response was truncated.")
    return b"".join(blocks)


def _retry_delay(value: str | None, attempt: int) -> float:
    delay = [2.0, 5.0][attempt]
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                delay = (parsed - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    return min(30.0, max(0.0, delay))


def _error_summary(response, key: str) -> str:
    """Small, redacted error fields; never expose raw gateway JSON or URLs."""
    try:
        # Errors may echo a large request. Read only a prefix rather than reject
        # their Content-Length and silently lose the actual upstream cause.
        raw = response.read(32 * 1024)
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            result = raw.decode("utf-8", errors="replace")
            if not result.lstrip().startswith("{"):
                return " Gateway returned a non-JSON or truncated error body."
        error = result.get("error", result) if isinstance(result, dict) else result
        if isinstance(error, dict):
            fields = [str(error[name]) for name in ("code", "type", "message", "detail")
                      if isinstance(error.get(name), (str, int))]
        elif isinstance(error, str):
            fields = [error]
        else:
            return " Gateway returned an unrecognized error object."
        message = "; ".join(fields).replace("\\/", "/").replace(key, "[credential redacted]")
        message = re.sub(r"(?i)Bearer\s+\S+|sk-[A-Za-z0-9_-]+", "[credential redacted]", message)
        message = re.sub(r"(?i)data:image/[^\s\"']+|https?://[^\s\"']+", "[URL/data redacted]", message)
        message = re.sub(r"[A-Za-z0-9+/=_-]{48,}", "[long token redacted]", message)
        message = " ".join(message.split())[:400]
        return f" Gateway detail: {message}" if message else ""
    except (ImageAPIError, ValueError, OSError, http.client.HTTPException, RecursionError):
        return ""


def post_edit(body: dict, key: str, *, connection_factory=None, sleep=time.sleep) -> dict:
    factory = connection_factory or http.client.HTTPSConnection
    model = body.get("model")
    if model not in ENDPOINTS:
        raise ImageAPIError("Choose a supported image model.")
    host, path = ENDPOINTS[model]
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(3):
        connection = factory(host, timeout=10)
        delay = None
        phase = "connect"
        try:
            connection.connect()
            connection.sock.settimeout(180)
            phase = "send"
            connection.request("POST", path, payload, {
                "Content-Type": "application/json", "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            })
            phase = "response"
            response = connection.getresponse()
            if response.status == 429 and attempt < 2:
                delay = _retry_delay(response.getheader("Retry-After"), attempt)
            elif response.status != 200:
                # Never include response JSON, request headers, or signed URLs.
                detail = _error_summary(response, key)
                raise ImageAPIError(f"Image API returned HTTP {response.status}; model was not changed.{detail}")
            else:
                if model == "doubao-seedream-5-0-260128" and "text/event-stream" in (response.getheader("Content-Type") or ""):
                    return _ark_stream_result(response)
                raw = _bounded_read(response, JSON_LIMIT)
                try:
                    result = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise ImageAPIError("Image API returned invalid JSON.") from None
                if not isinstance(result, dict):
                    raise ImageAPIError("Image API returned an invalid result object.")
                return _grs_result(result, key, host, connection_factory=connection_factory, sleep=sleep) if model == "gpt-image-2" else result
        except (OSError, http.client.HTTPException) as error:
            raise ImageAPIError(
                f"Image API transport failed during {phase} ({type(error).__name__}). "
                "The request may have been billed; it was not automatically repeated."
            ) from None
        finally:
            connection.close()
        if delay is not None:
            sleep(delay)
    raise ImageAPIError("Image API retry limit reached.")


def _ark_stream_result(response) -> dict:
    total = 0
    for _ in range(1000):
        line = response.readline(JSON_LIMIT + 1)
        if not line:
            break
        total += len(line)
        if total > JSON_LIMIT:
            raise ImageAPIError("Volcengine image stream exceeds the response limit.")
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            event = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            raise ImageAPIError("Volcengine returned invalid image stream data.") from None
        if not isinstance(event, dict):
            raise ImageAPIError("Volcengine returned an invalid image event.")
        if event.get("type") == "image_generation.partial_succeeded":
            if isinstance(event.get("url"), str) and event["url"]:
                return {"data": [{"url": event["url"]}]}
            if isinstance(event.get("b64_json"), str) and event["b64_json"]:
                return {"data": [{"b64_json": event["b64_json"]}]}
            raise ImageAPIError("Volcengine success event contained no image.")
        if event.get("type") == "image_generation.partial_failed":
            raise ImageAPIError("Volcengine image generation failed.")
    raise ImageAPIError("Volcengine image stream ended without an image.")


def _grs_result(result: dict, key: str, host: str, *, connection_factory=None, sleep=time.sleep) -> dict:
    if result.get("status") in ("succeeded", "success"):
        return result
    if result.get("status") not in ("running", "pending", "processing", "queued"):
        raise ImageAPIError("GrsAI image task failed or returned an unknown status.")
    task_id = result.get("id")
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_id):
        raise ImageAPIError("GrsAI did not return a valid task id.")
    factory = connection_factory or http.client.HTTPSConnection
    for _ in range(120):
        sleep(5)
        connection = factory(host, timeout=10)
        try:
            connection.connect(); connection.sock.settimeout(60)
            connection.request("GET", "/v1/api/result?id=" + task_id, headers={"Accept": "application/json", "Authorization": f"Bearer {key}"})
            response = connection.getresponse()
            if response.status != 200:
                raise ImageAPIError(f"GrsAI result query returned HTTP {response.status}.")
            result = json.loads(_bounded_read(response, JSON_LIMIT))
            if not isinstance(result, dict):
                raise ImageAPIError("GrsAI returned an invalid task result.")
            if result.get("status") in ("succeeded", "success"):
                return result
            if result.get("status") not in ("running", "pending", "processing", "queued"):
                raise ImageAPIError("GrsAI image task failed or returned an unknown status.")
        except (OSError, http.client.HTTPException, ValueError):
            raise ImageAPIError("GrsAI result query was interrupted; generation was not repeated.") from None
        finally:
            connection.close()
    raise ImageAPIError("GrsAI image task is still pending after ten minutes; generation was not repeated.")


def _public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not any((address.is_multicast, address.is_reserved,
                                         address.is_loopback, address.is_link_local,
                                         address.is_unspecified))


def _allowed_download_ip(host: str, value: str) -> bool:
    if _public_ip(value):
        return True
    # The managed Windows egress proxy resolves provider CDNs to 198.18/15.
    # Only the documented result hosts may use that synthetic address range.
    provider_host = (re.fullmatch(r"file[0-9]+\.aitohumanize\.com", host, re.IGNORECASE)
                     or host.lower() == "ark-content-generation-v2-cn-beijing.tos-cn-beijing.volces.com")
    return bool(provider_host) and ipaddress.ip_address(value) in ipaddress.ip_network("198.18.0.0/15")


def public_target(url: str) -> tuple[str, str, int, str, list[str]]:
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise ImageAPIError("Image download URL is invalid.")
    if any(ord(c) <= 32 or ord(c) > 126 for c in url) or "\\" in url:
        raise ImageAPIError("Image download URL contains forbidden characters.")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if (parsed.scheme not in ("http", "https") or not host or parsed.username is not None
                or parsed.password is not None or parsed.fragment or "%" in host
                or port != (443 if parsed.scheme == "https" else 80)):
            raise ValueError()
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
        if not addresses or not all(_allowed_download_ip(host, ip) for ip in addresses):
            raise ValueError()
    except (ValueError, OSError):
        raise ImageAPIError("Image download target is not a valid public HTTP(S) address.") from None
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return parsed.scheme, host, port, target, addresses


def _pinned_connection(scheme: str, host: str, port: int, addresses: list[str]):
    factory = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    connection = factory(host, port, timeout=10)
    # Resolve once, connect to the checked IP, keep the original Host and TLS SNI.
    raw_socket = socket.create_connection((addresses[0], port), timeout=10)
    try:
        peer = raw_socket.getpeername()[0]
        if peer not in addresses or not _allowed_download_ip(host, peer):
            raise ImageAPIError("Image download peer failed the public-address check.")
        if scheme == "https":
            raw_socket = ssl.create_default_context().wrap_socket(raw_socket, server_hostname=host)
        raw_socket.settimeout(180)
        connection.sock = raw_socket
        return connection
    except BaseException:
        raw_socket.close()
        raise


def download_image(url: str, *, connect=None) -> bytes:
    connect = connect or _pinned_connection
    # CDN links in the company documentation may be http://. Prefer authenticated
    # TLS for those links. Query/path are retained; there is no downgrade fallback.
    if isinstance(url, str) and url.startswith("http://"):
        # Validate the supplied URL first, then validate the upgraded target again.
        public_target(url)
        url = "https://" + url[len("http://"):]
    for hop in range(4):
        scheme, host, port, target, addresses = public_target(url)
        connection = None
        try:
            connection = connect(scheme, host, port, addresses)
            # A completely separate connection. API authorization is never forwarded.
            connection.request("GET", target, headers={"Accept": "image/*, application/octet-stream", "User-Agent": "GaussianRepairStudio/0.1"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if hop == 3 or not location:
                    raise ImageAPIError("Image download has too many or invalid redirects.")
                next_url = urljoin(url, location)
                if scheme == "https" and urlsplit(next_url).scheme != "https":
                    raise ImageAPIError("Image download refused an HTTPS downgrade redirect.")
                url = next_url
                continue
            if response.status != 200:
                raise ImageDownloadError(f"Image download returned HTTP {response.status} from {scheme}://{host} (path omitted).")
            return _bounded_read(response, IMAGE_LIMIT)
        except (OSError, http.client.HTTPException):
            raise ImageDownloadError("Image download failed or timed out; the existing generated image can be downloaded again without a new generation request.") from None
        finally:
            if connection:
                connection.close()
    raise ImageAPIError("Image download redirect limit reached.")


def result_bytes(result: dict, *, download=None) -> bytes:
    download = download or download_image
    data = result.get("results") if "results" in result else result.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ImageAPIError("Image API returned no image.")
    if isinstance(data[0].get("url"), str) and data[0]["url"]:
        return download(data[0]["url"])
    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ImageAPIError("Image API result has no URL or base64 image.")
    if encoded.startswith("data:"):
        prefix, separator, encoded = encoded.partition(",")
        if not separator or prefix not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
            raise ImageAPIError("Image API returned an unsupported image data URL.")
    if len(encoded) > (IMAGE_LIMIT + 2) // 3 * 4:
        raise ImageAPIError("Image API base64 result exceeds the size limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageAPIError("Image API returned invalid base64.") from None
    if len(raw) > IMAGE_LIMIT:
        raise ImageAPIError("Image API result exceeds the size limit.")
    return raw


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > IMAGE_LIMIT:
        raise ImageAPIError("Image is empty or exceeds 32 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                w, h = opened.size
                if min(w, h) < 1 or max(w, h) > 8192 or w * h > PIXEL_LIMIT:
                    raise ImageAPIError("Image exceeds 8192 pixels per side or 32 million pixels.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ImageAPIError("Animated API images are not supported.")
                opened.verify()
            with Image.open(io.BytesIO(raw)) as opened:
                normalized = ImageOps.exif_transpose(opened)
                normalized.load()
                return normalized.convert("RGBA")
    except ImageAPIError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageAPIError("Image data is corrupt, truncated, or unsafe to decode.") from None


def normalize_image(raw: bytes, source: Image.Image) -> tuple[Image.Image, dict]:
    edited = decode_image(raw)
    src_w, src_h = source.size
    api_w, api_h = edited.size
    # Integer comparison makes the inclusive 2% boundary exact.
    numerator = abs(api_w * src_h - src_w * api_h)
    denominator = src_w * api_h
    if numerator * 100 > denominator * 2:
        raise ImageAPIError(f"API image ratio differs by more than 2% ({api_w}x{api_h} vs {src_w}x{src_h}); no crop or stretch was applied.")
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    output = Image.alpha_composite(source.convert("RGB").convert("RGBA"), edited).convert("RGB")
    return output, {"input_size": [src_w, src_h], "api_size": [api_w, api_h],
                    "ratio_error": numerator / denominator, "output_size": list(output.size)}


def repair_image(source_path: Path, output_dir: Path, model: str, prompt: str, key: str,
                 *, result_cache: dict | None = None) -> Path:
    if source_path.stat().st_size > IMAGE_LIMIT:
        raise ImageAPIError("Input image exceeds the local 32 MiB safety limit.")
    source_raw = source_path.read_bytes()
    if not source_raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageAPIError("The locked API input must be a PNG image.")
    source = decode_image(source_raw)
    signature = hashlib.sha256(source_raw + model.encode("utf-8") + validate_prompt(prompt).encode("utf-8")).hexdigest()
    size = model_size(model, *source.size)
    data_url = "data:image/png;base64," + base64.b64encode(source_raw).decode("ascii")
    if model == "gpt-image-2":
        body = {"model": model, "prompt": validate_prompt(prompt), "images": [data_url],
                "aspectRatio": size, "quality": "auto", "replyType": "async"}
    else:
        body = {"model": model, "prompt": validate_prompt(prompt), "image": data_url,
                "size": size, "response_format": "url", "output_format": "png",
                "sequential_image_generation": "disabled", "watermark": False,
                "stream": True}
    result = result_cache.get("result") if result_cache is not None else None
    if result is not None and result_cache.get("signature") != signature:
        raise ImageAPIError("The locked input changed after generation; the cached result cannot be reused.")
    if result is None:
        model_preflight(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "api_input.png").write_bytes(source_raw)
        info = {"model": model, "provider": "grsai" if model == "gpt-image-2" else "volcengine",
                "input_size": list(source.size), "request_size": size,
                "input_bytes": len(source_raw), "input_sha256": hashlib.sha256(source_raw).hexdigest()}
        (output_dir / "api_request_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        result = post_edit(body, key)
        if result_cache is not None:
            result_cache["result"] = result  # memory only; permits GET retry without another paid POST
            result_cache["signature"] = signature
            result_cache["received_at"] = time.time()
    raw = result_bytes(result)
    repaired, metadata = normalize_image(raw, source)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Store image bytes for diagnosis, never the HTTP response or signed URL.
    (output_dir / "api_original.image").write_bytes(raw)
    output = output_dir / "repaired_rgb.png"
    repaired.save(output)
    metadata.update({"model": model, "request_size": size, "prompt": prompt,
                     "provider": "grsai" if model == "gpt-image-2" else "volcengine",
                     "source_sha256": hashlib.sha256(source_raw).hexdigest(),
                     "api_sha256": hashlib.sha256(raw).hexdigest(),
                     "repaired_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (output_dir / "image_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
