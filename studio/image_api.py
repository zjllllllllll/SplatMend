"""Company image-edit gateway. No Torch, browser secrets, or request-body logs."""

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
from studio import oss_input

ENDPOINT_HOST = "oneapi.qunhequnhe.com"
ENDPOINT_PATH = "/v1/images/edits"
IMAGE_LIMIT = 32 * 1024 * 1024
# GPT/Gemini may return base64 despite response_format=url. Bound that JSON too.
JSON_LIMIT = 48 * 1024 * 1024
PIXEL_LIMIT = 32_000_000
MODELS = {
    "doubao-seedream-5-0-260128": {
        "label": "Doubao · Seedream 5.0",
        "input_transport": "data_url",
        "response_format": "url",
        "sizes": [(2848, 1600), (2304, 1728), (2048, 2048), (1728, 2304), (1600, 2848)],
    },
    "gpt-image-2": {
        "label": "GPT Image 2",
        "input_transport": "oss_url",
        "response_format": "url",
        "sizes": [(2048, 1152), (2048, 1536), (2048, 2048), (1536, 2048), (1152, 2048)],
    },
    "gemini-3-pro-image-preview": {
        "label": "Gemini 3 Pro · Image",
        "input_transport": "oss_url",
        "response_format": "url",
        "sizes": [(2752, 1536), (2400, 1792), (2048, 2048), (1792, 2400), (1536, 2752)],
    },
}


class ImageDownloadError(ImageAPIError):
    """Generation finished; retrying the download never creates another image."""


def read_key(root: Path) -> str:
    path = root / "api_key.txt"
    # The user explicitly supplies this checkout's key. A stale credential in
    # an already-running desktop app must not silently override that file.
    if path.is_file():
        if path.stat().st_size > 4096:
            raise ImageAPIError("api_key.txt is unexpectedly large.")
        key = path.read_text(encoding="utf-8-sig").strip()
    else:
        key = os.environ.get("CLICKGS_IMAGE_API_KEY", "").strip()
    if not key:
        raise ImageAPIError("Add the API key to api_key.txt or CLICKGS_IMAGE_API_KEY.")
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
    return f"{w}x{h}"


def model_preflight(model: str) -> None:
    if model not in MODELS:
        raise ImageAPIError("Choose a supported image model; no automatic fallback is used.")
    if MODELS[model]["input_transport"] == "oss_url":
        oss_input.check_ready()


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
        for secret in (os.environ.get(name, "") for name in oss_input.CREDENTIAL_NAMES):
            if secret:
                message = message.replace(secret, "[credential redacted]")
        message = re.sub(r"(?i)Bearer\s+\S+|sk-[A-Za-z0-9_-]+", "[credential redacted]", message)
        message = re.sub(r"(?i)data:image/[^\s\"']+|https?://[^\s\"']+", "[URL/data redacted]", message)
        message = re.sub(r"[A-Za-z0-9+/=_-]{48,}", "[long token redacted]", message)
        message = " ".join(message.split())[:400]
        return f" Gateway detail: {message}" if message else ""
    except (ImageAPIError, ValueError, OSError, http.client.HTTPException, RecursionError):
        return ""


def post_edit(body: dict, key: str, *, connection_factory=None, sleep=time.sleep) -> dict:
    factory = connection_factory or http.client.HTTPSConnection
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(3):
        connection = factory(ENDPOINT_HOST, timeout=10)
        delay = None
        try:
            connection.connect()
            connection.sock.settimeout(180)
            connection.request("POST", ENDPOINT_PATH, payload, {
                "Content-Type": "application/json", "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            })
            response = connection.getresponse()
            if response.status in (429, 502, 503, 504) and attempt < 2:
                delay = _retry_delay(response.getheader("Retry-After"), attempt)
            elif response.status != 200:
                # Never include response JSON, request headers, or signed URLs.
                detail = _error_summary(response, key)
                raise ImageAPIError(f"Image API returned HTTP {response.status}; model was not changed.{detail}")
            else:
                raw = _bounded_read(response, JSON_LIMIT)
                try:
                    result = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise ImageAPIError("Image API returned invalid JSON.") from None
                if not isinstance(result, dict):
                    raise ImageAPIError("Image API returned an invalid result object.")
                return result
        except (OSError, http.client.HTTPException):
            raise ImageAPIError(
                "Image API connection interrupted or timed out. It may have been billed; "
                "the request was not automatically repeated."
            ) from None
        finally:
            connection.close()
        if delay is not None:
            sleep(delay)
    raise ImageAPIError("Image API retry limit reached.")


def _public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not any((address.is_multicast, address.is_reserved,
                                         address.is_loopback, address.is_link_local,
                                         address.is_unspecified))


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
        if not addresses or not all(_public_ip(ip) for ip in addresses):
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
        if peer not in addresses or not _public_ip(peer):
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
    data = result.get("data")
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
    body = {
        "model": model, "prompt": validate_prompt(prompt),
        "n": 1, "size": model_size(model, *source.size),
        "response_format": MODELS[model]["response_format"], "output_format": "png",
    }
    result = result_cache.get("result") if result_cache is not None else None
    if result is not None and result_cache.get("signature") != signature:
        raise ImageAPIError("The locked input changed after generation; the cached result cannot be reused.")
    if result is None:
        model_preflight(model)
        transport = MODELS[model]["input_transport"]
        if transport == "oss_url":
            # The alpha mask belongs to the geometric pipeline. Give the image
            # gateway a conventional RGB PNG, keeping every RGB channel and
            # pixel coordinate unchanged. Never resize/crop the locked inputs.
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format="PNG")
            input_raw = buffer.getvalue()
            if len(input_raw) > IMAGE_LIMIT:
                raise ImageAPIError("Encoded API input exceeds the local 32 MiB safety limit.")
        else:
            input_raw = source_raw
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "api_input.png").write_bytes(input_raw)
        info = {"model": model, "input_transport": transport, "input_size": list(source.size),
                "request_size": body["size"], "response_format": body["response_format"],
                "output_format": "png", "n": 1, "input_bytes": len(input_raw),
                "input_sha256": hashlib.sha256(input_raw).hexdigest()}
        (output_dir / "api_request_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        image_url = (oss_input.upload_png(input_raw) if transport == "oss_url" else
                     "data:image/png;base64," + base64.b64encode(input_raw).decode("ascii"))
        body["images"] = [{"image_url": image_url}]
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
    metadata.update({"model": model, "request_size": body["size"], "prompt": prompt,
                     "input_transport": MODELS[model]["input_transport"],
                     "response_format": body["response_format"],
                     "source_sha256": hashlib.sha256(source_raw).hexdigest(),
                     "api_sha256": hashlib.sha256(raw).hexdigest(),
                     "repaired_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (output_dir / "image_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
