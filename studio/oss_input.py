"""Private input-image upload, matching the user's Alibaba OSS reference.

No Torch, arbitrary destinations, bucket mutations, URL logs, or cloud deletes.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import timedelta
from urllib.parse import urlsplit
from uuid import uuid4

from studio.errors import ImageAPIError

OSS_BUCKET = "fengshuiimgs"
OSS_REGION = "cn-hangzhou"
OSS_PREFIX = "ai-enhance-input/gaussian-repair"
URL_TTL_SECONDS = 3600
INPUT_LIMIT = 32 * 1024 * 1024
CREDENTIAL_NAMES = ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_SESSION_TOKEN")


def _user_credentials() -> tuple[str, str, str]:
    if sys.platform != "win32":
        return ("", "", "")
    import winreg
    values = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
            for name in CREDENTIAL_NAMES:
                try:
                    value = winreg.QueryValueEx(registry, name)[0]
                    values.append(value.strip() if isinstance(value, str) else "")
                except OSError:
                    values.append("")
    except OSError:
        return ("", "", "")
    return tuple(values)


def _credentials() -> tuple[str, str, str]:
    # Select the entire credential set; never combine half of a process key
    # with a different Windows user's key or a stale STS session token.
    values = tuple(os.environ.get(name, "").strip() for name in CREDENTIAL_NAMES)
    if not any(values):
        values = _user_credentials()
    missing = [name for name, value in zip(CREDENTIAL_NAMES[:2], values[:2]) if not value]
    if missing:
        raise ImageAPIError("OSS input upload requires " + ", ".join(missing) + ". Configure them locally, not in the prompt.")
    if values[0].startswith("STS.") and not values[2]:
        raise ImageAPIError("Temporary OSS credentials also require OSS_SESSION_TOKEN.")
    if any(any(ord(char) < 33 or ord(char) > 126 for char in value) for value in values):
        raise ImageAPIError("OSS credentials contain invalid whitespace or characters.")
    return values


def _sdk():
    try:
        import alibabacloud_oss_v2 as oss
        return oss
    except ImportError:
        raise ImageAPIError("OSS SDK is missing. Install the pinned OSS dependencies listed in requirements.txt in the studio environment.") from None


def check_ready() -> None:
    """Local-only check. No network, no uploads, no credential output."""
    _credentials()
    _sdk()


def preflight() -> dict:
    try:
        check_ready()
        return {"ready": True, "message": "OSS upload configured (permissions checked on upload)."}
    except ImageAPIError as error:
        return {"ready": False, "message": str(error)}


def _client():
    credentials = _credentials()
    oss = _sdk()
    cfg = oss.config.load_default()
    cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(*credentials)
    cfg.region = OSS_REGION
    cfg.endpoint = f"https://oss-{OSS_REGION}.aliyuncs.com"
    cfg.connect_timeout = 10
    cfg.readwrite_timeout = 60
    # A timeout after PUT may already have created the private object. Do not
    # retry a forbid-overwrite PUT and mistake the resulting 409 for failure.
    cfg.retry_max_attempts = 1
    cfg.enabled_redirect = False
    return oss, oss.Client(cfg)


def upload_png(raw: bytes) -> str:
    """Return the signed HTTPS URL in memory only; upload once before POST."""
    if not raw.startswith(b"\x89PNG\r\n\x1a\n") or len(raw) > INPUT_LIMIT:
        raise ImageAPIError("OSS input must be a validated PNG no larger than 32 MiB.")
    oss, client = _client()
    key = f"{OSS_PREFIX}/{uuid4().hex}.png"
    try:
        client.put_object(oss.PutObjectRequest(
            bucket=OSS_BUCKET, key=key, body=raw, content_type="image/png",
            acl="private", forbid_overwrite=True,
        ))
        signed = client.presign(oss.GetObjectRequest(bucket=OSS_BUCKET, key=key),
                                expires=timedelta(seconds=URL_TTL_SECONDS))
        url = signed.url
        target = urlsplit(url)
        expected_host = f"{OSS_BUCKET}.oss-{OSS_REGION}.aliyuncs.com"
        if (target.scheme != "https" or target.hostname != expected_host
                or target.username or target.password or target.port not in (None, 443)
                or target.path != "/" + key or not target.query or target.fragment
                or any(ord(char) < 33 or ord(char) > 126 for char in url) or "\\" in url):
            raise ImageAPIError("OSS returned an unexpected signed URL; no image-generation request was sent.")
        return url
    except ImageAPIError:
        raise
    except Exception as error:
        # SDK exceptions can contain Authorization and complete signed URLs.
        # Expose only allowlisted structural fields, never str(error).
        code = getattr(error, "code", None)
        detail = f" ({code})" if isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", code) else ""
        raise ImageAPIError("OSS input upload/signing failed" + detail + "; no image-generation request was sent. Check credentials and scoped PutObject/GetObject permissions.") from None
