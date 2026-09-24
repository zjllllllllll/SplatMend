"""Loopback-only studio server. Only explicit public assets are served."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import mimetypes
import secrets
import shutil
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from studio.image_api import ImageAPIError, public_models, validate_prompt
from studio.jobs import JobManager, identifier, atomic_json
from studio.lifecycle import ServiceLease, StudioAlreadyRunning, process_alive

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_LIMIT = 1536 * 1024 * 1024
INPUT_UPLOADS = {"point_cloud.png": 32 * 1024 * 1024,
                 "point_cloud.depth.npy": 80 * 1024 * 1024, "deleted.bin": 10_000_000}


def workspace_id(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).casefold().encode("utf-8")).hexdigest()


def sample_available(root: Path) -> bool:
    assets = root / "assets"
    return all((assets / name).is_file() for name in ("point_cloud.ply", "point_cloud.camera.json"))


def existing_studio_url(root: Path) -> str:
    """Verify a live server from this checkout without opening an arbitrary URL."""
    try:
        path = root / "outputs" / "studio" / "server_instance.json"
        if path.stat().st_size > 4096:
            raise ValueError("Invalid server identity file.")
        record = json.loads(path.read_text(encoding="utf-8"))
        port = record["port"]
        if (record.get("workspace_id") != workspace_id(root)
                or type(port) is not int or not 1 <= port <= 65535
                or process_alive(record.get("process")) is not True):
            raise ValueError("Unconfirmed server identity.")
        # Always loopback, regardless of any hostname/URL inserted into the file.
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("GET", "/api/config")
            response = connection.getresponse()
            raw = response.read(128 * 1024 + 1)
            if response.status != 200 or len(raw) > 128 * 1024:
                raise ValueError("Unexpected local service response.")
            if json.loads(raw).get("workspace_id") != workspace_id(root):
                raise ValueError("Different local workspace.")
        finally:
            connection.close()
        return f"http://127.0.0.1:{port}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, http.client.HTTPException):
        raise RuntimeError("Studio is already running, but its address could not be verified. Use the existing window; no second server or task was started.") from None


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root: Path):
        self.lease = ServiceLease(root / "outputs" / "studio" / "server.lock")
        try:
            self.manager = JobManager(root)
            self.token = secrets.token_urlsafe(32)
            self.dist = root / "studio" / "web" / "dist"
            super().__init__(address, Handler)
            self.manager.recover_interrupted()
            atomic_json(self.manager.storage / "server_instance.json", {
                "workspace_id": workspace_id(root), "port": self.server_port,
                "process": self.manager.owner})
        except BaseException:
            self.lease.close()
            raise

    def server_close(self):
        if hasattr(self, "manager"):
            self.manager.stop_recovery.set()
        super().server_close()
        self.lease.close()


class Handler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, format, *args):
        # URL query strings and bodies never enter the application log.
        pass

    def _check_request(self, mutation=False):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed:
            raise PermissionError("Unrecognized Host header.")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            raise PermissionError("Cross-origin access is not permitted.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site access is not permitted.")
        if mutation and not secrets.compare_digest(self.headers.get("X-Studio-Token", ""), self.server.token):
            raise PermissionError("Refresh the local tool before submitting changes.")

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.end_headers()

    def _json(self, status, data):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(raw))
        self.wfile.write(raw)

    def _file(self, path: Path):
        if not path.is_file():
            self._json(404, {"error": "File is not available."})
            return
        kind = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                ".log": "text/plain; charset=utf-8", ".ply": "application/octet-stream"}.get(path.suffix)
        self._headers(200, kind or mimetypes.guess_type(path.name)[0] or "application/octet-stream", path.stat().st_size)
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def _body_length(self, limit):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked uploads are not supported.")
        length = int(self.headers.get("Content-Length", "-1"))
        if length < 0 or length > limit:
            raise ValueError("Upload size is missing or exceeds the limit.")
        return length

    def _body_json(self):
        size = self._body_length(256 * 1024)
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def _upload(self, path: Path, limit: int):
        length = self._body_length(limit)
        if shutil.disk_usage(path.parent).free < length + 1024 ** 3:
            raise ValueError("Insufficient free disk space for this upload.")
        temporary = path.with_suffix(path.suffix + ".uploading")
        # Exclusive creation prevents overlapping retries from corrupting an upload.
        with temporary.open("xb") as output:
            remaining = length
            self.connection.settimeout(90)
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("Upload was interrupted. Create a fresh job and retry.")
                output.write(block)
                remaining -= len(block)
        temporary.replace(path)

    def do_GET(self):
        try:
            self._check_request()
            path = unquote(urlsplit(self.path).path)
            segments = path.strip("/").split("/")
            manager = self.server.manager
            if path == "/api/config":
                prompt_path = manager.root / "prompt.txt"
                prompt = prompt_path.read_text(encoding="utf-8-sig") if prompt_path.is_file() else "Repair only the verified holes. Keep everything else unchanged."
                return self._json(200, {"token": self.server.token, "workspace_id": workspace_id(manager.root),
                                        "prompt": validate_prompt(prompt),
                                        "models": public_models(),
                                        "sample_available": sample_available(manager.root), "preflight": manager.preflight(), "active_job": manager.active})
            if len(segments) == 4 and segments[:2] == ["api", "scenes"] and segments[3] == "scene.ply":
                return self._file(manager.scenes / identifier(segments[2]) / "scene.ply")
            if len(segments) >= 3 and segments[:2] == ["api", "jobs"]:
                job_id = identifier(segments[2])
                if len(segments) == 3:
                    return self._json(200, manager.state(job_id))
                if len(segments) == 4:
                    if segments[3] == "view":
                        return self._json(200, manager.view(job_id))
                    if segments[3] == "result.ply":
                        return self._file(manager.result(job_id))
                    public_files = {"source.png": "input/point_cloud.png", "hole.png": "input/hole_preview.png",
                                    "repaired.png": "appearance/repaired_rgb.png", "pipeline.log": "pipeline.log",
                                    "corrected.png": "appearance/color_corrected_rgb.png",
                                    "deleted.bin": "input/deleted.bin",
                                    "acceptance.json": "pipeline/sample_acceptance.json"}
                    if segments[3] in public_files:
                        return self._file(manager.jobs / job_id / public_files[segments[3]])
            if path.startswith("/api/"):
                return self._json(404, {"error": "Unknown API route."})
            file = (self.server.dist / (path.lstrip("/") or "index.html")).resolve()
            if not file.is_relative_to(self.server.dist.resolve()) or any(p.startswith(".") for p in Path(path).parts):
                raise PermissionError("Access denied.")
            return self._file(file)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as error:
            self._error(error)

    def do_POST(self):
        try:
            self._check_request(mutation=True)
            parsed = urlsplit(self.path)
            segments = parsed.path.strip("/").split("/")
            manager = self.server.manager
            if parsed.path == "/api/sample":
                if not sample_available(manager.root):
                    raise ValueError("Local sample files are unavailable.")
                scene_id = uuid.uuid4().hex
                directory = manager.scenes / scene_id
                directory.mkdir()
                shutil.copyfile(manager.root / "assets" / "point_cloud.ply", directory / "upload.ply")
                sample = manager.register_scene(directory, "Sample · point_cloud.ply")
                sample["camera"] = json.loads((manager.root / "assets" / "point_cloud.camera.json").read_text(encoding="utf-8"))
                return self._json(201, sample)
            if parsed.path == "/api/scenes":
                scene_id = uuid.uuid4().hex
                directory = manager.scenes / scene_id
                directory.mkdir()
                self._upload(directory / "upload.ply", UPLOAD_LIMIT)
                name = parse_qs(parsed.query).get("name", ["scene.ply"])[0]
                return self._json(201, manager.register_scene(directory, name))
            if parsed.path == "/api/jobs":
                return self._json(201, manager.create(self._body_json()))
            if len(segments) == 4 and segments[:2] == ["api", "jobs"] and segments[3] == "start":
                return self._json(202, manager.submit(identifier(segments[2])))
            if len(segments) == 4 and segments[:2] == ["api", "jobs"] and segments[3] == "retry-download":
                return self._json(202, manager.retry_download(identifier(segments[2])))
            if len(segments) == 4 and segments[:2] == ["api", "jobs"] and segments[3] == "retry-pipeline":
                return self._json(202, manager.retry_pipeline(identifier(segments[2])))
            if len(segments) == 5 and segments[:2] == ["api", "jobs"] and segments[3] == "input":
                job_id, filename = identifier(segments[2]), segments[4]
                # Hold the state lock so start cannot race with an unfinished upload.
                with manager.lock:
                    if manager.state(job_id)["status"] != "uploading" or filename not in INPUT_UPLOADS:
                        raise ValueError("This input cannot be uploaded now.")
                    self._upload(manager.jobs / job_id / "input" / filename, INPUT_UPLOADS[filename])
                return self._json(200, {"saved": filename})
            return self._json(404, {"error": "Unknown API route."})
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as error:
            self._error(error)

    def _error(self, error):
        # Do not expose arbitrary exception text (URLs, paths, HTTP responses).
        if isinstance(error, PermissionError):
            code, message = 403, "Local request authentication failed. Refresh the tool."
        elif isinstance(error, (ValueError, ImageAPIError)):
            code, message = 400, str(error)[:1000]
        else:
            code, message = 500, f"Local operation failed ({type(error).__name__})."
        self.close_connection = True
        self._json(code, {"error": message})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = StudioServer(("127.0.0.1", args.port), ROOT)
    except StudioAlreadyRunning:
        try:
            url = existing_studio_url(ROOT)
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
        print(f"Existing Gaussian Repair Studio: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    if not (server.dist / "index.html").is_file():
        raise SystemExit("Frontend is not built. Run build_studio.cmd first.")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Gaussian Repair Studio: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if server.manager.active:
            print("Finishing the current task before stopping. Press Ctrl+C again to stop waiting; inputs and process identity remain saved.", flush=True)
            try:
                while not server.manager.idle.wait(.5):
                    pass
            except KeyboardInterrupt:
                pass
        print("Server stopped. Existing input files and diagnostics remain under outputs/studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
