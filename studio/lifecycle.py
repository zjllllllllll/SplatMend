"""Windows service ownership and read-only process checks (no CUDA imports)."""
from __future__ import annotations

import os
from pathlib import Path

import psutil


class StudioAlreadyRunning(RuntimeError):
    pass


class ServiceLease:
    """One server per checkout, including servers launched on different ports."""

    def __init__(self, path: Path):
        import msvcrt
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        try:
            if path.stat().st_size == 0:
                self.handle.write(b"\0")
                self.handle.flush()
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            self.handle = None
            raise StudioAlreadyRunning("This checkout already has a Studio server. Use its existing browser window.") from None

    def close(self):
        if self.handle is not None:
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()
            self.handle = None


def process_identity(pid: int) -> dict | None:
    if type(pid) is not int or pid <= 0:
        return None
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "created": process.create_time()}
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied:
        return {"pid": pid, "created": None}


def process_alive(identity: dict) -> bool | None:
    """None means unobservable, never terminal. Compare creation time against PID reuse."""
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int:
        return None
    try:
        process = psutil.Process(identity["pid"])
        if identity.get("created") is None:
            return None
        if abs(process.create_time() - identity["created"]) > .001:
            return False
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, TypeError):
        return None


def matching_job_processes(folder: Path) -> tuple[list[dict], bool]:
    """Detect a surviving launcher/child without printing unrelated command lines."""
    needle = str(folder.resolve()).replace("/", "\\").casefold()
    identities = []
    uncertain = False
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid == os.getpid() or (process.info.get("name") or "").lower() not in (
                "cmd.exe", "python.exe", "pythonw.exe"):
            continue
        try:
            command = " ".join(process.cmdline()).replace("/", "\\").casefold()
            if needle in command:
                identities.append({"pid": process.pid, "created": process.create_time()})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            uncertain = True
    return identities, uncertain
