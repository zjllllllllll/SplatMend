"""Disposable actual HTTP server; simulates saved task state, never API/CUDA work."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from studio import image_api, oss_input
from studio.jobs import atomic_json, digest
from studio.lifecycle import process_identity
from studio.server import StudioServer


def forbidden(*args, **kwargs):
    raise AssertionError("Recovery fixture must never upload or generate images.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("image_api", "pipeline"))
    args = parser.parse_args()
    image_api.post_edit = forbidden
    oss_input.upload_png = forbidden
    server = StudioServer(("127.0.0.1", 0), args.root)
    job_id = "f" * 32
    folder = server.manager.jobs / job_id
    folder.mkdir()
    state = {"id": job_id, "status": args.mode, "stage": 0,
             "message": "Disposable local recovery fixture.", "created_at": 1,
             "model": "gpt-image-2", "server_process": process_identity(os.getpid())}
    if args.mode == "pipeline":
        (folder / "input").mkdir()
        (folder / "appearance").mkdir()
        Image.new("RGB", (16, 16), (1, 2, 3)).save(folder / "input" / "point_cloud.png")
        Image.new("RGB", (16, 16), (1, 2, 3)).save(folder / "appearance" / "repaired_rgb.png")
        atomic_json(folder / "request.json", {"model": "gpt-image-2", "prompt": "fixture"})
        atomic_json(folder / "appearance" / "image_manifest.json", {
            "model": "gpt-image-2", "prompt": "fixture",
            "source_sha256": digest(folder / "input" / "point_cloud.png")["sha256"],
            "repaired_sha256": digest(folder / "appearance" / "repaired_rgb.png")["sha256"]})
        # This worker imports only stdlib. The fixture itself is launched with
        # the checkout's authoritative run_cuda_python.cmd by the parent test.
        worker = subprocess.Popen([sys.executable, str(Path(__file__).with_name("recovery_worker.py")), str(folder)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state["pipeline_process"] = process_identity(worker.pid)
    atomic_json(folder / "status.json", state)
    server.manager.active = job_id
    server.manager.idle.clear()
    atomic_json(args.root / "fixture-ready.json", {"owner": process_identity(os.getpid()),
                "worker": state.get("pipeline_process"), "port": server.server_port, "job_id": job_id})
    server.serve_forever()


if __name__ == "__main__":
    main()
