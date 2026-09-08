"""Real local server crashes + surviving child processes. No models/API calls."""
import http.client
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

from studio.lifecycle import process_alive
from studio.server import StudioServer

ROOT = Path(__file__).resolve().parents[2]


def wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("Local recovery fixture did not reach the expected state.")


class RecoveryIntegrationTests(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory(prefix="studio_recovery_test_", dir=ROOT / "outputs") as directory:
            root = Path(directory)
            launcher = subprocess.Popen([str(ROOT / "run_cuda_python.cmd"), "-m", "studio.tests.recovery_fixture", str(root), mode],
                                        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            ready = None
            server = None
            try:
                wait_until(lambda: (root / "fixture-ready.json").is_file() or launcher.poll() is not None)
                if launcher.poll() is not None:
                    self.fail("Recovery fixture launcher exited before ready: " + launcher.stdout.read(4000).decode("utf-8", errors="replace"))
                ready = json.loads((root / "fixture-ready.json").read_text())
                self.assertTrue(process_alive(ready["owner"]))
                client = http.client.HTTPConnection("127.0.0.1", ready["port"], timeout=5)
                try:
                    client.request("GET", "/api/config")
                    response = client.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read())["active_job"], ready["job_id"])
                finally:
                    client.close()
                # Terminate ONLY the exact disposable server identity returned
                # above, never the real Studio server or an arbitrary process.
                psutil.Process(ready["owner"]["pid"]).kill()
                wait_until(lambda: process_alive(ready["owner"]) is False)
                launcher.wait(timeout=5)
                if ready["worker"]:
                    self.assertTrue(process_alive(ready["worker"]))
                with patch("studio.jobs.repair_image") as repair, patch("studio.jobs.subprocess.Popen") as spawn:
                    server = StudioServer(("127.0.0.1", 0), root)
                    folder = server.manager.jobs / ready["job_id"]
                    if mode == "pipeline":
                        self.assertEqual(server.manager.state(ready["job_id"])["status"], "recovering")
                        self.assertEqual(server.manager.active, ready["job_id"])
                        # A local synchronization marker lets this owned inert
                        # child finish normally; no real pipeline is terminated.
                        (folder / "release-worker").touch()
                        wait_until(lambda: process_alive(ready["worker"]) is False)
                        wait_until(lambda: server.manager.state(ready["job_id"])["status"] == "failed")
                    state = server.manager.state(ready["job_id"])
                    self.assertEqual(state["status"], "failed")
                    self.assertEqual(state["can_retry_pipeline"], mode == "pipeline")
                    self.assertFalse(state["can_retry_download"])
                    self.assertFalse((folder / "studio_success.json").exists())
                    self.assertIsNone(server.manager.active)
                    repair.assert_not_called()
                    spawn.assert_not_called()
                    with self.assertRaises(ValueError):
                        server.manager.result(ready["job_id"])
            finally:
                if server:
                    server.server_close()
                if ready:
                    for identity in (ready["owner"], ready["worker"]):
                        if identity and process_alive(identity) is True:
                            process = psutil.Process(identity["pid"])
                            process.kill()
                            process.wait(timeout=5)
                if launcher.poll() is None:
                    launcher.terminate()
                    launcher.wait(timeout=5)
                launcher.stdout.close()

    def test_crash_during_image_request_never_resubmits(self):
        self.exercise("image_api")

    def test_crash_with_surviving_pipeline_waits_then_offers_image_reuse(self):
        self.exercise("pipeline")
