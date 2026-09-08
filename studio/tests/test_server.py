import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.server import StudioServer, existing_studio_url, workspace_id


class ServerTests(unittest.TestCase):
    def setUp(self):
        outputs = Path(__file__).resolve().parents[2] / "outputs"
        outputs.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="studio_http_test_", dir=outputs)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        dist = self.root / "studio" / "web" / "dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text("<html>local viewer</html>")
        (self.root / "api_key.txt").write_text("SECRET_MUST_NOT_BE_SERVED")
        self.server = StudioServer(("127.0.0.1", 0), self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method="GET", path="/", body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_config_and_static_content_never_expose_key(self):
        code, headers, raw = self.request(path="/api/config")
        self.assertEqual(code, 200)
        config = json.loads(raw)
        self.assertEqual({item["id"] for item in config["models"]}, {"doubao-seedream-5-0-260128", "gpt-image-2"})
        self.assertTrue(config["preflight"]["key_ready"])
        self.assertNotIn(b"SECRET_MUST_NOT_BE_SERVED", raw)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.request()[0], 200)

    def test_host_origin_and_mutation_token_enforced(self):
        for headers in [{"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}]:
            self.assertEqual(self.request(path="/api/config", headers=headers)[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", b"{}")[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", b"{}", {"X-Studio-Token": "incorrect"})[0], 403)

    def test_traversal_private_files_and_oversized_json_rejected(self):
        for path in ["/api_key.txt", "/../../api_key.txt", "/%2e%2e/%2e%2e/api_key.txt", "/.git/config", "/api/jobs/../../api_key.txt"]:
            code, _, raw = self.request(path=path)
            self.assertNotEqual(code, 200, path)
            self.assertNotIn(b"SECRET_MUST_NOT_BE_SERVED", raw)
        code, _, _ = self.request("POST", "/api/jobs", b"", {
            "X-Studio-Token": self.server.token, "Content-Length": str(256 * 1024 + 1)})
        self.assertEqual(code, 400)

    def test_read_job_and_view_never_start_or_retry_work(self):
        job_id = "a" * 32
        with patch.object(self.server.manager, "state", return_value={"id": job_id, "status": "pipeline"}), \
             patch.object(self.server.manager, "view", return_value={"model": "gpt-image-2"}), \
             patch.object(self.server.manager, "submit") as submit, \
             patch.object(self.server.manager, "retry_download") as retry:
            self.assertEqual(json.loads(self.request(path=f"/api/jobs/{job_id}")[2])["id"], job_id)
            self.assertEqual(json.loads(self.request(path=f"/api/jobs/{job_id}/view")[2])["model"], "gpt-image-2")
        submit.assert_not_called()
        retry.assert_not_called()

    def test_repeated_launcher_uses_verified_existing_server_without_starting_work(self):
        with patch.object(self.server.manager, "submit") as submit, patch.object(self.server.manager, "retry_pipeline") as retry:
            self.assertEqual(existing_studio_url(self.root), f"http://127.0.0.1:{self.server.server_port}")
        submit.assert_not_called()
        retry.assert_not_called()
        path = self.root / "outputs" / "studio" / "server_instance.json"
        record = json.loads(path.read_text())
        self.assertEqual(record["workspace_id"], workspace_id(self.root))
        self.assertNotIn("SECRET_MUST_NOT_BE_SERVED", path.read_text())
        self.assertNotIn(self.server.token, path.read_text())

    def test_existing_server_identity_and_port_must_not_be_guessed(self):
        path = self.root / "outputs" / "studio" / "server_instance.json"
        original = json.loads(path.read_text())
        for changed in [{**original, "workspace_id": "different"}, {**original, "port": "8765"},
                        {**original, "port": 0}, {**original, "process": {"pid": 0}}]:
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "could not be verified"):
                existing_studio_url(self.root)
        path.write_text(json.dumps(original))
        with patch("studio.server.process_alive", return_value=None), self.assertRaises(RuntimeError):
            existing_studio_url(self.root)


if __name__ == "__main__":
    unittest.main()
