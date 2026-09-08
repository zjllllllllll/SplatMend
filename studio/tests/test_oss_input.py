import io
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock, patch

from PIL import Image

from studio import oss_input as api
from studio.errors import ImageAPIError


class OSSTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"OSS_ACCESS_KEY_ID": "TEST_ID", "OSS_ACCESS_KEY_SECRET": "TEST_SECRET", "OSS_SESSION_TOKEN": ""})
        env.start()
        self.addCleanup(env.stop)
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        self.raw = buffer.getvalue()

    def test_missing_and_partial_credentials_fail_closed(self):
        for values in [{name: "" for name in api.CREDENTIAL_NAMES},
                       {"OSS_ACCESS_KEY_ID": "STS.TEST", "OSS_ACCESS_KEY_SECRET": "TEST_SECRET", "OSS_SESSION_TOKEN": ""},
                       {"OSS_ACCESS_KEY_ID": "TEST_ID", "OSS_ACCESS_KEY_SECRET": "", "OSS_SESSION_TOKEN": ""}]:
            with patch.dict("os.environ", values), patch.object(api, "_user_credentials", return_value=("", "", "")):
                with self.assertRaises(ImageAPIError):
                    api._credentials()

    def test_user_credentials_are_loaded_as_a_set(self):
        with patch.dict("os.environ", {name: "" for name in api.CREDENTIAL_NAMES}), \
                patch.object(api, "_user_credentials", return_value=("USER_ID", "USER_SECRET", "")):
            self.assertEqual(api._credentials(), ("USER_ID", "USER_SECRET", ""))

    def test_real_sdk_presigns_only_private_https_put_is_mocked(self):
        # Real SDK model/credential/config constructors, no actual network calls.
        oss, client = api._client()
        with patch.object(api, "_client", return_value=(oss, client)), patch.object(client, "put_object") as put:
            url = api.upload_png(self.raw)
        request = put.call_args.args[0]
        self.assertEqual(request.body, self.raw)
        self.assertEqual(request.bucket, api.OSS_BUCKET)
        self.assertEqual(request.acl, "private")
        self.assertTrue(request.forbid_overwrite)
        self.assertEqual(request.content_type, "image/png")
        self.assertTrue(url.startswith(f"https://{api.OSS_BUCKET}.oss-{api.OSS_REGION}.aliyuncs.com/{api.OSS_PREFIX}/"))
        # SDK rounds the two timestamps separately; one second may elapse.
        ttl = int(parse_qs(urlsplit(url).query)["x-oss-expires"][0])
        self.assertGreaterEqual(ttl, 3598)
        self.assertLessEqual(ttl, 3600)
        self.assertNotIn("TEST_SECRET", url)

    def test_invalid_bytes_do_not_even_create_client(self):
        with patch.object(api, "_client") as client:
            for raw in (b"JPEG", b"", b"\x89PNG\r\n\x1a\n" + b"x" * api.INPUT_LIMIT):
                with self.assertRaises(ImageAPIError):
                    api.upload_png(raw)
            client.assert_not_called()

    def test_sdk_failure_never_leaks_credentials_or_retries(self):
        client = Mock()
        error = RuntimeError("TEST_SECRET Authorization https://example.com/?sig=PRIVATE")
        error.code = "AccessDenied"
        client.put_object.side_effect = error
        with patch.object(api, "_client", return_value=(Mock(), client)):
            with self.assertRaises(ImageAPIError) as caught:
                api.upload_png(self.raw)
        text = str(caught.exception)
        self.assertIn("AccessDenied", text)
        for forbidden in ("TEST_SECRET", "Authorization", "PRIVATE", "https://"):
            self.assertNotIn(forbidden, text)
        client.put_object.assert_called_once()
        client.presign.assert_not_called()

    def test_unexpected_signed_host_protocol_or_path_is_rejected(self):
        client = Mock()
        for url in ["http://example.com/x", "https://127.0.0.1/x", "https://evil.example/x",
                    f"https://{api.OSS_BUCKET}.oss-{api.OSS_REGION}.aliyuncs.com/wrong?sig=x"]:
            client.presign.return_value = SimpleNamespace(url=url)
            with patch.object(api, "_client", return_value=(Mock(), client)), self.assertRaises(ImageAPIError):
                api.upload_png(self.raw)


if __name__ == "__main__":
    unittest.main()
