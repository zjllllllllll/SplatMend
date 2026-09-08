import base64
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from studio import image_api as api


def png(size=(100, 100), color=(40, 80, 120, 255)):
    buffer = io.BytesIO()
    Image.new("RGBA", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class Response:
    def __init__(self, status=200, body=b'{}', headers=None):
        self.status = status
        self.stream = io.BytesIO(body)
        self.headers = headers or {}

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, size=-1):
        return self.stream.read(size)


class Connection:
    def __init__(self, response):
        self.response = response
        self.sock = Mock()
        self.request = Mock()
        self.close = Mock()

    def connect(self):
        pass

    def getresponse(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ImageTests(unittest.TestCase):
    def test_size_normalization_inclusive_boundary(self):
        source = Image.new("RGB", (100, 100), (1, 2, 3))
        for size in [(100, 100), (200, 200), (101, 100), (102, 100)]:
            with self.subTest(size=size):
                image, info = api.normalize_image(png(size), source)
                self.assertEqual(image.size, source.size)
                self.assertLessEqual(info["ratio_error"], .02)

    def test_wrong_ratios_fail(self):
        source = Image.new("RGB", (100, 100))
        for size in [(103, 100), (400, 300), (160, 90)]:
            with self.subTest(size=size), self.assertRaises(api.ImageAPIError):
                api.normalize_image(png(size), source)

    def test_alpha_preserves_original(self):
        source = Image.new("RGB", (100, 100), (1, 2, 3))
        image, _ = api.normalize_image(png(color=(200, 200, 200, 0)), source)
        self.assertEqual(image.tobytes(), source.tobytes())

    def test_exif_orientation(self):
        raw = io.BytesIO()
        image = Image.new("RGB", (200, 100))
        exif = image.getexif()
        exif[274] = 6
        image.save(raw, format="JPEG", exif=exif)
        self.assertEqual(api.decode_image(raw.getvalue()).size, (100, 200))

    def test_corrupt_truncated_and_oversized_images(self):
        for raw in [b"", b"not an image", png()[:30], png()[:-30], png((8193, 1))]:
            with self.subTest(length=len(raw)), self.assertRaises(api.ImageAPIError):
                api.decode_image(raw)
        with patch.object(api, "PIXEL_LIMIT", 100), self.assertRaises(api.ImageAPIError):
            api.decode_image(png())

    def test_unicode_prompt_is_allowed(self):
        self.assertEqual(api.validate_prompt("  修复地面 real‑world  "), "修复地面 real‑world")
        for value in [" ", None, "x" * 16001, "x\x00"]:
            with self.assertRaises(api.ImageAPIError):
                api.validate_prompt(value)

    def test_all_models_support_view_sizes(self):
        for model in api.MODELS:
            for width, height in [(2560, 1440), (2048, 1536), (2048, 2048)]:
                self.assertIn("x", api.model_size(model, width, height))
        with self.assertRaises(api.ImageAPIError):
            api.model_size("not-a-model", 100, 100)

    def test_exact_per_model_2k_sizes_from_company_table(self):
        expected = {"doubao-seedream-5-0-260128": ["2848x1600", "2304x1728", "2048x2048"],
                    "gpt-image-2": ["2048x1152", "2048x1536", "2048x2048"],
                    "gemini-3-pro-image-preview": ["2752x1536", "2400x1792", "2048x2048"]}
        self.assertEqual(set(api.MODELS), set(expected))
        for model, sizes in expected.items():
            actual = [api.model_size(model, *size) for size in [(2560, 1440), (2048, 1536), (2048, 2048)]]
            self.assertEqual(actual, sizes)

    def test_checkout_key_wins_over_stale_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "api_key.txt").write_text("FILE_KEY", encoding="utf-8")
            with patch.dict("os.environ", {"CLICKGS_IMAGE_API_KEY": "STALE_KEY"}):
                self.assertEqual(api.read_key(root), "FILE_KEY")
                (root / "api_key.txt").write_text("", encoding="utf-8")
                with self.assertRaises(api.ImageAPIError):
                    api.read_key(root)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        ready = patch.object(api.oss_input, "check_ready")
        ready.start()
        self.addCleanup(ready.stop)
        upload = patch.object(api.oss_input, "upload_png", return_value="https://example.com/input.png?sign=PRIVATE_INPUT")
        self.upload = upload.start()
        self.addCleanup(upload.stop)

    def post(self, responses):
        connections = [Connection(response) for response in responses]
        factory = Mock(side_effect=connections)
        sleeper = Mock()
        result = api.post_edit({"model": "test"}, "TEST_SECRET", connection_factory=factory, sleep=sleeper)
        return result, connections, sleeper

    def test_post_success(self):
        result, connections, _ = self.post([Response(body=b'{"data":[{"url":"https://example.com/image.png"}]}')])
        self.assertEqual(len(result["data"]), 1)
        request = connections[0].request.call_args.args
        self.assertEqual(request[0:2], ("POST", api.ENDPOINT_PATH))
        self.assertEqual(request[3]["Authorization"], "Bearer TEST_SECRET")

    def test_retry_only_explicit_transient_statuses(self):
        for code in (429, 502, 503, 504):
            _, connections, sleep = self.post([Response(code), Response(code), Response()])
            self.assertEqual(len(connections), 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 5])
        _, _, sleep = self.post([Response(429, headers={"Retry-After": "600"}), Response()])
        sleep.assert_called_once_with(30)

    def test_permanent_http_failures_do_not_retry(self):
        for code in (400, 401, 403, 500, 302):
            connection = Connection(Response(code))
            factory = Mock(return_value=connection)
            with self.assertRaises(api.ImageAPIError):
                api.post_edit({}, "TEST_SECRET", connection_factory=factory)
            self.assertEqual(factory.call_count, 1)

    def test_gateway_error_details_are_bounded_and_redacted(self):
        body = {"error": {"code": "unsupported_size", "type": "upstream_error",
                          "message": "Bad size. TEST_SECRET Bearer hidden sk-hidden https://host/private?sig=hidden data:image/png;base64," + "A" * 100}}
        factory = Mock(return_value=Connection(Response(500, json.dumps(body).encode())))
        with self.assertRaises(api.ImageAPIError) as caught:
            api.post_edit({}, "TEST_SECRET", connection_factory=factory)
        text = str(caught.exception)
        self.assertIn("unsupported_size", text)
        self.assertIn("Bad size", text)
        for forbidden in ("TEST_SECRET", "Bearer hidden", "sk-hidden", "sig=hidden", "A" * 100):
            self.assertNotIn(forbidden, text)
        factory.assert_called_once()

    def test_gateway_error_string_and_top_level_message(self):
        for body in [{"error": "upstream unavailable"}, {"message": "upstream unavailable"}]:
            message = api._error_summary(Response(body=json.dumps(body).encode()), "SECRET")
            self.assertIn("upstream unavailable", message)
        self.assertIn("non-JSON", api._error_summary(Response(body=b"<html>Bad Gateway</html>"), "SECRET"))

    def test_model_specific_inputs_and_formats_match_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(png())
            for model in api.MODELS:
                result = {"data": [{"b64_json": base64.b64encode(png()).decode()}]}
                with patch.object(api, "post_edit", return_value=result) as post:
                    api.repair_image(source, root / model, model, "repair", "SECRET")
                body = post.call_args.args[0]
                self.assertEqual(body["model"], model)
                self.assertEqual(body["response_format"], "url")
                if model == "doubao-seedream-5-0-260128":
                    self.assertTrue(body["images"][0]["image_url"].startswith("data:image/png;base64,"))
                else:
                    self.assertEqual(body["images"][0]["image_url"], self.upload.return_value)
                    with Image.open(io.BytesIO(self.upload.call_args.args[0])) as image:
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, (100, 100))
                        self.assertEqual(image.getpixel((0, 0)), (40, 80, 120))
                for record in (root / model).glob("*.json"):
                    self.assertNotIn("PRIVATE_INPUT", record.read_text())

    def test_bad_input_or_oss_failure_never_calls_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            source.write_bytes(png((160, 100)))
            with patch.object(api, "post_edit") as post, self.assertRaises(api.ImageAPIError):
                api.repair_image(source, root / "out", "gpt-image-2", "repair", "SECRET")
            post.assert_not_called()
            self.upload.assert_not_called()
            source.write_bytes(png())
            self.upload.side_effect = api.ImageAPIError("OSS unavailable")
            with patch.object(api, "post_edit") as post, self.assertRaises(api.ImageAPIError):
                api.repair_image(source, root / "out", "gpt-image-2", "repair", "SECRET")
            post.assert_not_called()

    def test_large_error_body_retains_redacted_upstream_reason(self):
        data = {"error": {"message": "TencentVOD InvalidParameterValue FileId data:image/png;base64," + "A" * 100000}}
        text = api._error_summary(Response(body=json.dumps(data).encode(), headers={"Content-Length": "100200"}), "SECRET")
        self.assertIn("InvalidParameterValue", text)
        self.assertNotIn("A" * 100, text)

    def test_ambiguous_post_failures_do_not_retry_or_leak(self):
        for error in (socket.timeout("SECRET_URL"), ConnectionResetError("SECRET_URL"), socket.gaierror("SECRET_URL")):
            factory = Mock(return_value=Connection(error))
            with self.assertRaises(api.ImageAPIError) as caught:
                api.post_edit({}, "TEST_SECRET", connection_factory=factory)
            self.assertEqual(factory.call_count, 1)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_retry_limit(self):
        factory = Mock(side_effect=[Connection(Response(503)) for _ in range(3)])
        with self.assertRaises(api.ImageAPIError):
            api.post_edit({}, "TEST_SECRET", connection_factory=factory, sleep=Mock())
        self.assertEqual(factory.call_count, 3)

    def test_bad_json_and_bounded_responses(self):
        for body in [b"not-json", b"[]"]:
            with self.assertRaises(api.ImageAPIError):
                self.post([Response(body=body)])
        with self.assertRaises(api.ImageAPIError):
            api._bounded_read(Response(body=b"123456"), 5)
        with self.assertRaises(api.ImageAPIError):
            api._bounded_read(Response(headers={"Content-Length": "100"}), 5)
        with self.assertRaises(api.ImageAPIError):
            api._bounded_read(Response(body=b"1", headers={"Content-Length": "2"}), 5)

    def test_results_url_and_b64(self):
        raw = png()
        download = Mock(return_value=raw)
        self.assertEqual(api.result_bytes({"data": [{"url": "https://example.com/img"}]}, download=download), raw)
        for prefix in ["", "data:image/png;base64,"]:
            result = {"data": [{"b64_json": prefix + base64.b64encode(raw).decode()}]}
            self.assertEqual(api.result_bytes(result), raw)
        for result in [{}, {"data": []}, {"data": [{}]}, {"data": [{"b64_json": "invalid!!"}]}]:
            with self.assertRaises(api.ImageAPIError):
                api.result_bytes(result)

    def test_public_addresses_only(self):
        for ip in ["127.0.0.1", "10.1.2.3", "169.254.169.254", "0.0.0.0", "::1", "::ffff:127.0.0.1", "224.0.0.1", "192.0.2.1"]:
            self.assertFalse(api._public_ip(ip), ip)
        for url in ["file:///image", "https://user:password@host/image", "http://host:8080/image", "https://host\\internal/x"]:
            with self.assertRaises(api.ImageAPIError):
                api.public_target(url)
        with patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.1", 443))]):
            with self.assertRaises(api.ImageAPIError):
                api.public_target("https://download.example/image")

    def test_peer_is_checked(self):
        fake = Mock()
        fake.getpeername.return_value = ("127.0.0.1", 443)
        with patch.object(socket, "create_connection", return_value=fake), self.assertRaises(api.ImageAPIError):
            api._pinned_connection("https", "example.com", 443, ["8.8.8.8"])
        fake.close.assert_called_once()

    def test_download_has_no_authorization(self):
        connection = Connection(Response(body=png()))
        with patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            self.assertEqual(api.download_image("https://download.example/x", connect=Mock(return_value=connection)), png())
        self.assertNotIn("Authorization", connection.request.call_args.kwargs["headers"])

    def test_http_cdn_upgrade_keeps_path_and_signature(self):
        connection = Connection(Response(body=png()))
        connect = Mock(return_value=connection)
        with patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            api.download_image("http://download.example/image.png?sign=PRIVATE", connect=connect)
        self.assertEqual(connect.call_args.args[:3], ("https", "download.example", 443))
        self.assertEqual(connection.request.call_args.args, ("GET", "/image.png?sign=PRIVATE"))
        self.assertNotIn("Authorization", connection.request.call_args.kwargs["headers"])

    def test_https_host_header_has_no_erroneous_port(self):
        raw = Mock()
        raw.getpeername.return_value = ("8.8.8.8", 443)
        tls = Mock()
        context = Mock()
        context.wrap_socket.return_value = tls
        with patch.object(socket, "create_connection", return_value=raw), patch.object(api.ssl, "create_default_context", return_value=context):
            conn = api._pinned_connection("https", "download.example", 443, ["8.8.8.8"])
            conn.request("GET", "/test")
        request = b"".join(call.args[0] for call in tls.sendall.call_args_list)
        self.assertIn(b"Host: download.example\r\n", request)
        self.assertNotIn(b"download.example:443", request)
        context.wrap_socket.assert_called_once_with(raw, server_hostname="download.example")

    def test_download_error_omits_signed_path(self):
        conn = Connection(Response(403))
        with patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            with self.assertRaises(api.ImageDownloadError) as caught:
                api.download_image("https://download.example/SECRET?sign=SECRET", connect=Mock(return_value=conn))
        self.assertNotIn("SECRET", str(caught.exception))

    def test_https_downgrade_and_excess_redirects_rejected(self):
        with patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            with self.assertRaisesRegex(api.ImageAPIError, "downgrade"):
                api.download_image("https://example.com/x", connect=Mock(return_value=Connection(Response(302, headers={"Location": "http://example.com/x"}))))
            connect = Mock(side_effect=lambda *args: Connection(Response(302, headers={"Location": "/next"})))
            with self.assertRaisesRegex(api.ImageAPIError, "redirects"):
                api.download_image("https://example.com/x", connect=connect)
            self.assertEqual(connect.call_count, 4)

    def test_cached_result_retries_get_without_post_and_binds_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(png())
            cache = {}
            result = {"data": [{"url": "https://example.com/PRIVATE"}]}
            with patch.object(api, "post_edit", return_value=result) as post, patch.object(api, "download_image", side_effect=[api.ImageDownloadError("HTTP 403"), png()]) as download:
                with self.assertRaises(api.ImageDownloadError):
                    api.repair_image(source, root / "out", "gpt-image-2", "repair", "SECRET", result_cache=cache)
                self.assertTrue(api.repair_image(source, root / "out", "gpt-image-2", "repair", "SECRET", result_cache=cache).is_file())
                with self.assertRaisesRegex(api.ImageAPIError, "changed"):
                    api.repair_image(source, root / "out", "gpt-image-2", "different", "SECRET", result_cache=cache)
            post.assert_called_once()
            self.upload.assert_called_once()
            self.assertEqual(download.call_count, 2)
            for path in (root / "out").glob("*.json"):
                self.assertNotIn("PRIVATE", path.read_text())
                self.assertNotIn("SECRET", path.read_text())

    def test_redirect_revalidates_dns(self):
        connection = Connection(Response(302, headers={"Location": "http://internal.example/secret"}))
        answers = [[(2, 1, 6, "", ("8.8.8.8", 443))], [(2, 1, 6, "", ("127.0.0.1", 80))]]
        with patch.object(socket, "getaddrinfo", side_effect=answers), self.assertRaises(api.ImageAPIError):
            api.download_image("https://public.example/x", connect=Mock(return_value=connection))


if __name__ == "__main__":
    unittest.main()
