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

    def readline(self, size=-1):
        return self.stream.readline(size)


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

    def test_provider_supported_aspect_ratios_and_sizes(self):
        expected = {"doubao-seedream-5-0-260128": ["2848x1600", "2304x1728", "2048x2048"],
                    "gpt-image-2": ["1672x941", "1443x1090", "1024x1024"]}
        self.assertEqual(set(api.MODELS), set(expected))
        for model, sizes in expected.items():
            actual = [api.model_size(model, *size) for size in [(2560, 1440), (2048, 1536), (2048, 2048)]]
            self.assertEqual(actual, sizes)

    def test_provider_keys_are_separate_and_local_files_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "grs-key.txt").write_text("GRS_FILE_KEY", encoding="utf-8")
            (root / "ark-key.txt").write_text("ARK_FILE_KEY", encoding="utf-8")
            with patch.dict("os.environ", {"GRSAI_API_KEY": "STALE_GRS", "ARK_API_KEY": "STALE_ARK"}):
                self.assertEqual(api.read_key(root, "gpt-image-2"), "GRS_FILE_KEY")
                self.assertEqual(api.read_key(root, "doubao-seedream-5-0-260128"), "ARK_FILE_KEY")
                (root / "grs-key.txt").write_text("", encoding="utf-8")
                with self.assertRaises(api.ImageAPIError):
                    api.read_key(root, "gpt-image-2")
                self.assertEqual(api.read_key(root, "doubao-seedream-5-0-260128"), "ARK_FILE_KEY")


class NetworkTests(unittest.TestCase):
    def post(self, responses):
        connections = [Connection(response) for response in responses]
        factory = Mock(side_effect=connections)
        sleeper = Mock()
        result = api.post_edit({"model": "doubao-seedream-5-0-260128"}, "TEST_SECRET", connection_factory=factory, sleep=sleeper)
        return result, connections, sleeper

    def test_post_success(self):
        result, connections, _ = self.post([Response(body=b'{"data":[{"url":"https://example.com/image.png"}]}')])
        self.assertEqual(len(result["data"]), 1)
        request = connections[0].request.call_args.args
        self.assertEqual(request[0:2], ("POST", api.ENDPOINTS["doubao-seedream-5-0-260128"][1]))
        self.assertEqual(request[3]["Authorization"], "Bearer TEST_SECRET")

    def test_volcengine_stream_returns_first_completed_image(self):
        event = b'data: {"type":"image_generation.partial_succeeded","url":"https://example.com/result.png"}\n\n'
        connection = Connection(Response(body=event, headers={"Content-Type": "text/event-stream"}))
        result = api.post_edit({"model": "doubao-seedream-5-0-260128", "stream": True}, "TEST_SECRET",
                               connection_factory=Mock(return_value=connection))
        self.assertEqual(result["data"][0]["url"], "https://example.com/result.png")
        self.assertEqual(connection.request.call_args.args[1], api.ENDPOINTS["doubao-seedream-5-0-260128"][1])

    def test_grsai_completed_and_pending_responses(self):
        completed = {"id": "task-1", "status": "succeeded", "results": [{"url": "https://file1.aitohumanize.com/x.png"}]}
        connection = Connection(Response(body=json.dumps(completed).encode()))
        result = api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=Mock(return_value=connection))
        self.assertEqual(result, completed)
        self.assertEqual(connection.request.call_args.args[1], api.ENDPOINTS["gpt-image-2"][1])
        pending = {"id": "task-1", "status": "running"}
        factory = Mock(side_effect=[Connection(Response(body=json.dumps(pending).encode())),
                                    Connection(Response(body=json.dumps(completed).encode()))])
        self.assertEqual(api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory, sleep=Mock()), completed)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args_list[1].args[0], api.ENDPOINTS["gpt-image-2"][0])

    def test_grsai_host_is_configurable_only_to_official_nodes(self):
        completed = {"id": "task-1", "status": "succeeded", "results": []}
        with patch.dict("os.environ", {"GRSAI_API_HOST": "grsai.dakka.com.cn"}):
            factory = Mock(return_value=Connection(Response(body=json.dumps(completed).encode())))
            self.assertEqual(api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory), completed)
            factory.assert_called_once_with("grsai.dakka.com.cn", timeout=10)
        with patch.dict("os.environ", {"GRSAI_API_HOST": "untrusted.example"}):
            factory = Mock()
            with self.assertRaises(api.ImageAPIError):
                api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory)
            factory.assert_not_called()
    def test_grsai_falls_back_only_before_submission(self):
        completed = {"id": "task-1", "status": "succeeded", "results": []}
        first = Connection(Response())
        first.connect = Mock(side_effect=PermissionError("SECRET_URL"))
        second = Connection(Response(body=json.dumps(completed).encode()))
        factory = Mock(side_effect=[first, second])
        self.assertEqual(api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory), completed)
        self.assertEqual([call.args[0] for call in factory.call_args_list],
                         ["grsaiapi.com", "grsai.dakka.com.cn"])
        first.request.assert_not_called()
        second.request.assert_called_once()
        first.close.assert_called_once()
        second.close.assert_called_once()

        blocked = [Connection(Response()) for _ in range(2)]
        for connection in blocked:
            connection.connect = Mock(side_effect=PermissionError("SECRET_URL"))
        factory = Mock(side_effect=blocked)
        with self.assertRaises(api.ImageAPIError) as caught:
            api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory)
        self.assertIn("No generation request was sent", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertTrue(all(not connection.request.called for connection in blocked))

        sent = Connection(Response())
        sent.request.side_effect = ConnectionResetError("SECRET_URL")
        factory = Mock(return_value=sent)
        with self.assertRaises(api.ImageAPIError) as caught:
            api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory)
        factory.assert_called_once()
        self.assertIn("may have been billed", str(caught.exception))
    def test_grsai_result_and_proxy_cdn_restriction(self):
        raw = png()
        url = "https://file1.aitohumanize.com/image.png"
        self.assertEqual(api.result_bytes({"results": [{"url": url}]}, download=lambda _: raw), raw)
        self.assertTrue(api._allowed_download_ip("file1.aitohumanize.com", "198.18.0.88"))
        self.assertTrue(api._allowed_download_ip("ark-content-generation-v2-cn-beijing.tos-cn-beijing.volces.com", "198.18.0.88"))
        self.assertFalse(api._allowed_download_ip("other.tos-cn-beijing.volces.com", "198.18.0.88"))
        self.assertFalse(api._allowed_download_ip("internal.example", "198.18.0.88"))
        self.assertFalse(api._allowed_download_ip("file1.aitohumanize.com", "127.0.0.1"))

    def test_grsai_query_disconnect_retries_same_task_without_another_post(self):
        pending = Response(body=b'{"id":"task-1","status":"running"}')
        completed = {"id": "task-1", "status": "succeeded", "results": []}
        connections = [Connection(pending), Connection(api.http.client.RemoteDisconnected("SECRET_URL")),
                       Connection(Response(body=json.dumps(completed).encode()))]
        factory = Mock(side_effect=connections)
        sleeper = Mock()
        result = api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory, sleep=sleeper)
        self.assertEqual(result, completed)
        self.assertEqual([c.request.call_args.args[0] for c in connections], ["POST", "GET", "GET"])
        self.assertTrue(all(c.request.call_args.args[1] == "/v1/api/result?id=task-1" for c in connections[1:]))
        self.assertEqual([c.args[0] for c in sleeper.call_args_list], [5, 2])
        for connection in connections:
            connection.close.assert_called_once()

    def test_grsai_query_transient_http_retries_but_permanent_errors_stop(self):
        for code in (429, 500, 502, 503, 504):
            with self.subTest(code=code):
                factory = Mock(side_effect=[Connection(Response(code, headers={"Retry-After": "600"})),
                                           Connection(Response(body=b'{"status":"succeeded"}'))])
                sleeper = Mock()
                result = api._query_grs_task("task-1", "TEST_SECRET", "gateway.example", connection_factory=factory, sleep=sleeper)
                self.assertEqual(result["status"], "succeeded")
                sleeper.assert_called_once_with(30)
        for code in (400, 401, 403, 404):
            factory = Mock(return_value=Connection(Response(code)))
            with self.subTest(code=code), self.assertRaises(api.ImageAPIError):
                api._query_grs_task("task-1", "TEST_SECRET", "gateway.example", connection_factory=factory, sleep=Mock())
            factory.assert_called_once()

    def test_grsai_query_retry_exhaustion_is_bounded_and_redacted(self):
        connections = [Connection(Response(body=b'{"id":"task-1","status":"running"}'))]
        connections.extend(Connection(socket.timeout("SECRET_URL")) for _ in range(3))
        factory = Mock(side_effect=connections)
        with self.assertRaises(api.ImageAPIError) as caught:
            api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory, sleep=Mock())
        self.assertIn("after 3 query attempts", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual([c.request.call_args.args[0] for c in connections], ["POST", "GET", "GET", "GET"])

    def test_grsai_submission_disconnect_is_not_automatically_resubmitted(self):
        factory = Mock(return_value=Connection(api.http.client.RemoteDisconnected("SECRET_URL")))
        with self.assertRaises(api.ImageAPIError):
            api.post_edit({"model": "gpt-image-2"}, "TEST_SECRET", connection_factory=factory, sleep=Mock())
        factory.assert_called_once()

    def test_retry_only_explicit_transient_statuses(self):
        for code in (429,):
            _, connections, sleep = self.post([Response(code), Response(code), Response()])
            self.assertEqual(len(connections), 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 5])
        _, _, sleep = self.post([Response(429, headers={"Retry-After": "600"}), Response()])
        sleep.assert_called_once_with(30)

    def test_permanent_http_failures_do_not_retry(self):
        for code in (400, 401, 403, 500, 502, 503, 504, 302):
            connection = Connection(Response(code))
            factory = Mock(return_value=connection)
            with self.assertRaises(api.ImageAPIError):
                api.post_edit({"model": "doubao-seedream-5-0-260128"}, "TEST_SECRET", connection_factory=factory)
            self.assertEqual(factory.call_count, 1)

    def test_gateway_error_details_are_bounded_and_redacted(self):
        body = {"error": {"code": "unsupported_size", "type": "upstream_error",
                          "message": "Bad size. TEST_SECRET Bearer hidden sk-hidden https://host/private?sig=hidden data:image/png;base64," + "A" * 100}}
        factory = Mock(return_value=Connection(Response(500, json.dumps(body).encode())))
        with self.assertRaises(api.ImageAPIError) as caught:
            api.post_edit({"model": "doubao-seedream-5-0-260128"}, "TEST_SECRET", connection_factory=factory)
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

    def test_provider_specific_bodies_and_no_secret_persistence(self):
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
                if model == "gpt-image-2":
                    self.assertTrue(body["images"][0].startswith("data:image/png;base64,"))
                    self.assertEqual(body["aspectRatio"], "1024x1024")
                    self.assertEqual(body["replyType"], "async")
                else:
                    self.assertTrue(body["image"].startswith("data:image/png;base64,"))
                    self.assertEqual(body["size"], "2048x2048")
                    self.assertFalse(body["watermark"])
                for record in (root / model).glob("*.json"):
                    self.assertNotIn("SECRET", record.read_text())
                    self.assertNotIn("data:image", record.read_text())

    def test_invalid_input_never_calls_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            source.write_bytes(png((160, 100)))
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
                api.post_edit({"model": "doubao-seedream-5-0-260128"}, "TEST_SECRET", connection_factory=factory)
            self.assertEqual(factory.call_count, 1)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_retry_limit(self):
        factory = Mock(side_effect=[Connection(Response(429)) for _ in range(3)])
        with self.assertRaises(api.ImageAPIError):
            api.post_edit({"model": "doubao-seedream-5-0-260128"}, "TEST_SECRET", connection_factory=factory, sleep=Mock())
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
