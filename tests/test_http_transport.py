import json
import pathlib
import ssl
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from scripts import http_transport
from scripts import rewrite_learner_markdown
from scripts import transcribe_video_audio


class FakeResponse:
    def __init__(self, status: int, body: bytes, headers=None, reason="response"):
        self.status = status
        self.reason = reason
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body


class FakeConnection:
    responses = []
    instances = []

    def __init__(self, host, port=None, timeout=None, context=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.context = context
        self.requests = []
        self.closed = False
        type(self).instances.append(self)

    def request(self, method, target, body=None, headers=None):
        self.requests.append((method, target, body, headers or {}))

    def getresponse(self):
        return type(self).responses.pop(0)

    def close(self):
        self.closed = True


class HttpTransportTests(unittest.TestCase):
    def setUp(self):
        FakeConnection.responses = []
        FakeConnection.instances = []

    def test_open_url_uses_verified_https_connection_and_preserves_request(self):
        FakeConnection.responses = [FakeResponse(200, b'{"ok": true}')]
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/path/to/api?x=1",
            data=b'{"model":"test"}',
            headers={"Authorization": "Bearer test-only", "Content-Type": "application/json"},
            method="POST",
        )

        with mock.patch.object(
            http_transport.http.client,
            "HTTPSConnection",
            FakeConnection,
        ):
            with http_transport.open_url(request, timeout=123) as response:
                self.assertEqual(json.loads(response.read()), {"ok": True})

        connection = FakeConnection.instances[0]
        self.assertEqual(connection.host, "dashscope.aliyuncs.com")
        self.assertEqual(connection.port, 443)
        self.assertEqual(connection.timeout, 123)
        self.assertIsInstance(connection.context, ssl.SSLContext)
        self.assertTrue(connection.context.check_hostname)
        self.assertEqual(connection.context.verify_mode, ssl.CERT_REQUIRED)
        method, target, body, headers = connection.requests[0]
        self.assertEqual(method, "POST")
        self.assertEqual(target, "/path/to/api?x=1")
        self.assertEqual(body, b'{"model":"test"}')
        self.assertEqual(headers["Authorization"], "Bearer test-only")
        self.assertTrue(connection.closed)

    def test_http_error_retains_status_and_response_body(self):
        FakeConnection.responses = [FakeResponse(429, b'{"message":"limited"}', reason="limited")]
        request = urllib.request.Request("https://dashscope.aliyuncs.com/api")

        with mock.patch.object(
            http_transport.http.client,
            "HTTPSConnection",
            FakeConnection,
        ):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                http_transport.open_url(request, timeout=10)

        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.read(), b'{"message":"limited"}')

    def test_cross_origin_redirect_does_not_forward_authorization(self):
        FakeConnection.responses = [
            FakeResponse(
                302,
                b"",
                headers={"Location": "https://download.example/result.json"},
            ),
            FakeResponse(200, b"{}"),
        ]
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/result",
            headers={"Authorization": "Bearer test-only"},
        )

        with mock.patch.object(
            http_transport.http.client,
            "HTTPSConnection",
            FakeConnection,
        ):
            with http_transport.open_url(request, timeout=10) as response:
                self.assertEqual(response.read(), b"{}")

        self.assertEqual(len(FakeConnection.instances), 2)
        redirected_headers = FakeConnection.instances[1].requests[0][3]
        self.assertNotIn("Authorization", redirected_headers)

    def test_learner_model_call_uses_shared_transport(self):
        response = FakeResponse(
            200,
            json.dumps(
                {
                    "choices": [{"message": {"content": "# Learner"}}],
                    "usage": {"total_tokens": 7},
                }
            ).encode("utf-8"),
        )
        with mock.patch.object(
            rewrite_learner_markdown,
            "open_url",
            return_value=http_transport.BufferedResponse(
                response.read(), response.status, response.reason, response.headers
            ),
        ) as open_url:
            content, usage = rewrite_learner_markdown.call_model(
                "test-only", "qwen-plus", "prompt", 0.2
            )

        self.assertEqual(content, "# Learner")
        self.assertEqual(usage["total_tokens"], 7)
        open_url.assert_called_once()

    def test_asr_json_request_uses_shared_transport(self):
        response = http_transport.BufferedResponse(
            b'{"output":{"task_id":"task-1"}}', 200, "ok", {}
        )
        with mock.patch.object(
            transcribe_video_audio,
            "open_url",
            return_value=response,
        ) as open_url:
            result = transcribe_video_audio.request_json(
                "https://dashscope.aliyuncs.com/asr",
                "test-only",
                method="POST",
                payload={"model": "paraformer-v2"},
            )

        self.assertEqual(result["output"]["task_id"], "task-1")
        open_url.assert_called_once()

    def test_asr_audio_upload_avoids_sdk_requests_transport(self):
        upload_info = {
            "oss_access_key_id": "access-id",
            "signature": "signature",
            "policy": "policy",
            "upload_dir": "upload-dir",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
            "upload_host": "https://upload.example",
        }
        with tempfile.TemporaryDirectory() as raw:
            audio = pathlib.Path(raw) / "audio.mp3"
            audio.write_bytes(b"audio-bytes")
            with mock.patch.object(
                transcribe_video_audio,
                "request_json",
                return_value={"data": upload_info},
            ) as request_json:
                with mock.patch.object(
                    transcribe_video_audio,
                    "open_url",
                    return_value=http_transport.BufferedResponse(b"", 200, "ok", {}),
                ) as open_url:
                    oss_url, certificate = transcribe_video_audio.SecureOssUtils.upload(
                        model="paraformer-v2",
                        file_path=str(audio),
                        api_key="test-only",
                    )

        self.assertEqual(oss_url, "oss://upload-dir/audio.mp3")
        self.assertEqual(certificate, upload_info)
        request_json.assert_called_once()
        upload_request = open_url.call_args.args[0]
        self.assertEqual(upload_request.full_url, "https://upload.example")
        self.assertEqual(upload_request.get_method(), "POST")
        self.assertIn("multipart/form-data", upload_request.get_header("Content-type"))
        self.assertNotIn("Authorization", dict(upload_request.header_items()))
        self.assertIn(b"audio-bytes", upload_request.data)

    def test_quality_audit_ignores_frame_names_inside_valid_image_links(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            image = root / "frame_0001.jpg"
            image.write_bytes(b"image")
            output = root / "course.md"
            clean = rewrite_learner_markdown.output_quality(
                "正文\n\n![课程画面](frame_0001.jpg)\n",
                output,
            )
            leaked = rewrite_learner_markdown.output_quality(
                "正文错误地提到了 frame_0002。\n\n![课程画面](frame_0001.jpg)\n",
                output,
            )

        self.assertEqual(clean["technical_leakage"], [])
        self.assertEqual(clean["missing_images"], [])
        self.assertEqual(leaked["technical_leakage"], ["frame_0002"])


if __name__ == "__main__":
    unittest.main()
