import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.env_config import (
    get_asr_api_key,
    get_qwen_api_key,
    load_dashscope_environment,
    parse_env_file,
)


class EnvConfigTests(unittest.TestCase):
    def test_parser_accepts_comments_blank_lines_and_quotes(self):
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / ".env"
            path.write_text(
                "# comment\n\nDASHSCOPE_API_KEY='sk-test'\nOTHER=\"value\"\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_env_file(path),
                {"DASHSCOPE_API_KEY": "sk-test", "OTHER": "value"},
            )

    def test_existing_environment_wins_without_reading_explicit_file(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "existing"}, clear=True):
            result = load_dashscope_environment("missing.env", pathlib.Path.cwd())
            self.assertEqual(
                result,
                {
                    "source": "environment",
                    "api_key_set": True,
                    "qwen_api_key_set": True,
                    "asr_api_key_set": True,
                },
            )
            self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "existing")

    def test_explicit_file_wins_over_project_and_user_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            explicit = root / "chosen.env"
            explicit.write_text("DASHSCOPE_API_KEY=explicit\n", encoding="utf-8")
            (root / ".env").write_text("DASHSCOPE_API_KEY=project\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(str(explicit), root, str(root / "local"))
                self.assertEqual(result["source"], str(explicit.resolve()))
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "explicit")

    def test_explicit_file_without_key_does_not_fall_back_to_automatic_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            explicit = root / "chosen.env"
            explicit.write_text("OTHER=value\n", encoding="utf-8")
            (root / ".env").write_text("DASHSCOPE_API_KEY=project\n", encoding="utf-8")
            local = root / "local"
            private = local / "Codex" / "video-course-to-markdown" / ".env"
            private.parent.mkdir(parents=True)
            private.write_text("DASHSCOPE_API_KEY=private\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(str(explicit), root, str(local))
                self.assertEqual(
                    result,
                    {
                        "source": str(explicit.resolve()),
                        "api_key_set": False,
                        "qwen_api_key_set": False,
                        "asr_api_key_set": False,
                    },
                )
                self.assertNotIn("DASHSCOPE_API_KEY", os.environ)

    def test_project_file_wins_over_user_private_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            local = root / "local"
            private = local / "Codex" / "video-course-to-markdown" / ".env"
            private.parent.mkdir(parents=True)
            private.write_text("DASHSCOPE_API_KEY=private\n", encoding="utf-8")
            (root / ".env").write_text("DASHSCOPE_API_KEY=project\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(None, root, str(local))
                self.assertEqual(result["source"], str((root / ".env").resolve()))
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "project")

    def test_private_file_is_last_automatic_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            local = root / "local"
            private = local / "Codex" / "video-course-to-markdown" / ".env"
            private.parent.mkdir(parents=True)
            private.write_text("DASHSCOPE_API_KEY=private\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(None, root, str(local))
                self.assertEqual(result["source"], str(private.resolve()))
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "private")

    def test_malformed_line_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / ".env"
            path.write_text("DASHSCOPE_API_KEY=ok\nnot-valid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, rf"{path.name}:2"):
                parse_env_file(path)

    def test_missing_explicit_file_is_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "missing.env"):
                load_dashscope_environment("missing.env", pathlib.Path.cwd())

    def test_automatic_discovery_can_find_no_key(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(None, pathlib.Path(raw), str(pathlib.Path(raw) / "local"))
                self.assertEqual(
                    result,
                    {
                        "source": None,
                        "api_key_set": False,
                        "qwen_api_key_set": False,
                        "asr_api_key_set": False,
                    },
                )

    def test_dedicated_qwen_and_asr_keys_are_detected_independently(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            explicit = root / "chosen.env"
            explicit.write_text(
                "DASHSCOPE_QWEN_API_KEY=qwen-key\nDASHSCOPE_ASR_API_KEY=asr-key\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(str(explicit), root)
                self.assertTrue(result["qwen_api_key_set"])
                self.assertTrue(result["asr_api_key_set"])
                self.assertTrue(result["api_key_set"])
                self.assertEqual(os.environ["DASHSCOPE_QWEN_API_KEY"], "qwen-key")
                self.assertEqual(os.environ["DASHSCOPE_ASR_API_KEY"], "asr-key")

    def test_shared_key_is_fallback_for_both_model_families(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "shared"}, clear=True):
            result = load_dashscope_environment(None, pathlib.Path.cwd())
            self.assertTrue(result["qwen_api_key_set"])
            self.assertTrue(result["asr_api_key_set"])
            self.assertEqual(get_qwen_api_key(), "shared")
            self.assertEqual(get_asr_api_key(), "shared")

    def test_dedicated_keys_override_shared_key_per_model_family(self):
        with mock.patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "shared",
                "DASHSCOPE_QWEN_API_KEY": "qwen-dedicated",
                "DASHSCOPE_ASR_API_KEY": "asr-dedicated",
            },
            clear=True,
        ):
            self.assertEqual(get_qwen_api_key(), "qwen-dedicated")
            self.assertEqual(get_asr_api_key(), "asr-dedicated")

    def test_partial_dedicated_configuration_reports_missing_asr(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            explicit = root / "chosen.env"
            explicit.write_text("DASHSCOPE_QWEN_API_KEY=qwen-only\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                result = load_dashscope_environment(str(explicit), root)
                self.assertTrue(result["qwen_api_key_set"])
                self.assertFalse(result["asr_api_key_set"])
                self.assertFalse(result["api_key_set"])

    def test_existing_dedicated_key_is_not_overwritten_by_env_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            explicit = root / "chosen.env"
            explicit.write_text(
                "DASHSCOPE_QWEN_API_KEY=file-qwen\nDASHSCOPE_ASR_API_KEY=file-asr\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"DASHSCOPE_QWEN_API_KEY": "process-qwen"},
                clear=True,
            ):
                result = load_dashscope_environment(str(explicit), root)
                self.assertEqual(os.environ["DASHSCOPE_QWEN_API_KEY"], "process-qwen")
                self.assertEqual(os.environ["DASHSCOPE_ASR_API_KEY"], "file-asr")
                self.assertEqual(result["source"], str(explicit.resolve()))
