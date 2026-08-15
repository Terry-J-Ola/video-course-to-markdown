import csv
import pathlib
import tempfile
import unittest

from scripts import processing_metrics


class ProcessingMetricsTests(unittest.TestCase):
    def test_summarizes_fresh_visual_and_all_text_usage(self):
        visual = {
            "groups": [
                {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    }
                },
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "resumed": True,
                },
                {"usage": {"input_tokens": 3, "output_tokens": 2}},
            ]
        }
        audit = {
            "usage": {
                "initial": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
                "repairs": [{"input_tokens": 2, "output_tokens": 1}],
            }
        }

        summary = processing_metrics.summarize_token_usage(visual, audit)

        self.assertEqual(
            summary,
            {
                "visual": {"input_tokens": 13, "output_tokens": 6, "total_tokens": 19},
                "text": {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
                "qwen_total_tokens": 34,
            },
        )

    def test_transcription_duration_uses_declared_or_sentence_duration(self):
        transcription = {
            "transcripts": [
                {
                    "content_duration_in_milliseconds": 12_345,
                    "sentences": [{"end_time": 9_000}],
                },
                {"sentences": [{"end_time": 15_500}]},
            ]
        }
        self.assertEqual(
            processing_metrics.transcription_duration_seconds(transcription),
            15.5,
        )

    def test_builds_row_with_stage_timings_and_usage(self):
        report = {
            "video": "D:/courses/a.mp4",
            "status": "complete",
            "models": {"visual": "vl", "asr": "asr", "text": "text"},
            "stages": [
                {"name": "adaptive-keyframes", "elapsed_seconds": 1.25},
                {"name": "visual-analysis", "elapsed_seconds": 2.5},
            ],
        }
        usage = {
            "visual": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            "text": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
            "qwen_total_tokens": 26,
        }

        row = processing_metrics.build_processing_row(
            report,
            usage,
            asr_duration_seconds=9.5,
            started_at="2026-08-14T10:00:00+08:00",
            finished_at="2026-08-14T10:00:10+08:00",
            total_elapsed_seconds=10.0,
            report_path=pathlib.Path("D:/outputs/a_report.json"),
        )

        self.assertEqual(row["视频路径"], "D:/courses/a.mp4")
        self.assertEqual(row["总耗时（秒）"], 10.0)
        self.assertEqual(row["抽帧耗时（秒）"], 1.25)
        self.assertEqual(row["视觉分析耗时（秒）"], 2.5)
        self.assertEqual(row["Qwen总Token"], 26)
        self.assertEqual(row["ASR音频时长（秒）"], 9.5)

    def test_csv_is_excel_friendly_and_upserts_by_video_path(self):
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "视频处理统计.csv"
            base = {header: "" for header in processing_metrics.CSV_HEADERS}
            first = {**base, "视频路径": "D:/a.mp4", "状态": "complete", "Qwen总Token": 10}
            second = {**base, "视频路径": "D:/b.mp4", "状态": "complete", "Qwen总Token": 20}
            updated = {**base, "视频路径": "D:/a.mp4", "状态": "complete", "Qwen总Token": 30}

            processing_metrics.upsert_processing_stats(output, first)
            processing_metrics.upsert_processing_stats(output, second)
            processing_metrics.upsert_processing_stats(output, updated)

            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            by_video = {row["视频路径"]: row for row in rows}
            self.assertEqual(by_video["D:/a.mp4"]["Qwen总Token"], "30")
            self.assertEqual(by_video["D:/b.mp4"]["Qwen总Token"], "20")


if __name__ == "__main__":
    unittest.main()
