import csv
import os
import pathlib
import sys
import time
from collections.abc import Iterable


CSV_HEADERS = [
    "视频路径",
    "状态",
    "开始时间",
    "结束时间",
    "总耗时（秒）",
    "抽帧耗时（秒）",
    "关键帧总览耗时（秒）",
    "视觉分析耗时（秒）",
    "音频转写耗时（秒）",
    "证据稿耗时（秒）",
    "业务讲义耗时（秒）",
    "视觉输入Token",
    "视觉输出Token",
    "视觉总Token",
    "文本输入Token",
    "文本输出Token",
    "文本总Token",
    "Qwen总Token",
    "ASR音频时长（秒）",
    "视觉模型",
    "ASR模型",
    "文本模型",
    "处理报告",
    "错误信息",
]


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def normalize_token_usage(usage: object) -> dict[str, int]:
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _nonnegative_int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0))
    )
    output_tokens = _nonnegative_int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    total_value = usage.get("total_tokens")
    total_tokens = (
        _nonnegative_int(total_value)
        if total_value is not None
        else input_tokens + output_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _sum_token_usage(usages: Iterable[object]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usages:
        normalized = normalize_token_usage(usage)
        for key in total:
            total[key] += normalized[key]
    return total


def summarize_token_usage(
    visual_data: object,
    audit_data: object,
) -> dict[str, object]:
    visual_groups = visual_data.get("groups", []) if isinstance(visual_data, dict) else []
    visual_usages = [
        group.get("usage", {})
        for group in visual_groups
        if isinstance(group, dict) and not group.get("resumed")
    ]

    text_usages: list[object] = []
    if isinstance(audit_data, dict) and isinstance(audit_data.get("usage"), dict):
        usage = audit_data["usage"]
        text_usages.append(usage.get("initial", {}))
        repairs = usage.get("repairs", [])
        if isinstance(repairs, list):
            text_usages.extend(repairs)

    visual = _sum_token_usage(visual_usages)
    text = _sum_token_usage(text_usages)
    return {
        "visual": visual,
        "text": text,
        "qwen_total_tokens": visual["total_tokens"] + text["total_tokens"],
    }


def transcription_duration_seconds(transcription: object) -> float:
    if not isinstance(transcription, dict):
        return 0.0
    duration_ms = 0
    transcripts = transcription.get("transcripts", [])
    if not isinstance(transcripts, list):
        return 0.0
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        duration_ms = max(
            duration_ms,
            _nonnegative_int(transcript.get("content_duration_in_milliseconds", 0)),
        )
        sentences = transcript.get("sentences", [])
        if not isinstance(sentences, list):
            continue
        for sentence in sentences:
            if isinstance(sentence, dict):
                duration_ms = max(duration_ms, _nonnegative_int(sentence.get("end_time", 0)))
    return round(duration_ms / 1000, 3)


def _stage_elapsed(report: dict, name: str) -> float:
    elapsed = 0.0
    for stage in report.get("stages", []):
        if isinstance(stage, dict) and stage.get("name") == name:
            value = stage.get("elapsed_seconds", 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                elapsed += max(0.0, float(value))
    return round(elapsed, 2)


def build_processing_row(
    report: dict,
    usage: dict,
    *,
    asr_duration_seconds: float,
    started_at: str,
    finished_at: str,
    total_elapsed_seconds: float,
    report_path: pathlib.Path,
) -> dict[str, object]:
    visual = usage.get("visual", {})
    text = usage.get("text", {})
    models = report.get("models", {})
    return {
        "视频路径": report.get("video", ""),
        "状态": report.get("status", ""),
        "开始时间": started_at,
        "结束时间": finished_at,
        "总耗时（秒）": round(max(0.0, total_elapsed_seconds), 2),
        "抽帧耗时（秒）": _stage_elapsed(report, "adaptive-keyframes"),
        "关键帧总览耗时（秒）": _stage_elapsed(report, "contact-sheet"),
        "视觉分析耗时（秒）": _stage_elapsed(report, "visual-analysis"),
        "音频转写耗时（秒）": _stage_elapsed(report, "audio-transcription"),
        "证据稿耗时（秒）": _stage_elapsed(report, "evidence-markdown"),
        "业务讲义耗时（秒）": _stage_elapsed(report, "learner-markdown-and-audit"),
        "视觉输入Token": visual.get("input_tokens", 0),
        "视觉输出Token": visual.get("output_tokens", 0),
        "视觉总Token": visual.get("total_tokens", 0),
        "文本输入Token": text.get("input_tokens", 0),
        "文本输出Token": text.get("output_tokens", 0),
        "文本总Token": text.get("total_tokens", 0),
        "Qwen总Token": usage.get("qwen_total_tokens", 0),
        "ASR音频时长（秒）": round(max(0.0, asr_duration_seconds), 3),
        "视觉模型": models.get("visual") or "",
        "ASR模型": models.get("asr") or "",
        "文本模型": models.get("text") or "",
        "处理报告": str(report_path),
        "错误信息": "",
    }


def _replace_stats_file(temporary: pathlib.Path, path: pathlib.Path) -> None:
    """Replace ``path`` with ``temporary``, tolerating brief file locks.

    On Windows ``os.replace`` can raise ``PermissionError`` (WinError 5) when
    the target is momentarily held open by another process — an editor
    refreshing its file watch, an antivirus scan, or a sync client. We retry a
    few times with backoff for those transient cases. If the lock persists
    (e.g. the CSV is open in Excel, which takes an exclusive lock), we warn
    loudly and move on rather than aborting the whole batch: the stats file is
    a reporting artifact, not a primary output, and one missed row is far
    cheaper than losing all subsequent video processing.
    """
    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.2 * (attempt + 1))
    print(
        f"[processing_metrics] 警告：无法更新统计文件 {path}（{last_error}）。"
        f"该文件可能正被其他程序占用（如 Excel 或编辑器），请关闭后重跑以补回本条记录；"
        f"本次统计未写入，但不会影响后续视频处理。",
        file=sys.stderr,
        flush=True,
    )


def upsert_processing_stats(path: pathlib.Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for existing in csv.DictReader(handle):
                if existing.get("视频路径") != row.get("视频路径"):
                    rows.append(existing)
    rows.append({header: row.get(header, "") for header in CSV_HEADERS})

    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        _replace_stats_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
