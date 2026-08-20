import argparse
import datetime
import hashlib
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque

try:
    from .checkpoint_provenance import file_identity, read_checkpoint_fingerprint, write_checkpoint
    from .env_config import load_dashscope_environment
    from .processing_metrics import (
        build_processing_row,
        summarize_token_usage,
        transcription_duration_seconds,
        upsert_processing_stats,
        validate_xlsx_support,
    )
except ImportError:
    from checkpoint_provenance import file_identity, read_checkpoint_fingerprint, write_checkpoint
    from env_config import load_dashscope_environment
    from processing_metrics import (
        build_processing_row,
        summarize_token_usage,
        transcription_duration_seconds,
        upsert_processing_stats,
        validate_xlsx_support,
    )


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PROJECT_ROOT = SCRIPT_DIR.parent
BUSINESS_LECTURES_DIRNAME = "业务讲义汇总"
BUSINESS_LECTURES_MANIFEST = "业务讲义汇总清单.json"
MAX_COLLECTED_FILENAME_LENGTH = 220
ACTIVE_LOG_FILE: pathlib.Path | None = None
ACTIVE_JSON_OUTPUT = False
ACTIVE_RUN_ID: str | None = None
ACTIVE_REPORT_CONTEXT: dict | None = None
LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUPS = 5
LOG_WRITE_LOCK = threading.Lock()
API_PROTOCOLS = {
    "visual": "dashscope-native-multimodal-v1",
    "asr": "dashscope-native-asr-transcription-v1",
    "text": "dashscope-native-text-v1",
}
STAGE_LABELS = {
    "adaptive-keyframes": "自适应关键帧",
    "contact-sheet": "关键帧总览",
    "visual-analysis": "画面分析",
    "audio-transcription": "音频转写",
    "evidence-markdown": "技术证据稿",
    "learner-markdown-and-audit": "业务讲义与校验",
}


class StageExecutionError(subprocess.CalledProcessError):
    def __init__(self, returncode, cmd, *, error_detail=None, output=None, stderr=None):
        super().__init__(returncode, cmd, output=output, stderr=stderr)
        self.error_detail = error_detail or {}


def structured_dashscope_error(lines: list[str]) -> dict | None:
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and value.get("event") == "dashscope_error":
            return value
    return None


def timestamp_now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def configured_secrets() -> tuple[str, ...]:
    values = {
        os.environ.get("DASHSCOPE_API_KEY", ""),
        os.environ.get("DASHSCOPE_QWEN_API_KEY", ""),
        os.environ.get("DASHSCOPE_ASR_API_KEY", ""),
    }
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def redact_text(value: str) -> str:
    redacted = value
    for secret in configured_secrets():
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_value(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_value(item) for key, item in value.items()}
    return value


def rotate_log_if_needed(path: pathlib.Path, incoming_bytes: int) -> None:
    if not path.exists() or path.stat().st_size + incoming_bytes <= LOG_MAX_BYTES:
        return
    for index in range(LOG_BACKUPS, 0, -1):
        source = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
        destination = path.with_name(f"{path.name}.{index}")
        if source.exists():
            os.replace(source, destination)


def append_log(path: pathlib.Path | None, event: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = redact_value(
        {
            "timestamp": timestamp_now(),
            "run_id": ACTIVE_RUN_ID,
            "pid": os.getpid(),
            **event,
        }
    )
    line = json.dumps(record, ensure_ascii=False) + "\n"
    encoded_size = len(line.encode("utf-8"))
    with LOG_WRITE_LOCK:
        rotate_log_if_needed(path, encoded_size)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()


def runtime_log_path(report: dict) -> pathlib.Path | None:
    value = report.get("_runtime", {}).get("log_file")
    return pathlib.Path(value) if value else None


def runtime_json_output(report: dict) -> bool:
    return bool(report.get("_runtime", {}).get("json_output"))


def emit(json_output: bool, payload: dict, human_text: str) -> None:
    if json_output:
        print(json.dumps(redact_value(payload), ensure_ascii=False), flush=True)
    else:
        print(redact_text(human_text), flush=True)


def compact_path(path: pathlib.Path | str, limit: int = 72) -> str:
    text = str(path)
    if len(text) <= limit:
        return text
    return f"…{text[-(limit - 1):]}"


def selected_api_protocols(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "visual": API_PROTOCOLS["visual"],
        "asr": None if args.skip_asr else API_PROTOCOLS["asr"],
        "text": None if args.skip_business else API_PROTOCOLS["text"],
    }


def validate_required_api_keys(args: argparse.Namespace, configuration: dict) -> None:
    if args.dry_run:
        return
    if not configuration["qwen_api_key_set"]:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not configured. Set it with --api-key, in the "
            "environment, --env-file, project .env, or user-private .env."
        )
    if not args.skip_asr and not configuration["asr_api_key_set"]:
        raise RuntimeError(
            "DASHSCOPE_API_KEY or legacy DASHSCOPE_ASR_API_KEY is not configured "
            "for ASR. Set it with --api-key, in the environment, --env-file, "
            "project .env, or user-private .env."
        )


def concise_error(exc: Exception) -> str:
    if isinstance(exc, StageExecutionError) and exc.error_detail.get("message"):
        return redact_text(str(exc.error_detail["message"]))
    if isinstance(exc, subprocess.CalledProcessError):
        return f"子进程退出码 {exc.returncode}"
    text = redact_text(str(exc)).replace("\r", " ").replace("\n", " ")
    if len(text) <= 160:
        return text
    return f"{text[:80]}…{text[-79:]}"


def concise_stage_progress(stage: str, line: str) -> str | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if stage == "visual-analysis" and "completed_group" in payload:
        return f"    [进度] 画面分析 {payload['completed_group']}/{payload.get('total_groups', '?')}"
    if stage == "audio-transcription":
        labels = {
            "audio_extracted": "音频已提取",
            "audio_uploaded": "音频已上传",
            "task_submitted": "转写任务已提交",
        }
        label = labels.get(payload.get("stage"))
        if label:
            return f"    [进度] {label}"
    return None


def run_streamed_command(
    command: list[str],
    environment: dict[str, str],
    on_line,
) -> int:
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def pump(stream, stream_name: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((stream_name, line.rstrip("\r\n")))
        finally:
            stream.close()
            output_queue.put((stream_name, None))

    threads = [
        threading.Thread(target=pump, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=pump, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    completed_streams = 0
    try:
        while completed_streams < len(threads):
            stream_name, line = output_queue.get()
            if line is None:
                completed_streams += 1
            else:
                on_line(stream_name, line)
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise


def validate_video(path: pathlib.Path) -> pathlib.Path:
    video = path.expanduser().resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"Input video does not exist or is not a file: {video}")
    if video.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {video.suffix!r}: {video}. Supported: {supported}")
    return video


def validate_output_dir(path: pathlib.Path, video: pathlib.Path) -> pathlib.Path:
    output = path.expanduser().resolve()
    if output == video:
        raise ValueError(f"Output directory cannot equal the input video path: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is a file, not a directory: {output}")
    return output


def safe_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" .")
    return cleaned or "video-course"


def discover_pydeps(explicit: str | None) -> pathlib.Path | None:
    candidates = [
        explicit,
        os.environ.get("VIDEO_COURSE_PYDEPS"),
    ]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(
            str(
                pathlib.Path(local_appdata)
                / "Codex"
                / "video-course-to-markdown"
                / "pydeps"
            )
        )
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return pathlib.Path(candidate).resolve()
    return None


def utf8_child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def run_stage(
    name: str,
    command: list[str],
    report: dict,
    dry_run: bool,
) -> None:
    report["stages"].append({"name": name, "command": command, "status": "planned"})
    json_output = runtime_json_output(report)
    log_path = runtime_log_path(report)
    label = STAGE_LABELS.get(name, name)
    if dry_run:
        emit(
            json_output,
            {"stage": name, "command": command},
            f"  [计划] {label}",
        )
        return
    started = time.monotonic()
    append_log(log_path, {"event": "stage_started", "stage": name, "command": command})
    emit(
        json_output,
        {"stage": name, "status": "running"},
        f"  [开始] {label}",
    )
    child_environment = utf8_child_environment()
    pydeps = report.get("pydeps")
    if pydeps:
        inherited = child_environment.get("PYTHONPATH")
        child_environment["PYTHONPATH"] = (
            os.pathsep.join([pydeps, inherited]) if inherited else pydeps
        )
    captured_output: dict[str, deque[str]] = {
        "stdout": deque(maxlen=200),
        "stderr": deque(maxlen=200),
    }

    def handle_line(stream_name: str, line: str) -> None:
        captured_output[stream_name].append(line)
        append_log(
            log_path,
            {
                "event": "stage_output",
                "stage": name,
                "stream": stream_name,
                "content": line,
            },
        )
        if json_output:
            print(
                redact_text(line),
                file=sys.stderr if stream_name == "stderr" else sys.stdout,
                flush=True,
            )
        else:
            progress = concise_stage_progress(name, line)
            if progress:
                print(progress, flush=True)

    returncode = run_streamed_command(command, child_environment, handle_line)
    elapsed = round(time.monotonic() - started, 2)
    if returncode != 0:
        report["stages"][-1]["status"] = "failed"
        report["stages"][-1]["elapsed_seconds"] = elapsed
        append_log(
            log_path,
            {
                "event": "stage_failed",
                "stage": name,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            },
        )
        emit(
            json_output,
            {"stage": name, "status": "failed", "returncode": returncode},
            f"  [失败] {label}（退出码 {returncode}）",
        )
        output = "\n".join(captured_output["stdout"])
        stderr = "\n".join(captured_output["stderr"])
        detail = structured_dashscope_error(
            list(captured_output["stderr"]) + list(captured_output["stdout"])
        )
        if detail:
            raise StageExecutionError(
                returncode,
                command,
                error_detail=detail,
                output=output,
                stderr=stderr,
            )
        raise subprocess.CalledProcessError(
            returncode, command, output=output, stderr=stderr
        )
    report["stages"][-1]["status"] = "complete"
    report["stages"][-1]["elapsed_seconds"] = elapsed
    append_log(
        log_path,
        {"event": "stage_completed", "stage": name, "elapsed_seconds": elapsed},
    )
    if json_output:
        print(
            json.dumps(
                {"stage": name, "status": "complete", "elapsed_seconds": elapsed},
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(f"  [完成] {label}（{elapsed:.2f} 秒）", flush=True)


def script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def add_pydeps(command: list[str], pydeps: pathlib.Path | None) -> None:
    if pydeps:
        command.extend(["--pydeps", str(pydeps)])


def write_empty_transcription(path: pathlib.Path, video: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"transcripts": []}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checkpoint(
        path.with_name("transcription.checkpoint.json"),
        {"stage": "asr", "input": file_identity(video), "state": "skipped", "model": None},
    )


def read_optional_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def exception_metadata(exc: Exception) -> dict:
    if isinstance(exc, StageExecutionError) and exc.error_detail:
        detail = exc.error_detail
        return {
            "error": redact_text(str(detail.get("message") or concise_error(exc))),
            "error_category": detail.get("category", "stage"),
            "http_status": detail.get("http_status"),
            "provider_code": detail.get("provider_code"),
            "retryable": bool(detail.get("retryable", False)),
            "service": detail.get("service"),
        }
    return {
        "error": redact_text(str(exc)),
        "error_category": "internal",
        "http_status": None,
        "provider_code": None,
        "retryable": False,
        "service": None,
    }


def persist_active_failure_report(exc: Exception) -> pathlib.Path | None:
    context = ACTIVE_REPORT_CONTEXT
    if not context:
        return None
    report_path = pathlib.Path(context["report_path"])
    report = redact_value(context["report"])
    report.pop("_runtime", None)
    metadata = exception_metadata(exc)
    finished_at = timestamp_now()
    total_elapsed = round(time.monotonic() - context["started_monotonic"], 2)
    report.update(
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            **metadata,
            "timing": {
                "started_at": context["started_at"],
                "finished_at": finished_at,
                "total_elapsed_seconds": total_elapsed,
            },
        }
    )
    write_json_atomic(report_path, report)
    try:
        visual_data = read_optional_json(pathlib.Path(context["visual_data"]))
        audit_data = read_optional_json(pathlib.Path(context["audit_data"]))
        transcription_data = read_optional_json(pathlib.Path(context["transcription_data"]))
        usage = summarize_token_usage(visual_data, audit_data)
        row = build_processing_row(
            report,
            usage,
            asr_duration_seconds=transcription_duration_seconds(transcription_data),
            started_at=context["started_at"],
            finished_at=finished_at,
            total_elapsed_seconds=total_elapsed,
            report_path=report_path,
        )
        row["错误信息"] = metadata["error"]
        upsert_processing_stats(pathlib.Path(context["processing_stats"]), row)
    except Exception as stats_exc:
        append_log(
            ACTIVE_LOG_FILE,
            {
                "event": "failure_stats_write_failed",
                "error_type": type(stats_exc).__name__,
                "error": str(stats_exc),
            },
        )
    return report_path


def discover_batch_videos(input_dir: pathlib.Path, recursive: bool) -> list[pathlib.Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(input_dir)).casefold(),
    )


def validate_batch_output_dir(path: pathlib.Path, input_dir: pathlib.Path) -> pathlib.Path:
    output = path.expanduser().resolve()
    if output == input_dir:
        raise ValueError(f"Batch output directory cannot equal the input directory: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is a file, not a directory: {output}")
    return output


def batch_video_output_dir(
    video: pathlib.Path,
    input_dir: pathlib.Path,
    output_root: pathlib.Path,
    used: set[str],
) -> pathlib.Path:
    relative = video.relative_to(input_dir)
    candidate = output_root / relative.parent / safe_stem(video.stem)
    key = str(candidate).casefold()
    if key in used:
        candidate = candidate.with_name(f"{candidate.name}_{video.suffix.lower().lstrip('.')}")
        key = str(candidate).casefold()
    used.add(key)
    return candidate


def batch_business_lecture_paths(
    video: pathlib.Path,
    video_output: pathlib.Path,
    output_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    source = video_output / f"{safe_stem(video.stem)}_业务讲义.md"
    relative_output = video_output.relative_to(output_root)
    relative_identity = f"{relative_output.as_posix()}|{video.suffix.lower()}"
    digest = hashlib.sha256(relative_identity.encode("utf-8")).hexdigest()[:10]
    suffix = f"_业务讲义_{digest}.md"
    readable_limit = MAX_COLLECTED_FILENAME_LENGTH - len(suffix)
    collected_stem = safe_stem("__".join(relative_output.parts))[:readable_limit].rstrip(" ._")
    collected_stem = collected_stem or "video-course"
    destination = (
        output_root
        / BUSINESS_LECTURES_DIRNAME
        / f"{collected_stem}{suffix}"
    )
    return source, destination


def collect_batch_business_lecture(
    video: pathlib.Path,
    video_output: pathlib.Path,
    output_root: pathlib.Path,
) -> dict[str, str]:
    source, destination = batch_business_lecture_paths(video, video_output, output_root)
    if not source.is_file():
        raise FileNotFoundError(f"Expected business lecture was not created: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{ACTIVE_RUN_ID or 'run'}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "copied",
        "source": str(source),
        "destination": str(destination),
    }


def read_business_lecture_manifest(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def managed_collection_path(collection_dir: pathlib.Path, filename: str) -> pathlib.Path | None:
    if not filename or pathlib.Path(filename).name != filename:
        return None
    candidate = (collection_dir / filename).resolve()
    if candidate.parent != collection_dir.resolve():
        return None
    return candidate


def write_json_atomic(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_business_lecture_manifest(
    manifest_path: pathlib.Path,
    videos: list[pathlib.Path],
    results: list[dict],
) -> tuple[dict, list[str]]:
    collection_dir = manifest_path.parent
    previous = read_business_lecture_manifest(manifest_path)
    previous_entries = {
        item.get("video"): item
        for item in previous.get("entries", [])
        if isinstance(item, dict) and item.get("video")
    }
    current_videos = {str(video) for video in videos}
    current_entries: dict[str, dict] = {}
    cleanup_errors: list[str] = []
    cleanup_candidates = {
        filename
        for filename in previous.get("pending_cleanup", [])
        if isinstance(filename, str)
    }

    for result in results:
        lecture = result.get("business_lecture", {})
        if lecture.get("status") != "copied":
            continue
        destination = pathlib.Path(lecture["destination"])
        current_entries[result["video"]] = {
            "video": result["video"],
            "filename": destination.name,
            "source": lecture["source"],
            "status": "current",
        }

    for video, entry in previous_entries.items():
        filename = entry.get("filename")
        if not isinstance(filename, str):
            filename = None
        managed_path = managed_collection_path(collection_dir, filename) if filename else None
        if video not in current_videos:
            if managed_path:
                cleanup_candidates.add(filename)
            continue
        current_filename = current_entries.get(video, {}).get("filename")
        if current_filename:
            if managed_path and filename != current_filename:
                cleanup_candidates.add(filename)
            continue
        if video not in current_entries and managed_path and managed_path.is_file():
            retained = dict(entry)
            retained["status"] = "retained_after_failure"
            current_entries[video] = retained

    current_filenames = {entry["filename"] for entry in current_entries.values()}
    pending_cleanup: list[str] = []
    for filename in sorted(cleanup_candidates, key=str.casefold):
        if filename in current_filenames:
            continue
        managed_path = managed_collection_path(collection_dir, filename)
        if managed_path is None or not managed_path.exists():
            continue
        try:
            managed_path.unlink()
        except OSError as exc:
            cleanup_errors.append(f"{managed_path}: {exc}")
            pending_cleanup.append(filename)

    manifest = {
        "schema_version": 1,
        "run_id": ACTIVE_RUN_ID,
        "generated_at": timestamp_now(),
        "entries": [current_entries[key] for key in sorted(current_entries, key=str.casefold)],
        "pending_cleanup": pending_cleanup,
        "cleanup_errors": cleanup_errors,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest, cleanup_errors


def build_batch_child_command(
    args: argparse.Namespace,
    video: pathlib.Path,
    output_dir: pathlib.Path,
    stats_path: pathlib.Path,
) -> list[str]:
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        str(video),
        str(output_dir),
        "--mode",
        args.mode,
        "--scan-fps",
        str(args.scan_fps),
        "--visual-model",
        args.visual_model,
        "--asr-model",
        args.asr_model,
        "--text-model",
        args.text_model,
        "--group-size",
        str(args.group_size),
        "--workers",
        str(args.workers),
        "--stats-file",
        str(stats_path),
        "--batch-mode",
    ]
    for option, value in (
        ("--env-file", args.env_file),
        ("--min-spacing", args.min_spacing),
        ("--pydeps", args.pydeps),
        ("--corrections", args.corrections),
        ("--log-file", args.log_file),
        ("--run-id", args.run_id),
        ("--configuration-source", args.configuration_source),
        ("--log-max-mb", args.log_max_mb),
        ("--log-backups", args.log_backups),
    ):
        if value is not None:
            command.extend([option, str(value)])
    for flag, enabled in (
        ("--skip-asr", args.skip_asr),
        ("--skip-business", args.skip_business),
        ("--dry-run", args.dry_run),
        ("--json-output", args.json_output),
    ):
        if enabled:
            command.append(flag)
    return command


def run_batch_child(command: list[str]) -> int:
    return subprocess.run(
        command,
        check=False,
        env=utf8_child_environment(),
    ).returncode


def run_batch(args: argparse.Namespace, input_dir: pathlib.Path) -> int:
    global ACTIVE_JSON_OUTPUT, ACTIVE_LOG_FILE
    if args.title:
        raise ValueError("--title can only be used with a single input video")
    output_root = validate_batch_output_dir(pathlib.Path(args.output_dir), input_dir)
    videos = discover_batch_videos(input_dir, args.recursive)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in input directory: {input_dir}")

    stats_path = (
        pathlib.Path(args.stats_file).expanduser().resolve()
        if args.stats_file
        else output_root / "视频处理统计.xlsx"
    )
    batch_report_path = output_root / "批量处理报告.json"
    business_lectures_dir = (
        None if args.skip_business else output_root / BUSINESS_LECTURES_DIRNAME
    )
    business_lectures_manifest = (
        None
        if business_lectures_dir is None
        else business_lectures_dir / BUSINESS_LECTURES_MANIFEST
    )
    log_path = (
        pathlib.Path(args.log_file).expanduser().resolve()
        if args.log_file
        else output_root / "视频处理日志.jsonl"
    )
    args.log_file = str(log_path)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        ACTIVE_LOG_FILE = log_path
        ACTIVE_JSON_OUTPUT = args.json_output
        append_log(
            log_path,
            {
                "event": "batch_started",
                "input_dir": str(input_dir),
                "output_dir": str(output_root),
                "total_videos": len(videos),
            },
        )
    if not args.json_output:
        print(f"[批次] 共 {len(videos)} 个视频", flush=True)
        if not args.dry_run:
            print(f"[日志] {log_path.name}", flush=True)

    batch_started_monotonic = time.monotonic()
    batch_started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    results = []
    used_outputs: set[str] = set()
    stop_detail: dict | None = None
    for index, video in enumerate(videos, start=1):
        video_output = batch_video_output_dir(video, input_dir, output_root, used_outputs)
        if stop_detail is not None:
            skipped = {
                "video": str(video),
                "output_dir": str(video_output),
                "status": "skipped",
                "returncode": None,
                "elapsed_seconds": 0.0,
                "error": stop_detail["error"],
                "error_category": stop_detail["error_category"],
                "http_status": stop_detail.get("http_status"),
                "retryable": False,
                "reason": "batch stopped after a permanent credential failure",
            }
            if not args.skip_business:
                source, destination = batch_business_lecture_paths(
                    video, video_output, output_root
                )
                skipped["business_lecture"] = {
                    "status": "not-created",
                    "source": str(source),
                    "destination": str(destination),
                    "reason": skipped["reason"],
                }
            results.append(skipped)
            append_log(log_path, {"event": "batch_video_skipped", **skipped})
            emit(
                args.json_output,
                {"stage": "batch-video", "current": index, "total": len(videos), **skipped},
                f"[{index}/{len(videos)}] 跳过（凭证错误）",
            )
            continue
        command = build_batch_child_command(args, video, video_output, stats_path)
        relative_video = video.relative_to(input_dir)
        batch_video_event = {
            "stage": "batch-video",
            "current": index,
            "total": len(videos),
            "video": str(video),
            "output_dir": str(video_output),
        }
        append_log(
            None if args.dry_run else log_path,
            {"event": "batch_video_started", **batch_video_event},
        )
        emit(
            args.json_output,
            batch_video_event,
            f"[{index}/{len(videos)}] {compact_path(relative_video)}",
        )
        child_started_monotonic = time.monotonic()
        child_started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        returncode = run_batch_child(command)
        child_finished_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        child_elapsed = round(time.monotonic() - child_started_monotonic, 2)
        child_status = "planned" if args.dry_run and returncode == 0 else (
            "complete" if returncode == 0 else "failed"
        )
        result = {
            "video": str(video),
            "output_dir": str(video_output),
            "status": child_status,
            "returncode": returncode,
            "elapsed_seconds": child_elapsed,
        }
        failed_report_path = video_output / f"{safe_stem(video.stem)}_处理报告.json"
        child_failure = (
            read_optional_json(failed_report_path)
            if returncode != 0 and not args.dry_run
            else {}
        )
        if child_failure.get("status") == "failed":
            result.update(
                {
                    "error": child_failure.get(
                        "error", f"child process exited with exit code {returncode}"
                    ),
                    "error_category": child_failure.get("error_category", "child_process"),
                    "http_status": child_failure.get("http_status"),
                    "provider_code": child_failure.get("provider_code"),
                    "retryable": bool(child_failure.get("retryable", False)),
                    "failure_report": str(failed_report_path),
                }
            )
        elif returncode != 0 and not args.dry_run:
            result.update(
                {
                    "error": f"child process exited with exit code {returncode}",
                    "error_category": "child_process",
                    "http_status": None,
                    "retryable": False,
                    "failure_report": str(failed_report_path),
                }
            )
        if not args.skip_business:
            source, destination = batch_business_lecture_paths(video, video_output, output_root)
            if args.dry_run:
                result["business_lecture"] = {
                    "status": "planned",
                    "source": str(source),
                    "destination": str(destination),
                }
            elif returncode == 0:
                try:
                    result["business_lecture"] = collect_batch_business_lecture(
                        video,
                        video_output,
                        output_root,
                    )
                except OSError as exc:
                    result["business_lecture"] = {
                        "status": "failed",
                        "source": str(source),
                        "destination": str(destination),
                        "error": str(exc),
                    }
            else:
                result["business_lecture"] = {
                    "status": "not-created",
                    "source": str(source),
                    "destination": str(destination),
                    "reason": "video processing failed",
                }
        results.append(result)
        append_log(
            None if args.dry_run else log_path,
            {"event": "batch_video_finished", **result},
        )
        if not args.json_output:
            status_text = {
                "planned": "计划完成",
                "complete": "完成",
                "failed": "失败",
            }[child_status]
            print(
                f"[{index}/{len(videos)}] {status_text}（{child_elapsed:.2f} 秒）",
                flush=True,
            )
        if returncode != 0 and not args.dry_run:
            if not child_failure:
                error = f"child process exited with exit code {returncode}"
                result.update(
                    {
                        "error": error,
                        "error_category": "child_process",
                        "http_status": None,
                        "retryable": False,
                        "failure_report": str(failed_report_path),
                    }
                )
                failed_report = {
                    "video": str(video),
                    "output_dir": str(video_output),
                    "status": "failed",
                    "error": error,
                    "error_category": "child_process",
                    "http_status": None,
                    "retryable": False,
                    "models": {
                        "visual": args.visual_model,
                        "asr": None if args.skip_asr else args.asr_model,
                        "text": None if args.skip_business else args.text_model,
                    },
                    "stages": [],
                    "timing": {
                        "started_at": child_started_at,
                        "finished_at": child_finished_at,
                        "total_elapsed_seconds": child_elapsed,
                    },
                }
                video_output.mkdir(parents=True, exist_ok=True)
                write_json_atomic(failed_report_path, failed_report)
                zero_usage = summarize_token_usage({}, {})
                row = build_processing_row(
                    failed_report,
                    zero_usage,
                    asr_duration_seconds=0,
                    started_at=child_started_at,
                    finished_at=child_finished_at,
                    total_elapsed_seconds=child_elapsed,
                    report_path=failed_report_path,
                )
                row["错误信息"] = error
                upsert_processing_stats(stats_path, row)
            if result.get("error_category") in {"authentication", "permission"}:
                stop_detail = {
                    "error": result["error"],
                    "error_category": result["error_category"],
                    "http_status": result.get("http_status"),
                }

    manifest_errors: list[str] = []
    cleanup_errors: list[str] = []
    if not args.dry_run and business_lectures_manifest is not None:
        try:
            _, cleanup_errors = update_business_lecture_manifest(
                business_lectures_manifest,
                videos,
                results,
            )
        except OSError as exc:
            manifest_errors.append(str(exc))

    failed_count = sum(item["status"] == "failed" for item in results)
    business_lecture_failures = sum(
        item.get("business_lecture", {}).get("status") == "failed" for item in results
    ) + len(cleanup_errors) + len(manifest_errors)
    completed_count = sum(item["status"] == "complete" for item in results)
    planned_count = sum(item["status"] == "planned" for item in results)
    skipped_count = sum(item["status"] == "skipped" for item in results)
    batch_status = "planned" if args.dry_run and not failed_count else (
        "complete" if not failed_count and not business_lecture_failures else "partial"
    )
    batch_report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "status": batch_status,
        "started_at": batch_started_at,
        "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - batch_started_monotonic, 2),
        "total_videos": len(videos),
        "completed_videos": completed_count,
        "failed_videos": failed_count,
        "planned_videos": planned_count,
        "skipped_videos": skipped_count,
        "stopped_early": stop_detail is not None,
        "stop_category": stop_detail.get("error_category") if stop_detail else None,
        "stop_reason": stop_detail.get("error") if stop_detail else None,
        "processing_stats": str(stats_path),
        "log_file": str(log_path),
        "run_id": args.run_id,
        "configuration_source": args.configuration_source,
        "api_protocols": selected_api_protocols(args),
        "business_lectures_dir": (
            str(business_lectures_dir) if business_lectures_dir is not None else None
        ),
        "business_lectures_manifest": (
            str(business_lectures_manifest)
            if business_lectures_manifest is not None
            else None
        ),
        "business_lecture_failures": business_lecture_failures,
        "business_lecture_manifest_errors": manifest_errors,
        "business_lecture_cleanup_errors": cleanup_errors,
        "results": results,
    }
    if not args.dry_run:
        batch_report_path.write_text(
            json.dumps(batch_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        append_log(log_path, {"event": "batch_finished", **batch_report})
    batch_summary = {
        **batch_report,
        "batch_report": str(batch_report_path),
    }
    emit(
        args.json_output,
        batch_summary,
        (
            f"[批次完成] 成功 {completed_count}，失败 {failed_count}，"
            f"讲义汇总失败 {business_lecture_failures}"
            if not args.dry_run
            else f"[计划完成] 共 {len(videos)} 个视频"
        ),
    )
    return 1 if failed_count or business_lecture_failures else 0


def main() -> int:
    global ACTIVE_JSON_OUTPUT, ACTIVE_LOG_FILE, ACTIVE_REPORT_CONTEXT
    global ACTIVE_RUN_ID, LOG_BACKUPS, LOG_MAX_BYTES
    parser = argparse.ArgumentParser(
        description="Convert a course video into an evidence Markdown and learner-facing Markdown."
    )
    parser.add_argument("video", help="Local video file or directory containing videos")
    parser.add_argument("output_dir", help="Local output directory")
    parser.add_argument("--title")
    parser.add_argument("--mode", choices=("auto", "slides", "live"), default="auto")
    parser.add_argument("--scan-fps", type=float, default=2.0)
    parser.add_argument("--visual-model", default="qwen3-vl-flash")
    parser.add_argument("--asr-model", default="paraformer-v2")
    parser.add_argument("--text-model", default="qwen-plus")
    parser.add_argument("--env-file")
    parser.add_argument(
        "--api-key",
        help="One unified DashScope API Key for visual, ASR, and text models",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-spacing", type=float)
    parser.add_argument("--pydeps")
    parser.add_argument("--corrections")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-business", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--log-file",
        help="Persistent JSONL log path (default: 视频处理日志.jsonl in the output)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print machine-readable JSON events instead of concise human progress",
    )
    parser.add_argument(
        "--log-max-mb",
        type=float,
        default=20.0,
        help="Rotate the JSONL log after this many MiB (default: 20)",
    )
    parser.add_argument(
        "--log-backups",
        type=int,
        default=5,
        help="Number of rotated log files to keep (default: 5)",
    )
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--configuration-source", help=argparse.SUPPRESS)
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ACTIVE_LOG_FILE = None
    ACTIVE_REPORT_CONTEXT = None
    ACTIVE_JSON_OUTPUT = args.json_output
    ACTIVE_RUN_ID = args.run_id or uuid.uuid4().hex
    args.run_id = ACTIVE_RUN_ID
    if args.log_max_mb <= 0:
        raise ValueError("--log-max-mb must be greater than zero")
    if args.log_backups < 1:
        raise ValueError("--log-backups must be at least one")
    LOG_MAX_BYTES = max(1, int(args.log_max_mb * 1024 * 1024))
    LOG_BACKUPS = args.log_backups
    if args.api_key is not None:
        unified_key = args.api_key.strip()
        if not unified_key:
            raise ValueError("--api-key cannot be empty")
        os.environ["DASHSCOPE_API_KEY"] = unified_key
        os.environ["DASHSCOPE_QWEN_API_KEY"] = unified_key
        os.environ["DASHSCOPE_ASR_API_KEY"] = unified_key
        args.configuration_source = "cli --api-key"
    elif args.configuration_source is None:
        args.configuration_source = None
    if args.log_file and not args.dry_run:
        ACTIVE_LOG_FILE = pathlib.Path(args.log_file).expanduser().resolve()
    resolved_pydeps = discover_pydeps(args.pydeps)
    if resolved_pydeps:
        args.pydeps = str(resolved_pydeps)
        pydeps_text = str(resolved_pydeps)
        if pydeps_text not in sys.path:
            sys.path.insert(0, pydeps_text)
    if not args.dry_run:
        validate_xlsx_support()
    run_started_monotonic = time.monotonic()
    run_started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    input_path = pathlib.Path(args.video).expanduser().resolve()
    if input_path.is_dir():
        configuration = load_dashscope_environment(args.env_file, PROJECT_ROOT)
        if args.configuration_source:
            configuration["source"] = args.configuration_source
        else:
            args.configuration_source = configuration["source"]
        validate_required_api_keys(args, configuration)
        return run_batch(args, input_path)
    video = validate_video(input_path)
    output_dir = validate_output_dir(pathlib.Path(args.output_dir), video)
    configuration = load_dashscope_environment(args.env_file, PROJECT_ROOT)
    if args.configuration_source:
        configuration["source"] = args.configuration_source
    else:
        args.configuration_source = configuration["source"]
    validate_required_api_keys(args, configuration)
    title = args.title or video.stem
    stem = safe_stem(title)
    # 单跑模式下把所有产物收到 output_dir / stem 子目录下，与批处理模式
    # （父进程已通过 --batch-mode 传入分配好的 output_dir / stem）的产物结构
    # 保持一致，避免散落到 output_dir 根下产生孤儿目录。
    if not args.batch_mode:
        output_dir = output_dir / stem
    log_path = (
        pathlib.Path(args.log_file).expanduser().resolve()
        if args.log_file
        else output_dir / "视频处理日志.jsonl"
    )
    work_dir = output_dir / "_video_course_work" / stem
    frames_dir = work_dir / "frames"
    visual_dir = work_dir / "visual"
    asr_dir = work_dir / "asr"
    transcription = asr_dir / "transcription.json"
    audio_source = asr_dir / "audio_16k_mono.mp3"
    manifest = frames_dir / "manifest.json"
    contact_sheet = output_dir / f"{stem}_关键帧总览.jpg"
    evidence = output_dir / f"{stem}_技术证据稿.md"
    evidence_assets = output_dir / f"{stem}_技术证据稿_assets"
    visual_json = output_dir / f"{stem}_画面证据.json"
    asr_json = output_dir / f"{stem}_音频证据.json"
    audio_output = output_dir / f"{stem}_提取音频.mp3"
    learner = output_dir / f"{stem}_业务讲义.md"
    audit = output_dir / f"{stem}_内容保留校验.json"
    report_path = output_dir / f"{stem}_处理报告.json"
    processing_stats = (
        pathlib.Path(args.stats_file).expanduser().resolve()
        if args.stats_file
        else output_dir / "视频处理统计.xlsx"
    )
    pydeps = discover_pydeps(args.pydeps)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        ACTIVE_LOG_FILE = log_path
        ACTIVE_JSON_OUTPUT = args.json_output
        append_log(
            log_path,
            {
                "event": "pipeline_started",
                "video": str(video),
                "output_dir": str(output_dir),
                "models": {
                    "visual": args.visual_model,
                    "asr": None if args.skip_asr else args.asr_model,
                    "text": None if args.skip_business else args.text_model,
                },
            },
        )
    if not args.json_output and not args.batch_mode:
        print(f"[视频] {compact_path(video.name)}", flush=True)
        if not args.dry_run:
            print(f"[日志] {log_path.name}", flush=True)
    report = {
        "run_id": args.run_id,
        "video": str(video),
        "title": title,
        "output_dir": str(output_dir),
        "models": {
            "visual": args.visual_model,
            "asr": args.asr_model,
            "text": None if args.skip_business else args.text_model,
        },
        "api_protocols": selected_api_protocols(args),
        "configuration": {
            "env_source": configuration["source"],
            "api_key_set": configuration["api_key_set"],
            "qwen_api_key_set": configuration["qwen_api_key_set"],
            "asr_api_key_set": configuration["asr_api_key_set"],
        },
        "pydeps": str(pydeps) if pydeps else None,
        "_runtime": {
            "json_output": args.json_output,
            "log_file": None if args.dry_run else str(log_path),
        },
        "stages": [],
        "outputs": {
            "evidence_markdown": str(evidence),
            "learner_markdown": None if args.skip_business else str(learner),
            "preservation_audit": None if args.skip_business else str(audit),
            "contact_sheet": str(contact_sheet),
            "visual_json": str(visual_json),
            "asr_json": str(asr_json),
            "audio": None if args.skip_asr else str(audio_output),
            "processing_stats": str(processing_stats),
            "log_file": str(log_path),
        },
    }
    if not args.dry_run:
        ACTIVE_REPORT_CONTEXT = {
            "report": report,
            "report_path": str(report_path),
            "processing_stats": str(processing_stats),
            "visual_data": str(visual_dir / "consolidated.json"),
            "audit_data": str(audit),
            "transcription_data": str(transcription),
            "started_at": run_started_at,
            "started_monotonic": run_started_monotonic,
        }

    extract_command = [
        sys.executable,
        script("extract_adaptive_keyframes.py"),
        str(video),
        str(frames_dir),
        "--mode",
        args.mode,
        "--fps",
        str(args.scan_fps),
    ]
    add_pydeps(extract_command, pydeps)
    run_stage("adaptive-keyframes", extract_command, report, args.dry_run)

    if args.dry_run:
        resolved_mode = "slides" if args.mode == "auto" else args.mode
    else:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        resolved_mode = manifest_data["mode"]
    report["resolved_mode"] = resolved_mode

    run_stage(
        "contact-sheet",
        [
            sys.executable,
            script("make_contact_sheet.py"),
            str(frames_dir / "keyframes"),
            str(contact_sheet),
        ],
        report,
        args.dry_run,
    )

    min_spacing = args.min_spacing
    if min_spacing is None:
        min_spacing = 0.0 if resolved_mode == "slides" else 3.0
    run_stage(
        "visual-analysis",
        [
            sys.executable,
            script("analyze_video_frames.py"),
            str(manifest),
            str(visual_dir),
            "--model",
            args.visual_model,
            "--group-size",
            str(args.group_size),
            "--workers",
            str(args.workers),
            "--min-spacing",
            str(min_spacing),
        ],
        report,
        args.dry_run,
    )

    if args.skip_asr:
        if not args.dry_run:
            write_empty_transcription(transcription, video)
            audio_source.parent.mkdir(parents=True, exist_ok=True)
            audio_source.write_bytes(b"")
    else:
        asr_command = [
            sys.executable,
            script("transcribe_video_audio.py"),
            str(video),
            str(asr_dir),
            "--model",
            args.asr_model,
        ]
        add_pydeps(asr_command, pydeps)
        run_stage("audio-transcription", asr_command, report, args.dry_run)

    actual_visual_model = args.visual_model
    actual_asr_model = args.asr_model
    if not args.dry_run:
        visual_data = json.loads((visual_dir / "consolidated.json").read_text(encoding="utf-8"))
        actual_visual_model = visual_data.get("model", args.visual_model)
        transcription_data = read_optional_json(transcription)
        asr_metadata = transcription_data.get("_video_course_metadata")
        asr_fingerprint = read_checkpoint_fingerprint(
            transcription.with_name("transcription.checkpoint.json")
        )
        if asr_fingerprint and asr_fingerprint.get("state") == "skipped":
            actual_asr_model = None
        elif (
            isinstance(asr_fingerprint, dict)
            and asr_fingerprint.get("state") == "transcribed"
            and asr_fingerprint.get("model") == args.asr_model
            and isinstance(asr_metadata, dict)
            and asr_metadata.get("requested_model") == args.asr_model
            and isinstance(asr_metadata.get("producer_model"), str)
            and asr_metadata["producer_model"]
        ):
            actual_asr_model = asr_metadata["producer_model"]
        elif asr_fingerprint:
            actual_asr_model = asr_fingerprint.get("model", args.asr_model)
        report["models"]["visual"] = actual_visual_model
        report["models"]["asr"] = actual_asr_model

    assemble_command = [
        sys.executable,
        script("assemble_evidence_markdown.py"),
        "--title",
        title,
        "--mode",
        resolved_mode,
        "--visual",
        str(visual_dir / "consolidated.json"),
        "--asr",
        str(transcription),
        "--output",
        str(evidence),
        "--assets",
        str(evidence_assets),
        "--visual-json-output",
        str(visual_json),
        "--asr-json-output",
        str(asr_json),
        "--audio-source",
        str(audio_source),
        "--audio-output",
        str(audio_output),
        "--visual-model",
        actual_visual_model,
        "--asr-model",
        actual_asr_model or "未执行（--skip-asr）",
    ]
    if args.corrections:
        assemble_command.extend(["--corrections", str(pathlib.Path(args.corrections).resolve())])
    run_stage("evidence-markdown", assemble_command, report, args.dry_run)

    if not args.skip_business:
        run_stage(
            "learner-markdown-and-audit",
            [
                sys.executable,
                script("rewrite_learner_markdown.py"),
                str(evidence),
                str(learner),
                "--model",
                args.text_model,
                "--audit-output",
                str(audit),
            ],
            report,
            args.dry_run,
        )

    if not args.dry_run:
        report["status"] = "complete"
        finished_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        total_elapsed = round(time.monotonic() - run_started_monotonic, 2)
        audit_data = read_optional_json(audit)
        transcription_data = read_optional_json(transcription)
        if not args.skip_business:
            text_producer_model = audit_data.get("model")
            if isinstance(text_producer_model, str) and text_producer_model:
                report["models"]["text"] = text_producer_model
        usage = summarize_token_usage(visual_data, audit_data)
        asr_duration = transcription_duration_seconds(transcription_data)
        report["usage"] = {
            **usage,
            "asr_audio_seconds": asr_duration,
        }
        report["timing"] = {
            "started_at": run_started_at,
            "finished_at": finished_at,
            "total_elapsed_seconds": total_elapsed,
        }
        report.pop("_runtime", None)
        stats_row = build_processing_row(
            report,
            usage,
            asr_duration_seconds=asr_duration,
            started_at=run_started_at,
            finished_at=finished_at,
            total_elapsed_seconds=total_elapsed,
            report_path=report_path,
        )
        upsert_processing_stats(processing_stats, stats_row)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        append_log(
            log_path,
            {
                "event": "pipeline_finished",
                "video": str(video),
                "report": str(report_path),
                "outputs": report["outputs"],
                "elapsed_seconds": total_elapsed,
            },
        )
    summary = {
        "run_id": args.run_id,
        "status": "planned" if args.dry_run else "complete",
        "report": str(report_path),
        "outputs": report["outputs"],
        "models": report["models"],
        "api_protocols": report["api_protocols"],
        "configuration": report["configuration"],
    }
    if args.json_output or not args.batch_mode:
        emit(
            args.json_output,
            summary,
            (
                f"[计划完成] {compact_path(video.name)}"
                if args.dry_run
                else f"[处理完成] {compact_path(video.name)}（{total_elapsed:.2f} 秒）"
            ),
        )
    ACTIVE_REPORT_CONTEXT = None
    return 0


def cli() -> int:
    try:
        return main()
    except Exception as exc:
        safe_error = redact_text(str(exc))
        failure_report = None
        try:
            failure_report = persist_active_failure_report(exc)
        except Exception as report_exc:
            append_log(
                ACTIVE_LOG_FILE,
                {
                    "event": "failure_report_write_failed",
                    "error_type": type(report_exc).__name__,
                    "error": str(report_exc),
                },
            )
        metadata = exception_metadata(exc)
        append_log(
            ACTIVE_LOG_FILE,
            {
                "event": "pipeline_failed",
                "error_type": type(exc).__name__,
                "error": safe_error,
                **metadata,
                "failure_report": str(failure_report) if failure_report else None,
                "traceback": traceback.format_exc(),
            },
        )
        if ACTIVE_JSON_OUTPUT:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": safe_error,
                        **metadata,
                        "log_file": str(ACTIVE_LOG_FILE) if ACTIVE_LOG_FILE else None,
                        "failure_report": str(failure_report) if failure_report else None,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"[失败] {type(exc).__name__}: {concise_error(exc)}", file=sys.stderr)
            if ACTIVE_LOG_FILE is not None:
                print(f"[日志] {ACTIVE_LOG_FILE.name}", file=sys.stderr)
            if failure_report is not None:
                print(f"[失败报告] {failure_report.name}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
