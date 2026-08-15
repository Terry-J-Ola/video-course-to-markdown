import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time

try:
    from .checkpoint_provenance import file_identity, read_checkpoint_fingerprint, write_checkpoint
    from .env_config import load_dashscope_environment
    from .processing_metrics import (
        build_processing_row,
        summarize_token_usage,
        transcription_duration_seconds,
        upsert_processing_stats,
    )
except ImportError:
    from checkpoint_provenance import file_identity, read_checkpoint_fingerprint, write_checkpoint
    from env_config import load_dashscope_environment
    from processing_metrics import (
        build_processing_row,
        summarize_token_usage,
        transcription_duration_seconds,
        upsert_processing_stats,
    )


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PROJECT_ROOT = SCRIPT_DIR.parent


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


def run_stage(
    name: str,
    command: list[str],
    report: dict,
    dry_run: bool,
) -> None:
    report["stages"].append({"name": name, "command": command, "status": "planned"})
    if dry_run:
        print(json.dumps({"stage": name, "command": command}, ensure_ascii=False))
        return
    started = time.monotonic()
    print(json.dumps({"stage": name, "status": "running"}, ensure_ascii=False), flush=True)
    child_environment = os.environ.copy()
    pydeps = report.get("pydeps")
    if pydeps:
        inherited = child_environment.get("PYTHONPATH")
        child_environment["PYTHONPATH"] = (
            os.pathsep.join([pydeps, inherited]) if inherited else pydeps
        )
    subprocess.run(command, check=True, env=child_environment)
    report["stages"][-1]["status"] = "complete"
    report["stages"][-1]["elapsed_seconds"] = round(time.monotonic() - started, 2)


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
    ):
        if value is not None:
            command.extend([option, str(value)])
    for flag, enabled in (
        ("--skip-asr", args.skip_asr),
        ("--skip-business", args.skip_business),
        ("--dry-run", args.dry_run),
    ):
        if enabled:
            command.append(flag)
    return command


def run_batch_child(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def run_batch(args: argparse.Namespace, input_dir: pathlib.Path) -> int:
    if args.title:
        raise ValueError("--title can only be used with a single input video")
    output_root = validate_batch_output_dir(pathlib.Path(args.output_dir), input_dir)
    videos = discover_batch_videos(input_dir, args.recursive)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in input directory: {input_dir}")

    stats_path = (
        pathlib.Path(args.stats_file).expanduser().resolve()
        if args.stats_file
        else output_root / "视频处理统计.csv"
    )
    batch_report_path = output_root / "批量处理报告.json"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    batch_started_monotonic = time.monotonic()
    batch_started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    results = []
    used_outputs: set[str] = set()
    for index, video in enumerate(videos, start=1):
        video_output = batch_video_output_dir(video, input_dir, output_root, used_outputs)
        command = build_batch_child_command(args, video, video_output, stats_path)
        print(
            json.dumps(
                {
                    "stage": "batch-video",
                    "current": index,
                    "total": len(videos),
                    "video": str(video),
                    "output_dir": str(video_output),
                },
                ensure_ascii=False,
            ),
            flush=True,
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
        results.append(result)
        if returncode != 0 and not args.dry_run:
            error = f"child process exited with exit code {returncode}"
            result["error"] = error
            failed_report_path = video_output / f"{safe_stem(video.stem)}_处理报告.json"
            failed_report = {
                "video": str(video),
                "output_dir": str(video_output),
                "status": "failed",
                "error": error,
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
            failed_report_path.write_text(
                json.dumps(failed_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
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

    failed_count = sum(item["status"] == "failed" for item in results)
    completed_count = sum(item["status"] == "complete" for item in results)
    planned_count = sum(item["status"] == "planned" for item in results)
    batch_status = "planned" if args.dry_run and not failed_count else (
        "complete" if not failed_count else "partial"
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
        "processing_stats": str(stats_path),
        "results": results,
    }
    if not args.dry_run:
        batch_report_path.write_text(
            json.dumps(batch_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                **batch_report,
                "batch_report": str(batch_report_path),
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed_count else 0


def main() -> int:
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
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-spacing", type=float)
    parser.add_argument("--pydeps")
    parser.add_argument("--corrections")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-business", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_started_monotonic = time.monotonic()
    run_started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    input_path = pathlib.Path(args.video).expanduser().resolve()
    if input_path.is_dir():
        return run_batch(args, input_path)
    video = validate_video(input_path)
    output_dir = validate_output_dir(pathlib.Path(args.output_dir), video)
    configuration = load_dashscope_environment(args.env_file, PROJECT_ROOT)
    if not args.dry_run and not configuration["qwen_api_key_set"]:
        raise RuntimeError(
            "DASHSCOPE_QWEN_API_KEY or DASHSCOPE_API_KEY is not configured. "
            "Set it in the environment, --env-file, project .env, or user-private .env."
        )
    if not args.dry_run and not args.skip_asr and not configuration["asr_api_key_set"]:
        raise RuntimeError(
            "DASHSCOPE_ASR_API_KEY or DASHSCOPE_API_KEY is not configured. "
            "Set it in the environment, --env-file, project .env, or user-private .env."
        )
    title = args.title or video.stem
    stem = safe_stem(title)
    # 单跑模式下把所有产物收到 output_dir / stem 子目录下，与批处理模式
    # （父进程已通过 --batch-mode 传入分配好的 output_dir / stem）的产物结构
    # 保持一致，避免散落到 output_dir 根下产生孤儿目录。
    if not args.batch_mode:
        output_dir = output_dir / stem
    work_dir = output_dir / "_video_course_work" / stem
    frames_dir = work_dir / "frames"
    visual_dir = work_dir / "visual"
    asr_dir = work_dir / "asr"
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
        else output_dir / "视频处理统计.csv"
    )
    pydeps = discover_pydeps(args.pydeps)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "video": str(video),
        "title": title,
        "output_dir": str(output_dir),
        "models": {
            "visual": args.visual_model,
            "asr": args.asr_model,
            "text": args.text_model,
        },
        "configuration": {
            "env_source": configuration["source"],
            "api_key_set": configuration["api_key_set"],
            "qwen_api_key_set": configuration["qwen_api_key_set"],
            "asr_api_key_set": configuration["asr_api_key_set"],
        },
        "pydeps": str(pydeps) if pydeps else None,
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
        },
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

    transcription = asr_dir / "transcription.json"
    audio_source = asr_dir / "audio_16k_mono.mp3"
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
        asr_fingerprint = read_checkpoint_fingerprint(
            transcription.with_name("transcription.checkpoint.json")
        )
        if asr_fingerprint and asr_fingerprint.get("state") == "skipped":
            actual_asr_model = None
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
    print(
        json.dumps(
            {
                "status": "planned" if args.dry_run else "complete",
                "report": str(report_path),
                "outputs": report["outputs"],
                "models": report["models"],
                "configuration": report["configuration"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
