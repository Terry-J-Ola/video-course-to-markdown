import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    from .checkpoint_provenance import checkpoint_matches, file_identity, write_checkpoint
    from .env_config import get_asr_api_key
except ImportError:
    from checkpoint_provenance import checkpoint_matches, file_identity, write_checkpoint
    from env_config import get_asr_api_key


def configure_dependencies(pydeps: pathlib.Path | None):
    if pydeps:
        sys.path.insert(0, str(pydeps))
    try:
        from dashscope.utils.oss_utils import OssUtils
    except ImportError as exc:
        raise RuntimeError(
            "dashscope is unavailable. Run bootstrap_dependencies.py first."
        ) from exc
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg, OssUtils
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable. Install ffmpeg or run bootstrap_dependencies.py."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe(), OssUtils


def extract_audio(ffmpeg: str, video: pathlib.Path, audio: pathlib.Path) -> None:
    if audio.exists() and audio.stat().st_size > 0:
        return
    audio.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(audio),
    ]
    subprocess.run(command, check=True)


def audio_fingerprint(video: pathlib.Path) -> dict:
    return {"stage": "audio-extraction", "input": file_identity(video)}


def ensure_audio(
    ffmpeg: str,
    video: pathlib.Path,
    audio: pathlib.Path,
    checkpoint_path: pathlib.Path,
) -> None:
    fingerprint = audio_fingerprint(video)
    if audio.exists() and audio.stat().st_size > 0 and checkpoint_matches(
        checkpoint_path, fingerprint
    ):
        return
    if audio.exists():
        audio.unlink()
    extract_audio(ffmpeg, video, audio)
    write_checkpoint(checkpoint_path, fingerprint)


def request_json(url: str, api_key: str, method="GET", payload=None, oss_resolve=False) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if method == "POST":
        headers["X-DashScope-Async"] = "enable"
    if oss_resolve:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def transcription_sentences(transcription: dict) -> list[dict]:
    if not isinstance(transcription, dict):
        raise ValueError("transcription result must be an object")
    transcripts = transcription.get("transcripts")
    if not isinstance(transcripts, list):
        raise ValueError("transcription result must contain a transcripts list")
    sentences = []
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            raise ValueError("transcription entries must be objects")
        transcript_sentences = transcript.get("sentences", [])
        if not isinstance(transcript_sentences, list):
            raise ValueError("transcript sentences must be a list")
        if not all(isinstance(sentence, dict) for sentence in transcript_sentences):
            raise ValueError("transcript sentence entries must be objects")
        sentences.extend(transcript_sentences)
    return sentences


def persist_transcription(
    transcription_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    transcription: dict,
    fingerprint: dict,
) -> list[dict]:
    sentences = transcription_sentences(transcription)
    transcription_path.write_text(
        json.dumps(transcription, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_checkpoint(checkpoint_path, fingerprint)
    return sentences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--pydeps", default=None)
    parser.add_argument("--model", default="paraformer-v2")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    api_key = get_asr_api_key()
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_ASR_API_KEY or DASHSCOPE_API_KEY is not configured"
        )

    video = pathlib.Path(args.video)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcription_path = output_dir / "transcription.json"
    checkpoint_path = output_dir / "transcription.checkpoint.json"
    fingerprint = {
        "stage": "asr",
        "input": file_identity(video),
        "state": "transcribed",
        "model": args.model,
    }
    if (
        transcription_path.exists()
        and not args.no_resume
        and checkpoint_matches(checkpoint_path, fingerprint)
    ):
        try:
            transcription = json.loads(transcription_path.read_text(encoding="utf-8"))
            sentences = transcription_sentences(transcription)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            checkpoint_path.unlink(missing_ok=True)
        else:
            print(
                json.dumps(
                    {
                        "stage": "complete",
                        "resumed": True,
                        "sentences": len(sentences),
                        "duration_ms": max(
                            (item.get("end_time", 0) for item in sentences),
                            default=0,
                        ),
                    }
                ),
                flush=True,
            )
            return 0
    audio_path = output_dir / "audio_16k_mono.mp3"
    audio_checkpoint_path = output_dir / "audio.checkpoint.json"
    pydeps = pathlib.Path(args.pydeps) if args.pydeps else None
    ffmpeg, OssUtils = configure_dependencies(pydeps)
    ensure_audio(ffmpeg, video, audio_path, audio_checkpoint_path)
    print(json.dumps({"stage": "audio_extracted", "bytes": audio_path.stat().st_size}), flush=True)

    oss_url, _ = OssUtils.upload(
        model=args.model,
        file_path=str(audio_path),
        api_key=api_key,
    )
    print(json.dumps({"stage": "audio_uploaded", "scheme": "oss"}), flush=True)

    if args.model.startswith("qwen3-asr-flash-filetrans"):
        payload = {
            "model": args.model,
            "input": {"file_url": oss_url},
            "parameters": {
                "channel_id": [0],
                "enable_itn": True,
                "enable_words": True,
            },
        }
    else:
        payload = {
            "model": args.model,
            "input": {"file_urls": [oss_url]},
            "parameters": {
                "channel_id": [0],
                "language_hints": ["zh", "en"],
                "disfluency_removal_enabled": False,
                "timestamp_alignment_enabled": True,
            },
        }
    submit = request_json(
        "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        api_key,
        method="POST",
        payload=payload,
        oss_resolve=True,
    )
    (output_dir / "submit.json").write_text(
        json.dumps(submit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    task_id = submit.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"ASR task was not created: {submit}")
    print(json.dumps({"stage": "task_submitted", "task_id_present": True}), flush=True)

    deadline = time.monotonic() + args.timeout_seconds
    task_result = None
    while time.monotonic() < deadline:
        task_result = request_json(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
            api_key,
            oss_resolve=True,
        )
        status = task_result.get("output", {}).get("task_status")
        if status in {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}:
            break
        time.sleep(args.poll_interval)
    if task_result is None:
        raise RuntimeError("ASR task returned no status")
    (output_dir / "task_result.json").write_text(
        json.dumps(task_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = task_result.get("output", {}).get("task_status")
    if status != "SUCCEEDED":
        raise RuntimeError(f"ASR task ended with status {status}: {task_result}")

    output = task_result.get("output", {})
    transcription_url = output.get("result", {}).get("transcription_url")
    if not transcription_url:
        results = output.get("results", [])
        if results:
            transcription_url = results[0].get("transcription_url")
    if not transcription_url:
        raise RuntimeError(f"No transcription URL in task result: {task_result}")
    with urllib.request.urlopen(transcription_url, timeout=180) as response:
        transcription = json.loads(response.read().decode("utf-8"))
    sentences = persist_transcription(
        transcription_path, checkpoint_path, transcription, fingerprint
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "sentences": len(sentences),
                "duration_ms": max((item.get("end_time", 0) for item in sentences), default=0),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
