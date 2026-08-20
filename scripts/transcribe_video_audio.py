import argparse
import json
import mimetypes
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

try:
    from .checkpoint_provenance import (
        checkpoint_matches,
        file_identity,
        write_checkpoint,
        write_json_atomic,
    )
    from .dashscope_errors import DashScopeAPIError, run_with_retries
    from .env_config import get_asr_api_key
    from .http_transport import open_url
except ImportError:
    from checkpoint_provenance import (
        checkpoint_matches,
        file_identity,
        write_checkpoint,
        write_json_atomic,
    )
    from dashscope_errors import DashScopeAPIError, run_with_retries
    from env_config import get_asr_api_key
    from http_transport import open_url


def multipart_form_data(fields: dict[str, str], file_path: pathlib.Path) -> tuple[bytes, str]:
    boundary = f"----video-course-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode("ascii"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return bytes(body), boundary


class SecureOssUtils:
    REQUIRED_UPLOAD_FIELDS = {
        "oss_access_key_id",
        "signature",
        "policy",
        "upload_dir",
        "x_oss_object_acl",
        "x_oss_forbid_overwrite",
        "upload_host",
    }

    @classmethod
    def upload(
        cls,
        model: str,
        file_path: str,
        api_key: str,
        upload_certificate: dict | None = None,
        **_kwargs,
    ) -> tuple[str, dict]:
        audio_path = pathlib.Path(file_path)
        if upload_certificate is None:
            query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
            response = request_json(
                f"https://dashscope.aliyuncs.com/api/v1/uploads?{query}",
                api_key,
            )
            upload_info = response.get("data")
            if upload_info is None:
                upload_info = response.get("output")
        else:
            upload_info = upload_certificate
        if not isinstance(upload_info, dict) or not cls.REQUIRED_UPLOAD_FIELDS.issubset(
            upload_info
        ):
            raise RuntimeError("DashScope upload certificate is incomplete")

        object_key = f"{upload_info['upload_dir']}/{audio_path.name}"
        fields = {
            "OSSAccessKeyId": upload_info["oss_access_key_id"],
            "Signature": upload_info["signature"],
            "policy": upload_info["policy"],
            "key": object_key,
            "x-oss-object-acl": upload_info["x_oss_object_acl"],
            "x-oss-forbid-overwrite": upload_info["x_oss_forbid_overwrite"],
            "success_action_status": "200",
            "x-oss-content-type": mimetypes.guess_type(audio_path.name)[0]
            or "application/octet-stream",
        }
        body, boundary = multipart_form_data(fields, audio_path)
        request = urllib.request.Request(
            upload_info["upload_host"],
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        def upload_audio() -> None:
            with open_url(request, timeout=3600):
                pass

        run_with_retries(
            upload_audio,
            service="asr-upload",
            max_attempts=3,
            sleep=time.sleep,
            context={"operation": "audio-upload"},
        )
        return f"oss://{object_key}", upload_info


def configure_dependencies(pydeps: pathlib.Path | None):
    if pydeps:
        sys.path.insert(0, str(pydeps))
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg, SecureOssUtils
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable. Install ffmpeg or run bootstrap_dependencies.py."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe(), SecureOssUtils


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

    def request_and_parse() -> dict:
        with open_url(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))

    return run_with_retries(
        request_and_parse,
        service="asr",
        max_attempts=3,
        sleep=time.sleep,
        context={"method": method},
    )


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


def response_model(requested_model: str, *responses: dict) -> str:
    for response in responses:
        if not isinstance(response, dict):
            continue
        output = response.get("output")
        candidates = [
            response.get("model"),
            output.get("model") if isinstance(output, dict) else None,
        ]
        resolved = next(
            (value for value in candidates if isinstance(value, str) and value),
            None,
        )
        if resolved:
            return resolved
    return requested_model


def transcription_producer_model(transcription: dict, requested_model: str) -> str:
    metadata = transcription.get("_video_course_metadata")
    if not isinstance(metadata, dict):
        return requested_model
    if metadata.get("requested_model") != requested_model:
        return requested_model
    producer_model = metadata.get("producer_model")
    return (
        producer_model
        if isinstance(producer_model, str) and producer_model
        else requested_model
    )


def persist_transcription(
    transcription_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    transcription: dict,
    fingerprint: dict,
) -> list[dict]:
    sentences = transcription_sentences(transcription)
    write_json_atomic(transcription_path, transcription)
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
            producer_model = transcription_producer_model(transcription, args.model)
            print(
                json.dumps(
                    {
                        "stage": "complete",
                        "model": producer_model,
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
    def download_transcription() -> dict:
        with open_url(transcription_url, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("transcription result must be an object")
        return payload

    transcription = run_with_retries(
        download_transcription,
        service="asr-result",
        max_attempts=3,
        sleep=time.sleep,
        context={"operation": "transcription-download"},
    )
    asr_actual_model = response_model(
        args.model,
        task_result,
        submit,
        transcription,
    )
    transcription["_video_course_metadata"] = {
        "requested_model": args.model,
        "producer_model": asr_actual_model,
    }
    sentences = persist_transcription(
        transcription_path, checkpoint_path, transcription, fingerprint
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "model": asr_actual_model,
                "sentences": len(sentences),
                "duration_ms": max((item.get("end_time", 0) for item in sentences), default=0),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
