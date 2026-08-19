import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

try:
    from .checkpoint_provenance import (
        checkpoint_matches,
        file_identity,
        write_checkpoint,
        write_json_atomic,
    )
    from .env_config import get_qwen_api_key
    from .http_transport import open_url
except ImportError:
    from checkpoint_provenance import (
        checkpoint_matches,
        file_identity,
        write_checkpoint,
        write_json_atomic,
    )
    from env_config import get_qwen_api_key
    from http_transport import open_url


SYSTEM_PROMPT = r"""
你是视频课程关键帧信息抽取器。输入是按时间顺序排列的多张视频帧。
目标是完整还原每个独立画面状态，不是总结视频。

规则：
1. 每张帧独立处理，frame_index 和 timestamp_seconds 必须原样返回。
2. 逐字转录所有有业务意义的可见文字：标题、小节标题、正文、问题、选项、按钮、反馈、弹窗、字幕、标注。
3. 禁止概括、改写、润色或自动纠错；保持原始用词、英文、数字和标点。
4. 无法确认的字符标记 uncertain。被遮挡但按上下文补出的文字必须标记 inferred，并说明原因。
5. 非文字内容放在 visual_description，包括人物、动作、商品、布局和图文关系。
6. 若当前帧与同组前一帧承载的信息完全相同，只是人物轻微移动或动画位置变化，same_information_as_previous=true，visible_text_blocks 可为空。
7. 若文字新增、消失、替换、出现弹窗或进入新的动作阶段，same_information_as_previous=false，并完整输出当前帧文字。
8. 只输出合法 JSON，不使用 Markdown 代码围栏。

JSON 格式：
{
  "frames": [
    {
      "frame_index": 1,
      "timestamp_seconds": 0.0,
      "page_title": "",
      "section_title": "",
      "visible_text_blocks": [
        {
          "role": "title|body|question|option|feedback|button|caption|popup|label|other",
          "text": "逐字原文",
          "status": "confirmed|uncertain|inferred",
          "note": ""
        }
      ],
      "visual_description": "",
      "same_information_as_previous": false,
      "change_type": "new_page|text_added|text_changed|popup|action_changed|same",
      "uncertainties": []
    }
  ]
}
""".strip()


def data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def choose_frames(manifest: dict, min_spacing: float) -> list[dict]:
    frames = manifest["keyframes"]
    if min_spacing <= 0:
        return frames
    selected = []
    last_time = -10**9
    for frame in frames:
        timestamp = float(frame["timestamp_seconds"])
        if not selected or timestamp - last_time >= min_spacing:
            selected.append(frame)
            last_time = timestamp
    if selected[-1] != frames[-1]:
        selected.append(frames[-1])
    return selected


def parse_json_content(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as original_error:
        # 模型可能在结果前后加说明或示例对象。逐个尝试完整 JSON 对象，
        # 只接受满足视觉结果结构的对象，避免“第一个 { 到最后一个 }”误拼多个对象。
        decoder = json.JSONDecoder()
        parsed = None
        for match in re.finditer(r"{", content):
            try:
                candidate, _ = decoder.raw_decode(content[match.start() :])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("frames"), list)
                and all(isinstance(frame, dict) for frame in candidate["frames"])
            ):
                parsed = candidate
                break
        if parsed is None:
            raise original_error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("frames"), list):
        raise ValueError("visual result must be an object with a frames list")
    if not all(isinstance(frame, dict) for frame in parsed["frames"]):
        raise ValueError("visual result frames must contain objects")
    return parsed


def response_content(raw: dict) -> str:
    if not isinstance(raw, dict):
        raise ValueError("visual response must be an object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("visual response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("visual response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("visual response has no message content")
    return message["content"]


def invoke_group(
    api_key: str,
    model: str,
    group_id: int,
    frames: list[dict],
    output_dir: pathlib.Path,
    resume: bool,
    fingerprint: dict | None = None,
) -> dict:
    raw_path = output_dir / f"group_{group_id:04d}_raw.json"
    checkpoint_path = output_dir / f"group_{group_id:04d}_checkpoint.json"
    if resume and fingerprint is not None and raw_path.exists() and checkpoint_matches(
        checkpoint_path, fingerprint
    ):
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            message = response_content(raw)
            parsed = parse_json_content(message)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            checkpoint_path.unlink(missing_ok=True)
        else:
            return {
                "group_id": group_id,
                "source_frames": frames,
                "result": parsed,
                "usage": raw.get("usage", {}),
                "model": raw.get("model", model),
                "resumed": True,
            }

    content = [{"type": "text", "text": SYSTEM_PROMPT}]
    for frame_index, frame in enumerate(frames, start=1):
        timestamp = float(frame["timestamp_seconds"])
        content.append(
            {
                "type": "text",
                "text": f"FRAME {frame_index}; timestamp_seconds={timestamp:.3f}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url(pathlib.Path(frame["image"]))},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(1, 4):
        try:
            with open_url(request, timeout=240) as response:
                raw = json.loads(response.read().decode("utf-8"))
            write_json_atomic(raw_path, raw)
            message = response_content(raw)
            parsed = parse_json_content(message)
            if fingerprint is not None:
                write_checkpoint(checkpoint_path, fingerprint)
            return {
                "group_id": group_id,
                "source_frames": frames,
                "result": parsed,
                "usage": raw.get("usage", {}),
                "model": raw.get("model", model),
            }
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            error_type = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body}"
            else:
                last_error = repr(exc)
            print(
                json.dumps(
                    {
                        "group_id": group_id,
                        "attempt": attempt,
                        "max_attempts": 3,
                        "error_type": error_type,
                        "error_message": last_error[:200],
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"group {group_id} failed after 3 attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-spacing", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    api_key = get_qwen_api_key()
    if not api_key:
        print(
            "DASHSCOPE_QWEN_API_KEY or DASHSCOPE_API_KEY is not configured",
            file=sys.stderr,
        )
        return 2

    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = choose_frames(manifest, args.min_spacing)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = [frames[index : index + args.group_size] for index in range(0, len(frames), args.group_size)]
    stage_fingerprint = {
        "stage": "visual-analysis",
        "input": file_identity(pathlib.Path(manifest["video"])),
        "model": args.model,
        "group_size": args.group_size,
        "min_spacing": args.min_spacing,
        "selected_frames": [
            {
                "timestamp_seconds": float(frame["timestamp_seconds"]),
                "image": file_identity(pathlib.Path(frame["image"])),
            }
            for frame in frames
        ],
    }
    print(json.dumps({"selected_frames": len(frames), "groups": len(groups)}), flush=True)

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                invoke_group,
                api_key,
                args.model,
                group_id,
                group,
                output_dir,
                not args.no_resume,
                {**stage_fingerprint, "group_id": group_id},
            ): group_id
            for group_id, group in enumerate(groups, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            group_id = futures[future]
            result = future.result()
            completed.append(result)
            print(json.dumps({"completed_group": group_id, "total_groups": len(groups)}), flush=True)

    completed.sort(key=lambda item: item["group_id"])
    producer_models = sorted({item.get("model", args.model) for item in completed})
    producer_model = producer_models[0] if len(producer_models) == 1 else "mixed"
    consolidated = {
        "video": manifest["video"],
        "model": producer_model,
        "producer_models": producer_models,
        "selected_frame_count": len(frames),
        "group_count": len(groups),
        "groups": completed,
    }
    (output_dir / "consolidated.json").write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(output_dir / 'consolidated.json')}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
