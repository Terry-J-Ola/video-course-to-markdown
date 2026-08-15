import argparse
import json
import pathlib
import re
import shutil


def format_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def attach_batch_frames(consolidated: dict) -> list[dict]:
    frames = []
    for group in consolidated.get("groups", []):
        sources = group.get("source_frames", [])
        returned = group.get("result", {}).get("frames", [])
        for position, frame in enumerate(returned):
            try:
                index = int(frame.get("frame_index", position + 1)) - 1
            except (TypeError, ValueError):
                index = position
            if index < 0 or index >= len(sources):
                index = min(position, len(sources) - 1)
            source = sources[index]
            item = dict(frame)
            item["timestamp_seconds"] = float(source["timestamp_seconds"])
            item["source_image"] = source["image"]
            frames.append(item)
    frames.sort(key=lambda item: float(item["timestamp_seconds"]))
    return frames


def load_visual_frames(path: pathlib.Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frames" in data:
        return data["frames"], data
    return attach_batch_frames(data), data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    corrected = text or ""
    for original, replacement in corrections.items():
        corrected = corrected.replace(original, replacement)
    return corrected


def load_corrections(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("Corrections JSON must be a string-to-string object")
    return data


def reviewed_visual_block(block: dict, corrections: dict[str, str]) -> dict:
    reviewed = dict(block)
    original = reviewed.get("text", "")
    corrected = apply_corrections(original, corrections)
    if corrected != original:
        reviewed["text"] = corrected
        existing_note = reviewed.get("note", "")
        review_note = "依据复核修正表校正"
        reviewed["note"] = f"{existing_note}；{review_note}" if existing_note else review_note
    return reviewed


def reviewed_visual_description(text: str, corrections: dict[str, str]) -> str:
    return apply_corrections(text, corrections)


def select_visual_events(
    frames: list[dict],
    mode: str,
    corrections: dict[str, str],
) -> list[dict]:
    events = []
    last_signature = None
    for frame in frames:
        blocks = [
            reviewed_visual_block(block, corrections)
            for block in frame.get("visible_text_blocks", [])
            if block.get("text")
        ]
        same = bool(frame.get("same_information_as_previous"))
        change_type = frame.get("change_type", "")
        if same and not blocks and change_type != "action_changed":
            continue

        signature = (
            normalize_text(frame.get("page_title", "")),
            normalize_text(frame.get("section_title", "")),
            tuple((block.get("role", ""), normalize_text(block.get("text", ""))) for block in blocks),
        )
        if signature == last_signature and signature[2]:
            continue
        if mode == "slides" and not blocks and not frame.get("page_title") and events:
            # Decorative animation frame with no newly readable course content.
            continue

        event = dict(frame)
        event["visible_text_blocks"] = blocks
        event["visual_description"] = reviewed_visual_description(
            event.get("visual_description", ""), corrections
        )
        events.append(event)
        if signature[2] or signature[0] or signature[1]:
            last_signature = signature
    return events


def corrected_asr(text: str, corrections: dict[str, str]) -> str:
    return apply_corrections(text, corrections)


def load_sentences(path: pathlib.Path) -> tuple[list[dict], float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sentences = []
    duration_ms = 0
    for transcript in data.get("transcripts", []):
        duration_ms = max(duration_ms, int(transcript.get("content_duration_in_milliseconds", 0)))
        sentences.extend(transcript.get("sentences", []))
    sentences.sort(key=lambda item: int(item.get("begin_time", 0)))
    if sentences:
        duration_ms = max(duration_ms, max(int(item.get("end_time", 0)) for item in sentences))
    return sentences, duration_ms / 1000, data


def copy_event_images(events: list[dict], asset_dir: pathlib.Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for index, event in enumerate(events, start=1):
        source = pathlib.Path(event["source_image"])
        destination = asset_dir / f"frame_{index:04d}_t{int(float(event['timestamp_seconds']) * 1000):09d}.jpg"
        shutil.copy2(source, destination)
        event["asset_name"] = destination.name


def markdown_for_video(
    title: str,
    events: list[dict],
    sentences: list[dict],
    duration: float,
    asset_dir_name: str,
    visual_model: str,
    asr_model: str,
    corrections: dict[str, str],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"> 画面由 {visual_model} 关键帧分析生成；音频由 {asr_model} 转写并保留毫秒时间戳。",
        "> 本文是用于复核的技术证据稿；最终学员讲义由后续转换和完整性校验生成。",
        "",
    ]

    for index, event in enumerate(events):
        start = float(event["timestamp_seconds"])
        end = float(events[index + 1]["timestamp_seconds"]) if index + 1 < len(events) else duration
        if end < start:
            end = start
        lines.extend(
            [
                f"### [{format_time(start)} - {format_time(end)}]",
                "",
                f"![{title} {format_time(start)}]({asset_dir_name}/{event['asset_name']})",
                "",
            ]
        )

        page_title = event.get("page_title", "")
        section_title = event.get("section_title", "")
        if page_title:
            lines.append(f"> **页面标题：** {page_title}")
        if section_title:
            lines.append(f"> **小节标题：** {section_title}")
        blocks = event.get("visible_text_blocks", [])
        if blocks:
            lines.append("> **画面文字：**")
            for block in blocks:
                status = block.get("status", "confirmed")
                suffix = "" if status == "confirmed" else f"（{status}）"
                note = f"；{block.get('note')}" if block.get("note") else ""
                role = block.get("role", "other")
                lines.append(f"> - `{role}` {block.get('text', '')}{suffix}{note}")
        else:
            lines.append("> **画面文字：** 未识别到新增文字。")
        description = event.get("visual_description", "")
        if description:
            lines.append(f"> **画面说明：** {description}")

        audio_items = [
            item
            for item in sentences
            if start <= int(item.get("begin_time", 0)) / 1000 < end
        ]
        if audio_items:
            lines.append("> **音频：**")
            for item in audio_items:
                audio_start = int(item.get("begin_time", 0)) / 1000
                audio_end = int(item.get("end_time", 0)) / 1000
                text = corrected_asr(item.get("text", ""), corrections)
                lines.append(
                    f"> - `[{format_time(audio_start)} - {format_time(audio_end)}]` {text}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--mode", choices=("slides", "live"), required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--visual-json-output", required=True)
    parser.add_argument("--asr-json-output", required=True)
    parser.add_argument("--audio-source", required=True)
    parser.add_argument("--audio-output", required=True)
    parser.add_argument("--corrections")
    parser.add_argument("--visual-model", default="qwen3-vl-flash")
    parser.add_argument("--asr-model", default="paraformer-v2")
    args = parser.parse_args()

    frames, visual_raw = load_visual_frames(pathlib.Path(args.visual))
    corrections = load_corrections(args.corrections)
    events = select_visual_events(frames, args.mode, corrections)
    sentences, duration, asr_raw = load_sentences(pathlib.Path(args.asr))

    output = pathlib.Path(args.output)
    assets = pathlib.Path(args.assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    copy_event_images(events, assets)
    output.write_text(
        markdown_for_video(
            args.title,
            events,
            sentences,
            duration,
            assets.name,
            args.visual_model,
            args.asr_model,
            corrections,
        ),
        encoding="utf-8",
    )

    pathlib.Path(args.visual_json_output).write_text(
        json.dumps(
            {
                "video": visual_raw.get("video"),
                "model": visual_raw.get("model", "qwen3-vl-flash"),
                "input_frames": len(frames),
                "retained_events": len(events),
                "events": events,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pathlib.Path(args.asr_json_output).write_text(
        json.dumps(asr_raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.audio_source, args.audio_output)

    visible_blocks = sum(len(event.get("visible_text_blocks", [])) for event in events)
    visible_chars = sum(
        len(block.get("text", ""))
        for event in events
        for block in event.get("visible_text_blocks", [])
    )
    print(
        json.dumps(
            {
                "frames": len(frames),
                "events": len(events),
                "visible_blocks": visible_blocks,
                "visible_chars": visible_chars,
                "asr_sentences": len(sentences),
                "duration_seconds": duration,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
