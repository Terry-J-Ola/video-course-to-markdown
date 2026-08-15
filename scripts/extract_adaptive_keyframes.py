import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

try:
    from .checkpoint_provenance import checkpoint_matches, file_identity, write_checkpoint
except ImportError:
    from checkpoint_provenance import checkpoint_matches, file_identity, write_checkpoint


def extraction_fingerprint(
    video: pathlib.Path, mode: str, fps: float
) -> dict:
    return {
        "stage": "adaptive-keyframes",
        "input": file_identity(video),
        "mode": mode,
        "fps": fps,
    }


def scan_checkpoint_is_compatible(
    video: pathlib.Path, output_dir: pathlib.Path, mode: str, fps: float
) -> bool:
    scan_dir = output_dir / "scan"
    return bool(list(scan_dir.glob("scan_*.jpg"))) and checkpoint_matches(
        output_dir / "extraction.checkpoint.json",
        extraction_fingerprint(video, mode, fps),
    )


def clear_incompatible_extraction(output_dir: pathlib.Path) -> None:
    for directory, pattern in (
        (output_dir / "scan", "scan_*.jpg"),
        (output_dir / "keyframes", "key_*.jpg"),
    ):
        for path in directory.glob(pattern):
            path.unlink()
    for name in ("manifest.json", "extraction.checkpoint.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def ffmpeg_executable(pydeps: pathlib.Path | None) -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if pydeps:
        sys.path.insert(0, str(pydeps))
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable. Install ffmpeg or run bootstrap_dependencies.py."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_scan_frames(ffmpeg: str, video: pathlib.Path, scan_dir: pathlib.Path, fps: float) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(scan_dir.glob("scan_*.jpg"))
    if existing:
        return
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(scan_dir / "scan_%06d.jpg"),
    ]
    subprocess.run(command, check=True)


def grayscale_array(path: pathlib.Path, size=(256, 144)) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32)


def edge_array(gray: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    edge = image.filter(ImageFilter.FIND_EDGES)
    return np.asarray(edge, dtype=np.float32)


def difference_metrics(previous: np.ndarray, current: np.ndarray) -> dict:
    delta = np.abs(current - previous)
    height, width = delta.shape
    grid_scores = []
    for row in range(4):
        for col in range(4):
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = col * width // 4, (col + 1) * width // 4
            grid_scores.append(float(delta[y0:y1, x0:x1].mean()))
    edge_delta = np.abs(edge_array(current) - edge_array(previous))
    return {
        "global": float(delta.mean()),
        "local_max": max(grid_scores),
        "edge": float(edge_delta.mean()),
    }


def frame_sharpness(path: pathlib.Path) -> float:
    with Image.open(path) as image:
        gray = image.convert("L").resize((320, 180), Image.Resampling.BILINEAR)
        edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        return float(edges.var())


def select_slide_frames(frames, arrays, fps):
    changes = []
    metrics = [None]
    for index in range(1, len(frames)):
        metric = difference_metrics(arrays[index - 1], arrays[index])
        metrics.append(metric)
        if metric["global"] >= 3.0 or metric["local_max"] >= 6.0 or metric["edge"] >= 4.0:
            changes.append(index)

    groups = []
    for index in changes:
        if not groups or index - groups[-1][-1] > int(1.5 * fps):
            groups.append([index])
        else:
            groups[-1].append(index)

    selected = {0, len(frames) - 1}
    for group in groups:
        start = max(0, group[0] - 1)
        end = min(len(frames) - 1, group[-1] + max(1, int(0.75 * fps)))
        window = range(start, end + 1)
        sharpest = max(window, key=lambda idx: frame_sharpness(frames[idx]))
        selected.add(sharpest)

    # Recall safeguard: inspect each 5-second boundary, but keep it only if the
    # visual state differs materially from the most recent selected state.
    step = max(1, int(5 * fps))
    for index in range(step, len(frames), step):
        prior = max(idx for idx in selected if idx < index) if any(idx < index for idx in selected) else 0
        metric = difference_metrics(arrays[prior], arrays[index])
        if metric["global"] >= 2.0 or metric["local_max"] >= 4.0 or metric["edge"] >= 2.8:
            selected.add(index)

    return sorted(selected), metrics


def select_live_frames(frames, arrays, fps):
    selected = {0, len(frames) - 1}
    metrics = [None]
    forced_step = max(1, int(3.0 * fps))
    for index in range(1, len(frames)):
        metric = difference_metrics(arrays[index - 1], arrays[index])
        metrics.append(metric)
        hard_cut = metric["global"] >= 18.0 or metric["local_max"] >= 28.0 or metric["edge"] >= 16.0
        if hard_cut or index % forced_step == 0:
            start = index
            end = min(len(frames) - 1, index + max(1, int(0.5 * fps)))
            sharpest = max(range(start, end + 1), key=lambda idx: frame_sharpness(frames[idx]))
            selected.add(sharpest)
    return sorted(selected), metrics


def detect_mode(arrays: list[np.ndarray]) -> str:
    if len(arrays) < 3:
        return "slides"
    stride = max(1, len(arrays) // 120)
    sampled = arrays[::stride]
    global_changes = [
        difference_metrics(sampled[index - 1], sampled[index])["global"]
        for index in range(1, len(sampled))
    ]
    moving_ratio = sum(score >= 2.0 for score in global_changes) / len(global_changes)
    return "live" if moving_ratio >= 0.35 else "slides"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=("auto", "slides", "live"), default="auto")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--pydeps", default=None)
    args = parser.parse_args()

    video = pathlib.Path(args.video)
    output_dir = pathlib.Path(args.output_dir)
    scan_dir = output_dir / "scan"
    keyframe_dir = output_dir / "keyframes"
    fingerprint = extraction_fingerprint(video, args.mode, args.fps)
    if not scan_checkpoint_is_compatible(video, output_dir, args.mode, args.fps):
        clear_incompatible_extraction(output_dir)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    pydeps = pathlib.Path(args.pydeps) if args.pydeps else None
    ffmpeg = ffmpeg_executable(pydeps)
    extract_scan_frames(ffmpeg, video, scan_dir, args.fps)
    frames = sorted(scan_dir.glob("scan_*.jpg"))
    if not frames:
        raise RuntimeError("No frames extracted")

    arrays = [grayscale_array(frame) for frame in frames]
    resolved_mode = detect_mode(arrays) if args.mode == "auto" else args.mode
    if resolved_mode == "slides":
        selected, metrics = select_slide_frames(frames, arrays, args.fps)
    else:
        selected, metrics = select_live_frames(frames, arrays, args.fps)

    manifest = {
        "video": str(video),
        "requested_mode": args.mode,
        "mode": resolved_mode,
        "scan_fps": args.fps,
        "scan_frame_count": len(frames),
        "keyframe_count": len(selected),
        "keyframes": [],
    }
    for order, index in enumerate(selected, start=1):
        timestamp = index / args.fps
        destination = keyframe_dir / f"key_{order:04d}_t{int(round(timestamp * 1000)):09d}.jpg"
        shutil.copy2(frames[index], destination)
        manifest["keyframes"].append(
            {
                "order": order,
                "scan_index": index,
                "timestamp_seconds": round(timestamp, 3),
                "image": str(destination),
                "change_from_previous": metrics[index],
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_checkpoint(output_dir / "extraction.checkpoint.json", fingerprint)
    print(
        json.dumps(
            {
                "video": video.name,
                "mode": resolved_mode,
                "scan_frames": len(frames),
                "keyframes": len(selected),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
