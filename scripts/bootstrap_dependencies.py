import argparse
import json
import os
import pathlib
import subprocess
import sys


PACKAGES = ["pillow", "numpy", "imageio-ffmpeg", "certifi", "openpyxl"]


def default_target() -> pathlib.Path:
    base = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home()))
    return base / "Codex" / "video-course-to-markdown" / "pydeps"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install isolated Python dependencies for video-course-to-markdown."
    )
    parser.add_argument("--target", default=str(default_target()))
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()

    target = pathlib.Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
    ]
    if args.upgrade:
        command.append("--upgrade")
    command.extend(PACKAGES)
    subprocess.run(command, check=True)
    print(
        json.dumps(
            {
                "target": str(target),
                "packages": PACKAGES,
                "next": f"Set VIDEO_COURSE_PYDEPS or pass --pydeps {target}",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
