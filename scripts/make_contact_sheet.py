import argparse
import math
import pathlib

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=260)
    args = parser.parse_args()

    paths = sorted(pathlib.Path(args.input_dir).glob("*.jpg"))
    if not paths:
        raise RuntimeError("No JPG images found")

    label_height = 24
    first = Image.open(paths[0])
    ratio = first.height / first.width
    thumb_height = int(args.thumb_width * ratio)
    rows = math.ceil(len(paths) / args.columns)
    sheet = Image.new(
        "RGB",
        (args.columns * args.thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, path in enumerate(paths):
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((args.thumb_width, thumb_height), Image.Resampling.LANCZOS)
            x = (index % args.columns) * args.thumb_width
            y = (index // args.columns) * (thumb_height + label_height)
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + thumb_height + 4), path.stem, fill="black", font=font)

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
