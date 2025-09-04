import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from xml.etree import ElementTree
from PIL import Image, ImageDraw

# ------------------------- InkML parsing -------------------------
@dataclass
class Ink:
    strokes: List[Tuple[List[float], List[float], List[float]]]  # (x[], y[], t[])
    annotations: dict


def read_inkml_file(filename: Path) -> Ink:
    root = ElementTree.fromstring(Path(filename).read_text(encoding="utf-8"))
    ns_prefix = '{http://www.w3.org/2003/InkML}'
    strokes = []
    annotations = {}
    for element in root:
        tag_name = element.tag.replace(ns_prefix, '')
        if tag_name == 'annotation':
            annotations[element.attrib.get('type')] = element.text
        elif tag_name == 'trace':
            if not element.text:
                continue
            points = element.text.split(',')
            sx, sy, st = [], [], []
            for point in points:
                parts = point.strip().split(' ')
                if len(parts) < 3:
                    # Some files may have variable spacing
                    parts = [p for p in parts if p]
                    if len(parts) != 3:
                        continue
                x, y, t = parts
                sx.append(float(x))
                sy.append(float(y))
                st.append(float(t))
            if sx:
                strokes.append((sx, sy, st))
    return Ink(strokes=strokes, annotations=annotations)


# ------------------------- Rendering -------------------------

def render_ink(ink: Ink, margin: int = 10, stroke_width: int = 2,
               stroke_color=(0, 0, 0), background_color=(255, 255, 255)) -> Image.Image:
    # Compute bounds
    import math
    xs, ys = [], []
    for sx, sy, _ in ink.strokes:
        xs.extend(sx)
        ys.extend(sy)
    if not xs or not ys:
        return Image.new('RGB', (64, 64), background_color)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = int(math.ceil(xmax - xmin + 2 * margin))
    height = int(math.ceil(ymax - ymin + 2 * margin))
    width = max(width, 8)
    height = max(height, 8)

    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    def tx(x: float) -> float:
        return (x - xmin) + margin

    def ty(y: float) -> float:
        # Invert Y to image coords
        return (ymax - y) + margin

    for sx, sy, _ in ink.strokes:
        if len(sx) == 1:
            x, y = tx(sx[0]), ty(sy[0])
            # Draw a small dot
            r = stroke_width / 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=stroke_color)
        else:
            pts = [(tx(x), ty(y)) for x, y in zip(sx, sy)]
            draw.line(pts, fill=stroke_color, width=stroke_width, joint="curve")

    return img


# ------------------------- Conversion -------------------------

def convert_inkml_dir(src_dir: Path, out_img_dir: Path, out_jsonl_path: Path,
                      target_height: int = 256) -> int:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_jsonl_path.open('w', encoding='utf-8') as w:
        for inkml in sorted((src_dir).glob('*.inkml')):
            try:
                ink = read_inkml_file(inkml)
                label = ink.annotations.get('normalizedLabel') or ink.annotations.get('label')
                if not label:
                    continue
                img = render_ink(ink)
                # Resize to target height maintaining aspect
                w0, h0 = img.size
                if h0 != 0 and h0 != target_height:
                    new_w = max(16, int(w0 * (target_height / h0)))
                    img = img.resize((new_w, target_height), Image.BILINEAR)
                # Save image as PNG
                png_name = inkml.with_suffix('.png').name
                out_path = out_img_dir / png_name
                img.save(out_path)
                rec = {"image_path": str(out_path), "latex": label}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"[warn] failed on {inkml.name}: {e}")
    return count


def main():
    ap = argparse.ArgumentParser(description="Convert MathWriting InkML to images + JSONL for pix2tex training")
    ap.add_argument('--root', type=str, required=True, help='Path to mathwriting-2024[-excerpt] root')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'val', 'valid', 'test', 'trainval'])
    ap.add_argument('--out', type=str, default='data', help='Output dir under training/pix2tex_lora/')
    ap.add_argument('--height', type=int, default=256)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    img_dir = out_dir / f"images-{args.split}"
    jsonl_path = out_dir / f"{args.split}.jsonl"

    # Map 'val' to 'valid' on disk
    split_dir_name = 'valid' if args.split == 'val' else args.split
    src_dir = root / split_dir_name
    if not src_dir.exists():
        # Some excerpts might have only train; allow trainval to alias train
        if args.split == 'trainval' and (root / 'train').exists():
            src_dir = root / 'train'
        else:
            raise SystemExit(f"Split directory not found: {src_dir}")

    n = convert_inkml_dir(src_dir, img_dir, jsonl_path, target_height=args.height)
    print(f"Wrote {n} samples to {jsonl_path} with images in {img_dir}")


if __name__ == '__main__':
    main()
