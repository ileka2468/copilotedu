#!/usr/bin/env python3
"""
make_wds_shards.py

Create WebDataset shards from a dataset organized like:
  <root>/train/
      images/               # image files (*.png, *.jpg, ...)
      equations.txt         # one LaTeX line per image, sorted to match image filenames
  <root>/val/
      images/
      equations.txt

Output:
  <out_dir>/train/shard-000000.tar, ...
  <out_dir>/val/shard-000000.tar, ...
Each sample contains keys: {"__key__", "jpg" (or original ext), "txt"}

Example:
  python make_wds_shards.py \
    --root C:/data/mathwriting-2024 \
    --out  C:/data/mathwriting-2024-wds \
    --shard-size 2000

Notes:
- We align images to equations by sorting filenames in `images/` and zipping with lines from `equations.txt`.
- If counts mismatch, we take the minimum and warn.
- For PNG images we store as `png` key; for JPEG as `jpg`. Both are accepted by WebDataset.
- Requires: pip install webdataset
"""
import argparse
from pathlib import Path
from typing import List, Tuple
import webdataset as wds

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def read_equations(txt_path: Path) -> List[str]:
    lines: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    return lines


def load_split(split_dir: Path) -> Tuple[List[Path], List[str]]:
    images_dir = split_dir / "images"
    eq_path = split_dir / "equations.txt"
    assert images_dir.exists(), f"Missing images dir: {images_dir}"
    assert eq_path.exists(), f"Missing equations file: {eq_path}"

    equations = read_equations(eq_path)
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if len(image_files) != len(equations):
        print(f"[warn] {split_dir.name}: image count ({len(image_files)}) != equations count ({len(equations)}). Using min length.")
    n = min(len(image_files), len(equations))
    return image_files[:n], equations[:n]


def write_shards(images: List[Path], equations: List[str], out_dir: Path, shard_size: int, split: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")
    num_written = 0
    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        for idx, (img_path, eq) in enumerate(zip(images, equations)):
            key = f"{idx:09d}"
            ext = img_path.suffix.lower().lstrip(".")
            if ext == "jpeg":
                ext = "jpg"
            sample = {
                "__key__": key,
                ext: img_path.read_bytes(),
                "txt": eq,
            }
            sink.write(sample)
            num_written += 1
    print(f"[ok] {split}: wrote {num_written} samples into shards at {out_dir}")
    return num_written


def main():
    ap = argparse.ArgumentParser(description="Create WebDataset shards from images + equations.txt")
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing train/ and val/")
    ap.add_argument("--out", type=str, required=True, help="Output directory for shards")
    ap.add_argument("--shard-size", type=int, default=2000, help="Samples per shard")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    for split in ("train", "val"):
        split_dir = root / split
        imgs, eqs = load_split(split_dir)
        write_shards(imgs, eqs, out / split, args.shard_size, split)

    print("Done. Upload the 'out' directory to Hugging Face Hub or your object store.")


if __name__ == "__main__":
    main()
