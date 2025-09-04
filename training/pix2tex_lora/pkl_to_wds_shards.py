#!/usr/bin/env python3
"""
pkl_to_wds_shards.py

Convert the existing pix2tex pickles (train.pkl / val.pkl) into WebDataset shards.
This lets you reuse your already-working local dataset (copied images + pickles)
without touching InkML in Colab.

Inputs:
  --pkl       Path to train.pkl or val.pkl (a list of (image_path, equation) tuples)
  --base-dir  Base directory used to resolve relative image paths stored in the pkl
              (usually the folder that contains split/images/). If image paths are
              absolute (created with --no-copy), base-dir is ignored for those items.
  --out       Output directory for shards
  --shard-size  Number of samples per shard (default 2000)

Example:
  python pkl_to_wds_shards.py \
    --pkl training/pix2tex_lora/dataset/train.pkl \
    --base-dir training/pix2tex_lora/dataset \
    --out shards/train --shard-size 2000

  python pkl_to_wds_shards.py \
    --pkl training/pix2tex_lora/dataset/val.pkl \
    --base-dir training/pix2tex_lora/dataset \
    --out shards/val --shard-size 2000

Output:
  <out>/shard-000000.tar, ... with keys: {"__key__", ext (jpg/png), "txt"}
"""
import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import webdataset as wds

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_pkl(pkl_path: Path) -> List[Tuple[str, str]]:
    with pkl_path.open("rb") as f:
        items = pickle.load(f)
    # items is list of (image_path, equation)
    return items


def resolve_image_bytes(p: str, base_dir: Path) -> bytes:
    pp = Path(p)
    if not pp.is_absolute():
        pp = base_dir / p
    if not pp.exists():
        raise FileNotFoundError(f"Image not found: {pp}")
    return pp.read_bytes()


def write_shards(items: List[Tuple[str, str]], base_dir: Path, out_dir: Path, shard_size: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")
    num = 0
    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        for idx, (img_path, eq) in enumerate(items):
            key = f"{idx:09d}"
            ext = Path(img_path).suffix.lower().lstrip(".")
            if ext == "jpeg":
                ext = "jpg"
            if len(ext) == 0:
                ext = "jpg"
            if f".{ext}" not in IMG_EXTS:
                # default to jpg key if unknown
                ext = "jpg"
            img_bytes = resolve_image_bytes(img_path, base_dir)
            sample = {
                "__key__": key,
                ext: img_bytes,
                "txt": eq,
            }
            sink.write(sample)
            num += 1
    return num


def main():
    ap = argparse.ArgumentParser(description="Convert pix2tex pkl to WebDataset shards")
    ap.add_argument("--pkl", type=str, required=True, help="Path to train.pkl or val.pkl")
    ap.add_argument("--base-dir", type=str, required=True, help="Base dir to resolve relative image paths")
    ap.add_argument("--out", type=str, required=True, help="Output dir for shards")
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out)

    items = load_pkl(pkl_path)
    n = write_shards(items, base_dir, out_dir, args.shard_size)
    print(f"[ok] Wrote {n} samples into shards at {out_dir}")


if __name__ == "__main__":
    main()
