import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Convert JSONL (image_path, latex) into pix2tex dataset folder with images/ and equations.txt")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to JSONL file (train.jsonl or valid.jsonl)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dataset dir (will create images/ and equations.txt)")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of linking (recommended on Windows)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    eq_path = out_dir / "equations.txt"
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as r, eq_path.open("w", encoding="utf-8") as w:
        for i, line in enumerate(r):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            src = Path(obj["image_path"])  # may be relative or absolute
            if not src.exists():
                # try relative to jsonl
                src = (jsonl_path.parent / obj["image_path"]).resolve()
            if not src.exists():
                print(f"[warn] missing image: {obj['image_path']}")
                continue
            dst = img_dir / f"{i:06d}.png"
            if args.copy:
                shutil.copyfile(src, dst)
            else:
                try:
                    dst.symlink_to(src)
                except Exception:
                    shutil.copyfile(src, dst)
            w.write(obj["latex"].strip() + "\n")
            n += 1
    print(f"Wrote {n} samples to {out_dir} (images/, equations.txt)")


if __name__ == "__main__":
    main()
