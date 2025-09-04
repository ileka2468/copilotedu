import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Tuple

try:
    # Used only to locate default tokenizer shipped with pix2tex
    from pix2tex.utils import in_model_path
except Exception:
    in_model_path = None


def read_equations(txt_path: Path) -> List[str]:
    lines: List[str] = []
    with txt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line is None:
                continue
            lines.append(line)
    return lines


def build_split(images_dir: Path, equations_txt: Path, base_out_dir: Path, split: str, no_copy: bool = False) -> int:
    assert images_dir.exists(), f"Images dir not found: {images_dir}"
    assert equations_txt.exists(), f"Equations file not found: {equations_txt}"

    equations = read_equations(equations_txt)
    # Discover images case-insensitively and sort to keep alignment with equations.txt lines
    exts = {'.png', '.jpg', '.jpeg'}
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if len(image_files) != len(equations):
        print(f"[warn] {split}: image count ({len(image_files)}) != equations count ({len(equations)}). Will zip to min length.")
    n = min(len(image_files), len(equations))

    # Create relative paths under dataset folder for compatibility
    rel_prefix = f"{split}/images/"
    # Ensure images are accessible relative to the dataset root
    dest_img_dir = base_out_dir / split / 'images'
    dest_img_dir.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, str]] = []
    for i in range(n):
        src = image_files[i]
        if no_copy:
            # Store absolute image path directly (useful on Colab to avoid slow Drive->local copies)
            rel = str(src)
        else:
            rel = rel_prefix + src.name
            dst = dest_img_dir / src.name
            if not dst.exists():
                try:
                    # Copy to keep a self-contained dataset
                    shutil.copyfile(src, dst)
                except Exception as e:
                    print(f"[warn] failed to copy {src} -> {dst}: {e}")
        items.append((rel, equations[i]))

    out_pkl = base_out_dir / f"{split}.pkl"
    with out_pkl.open('wb') as f:
        pickle.dump(items, f)
    print(f"[ok] Wrote {n} items to {out_pkl}")
    return n


def copy_default_tokenizer(out_tokenizer: Path) -> bool:
    # Try to copy tokenizer.json shipped within pix2tex package
    if in_model_path is None:
        return False
    try:
        with in_model_path():
            # Common locations: settings/tokenizer.json
            candidates = [
                Path('model/dataset/tokenizer.json'),
                Path('tokenizer.json'),
            ]
            for c in candidates:
                if c.exists():
                    out_tokenizer.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(c, out_tokenizer)
                    print(f"[ok] Copied default tokenizer to {out_tokenizer}")
                    return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser(description='Build pix2tex pickles (train/val) and copy tokenizer.json')
    ap.add_argument('--train-dir', type=str, required=True, help='Path to dataset/train folder containing images/ and equations.txt')
    ap.add_argument('--val-dir',   type=str, required=True, help='Path to dataset/val   folder containing images/ and equations.txt')
    ap.add_argument('--out',       type=str, required=True, help='Output dataset root (e.g., training/pix2tex_lora/dataset)')
    ap.add_argument('--no-copy', action='store_true', help='Do not copy images; store absolute paths instead (fast on Colab).')
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out)

    # Build train
    n_train = build_split(train_dir / 'images', train_dir / 'equations.txt', out_dir, 'train', no_copy=args.no_copy)
    # Build val
    n_val = build_split(val_dir / 'images', val_dir / 'equations.txt', out_dir, 'val', no_copy=args.no_copy)

    # Tokenizer
    tok_path = out_dir / 'tokenizer.json'
    have_tok = copy_default_tokenizer(tok_path)
    if not have_tok:
        print('[warn] Could not locate default pix2tex tokenizer.json to copy. You may set tokenizer path in config.yaml to the one bundled with pix2tex (settings/tokenizer.json).')

    print(f"Done. train={n_train}, val={n_val}, tokenizer={'ok' if have_tok else 'missing'} -> {tok_path}")


if __name__ == '__main__':
    main()
