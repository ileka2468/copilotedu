# pix2tex LoRA Fine-tuning (RTX 3070 / 8GB)

This folder contains a reproducible scaffold to fine-tune LaTeX-OCR (pix2tex) for handwritten math using LoRA adapters.

## Why LoRA
- Trains fast on 8GB VRAM
- Keeps base weights intact
- Small adapter checkpoints; easy to swap/merge

## Structure
- `train_lora.py` — CLI training entrypoint (LoRA on decoder; encoder frozen)
- `requirements.txt` — minimal deps
- `data/` — place your dataset here
  - `train.jsonl`, `val.jsonl` lines: `{ "image_path": "path/to.png", "latex": "\\frac{...}" }`

## Quickstart (dry-run)
```bash
# inside your venv
pip install -r requirements.txt
python train_lora.py --dry-run 1
```

## Dataset prep
- Use CROHME + your own scans; optionally synthesize from LaTeX with handwriting-like fonts.
- Normalize images to height ~256 px; grayscale is fine.
- Store as JSONL (see above). Paths can be absolute or relative to this folder.

## Train (example config for 8GB)
```bash
python train_lora.py \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out runs/exp1 \
  --rank 16 --alpha 32 \
  --epochs 20 --bsz 2 --grad-accum 16 \
  --lr 2e-4 --wd 0.01 --fp16 1
```

## Checkpoints
- Saved every epoch under `--out/ckpt-epoch-XX/`
- LoRA-only weights are also saved separately under `--out/lora-epoch-XX/`

## Inference with adapters (concept)
```python
from PIL import Image
from pix2tex.cli import LatexOCR
# Load base
model = LatexOCR()
# Apply LoRA (to decoder) — see TODO in train_lora.py for exact target module names
# peft_model = PeftModel.from_pretrained(model.decoder, "path/to/lora-epoch-XX")
# peft_model.merge_and_unload()
print(model(Image.open("sample.png")))
```

## Notes
- This scaffold is conservative and stable-first. Customize target modules based on the actual decoder class in your installed pix2tex version.
- Some GitHub pages block automated scraping; use the repo locally for code reference.
