import os
import argparse
import logging
import yaml

import torch
from munch import Munch
from tqdm.auto import tqdm
import torch.nn as nn
import sys
import types
import imagesize
from transformers import PreTrainedTokenizerFast
import cv2
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt
import re
import time
import inspect

# Ensure wandb is disabled before any pix2tex modules import it
os.environ.setdefault("WANDB_DISABLED", "true")

# Provide a stub 'wandb' module before any importers see it
if 'wandb' not in sys.modules:
    wandb_stub = types.SimpleNamespace()
    wandb_stub.init = lambda *a, **k: None
    wandb_stub.watch = lambda *a, **k: None
    wandb_stub.log = lambda *a, **k: None
    wandb_stub.util = types.SimpleNamespace(generate_id=lambda: "disabled")
    sys.modules['wandb'] = wandb_stub

# Import pix2tex core modules (dataset, models, utils) from the installed package
try:
    import wandb  # type: ignore
    # no-op stubs
    wandb.init = lambda *a, **k: None
    wandb.watch = lambda *a, **k: None
    wandb.log = lambda *a, **k: None
except Exception:
    pass

from pix2tex.dataset.dataset import Im2LatexDataset
from pix2tex.dataset.transforms import test_transform
from pix2tex.models import get_model
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check


def evaluate_noop(*args, **kwargs):
    # Windows-safe placeholder to avoid torchtext dependency during training
    return 0.0, 0.0, 0.0


def _validate(model, valdataloader, args, device):
    """Validation: average loss on a limited number of batches and exact-match accuracy.
    - Uses tokenizer at `args.tokenizer`
    - Limits to `args.valbatches` batches if present, else all
    - Generates predictions for up to `val_samples_per_batch` items per batch for EM accuracy
    """
    model.eval()
    torch.set_grad_enabled(False)
    # Defaults
    max_batches = int(getattr(args, 'valbatches', 0) or 0)
    max_batches = max_batches if max_batches > 0 else len(valdataloader)
    val_samples_per_batch = int(getattr(args, 'val_samples_per_batch', 4) or 4)
    # Tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    except Exception:
        tokenizer = None
    # Generation kwargs (optional)
    gen_kwargs = {}
    for k in ['num_beams', 'no_repeat_ngram_size', 'max_new_tokens']:
        v = args.get(k, None)
        if isinstance(v, int) and v is not None and v > 0:
            gen_kwargs[k] = v
    for k in ['length_penalty', 'repetition_penalty']:
        v = args.get(k, None)
        if isinstance(v, (int, float)) and v is not None and v > 0:
            gen_kwargs[k] = float(v)

    total_loss = 0.0
    batches = 0
    em_match = 0
    em_total = 0
    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.testbatchsize
    # Try to discover vocab size to clamp predictions
    _num_tokens = None
    try:
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        _tv = tok.get_vocab() if hasattr(tok, 'get_vocab') else {}
        _num_tokens = (max(_tv.values()) + 1) if _tv else None
    except Exception:
        pass
    def _clamp_ids_tensor(t: torch.Tensor) -> torch.Tensor:
        try:
            if _num_tokens and isinstance(t, torch.Tensor):
                t = t.clamp_(0, max(0, _num_tokens - 1))
        except Exception:
            pass
        return t
    for bi, (seq, im) in enumerate(tqdm(iter(valdataloader), desc='Validating'), start=1):
        if bi > max_batches:
            break
        if seq is None or im is None:
            continue
        # Loss over full batch via microbatching
        batch_loss = 0.0
        for j in range(0, len(im), microbatch):
            tgt_seq = seq['input_ids'][j:j+microbatch].to(device)
            tgt_mask = seq['attention_mask'][j:j+microbatch].bool().to(device)
            with torch.no_grad():
                l = model.data_parallel(
                    im[j:j+microbatch].to(device),
                    device_ids=args.gpu_devices,
                    tgt_seq=tgt_seq,
                    mask=tgt_mask
                ) * microbatch / args.testbatchsize
            batch_loss += float(l.item())
        total_loss += batch_loss
        batches += 1
        # Exact match on a few samples per batch (if tokenizer available)
        if tokenizer is not None and val_samples_per_batch > 0:
            k = min(val_samples_per_batch, len(im))
            im_k = im[:k].float().to(device)
            try:
                gen = model.generate(im_k, **gen_kwargs) if gen_kwargs else model.generate(im_k)
            except Exception:
                gen = model.generate(im_k)
            for idx in range(k):
                # Pred text
                ids_t = gen[idx].detach().cpu()
                ids_t = _clamp_ids_tensor(ids_t)
                pred_txt = tokenizer.decode(ids_t.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True) if tokenizer else ''
                # GT text
                try:
                    gt_ids = seq['input_ids'][idx].detach().cpu().tolist()
                    gt_txt = tokenizer.decode(gt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) if tokenizer else ''
                except Exception:
                    gt_txt = ''
                if pred_txt.strip() == gt_txt.strip():
                    em_match += 1
                em_total += 1
    avg_val_loss = (total_loss / max(1, batches)) if batches else float('nan')
    em_acc = (em_match / max(1, em_total)) if em_total else 0.0
    model.train()
    torch.set_grad_enabled(True)
    return avg_val_loss, em_acc


def load_state_dict_forgiving(model: torch.nn.Module, state_dict: dict, prefix: str = "") -> None:
    """Load as many weights as possible; drop keys with shape mismatch and warn.
    Useful when tokenizer/num_tokens differ from the checkpoint.
    """
    try:
        msd = model.state_dict()
        filtered = {}
        dropped = []
        for k, v in state_dict.items():
            if k in msd and hasattr(v, 'shape') and hasattr(msd[k], 'shape'):
                if v.shape == msd[k].shape:
                    filtered[k] = v
                else:
                    dropped.append(k)
            elif k in msd:
                # Non-tensor buffers; try to keep if possible
                try:
                    if type(v) is type(msd[k]):
                        filtered[k] = v
                    else:
                        dropped.append(k)
                except Exception:
                    dropped.append(k)
        model.load_state_dict(filtered, strict=False)
        if dropped:
            head = ", ".join(dropped[:6]) + (" ..." if len(dropped) > 6 else "")
            print(f"[ckpt] {prefix}loaded with dropped keys (shape mismatch): {head}")
    except Exception as e:
        print(f"[ckpt] {prefix}forgiving load failed, falling back to strict: {e}")
        model.load_state_dict(state_dict)


def _sample_eval(args, max_samples=10):
    """Run a quick qualitative eval by printing predictions for a few val samples."""
    device = args.device
    # Build or load val dataloader
    # Always load the pickle and derive a flat list of (img_path, gt)
    loaded = Im2LatexDataset().load(args.valdata)
    cfg_dir = os.path.dirname(os.path.abspath(args.get('config', 'config.yaml')))
    items = []  # list of (abs_img_path, gt_eq)
    def _is_img_path(s: str) -> bool:
        s_low = s.lower()
        return s_low.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'))
    if isinstance(loaded, list):
        for tup in loaded:
            try:
                if not isinstance(tup, (list, tuple)) or len(tup) != 2:
                    continue
                a, b = tup
                if isinstance(a, str) and _is_img_path(a):
                    img_path, gt = a, b
                elif isinstance(b, str) and _is_img_path(b):
                    img_path, gt = b, a
                else:
                    continue
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(os.path.join(cfg_dir, 'dataset', img_path))
                items.append((img_path, gt))
            except Exception:
                continue
    elif isinstance(loaded, Im2LatexDataset):
        # Flatten dataset.data
        for _, lst in getattr(loaded, 'data', {}).items():
            for (gt, img_path) in lst:
                items.append((img_path, gt))
    else:
        items = []
    # Fallback: if empty, parse the pickle directly (handle dict or list formats)
    if not items:
        try:
            import pickle
            with open(args.valdata, 'rb') as f:
                raw = pickle.load(f)
            if isinstance(raw, dict):
                for _, lst in raw.items():
                    for gt, rel_img in lst:
                        img_path = rel_img if os.path.isabs(rel_img) else os.path.abspath(os.path.join(cfg_dir, 'dataset', rel_img))
                        items.append((img_path, gt))
            elif isinstance(raw, list):
                for tup in raw:
                    if not isinstance(tup, (list, tuple)) or len(tup) != 2:
                        continue
                    a, b = tup
                    if isinstance(a, str) and _is_img_path(a):
                        rel_img, gt = a, b
                    elif isinstance(b, str) and _is_img_path(b):
                        rel_img, gt = b, a
                    else:
                        continue
                    img_path = rel_img if os.path.isabs(rel_img) else os.path.abspath(os.path.join(cfg_dir, 'dataset', rel_img))
                    items.append((img_path, gt))
        except Exception as e:
            logging.warning(f"Fallback load of val.pkl failed: {e}")

    # Initialize tokenizer FIRST and align model vocab with it to avoid out-of-range indices
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    # Determine vocab size for safe clamping and model construction
    try:
        _vocab = tokenizer.get_vocab()
        _max_id = max(_vocab.values()) if _vocab else 0
    except Exception:
        _vocab, _max_id = {}, 0
    _tok_vocab = (_max_id + 1)
    _num_tokens_cfg = args.get('num_tokens', None)
    # Always prefer tokenizer vocab; set args.num_tokens so get_model() constructs matching heads/embeddings
    if _tok_vocab > 0:
        if (_num_tokens_cfg is None) or (int(_num_tokens_cfg) != _tok_vocab):
            print(f"[sample-eval] setting args.num_tokens={_tok_vocab} based on tokenizer (was { _num_tokens_cfg })")
        args.num_tokens = _tok_vocab
    _num_tokens = int(args.get('num_tokens', _tok_vocab)) if _tok_vocab > 0 else int(_num_tokens_cfg or 0)
    # Warn if tokenizer vocab size mismatches provided config
    try:
        if _num_tokens_cfg is not None and _tok_vocab not in (0, int(_num_tokens_cfg)):
            print(f"[sample-eval] WARNING: tokenizer vocab={_tok_vocab} != config.num_tokens={int(_num_tokens_cfg)}; using tokenizer size.")
    except Exception:
        pass

    # Build model AFTER num_tokens is aligned to tokenizer and load checkpoint
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    # Try to resize token embeddings to match tokenizer if supported
    try:
        if hasattr(model, 'resize_token_embeddings') and callable(getattr(model, 'resize_token_embeddings')) and _num_tokens:
            model.resize_token_embeddings(_num_tokens)
    except Exception:
        pass
    # Ensure model on the intended device
    try:
        model = model.to(device)
    except Exception:
        pass
    chk = args.get('load_chkpt', None)
    if chk is None:
        out_path = os.path.join(args.model_path, args.name)
        if os.path.isdir(out_path):
            pths = [os.path.join(out_path, f) for f in os.listdir(out_path) if f.endswith('.pth')]
            if pths:
                pths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                chk = pths[0]
                logging.warning(f"Auto-selected latest checkpoint: {chk}")
    if chk is not None and os.path.exists(chk):
        sd = torch.load(chk, map_location=device)
        load_state_dict_forgiving(model, sd, prefix="sample-eval: ")
    else:
        logging.warning("No checkpoint found; evaluating random-initialized model.")
    model.eval()
    # GPU info diagnostics
    if device != 'cpu' and torch.cuda.is_available():
        try:
            dev_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(dev_idx)
            print(f"[sample-eval] device={device} ({gpu_name}), checkpoint={(chk or 'none')}", flush=True)
        except Exception:
            print(f"[sample-eval] device={device}, checkpoint={(chk or 'none')}", flush=True)
    else:
        print(f"[sample-eval] device={device}, checkpoint={(chk or 'none')}", flush=True)

    # Build decoding parameters from args if present
    gen_kwargs = {}
    for k in ['num_beams', 'no_repeat_ngram_size', 'max_new_tokens']:
        v = args.get(k, None)
        if isinstance(v, int) and v is not None and v > 0:
            gen_kwargs[k] = v
    for k in ['length_penalty', 'repetition_penalty']:
        v = args.get(k, None)
        if isinstance(v, (int, float)) and v is not None and v > 0:
            gen_kwargs[k] = float(v)
    # Filter kwargs to only those accepted by model.generate
    gen_kwargs_use = gen_kwargs
    _gen_warned = {'unsupported': False, 'typeerror': False}
    try:
        sig = inspect.signature(getattr(model, 'generate'))
        allowed = set(sig.parameters.keys())
        gen_kwargs_use = {k: v for k, v in gen_kwargs.items() if k in allowed}
        if gen_kwargs and gen_kwargs_use.keys() != gen_kwargs.keys() and not _gen_warned['unsupported']:
            missing = [k for k in gen_kwargs.keys() if k not in allowed]
            print(f"[sample-eval] Note: model.generate does not support: {missing}. Using supported subset: {list(gen_kwargs_use.keys())}")
            _gen_warned['unsupported'] = True
    except Exception:
        pass

    # Simple LaTeX cleanup for decoded tokenizer output
    _space_token = 'Ġ'  # common GPT-2 style space marker
    def _clean_tex(s: str) -> str:
        if not s:
            return s
        # Replace GPT2 space token marker with real space
        s = s.replace(_space_token, ' ')
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s)
        # Remove spaces after a backslash: '\ frac' -> '\frac'
        s = re.sub(r"\\\s+", r"\\", s)
        # Join split command names: '\ f r a c' -> '\frac'
        def _join_cmd(m):
            letters = re.sub(r"\s+", "", m.group(1))
            return "\\" + letters
        s = re.sub(r"\\((?:[A-Za-z]\s*){2,})", _join_cmd, s)
        # Remove spaces before/after braces and math operators
        s = re.sub(r"\s*\{\s*", "{", s)
        s = re.sub(r"\s*\}\s*", "}", s)
        s = re.sub(r"\s*_\s*", "_", s)
        s = re.sub(r"\s*\^\s*", "^", s)
        s = re.sub(r"\s*\\left\s*", r"\\left ", s)
        s = re.sub(r"\s*\\right\s*", r"\\right ", s)
        # Fix split environment names inside begin/end: '\begin{ma tri x}' -> '\begin{matrix}'
        def _join_env(m):
            name = re.sub(r"\s+", "", m.group(1))
            return f"\\begin{{{name}}}"
        def _join_env_end(m):
            name = re.sub(r"\s+", "", m.group(1))
            return f"\\end{{{name}}}"
        s = re.sub(r"\\begin\{([^}]*)\}", _join_env, s)
        s = re.sub(r"\\end\{([^}]*)\}", _join_env_end, s)
        # Additional hygiene to cut obvious hallucinations
        s = re.sub(r"\\int(?=[A-Za-z])", r"\\int ", s)  # \intx -> \int x
        s = s.replace(r"\partialt", r"\partial t")      # \partialt -> \partial t
        s = re.sub(r"(\\cdot\s*){2,}", r"\\cdot ", s)  # dedup \cdot
        # Drop isolated caps adjacent to \cdot (noise like B\cdot BF)
        s = re.sub(r"\b[A-Z]\b\s*\\cdot\s*", " ", s)
        s = re.sub(r"(?<=\\cdot)\s*[A-Z](?![A-Za-z])", "", s)
        # Make differentials consistent and keep \cdot with proper spacing
        # e.g., x \\cdot dx -> x \\cdot \, dx ;  x\\cdot\, d t -> x \\cdot \, dt ; x \\cdot\\dx -> x \\cdot \, dx
        s = re.sub(r"\\cdot\s*\\,?\s*d\s*([A-Za-z])", r" \\cdot \\, d\1", s)
        s = re.sub(r"\\cdot\\?d\s*([A-Za-z])", r" \\cdot \\, d\1", s)
        # Very specific fallback for '\\cdotdx' with no spaces
        s = s.replace(r"\cdotdx", r" \cdot \, dx")
        s = re.sub(r"\\cdot\s*dx\b", r" \\cdot \\, dx", s)
        s = re.sub(r"\s+d\s*([A-Za-z])", r" \\, d\1", s)
        # Remove \cdot between two single-letter variables only if near an integral (heuristic)
        s = re.sub(r"(\\int[^$]{0,80}?)\b([a-zA-Z])\s*\\cdot\s*([a-zA-Z])\b", r"\1\2 \\, d\3", s)
        # Avoid visual double parentheses: if a pmatrix is wrapped in extra parentheses, drop them
        s = re.sub(r"\(\s*\\begin\{pmatrix\}", r"\\begin{pmatrix}", s)
        s = re.sub(r"\\end\{pmatrix\}\s*\)", r"\\end{pmatrix}", s)
        # Trim
        return s.strip()
    
    def _clamp_ids_tensor(t: torch.Tensor) -> torch.Tensor:
        """Clamp token IDs to valid vocab range to avoid device-side asserts during decode."""
        try:
            if _num_tokens and isinstance(t, torch.Tensor):
                t = t.clamp_(0, max(0, _num_tokens - 1))
        except Exception:
            pass
        return t
    # Prepare render directory if requested
    render_dir = getattr(args, 'sample_render_dir', None)
    if render_dir:
        os.makedirs(render_dir, exist_ok=True)

    def render_latex_png(tex: str, out_path: str):
        try:
            txt = tex.strip()
            # Ensure math mode
            if not (txt.startswith('$') and txt.endswith('$')):
                txt = f"$ {txt} $"
            # Matplotlib mathtext cannot render LaTeX environments like \begin{matrix}
            # Skip PNG render for those; rely on HTML MathJax instead
            if "\\begin{" in txt or "\\end{" in txt:
                raise RuntimeError("env_not_supported_by_mathtext")
            fig = plt.figure(figsize=(4, 1), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.text(0.5, 0.5, txt, fontsize=18, ha='center', va='center')
            fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            return True
        except Exception as e:
            print(f"[warn] render failed for {out_path}: {e}")
            return False
    shown = 0
    print("=== Sample predictions (val) ===", flush=True)
    report_rows = []
    # Iterate over derived items list directly
    print(f"[sample-eval] resolved {len(items)} validation items from {args.valdata}", flush=True)
    if not items:
        print("[sample-eval] No items found. Check that your pickle contains image paths relative to 'dataset/' or absolute paths.", flush=True)
    failures = 0
    # If a CUDA device-side assert happens during generate (e.g., vocab/embedding mismatch),
    # switch to CPU for the remainder of sample-eval so the run completes.
    cpu_fallback = False
    if items:
        for (img_path, gt_eq) in items:
            try:
                # Resolve to absolute if relative under dataset
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(os.path.join(cfg_dir, 'dataset', img_path))
                if not os.path.exists(img_path):
                    print(f"[sample-eval] WARNING: image path does not exist: {img_path}", flush=True)
                    failures += 1
                    continue
                im = cv2.imread(img_path)
                if im is None:
                    print(f"[sample-eval] WARNING: could not read image: {img_path}", flush=True)
                    failures += 1
                    continue
                # Use single-channel grayscale to match pix2tex eval pipeline expectations
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # Resize to fit within configured max dimensions BEFORE padding, preserving aspect ratio
                try:
                    max_h = int(getattr(args, 'max_height', 256) or 256)
                    max_w = int(getattr(args, 'max_width', 1024) or 1024)
                except Exception:
                    max_h, max_w = 256, 1024
                h, w = im.shape[:2]
                # If any dimension exceeds limits, scale down while preserving aspect
                scale = 1.0
                if h > max_h or w > max_w:
                    scale = min(max_h / max(h, 1), max_w / max(w, 1))
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    # Ensure even numbers for later patching/padding stability
                    if new_w % 2: new_w += 1
                    if new_h % 2: new_h += 1
                    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    h, w = im.shape[:2]
                # Pad to next multiple of 32 (H, W) while ensuring we do not exceed the max limits
                def _next_mul32(x):
                    return ((x + 31) // 32) * 32
                target_h, target_w = _next_mul32(h), _next_mul32(w)
                # Clamp targets to max limits
                target_h = min(target_h, max_h)
                target_w = min(target_w, max_w)
                pad_bottom = max(0, target_h - h)
                pad_right = max(0, target_w - w)
                if pad_bottom or pad_right:
                    im = cv2.copyMakeBorder(im, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=255)
                res = test_transform(image=im)['image']
                # Accept both numpy arrays and torch tensors from albumentations
                if torch.is_tensor(res):
                    im_t = res
                else:
                    im_t = torch.from_numpy(res)
                # Ensure CHW. If HxW, add channel. If more than 1 channel, keep first
                if im_t.ndim == 2:
                    im_t = im_t.unsqueeze(0)
                elif im_t.ndim == 3:
                    pass
                else:
                    raise RuntimeError(f"Unexpected image tensor shape: {tuple(im_t.shape)}")
                if im_t.shape[0] > 1:
                    im_t = im_t[:1]
                # Honor CPU fallback if previously triggered
                run_device = 'cpu' if cpu_fallback else device
                im_t = im_t.float().unsqueeze(0).to(run_device)
                with torch.no_grad():
                    try:
                        gen = model.to(run_device).generate(im_t, **gen_kwargs_use)
                    except TypeError as te:
                        if not _gen_warned['typeerror'] and gen_kwargs_use:
                            print(f"[sample-eval] Note: generation kwargs caused TypeError: {te}. Falling back to default generate().")
                            _gen_warned['typeerror'] = True
                        gen = model.to(run_device).generate(im_t)
                    except RuntimeError as re_err:
                        # Handle CUDA device-side asserts by switching to CPU
                        msg = str(re_err)
                        if (not cpu_fallback) and (('device-side assert' in msg) or ('IndexKernel' in msg)):
                            print("[sample-eval] CUDA device-side assert detected during generate; switching to CPU for the rest of sample-eval.")
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            cpu_fallback = True
                            run_device = 'cpu'
                            im_t = im_t.to('cpu')
                            model = model.to('cpu')
                            # Retry on CPU
                            try:
                                gen = model.generate(im_t, **gen_kwargs_use)
                            except TypeError:
                                gen = model.generate(im_t)
                        else:
                            raise
                    if run_device != 'cpu' and torch.cuda.is_available():
                        torch.cuda.synchronize()
                ids_t = gen[0].detach().cpu()
                ids_t = _clamp_ids_tensor(ids_t)
                ids = ids_t.tolist()
                # Prefer tokenizer BOS/EOS if available
                try:
                    bos_id = getattr(tokenizer, 'bos_token_id', None)
                    eos_id = getattr(tokenizer, 'eos_token_id', None)
                    if bos_id is None:
                        bos_id = 1
                    if eos_id is None:
                        eos_id = 2
                except Exception:
                    bos_id, eos_id = 1, 2
                if ids and ids[0] == bos_id:
                    ids = ids[1:]
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                pred = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                pred = _clean_tex(pred)
                print(f"Image: {img_path}", flush=True)
                print(f"GT:    {gt_eq}", flush=True)
                print(f"Pred[{shown+1}]: {pred}", flush=True)
                if render_dir:
                    gt_png = os.path.join(render_dir, f"gt_{shown+1}.png")
                    pr_png = os.path.join(render_dir, f"pred_{shown+1}.png")
                    if render_latex_png(gt_eq, gt_png):
                        print(f"GT PNG:   {gt_png}", flush=True)
                    if render_latex_png(pred, pr_png):
                        print(f"Pred PNG: {pr_png}", flush=True)
                report_rows.append({
                    'image': img_path,
                    'gt': gt_eq,
                    'pred': pred
                })
                shown += 1
                if shown >= max_samples:
                    break
            except Exception as e:
                failures += 1
                try:
                    print(f"[sample-eval] ERROR on {img_path}: {e}", flush=True)
                except Exception:
                    pass
                continue
    # fall-through to report writing
    print(f"[sample-eval] done: shown={shown}, failures={failures}", flush=True)

    # Write MathJax HTML report if requested
    report_path = getattr(args, 'sample_report', '')
    if report_path:
        try:
            os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("""<!doctype html><html><head>
<meta charset='utf-8'/>
<script>
window.MathJax = { tex: { inlineMath: [['$','$'], ['\\(','\\)']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
<style>table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}code{white-space:pre-wrap}</style>
</head><body>
<h2>Sample Eval</h2>
<table><thead><tr><th>#</th><th>Image</th><th>GT</th><th>Pred</th></tr></thead><tbody>
""")
                for i, row in enumerate(report_rows, 1):
                    img = row['image']
                    gt = row['gt']
                    pr = row['pred']
                    # escape HTML
                    def esc(s):
                        return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;') if s else ''
                    # wrap in $ for MathJax if not already
                    def wrap_math(s):
                        s = s.strip()
                        return s if (s.startswith('$') and s.endswith('$')) else f"$ {s} $"
                    # Show thumbnail preview of the source image plus its path
                    try:
                        rel_img = os.path.relpath(img, os.path.dirname(report_path) or '.') if img else ''
                    except Exception:
                        rel_img = img
                    thumb = f"<img src='{esc(rel_img)}' style='max-height:120px;max-width:180px;object-fit:contain;background:#fafafa;border:1px solid #eee;padding:2px' alt='img {i}'/>" if img else ''
                    img_cell = f"{thumb}<br><code>{esc(img)}</code>" if img else ''
                    f.write(f"<tr><td>{i}</td><td>{img_cell}</td><td>{wrap_math(esc(gt))}</td><td>{wrap_math(esc(pr))}</td></tr>\n")
                f.write("""</tbody></table></body></html>""")
            print(f"[report] Wrote {report_path}")
        except Exception as e:
            print(f"[warn] failed to write report {report_path}: {e}")
    return


def train(args):
    # Inspect pickles directly to ensure they contain items and paths resolve
    import pickle
    def _inspect_pickle(path, base_dir):
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            n = len(d) if hasattr(d, '__len__') else 'unknown'
            logging.warning(f"PICKLE {os.path.basename(path)} len={n} type={type(d)}")
            # peek up to 3 items
            try:
                items = list(d[:3]) if hasattr(d, '__getitem__') else [next(iter(d))]
            except Exception as e:
                logging.warning(f"PICKLE ITER ERROR: {e}")
                items = []
            for i, it in enumerate(items):
                if isinstance(it, dict):
                    img = it.get('image') or it.get('img') or it.get('path')
                    tex = it.get('latex') or it.get('formula')
                else:
                    try:
                        img, tex = it
                    except Exception:
                        img, tex = None, None
                if isinstance(img, str):
                    abs_img = img if os.path.isabs(img) else os.path.abspath(os.path.join(base_dir, 'dataset', img))
                    logging.warning(f"ITEM{i} image exists={os.path.exists(abs_img)} -> {abs_img}")
                if isinstance(tex, str):
                    logging.warning(f"ITEM{i} latex snippet: {tex[:80]}")
        except Exception as e:
            logging.error(f"Failed to inspect pickle {path}: {e}")

    cfg_dir = os.path.dirname(os.path.abspath(args.get('config', 'config.yaml')))
    _inspect_pickle(args.data, cfg_dir)
    _inspect_pickle(args.valdata, cfg_dir)

    def _build_dataset_from_list(items, base_dir, label, test=False):
        """Convert list[(rel_img, equation)] into Im2LatexDataset grouped by image dimensions.
        base_dir should be the directory containing the config.yaml; images are usually under base_dir/dataset/...
        """
        # Try cache first
        try:
            cache_path = os.path.abspath(os.path.join(base_dir, 'dataset', f'{label}_ds.pkl'))
            if os.path.exists(cache_path):
                logging.warning(f"Loading cached dataset: {cache_path}")
                cached = Im2LatexDataset().load(cache_path)
                if isinstance(cached, Im2LatexDataset):
                    # Ensure runtime params are up to date
                    cached.update(
                        tokenizer=args.tokenizer,
                        batchsize=(args.testbatchsize if test else args.batchsize),
                        keep_smaller_batches=True,
                        test=test,
                        pad=args.pad,
                        max_seq_len=args.max_seq_len,
                        max_dimensions=args.max_dimensions,
                        min_dimensions=args.min_dimensions,
                    )
                    return cached
        except Exception:
            pass
        ds = Im2LatexDataset()
        ds.data = {}
        added = 0
        from tqdm import tqdm as _tqdm
        for it in _tqdm(items, desc=f'Bucketing {label}', unit='it', miniters=1000):
            try:
                a, b = it
                if isinstance(a, str) and isinstance(b, str):
                    rel_img, eq = a, b
                else:
                    continue
                img_path = rel_img if os.path.isabs(rel_img) else os.path.abspath(os.path.join(base_dir, 'dataset', rel_img))
                if not os.path.exists(img_path):
                    alt = os.path.abspath(os.path.join(base_dir, rel_img))
                    img_path = alt if os.path.exists(alt) else img_path
                # Group by actual image dimensions so batches contain same-sized tensors
                w, h = imagesize.get(img_path)
                if w is None or h is None:
                    continue
                key = (w, h)
                ds.data.setdefault(key, []).append((eq, img_path))
                added += 1
            except Exception:
                continue
        logging.warning(f"Constructed dataset from list: items_in={len(items)} items_added={added} buckets={len(ds.data)}")
        # initialize tokenizer and batch settings via update()
        ds.update(
            tokenizer=args.tokenizer,
            batchsize=(args.testbatchsize if test else args.batchsize),
            keep_smaller_batches=True,
            test=test,
            pad=args.pad,
            max_seq_len=args.max_seq_len,
            max_dimensions=args.max_dimensions,
            min_dimensions=args.min_dimensions,
        )
        # Save cache for future runs
        try:
            ds.save(cache_path)
            logging.warning(f"Saved cached dataset: {cache_path}")
        except Exception:
            pass
        return ds

    dataloader = Im2LatexDataset().load(args.data)
    if isinstance(dataloader, list):
        cfg_dir = os.path.dirname(os.path.abspath(args.get('config', 'config.yaml')))
        dataloader = _build_dataset_from_list(dataloader, cfg_dir, label='train', test=False)
    else:
        # keep_smaller_batches to avoid zero batches when small dataset or heavy filtering
        dataloader.update(batchsize=args.batchsize, keep_smaller_batches=True, test=False, **{k: v for k, v in dict(args).items() if k not in ['batchsize']})

    valdataloader = Im2LatexDataset().load(args.valdata)
    if isinstance(valdataloader, list):
        cfg_dir = os.path.dirname(os.path.abspath(args.get('config', 'config.yaml')))
        valdataloader = _build_dataset_from_list(valdataloader, cfg_dir, label='val', test=True)
    else:
        valargs = args.copy()
        valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
        valdataloader.update(**valargs)
    try:
        logging.warning(f"Train loader batches: {len(dataloader)} | Val loader batches: {len(valdataloader)}")
    except Exception:
        pass
    if hasattr(dataloader, '__len__') and len(dataloader) == 0:
        logging.error("Train dataloader has 0 batches. Check config: max_height/max_width, tokenizer path, or dataset pickles.")
        return
    device = args.device
    # Ensure tokenizer/model vocab alignment during training as well
    try:
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        _tv = tok.get_vocab() if hasattr(tok, 'get_vocab') else {}
        _tok_size = (max(_tv.values()) + 1) if _tv else None
        if _tok_size and getattr(args, 'num_tokens', None) != _tok_size:
            logging.warning(f"[train] setting args.num_tokens={_tok_size} based on tokenizer (was {getattr(args, 'num_tokens', None)})")
            args.num_tokens = _tok_size
    except Exception:
        pass
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    max_bleu, max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    # If supported, resize embeddings to tokenizer size (helps when loading preexisting checkpoints with different heads)
    try:
        if hasattr(model, 'resize_token_embeddings') and callable(getattr(model, 'resize_token_embeddings')) and getattr(args, 'num_tokens', None):
            model.resize_token_embeddings(int(args.num_tokens))
    except Exception:
        pass
    load_chk = args.get('load_chkpt', None)
    resume_path = args.get('resume', None)
    
    if resume_path is not None and os.path.exists(resume_path):
        # Load full training state
        global_step = load_training_state(resume_path)
    elif load_chk is not None:
        # Load only model weights (old behavior)
        sd = torch.load(load_chk, map_location=device)
        load_state_dict_forgiving(model, sd, prefix="train: ")
        global_step = 0
    else:
        global_step = 0

    def save_models(e, step=0):
        torch.save(model.state_dict(), os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step)))
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+', encoding='utf-8'))
    
    def save_training_state(e, step):
        """Save full training state: model, optimizer, scheduler, epoch, global_step"""
        state = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': e,
            'global_step': step,
            'args': dict(args)
        }
        torch.save(state, os.path.join(out_path, '%s_e%02d_step%02d_state.pth' % (args.name, e+1, step)))
        print(f"[state] Saved training state to {out_path}/{args.name}_e{e+1:02d}_step{step:02d}_state.pth")
    
    def load_training_state(path):
        """Load full training state and restore model, optimizer, scheduler, epoch"""
        print(f"[state] Loading training state from {path}")
        state = torch.load(path, map_location=device)
        
        # Restore model
        model.load_state_dict(state['model'])
        
        # Restore optimizer
        opt.load_state_dict(state['optimizer'])
        
        # Restore scheduler
        scheduler.load_state_dict(state['scheduler'])
        
        # Update args with saved epoch
        args.epoch = state['epoch'] + 1  # Continue from next epoch
        global_step = state['global_step']
        
        print(f"[state] Restored to epoch {args.epoch}, global_step {global_step}")
        return global_step

    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    # Select scheduler with appropriate arguments
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = get_scheduler(args.scheduler)(opt, T_max=args.epochs)
    elif args.scheduler == 'StepLR':
        scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)
    else:
        # Fallback for other schedulers; rely on their defaults or config-covered params
        scheduler = get_scheduler(args.scheduler)(opt)

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    try:
        # Logging helpers
        prev_epoch_loss = None
        last_print_loss = None
        step_interval = getattr(args, 'log_interval', 100)
        ema_beta = getattr(args, 'ema_beta', 0.98)
        ema_loss = None
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            epoch_loss_sum, epoch_batches = 0.0, 0
            epoch_start = time.time()
            for i, (seq, im) in enumerate(dset, start=1):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0.0
                    last_grad_norm = None
                    for j in range(0, len(im), microbatch):
                        tgt_seq = seq['input_ids'][j:j+microbatch].to(device)
                        tgt_mask = seq['attention_mask'][j:j+microbatch].bool().to(device)
                        loss = model.data_parallel(
                            im[j:j+microbatch].to(device),
                            device_ids=args.gpu_devices,
                            tgt_seq=tgt_seq,
                            mask=tgt_mask
                        ) * microbatch / args.batchsize
                        loss.backward()
                        total_loss += float(loss.item())
                        try:
                            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1).item())
                        except Exception:
                            last_grad_norm = None
                    # Update EMA loss after aggregating microbatches
                    try:
                        if ema_loss is None:
                            ema_loss = total_loss
                        else:
                            ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * total_loss
                    except Exception:
                        pass
                    opt.step()
                    scheduler.step()
                    epoch_loss_sum += total_loss
                    epoch_batches += 1
                    # Update tqdm
                    try:
                        lr = opt.param_groups[0]['lr']
                    except Exception:
                        lr = None
                    ddesc = f"Loss: {total_loss:.4f}"
                    if ema_loss is not None:
                        ddesc += f" | EMA: {ema_loss:.4f}"
                    if lr is not None:
                        ddesc += f" | LR: {lr:.6f}"
                    if last_grad_norm is not None:
                        ddesc += f" | |g|: {last_grad_norm:.3f}"
                    dset.set_description(ddesc)
                    # Periodic plain-English trend
                    if i % step_interval == 0:
                        trend = ""
                        if last_print_loss is not None:
                            if total_loss < last_print_loss - 1e-6:
                                trend = "down vs last interval (good)"
                            elif total_loss > last_print_loss + 1e-6:
                                trend = "up vs last interval (watch)"
                            else:
                                trend = "flat vs last interval"
                        msg = f"[epoch {e+1} step {i}] loss={total_loss:.4f} ema={(f'{ema_loss:.4f}' if ema_loss is not None else 'n/a')} lr={(lr if lr is not None else 'n/a')} grad_norm={(f'{last_grad_norm:.3f}' if last_grad_norm is not None else 'n/a')} -> {trend}"
                        print(msg, flush=True)
                        last_print_loss = total_loss
                # Skip BLEU/token eval on Windows to avoid torchtext
                # bleu_score, edit_distance, token_accuracy = evaluate_noop()
            # Epoch-end summary
            epoch_time = time.time() - epoch_start
            avg_loss = (epoch_loss_sum / max(1, epoch_batches)) if epoch_batches else float('nan')
            trend_ep = ""
            if prev_epoch_loss is not None and not (avg_loss != avg_loss):  # check not NaN
                if avg_loss < prev_epoch_loss - 1e-6:
                    trend_ep = "Loss down vs last epoch (improving)."
                elif avg_loss > prev_epoch_loss + 1e-6:
                    trend_ep = "Loss up vs last epoch (worse)."
                else:
                    trend_ep = "Loss flat vs last epoch."
            print(f"[epoch {e+1}] avg_loss={avg_loss:.4f} ema={(f'{ema_loss:.4f}' if ema_loss is not None else 'n/a')} in {epoch_time:.1f}s. {trend_ep}", flush=True)
            prev_epoch_loss = avg_loss
            # Validation + early stopping + best checkpoint
            do_validate = True
            if do_validate:
                try:
                    val_loss, val_em = _validate(model, valdataloader, args, device)
                except Exception as ve:
                    print(f"[val] failed: {ve}")
                    val_loss, val_em = float('nan'), 0.0
                print(f"[epoch {e+1}] val_loss={val_loss:.4f} val_EM={val_em*100:.2f}%", flush=True)
                # Track best
                if not hasattr(args, '_best_val'):
                    args._best_val = {
                        'loss': None,
                        'epoch': None,
                        'em': None,
                    }
                min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0) or 0.0)
                patience = int(getattr(args, 'early_stopping_patience', 0) or 0)
                improved = False
                if val_loss == val_loss:  # not NaN
                    if args._best_val['loss'] is None or (val_loss < args._best_val['loss'] - min_delta):
                        improved = True
                        args._best_val['loss'] = val_loss
                        args._best_val['epoch'] = e + 1
                        args._best_val['em'] = val_em
                        # Save best checkpoint
                        try:
                            best_path = os.path.join(out_path, f"{args.name}_best.pth")
                            torch.save(model.state_dict(), best_path)
                            print(f"[best] Saved improved checkpoint: {best_path} (val_loss={val_loss:.4f}, EM={val_em*100:.2f}%)")
                        except Exception as se:
                            print(f"[best] save failed: {se}")
                # Early stopping bookkeeping
                if patience > 0:
                    if not hasattr(args, '_no_improve_epochs'):
                        args._no_improve_epochs = 0
                    if improved:
                        args._no_improve_epochs = 0
                    else:
                        args._no_improve_epochs += 1
                        print(f"[early-stopping] no improvement epochs={args._no_improve_epochs}/{patience}")
                        if args._no_improve_epochs >= patience:
                            print("[early-stopping] stopping training.")
                            save_models(e, step=len(dataloader))
                            save_training_state(e, step=len(dataloader))
                            return
            # Periodic save
            if (e+1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
                save_training_state(e, step=len(dataloader))
    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
            save_training_state(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))
    save_training_state(e, step=len(dataloader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model (Windows-safe, no torchtext eval)')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--sample-eval', type=int, default=0, help='Print N predictions from val set and exit')
    parser.add_argument('--sample-render-dir', type=str, default='', help='If set, save PNG renders of GT and Pred LaTeX to this directory')
    parser.add_argument('--sample-report', type=str, default='', help='Write an HTML report with MathJax-rendered GT and Pred')
    # Optional decoding parameters for sample-eval (also honored if set in config)
    parser.add_argument('--num-beams', type=int, default=None, help='Beam search width for generation')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=None, help='Disallow repeating n-grams of this size')
    parser.add_argument('--length-penalty', type=float, default=None, help='Length penalty for beam search')
    parser.add_argument('--repetition-penalty', type=float, default=None, help='Repetition penalty during generation')
    parser.add_argument('--max-new-tokens', type=int, default=None, help='Maximum new tokens to generate')
    parser.add_argument('--resume', type=str, default=None, help='Path to training state file (.pth) to resume training from')
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    # Resolve tokenizer and dataset paths
    try:
        cfg_dir = os.path.dirname(os.path.abspath(parsed_args.config))
        tok_path = args.get('tokenizer', None)
        if tok_path is not None:
            abs_tok = tok_path if os.path.isabs(tok_path) else os.path.abspath(os.path.join(cfg_dir, tok_path))
            args.tokenizer = abs_tok
            logging.warning(f"Using tokenizer: {abs_tok} (exists={os.path.exists(abs_tok)})")
        for key in ['data', 'valdata']:
            p = args.get(key, None)
            if p is not None:
                abs_p = p if os.path.isabs(p) else os.path.abspath(os.path.join(cfg_dir, p))
                args[key] = abs_p
                logging.warning(f"Using {key}: {abs_p} (exists={os.path.exists(abs_p)})")
    except Exception as e:
        logging.error(f"Tokenizer path resolution failed: {e}")

    if parsed_args.sample_eval and parsed_args.sample_eval > 0:
        # Plumb render dir into args
        if parsed_args.sample_render_dir:
            args.sample_render_dir = parsed_args.sample_render_dir
        if parsed_args.sample_report:
            args.sample_report = parsed_args.sample_report
        _sample_eval(args, max_samples=parsed_args.sample_eval)
    else:
        train(args)
