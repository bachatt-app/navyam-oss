#!/usr/bin/env python3
"""Per-domain validation loss for a Navya checkpoint.

Reads each domain bin in data/domains/ and evaluates mean next-token NLL on
its held-out tail — the same VAL_FRAC=0.005 slice build_stream.py excludes
from every training stream, reconstructed deterministically, so this is valid
for checkpoints trained on any stream built from these bins.

Reports loss + perplexity per domain: the first place data problems show up
(one domain's loss diverging = contamination, dedup failure, or mix bug).

Usage:
  python eval_domain_loss.py --ckpt ../04-training-stack/out/navya-1a/ckpt_last.pt \
      [--domains ../04-training-stack/data/domains] [--windows 200] [--json-out f.json]
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402

VAL_FRAC = 0.005      # must match build_stream.py


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def domain_loss(model, arr, seq_len, windows, batch, device, seed=1234):
    n_val = max(1, int(len(arr) * VAL_FRAC))
    val = arr[-n_val:]
    if len(val) < seq_len + 1:
        return None, n_val
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val) - seq_len - 1, size=windows)
    losses = []
    for i in range(0, windows, batch):
        chunk = starts[i:i + batch]
        x = torch.from_numpy(np.stack([val[s:s + seq_len] for s in chunk])
                             .astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[s + 1:s + seq_len + 1] for s in chunk])
                             .astype(np.int64)).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses), n_val


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domains", default=os.path.join(
        here, "..", "04-training-stack", "data", "domains"))
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"{args.ckpt} · step {ckpt.get('step')} · "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M · {device}")

    results = {}
    for bin_path in sorted(glob.glob(os.path.join(args.domains, "*.bin"))):
        name = os.path.basename(bin_path)[:-4]
        meta = json.load(open(bin_path + ".meta.json"))
        arr = np.memmap(bin_path, dtype=np.dtype(meta["dtype"]), mode="r")
        loss, n_val = domain_loss(model, arr, cfg.max_seq_len,
                                  args.windows, args.batch, device)
        if loss is None:
            print(f"{name:<20} val slice too small ({n_val} tokens) — skipped")
            continue
        results[name] = {"val_loss": round(loss, 4),
                         "ppl": round(math.exp(loss), 2),
                         "val_tokens": int(n_val)}
        print(f"{name:<20} loss {loss:.4f}   ppl {math.exp(loss):8.2f}   "
              f"({n_val/1e6:.2f}M val tokens)")

    if args.json_out:
        json.dump({"ckpt": os.path.abspath(args.ckpt),
                   "step": ckpt.get("step"), "domains": results},
                  open(args.json_out, "w"), indent=1)
        print(f"→ {args.json_out}")


if __name__ == "__main__":
    main()
