#!/usr/bin/env python3
"""Score a Navya checkpoint on BachattBench (MCQ, length-normalized loglikelihood).

Scoring matches lm-eval's acc_norm convention: for each option, compute the
model's mean per-token logprob of the option text following the prompt; pick
the highest. Reports overall + per-section accuracy and review coverage
(items with reviewed=false are still scored but flagged — the bench is only
authoritative once items are expert-reviewed).

Usage:
  python run_bachattbench.py --ckpt ../04-training-stack/out/navya-0/ckpt_last.pt \
      [--bench "bachattbench/*.jsonl"] [--json-out results/bb_navya0.json]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def option_logprob(model, tok, device, prompt, option):
    p_ids = tok.encode(prompt).ids
    o_ids = tok.encode(" " + option).ids
    x = torch.tensor([p_ids + o_ids], device=device)
    logits, _ = model(x, x)          # targets -> full-position logits
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    span = range(len(p_ids) - 1, len(p_ids) - 1 + len(o_ids))
    token_lps = [lp[i, x[0, i + 1]].item() for i in span]
    return sum(token_lps) / len(token_lps)     # length-normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default="bachattbench/*.jsonl")
    ap.add_argument("--tokenizer", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "01-tokenizer", "tokenizer-v0.3-64k.json"))
    ap.add_argument("--json-out")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)

    items = []
    for path in sorted(glob.glob(args.bench)):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if d.get("type", "mcq") == "mcq":
                    items.append(d)
    print(f"{len(items)} MCQ items from {args.bench}")

    per_section = defaultdict(lambda: [0, 0])
    correct = reviewed_n = 0
    for i, d in enumerate(items):
        prompt = d["prompt"] + "\nAnswer:"
        scores = {k: option_logprob(model, tok, device, prompt, v)
                  for k, v in d["options"].items()}
        pick = max(scores, key=scores.get)
        ok = pick == d["answer"]
        correct += ok
        sec = per_section[d.get("section", "misc")]
        sec[0] += ok
        sec[1] += 1
        reviewed_n += bool(d.get("reviewed", True))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)} …", file=sys.stderr)

    n = len(items)
    print(f"\noverall: {correct}/{n} = {correct/n:.3f}   "
          f"(reviewed items: {reviewed_n}/{n} — unreviewed results are provisional)")
    out = {"ckpt": os.path.abspath(args.ckpt), "overall_acc": correct / n,
           "n_items": n, "n_reviewed": reviewed_n, "sections": {}}
    for sec, (c, t) in sorted(per_section.items()):
        print(f"  {sec:<15} {c}/{t} = {c/t:.3f}")
        out["sections"][sec] = {"correct": c, "total": t, "acc": c / t}
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"→ {args.json_out}")


if __name__ == "__main__":
    main()
