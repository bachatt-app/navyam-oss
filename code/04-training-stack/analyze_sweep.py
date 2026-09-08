#!/usr/bin/env python3
"""Analyze the navya-1b LR sweep — selection is multi-signal, never final
loss alone.

Pulls each run's train log + per-domain loss from blob (or local files),
then reports per LR:
  - final val loss AND the val-loss slope over the last K eval points
    (linear fit; a steeper negative slope at equal loss = more headroom)
  - training-loss spike count (instability marker: loss[t] > loss[t-1]+0.15)
  - per-domain val loss profile (a diverging domain disqualifies)
Downstream evals: export each ckpt with export_hf.py and run the Tier-1
subset separately; paste those into the final decision.

Usage:
  python analyze_sweep.py --logs sweep_lr2e4.log sweep_lr4e4.log sweep_lr6e4.log
  (add --domains sweep_lr2e4_domains.json ... in the same order)
"""

import argparse
import json
import re

import numpy as np

STEP_RE = re.compile(r"step\s+(\d+) \| loss\s+([\d.]+)")
VAL_RE = re.compile(r"step\s+(\d+) \| VAL loss ([\d.]+)")


def analyze(log_path, k_last=5):
    text = open(log_path).read()
    train = [(int(s), float(v)) for s, v in STEP_RE.findall(text)]
    val = [(int(s), float(v)) for s, v in VAL_RE.findall(text)]
    spikes = sum(1 for (_, a), (_, b) in zip(train, train[1:]) if b > a + 0.15)
    out = {"final_train": train[-1][1] if train else None,
           "final_val": val[-1][1] if val else None,
           "val_points": len(val), "train_spikes": spikes}
    if len(val) >= k_last:
        xs = np.array([s for s, _ in val[-k_last:]], dtype=float)
        ys = np.array([v for _, v in val[-k_last:]], dtype=float)
        slope = np.polyfit(xs, ys, 1)[0] * 1000    # loss per 1000 steps
        out["val_slope_per_1k_steps"] = round(float(slope), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--domains", nargs="*", default=[])
    args = ap.parse_args()

    rows = {}
    for i, log in enumerate(args.logs):
        tag = re.search(r"lr\de\d", log)
        tag = tag.group() if tag else log
        rows[tag] = analyze(log)
        if i < len(args.domains):
            d = json.load(open(args.domains[i]))
            rows[tag]["domains"] = {k: v["val_loss"]
                                    for k, v in d["domains"].items()}

    print(f"{'run':<8} {'final val':>10} {'slope/1k':>10} {'spikes':>7}")
    for tag, r in rows.items():
        print(f"{tag:<8} {r.get('final_val', float('nan')):>10.4f} "
              f"{r.get('val_slope_per_1k_steps', float('nan')):>10.4f} "
              f"{r['train_spikes']:>7}")
    if any("domains" in r for r in rows.values()):
        doms = sorted(next(r["domains"] for r in rows.values()
                           if "domains" in r))
        print(f"\n{'domain':<20}" + "".join(f"{t:>10}" for t in rows))
        for d in doms:
            print(f"{d:<20}" + "".join(
                f"{rows[t].get('domains', {}).get(d, float('nan')):>10.3f}"
                for t in rows))
    print("\nDecision guide: prefer the LOWEST final val whose slope is still"
          "\nclearly negative and spike count ~0; a run that wins final loss"
          "\nbut has flattened (slope ~0) loses to one still descending."
          "\nCross-check per-domain profile and downstream evals before"
          "\ncommitting the 1b config.")
    json.dump(rows, open("sweep_analysis.json", "w"), indent=1)
    print("→ sweep_analysis.json")


if __name__ == "__main__":
    main()
