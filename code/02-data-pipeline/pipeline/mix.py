#!/usr/bin/env python3
"""Corpus mixing: turn per-domain pools into a versioned snapshot manifest.

Input: a JSON config mapping domain -> {paths: [...], tokens: N, weight: w}
where `tokens` is the measured clean-token count of the pool and `weight` is
the target share of the final corpus (charter mix as the starting hypothesis).

Output: a manifest with per-domain sampling rates (epochs over each pool) and
a snapshot ID = sha256 of the manifest content. Runs reference the snapshot ID;
the manifest is immutable once written.

A pool smaller than its target share is repeated (multiple epochs) up to
MAX_EPOCHS; beyond that the manifest records a shortfall instead of silently
over-repeating (repetition beyond ~4 epochs measurably hurts quality).

Usage:
  python mix.py --config mix_v1.json --total-tokens 100e9 --out snapshots/
"""

import argparse
import hashlib
import json
import os
import sys

MAX_EPOCHS = 4.0

# Charter starting hypothesis (README section 3), adjusted to sum to 1.0
# (the charter gives ranges; these sit inside every range).
DEFAULT_WEIGHTS = {
    "global_english": 0.50,
    "indian_english": 0.175,
    "indian_languages": 0.125,
    "code": 0.10,
    "finance_econ_law": 0.075,
    "math_reasoning": 0.025,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--total-tokens", type=float, required=True)
    ap.add_argument("--out", required=True, help="snapshot output directory")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    total = args.total_tokens
    wsum = sum(d.get("weight", DEFAULT_WEIGHTS.get(name, 0))
               for name, d in cfg.items())
    if abs(wsum - 1.0) > 1e-6:
        sys.exit(f"weights sum to {wsum:.4f}, must be 1.0")

    domains = {}
    shortfall = False
    for name, d in cfg.items():
        w = d.get("weight", DEFAULT_WEIGHTS.get(name, 0))
        target = w * total
        pool = float(d["tokens"])
        epochs = target / pool if pool else float("inf")
        capped = min(epochs, MAX_EPOCHS)
        got = capped * pool
        if epochs > MAX_EPOCHS:
            shortfall = True
        domains[name] = {
            "paths": d["paths"],
            "pool_tokens": pool,
            "weight": w,
            "target_tokens": target,
            "epochs": round(capped, 4),
            "sampled_tokens": round(got),
            "shortfall_tokens": round(max(0.0, target - got)),
        }

    manifest = {
        "total_tokens_requested": total,
        "max_epochs": MAX_EPOCHS,
        "domains": domains,
    }
    blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    snap_id = "snap-" + hashlib.sha256(blob).hexdigest()[:16]
    manifest["snapshot_id"] = snap_id

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{snap_id}.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists — snapshots are immutable, never overwrite")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"snapshot {snap_id} -> {path}")
    for name, d in domains.items():
        note = f"  SHORTFALL {d['shortfall_tokens']:.3g}" if d["shortfall_tokens"] else ""
        print(f"  {name:<18} w={d['weight']:.3f} epochs={d['epochs']:.2f}"
              f" sampled={d['sampled_tokens']:.4g}{note}")
    if shortfall:
        print("WARNING: shortfalls present — get more data or rebalance weights",
              file=sys.stderr)


if __name__ == "__main__":
    main()
