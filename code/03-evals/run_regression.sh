#!/usr/bin/env bash
# Base-vs-SFT regression battery. Run BEFORE promoting any SFT checkpoint:
# SFT must shift behaviour without destroying base capability.
#
#   1. Tier-1 subset (arc_easy, sciq, piqa) on the base and SFT HF exports —
#      the SFT model may dip slightly but a large drop = capability damage.
#   2. BachattBench MCQ on both checkpoints (native scorer).
#   3. Per-domain val loss on both (catastrophic-forgetting check: pretraining
#      domains should not blow up after SFT).
#
# Usage: bash run_regression.sh <base_ckpt.pt> <sft_ckpt.pt> [base_hf_dir] [sft_hf_dir]
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
BASE=${1:?usage: run_regression.sh base.pt sft.pt [base_hf] [sft_hf]}
SFT=${2:?}
BASE_HF=${3:-}
SFT_HF=${4:-}
STAMP=$(date +%Y%m%d-%H%M)
mkdir -p results/regression-$STAMP

echo "=== 1. BachattBench: base vs SFT ==="
$PY run_bachattbench.py --ckpt "$BASE" \
    --json-out results/regression-$STAMP/bb_base.json | tail -8
$PY run_bachattbench.py --ckpt "$SFT" \
    --json-out results/regression-$STAMP/bb_sft.json | tail -8

echo "=== 2. Per-domain val loss: base vs SFT (forgetting check) ==="
$PY eval_domain_loss.py --ckpt "$BASE" --windows 100 \
    --json-out results/regression-$STAMP/domains_base.json | tail -7
$PY eval_domain_loss.py --ckpt "$SFT" --windows 100 \
    --json-out results/regression-$STAMP/domains_sft.json | tail -7

if [ -n "$BASE_HF" ] && [ -n "$SFT_HF" ]; then
  echo "=== 3. Tier-1 subset via lm-eval (needs HF exports) ==="
  for M in "$BASE_HF:base" "$SFT_HF:sft"; do
    DIR=${M%%:*}; TAG=${M##*:}
    $PY -m lm_eval --model hf --model_args pretrained="$DIR",dtype=float32 \
        --tasks arc_easy,sciq,piqa --device mps --batch_size 16 \
        --output_path "results/regression-$STAMP/tier1_$TAG" 2>&1 | tail -8
  done
else
  echo "=== 3. skipped (pass HF export dirs to include Tier-1) ==="
fi

echo "results → results/regression-$STAMP/"
$PY - "$STAMP" <<'EOF'
import json, sys, glob, os
d = f"results/regression-{sys.argv[1]}"
bb = {t: json.load(open(f"{d}/bb_{t}.json")) for t in ("base", "sft")
      if os.path.exists(f"{d}/bb_{t}.json")}
if len(bb) == 2:
    print(f"BachattBench: base {bb['base']['overall_acc']:.3f} -> "
          f"sft {bb['sft']['overall_acc']:.3f}")
dm = {t: json.load(open(f"{d}/domains_{t}.json")) for t in ("base", "sft")
      if os.path.exists(f"{d}/domains_{t}.json")}
if len(dm) == 2:
    print("Domain val loss (base -> sft):")
    for k in dm["base"]["domains"]:
        b = dm["base"]["domains"][k]["val_loss"]
        s = dm["sft"]["domains"][k]["val_loss"]
        flag = "  <-- FORGETTING?" if s - b > 0.5 else ""
        print(f"  {k:<20} {b:.3f} -> {s:.3f}{flag}")
EOF
