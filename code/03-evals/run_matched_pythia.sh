#!/usr/bin/env bash
# Matched-token baselines: Pythia intermediate checkpoints on Tier-1.
#
# Pythia trains at ~2.1M tokens/step (batch 1024 x seq 2048), so:
#   step512  ~ 1.07B tokens  — brackets navya-0 (1.31B) from below
#   step1000 ~ 2.10B tokens  — brackets navya-0 from above
#   step2000 ~ 4.19B tokens  — matches navya-1a (4.0B)
#   main     =  300B tokens  — the fully trained reference
# Comparing navya to a SAME-TOKEN-BUDGET Pythia is the fair test of our data
# and stack; comparing to main measures the remaining scale gap.
#
# Usage: bash run_matched_pythia.sh [model] [device]
#   model:  EleutherAI/pythia-70m (default) or EleutherAI/pythia-160m
#   device: mps (default) / cuda / cpu — pythia segfaults on some mps builds;
#           rerun with cpu if that happens.
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
TASKS=hellaswag,arc_easy,piqa,sciq,winogrande,lambada_openai
MODEL=${1:-EleutherAI/pythia-70m}
DEVICE=${2:-mps}
SHORT=$(basename "$MODEL")
mkdir -p results

for REV in step512 step1000 step2000 main; do
  echo "=== $SHORT @ $REV ==="
  $PY -m lm_eval --model hf \
      --model_args "pretrained=$MODEL,revision=$REV,dtype=float32,device_map=cpu" \
      --tasks $TASKS --device "$DEVICE" --batch_size 16 \
      --output_path "results/matched_${SHORT}_${REV}" 2>&1 | tail -12
done
echo "done — results/matched_${SHORT}_*"
