#!/usr/bin/env bash
# Run the product-bar baselines for a navya stage (see BASELINES.md).
# Matched-token fairness baselines live in run_matched_pythia.sh — different
# question, different script.
#
# Usage: bash run_baselines.sh <stage> [device]
#   stage:  1a | 1b | 2 | 2.5 | 3
#   device: mps (default) / cuda / cpu
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
STAGE=${1:?usage: run_baselines.sh <1a|1b|2|2.5|3> [device]}
DEVICE=${2:-mps}
TASKS=hellaswag,arc_easy,piqa,sciq,winogrande,lambada_openai,hinglish_finance
mkdir -p results

case "$STAGE" in
  1a)  MODELS="HuggingFaceTB/SmolLM2-135M" ;;
  1b)  MODELS="LiquidAI/LFM2-350M Qwen/Qwen2.5-0.5B google/gemma-3-270m" ;;
  2)   MODELS="google/gemma-3-1b-pt Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base" ;;
  2.5) MODELS="Qwen/Qwen3-1.7B-Base HuggingFaceTB/SmolLM2-1.7B" ;;
  3)   MODELS="HuggingFaceTB/SmolLM3-3B Qwen/Qwen3-4B-Base" ;;
  *)   echo "unknown stage $STAGE"; exit 1 ;;
esac

for M in $MODELS; do
  SHORT=$(basename "$M")
  echo "=== $M (stage $STAGE product bar) ==="
  $PY -m lm_eval --model hf --model_args "pretrained=$M,dtype=${DTYPE:-float32},device_map=cpu" \
      --tasks $TASKS --include_path tasks/hinglish_finance \
      --device "$DEVICE" --batch_size 8 \
      --output_path "results/baseline_${STAGE}_${SHORT}" 2>&1 | tail -14
done
echo "done — compare against results/tier1_navya* and results/hinglish_navya*"
