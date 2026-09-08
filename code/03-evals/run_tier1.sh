#!/usr/bin/env bash
# Tier-1 benchmarks via lm-evaluation-harness, small-model-appropriate tasks.
# MMLU/GSM8K are deliberately excluded at <100M params (noise floor).
# Runs the target model AND a same-size open baseline with identical settings.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
TASKS=hellaswag,arc_easy,piqa,sciq,winogrande,lambada_openai
NAVYA=${1:-../04-training-stack/out/navya-0/hf}
mkdir -p results

echo "=== navya-0 ==="
$PY -m lm_eval --model hf --model_args pretrained="$NAVYA",dtype=float32 \
    --tasks $TASKS --device mps --batch_size 16 \
    --output_path results/tier1_navya0 2>&1 | tail -25

echo "=== pythia-70m baseline ==="
$PY -m lm_eval --model hf --model_args pretrained=EleutherAI/pythia-70m,dtype=float32,device_map=cpu \
    --tasks $TASKS --device mps --batch_size 16 \
    --output_path results/tier1_pythia70m 2>&1 | tail -25
