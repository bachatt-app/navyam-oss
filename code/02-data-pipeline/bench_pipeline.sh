#!/usr/bin/env bash
# Pipeline cleaning benchmark: sweep worker counts x chunk sizes on a
# representative pool BEFORE sizing the navya-1b ingest VM.
#
# Usage (on the ingest VM, with a real pool):
#   bash bench_pipeline.sh pools/indian_english/raw_bulk_v2.jsonl 1000000000
# Args: <input jsonl> [max-bytes of input to use, default 1e9]
#
# Reports per config: wall secs, MB/s, docs/s, peak RSS (parent+workers).
# Reading the results: if MB/s stops scaling with workers while parent CPU
# is <100%, the bottleneck is worker compute (bigger VM helps). If the
# parent pins a core, fix serial state/IPC first (PERF_TODO items 3-4).
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
INPUT=${1:?usage: bench_pipeline.sh <input.jsonl> [max_bytes]}
MAXB=${2:-1e9}
SAMPLE=/tmp/bench_pipe_sample.jsonl
EVALS="../03-evals/bachattbench/*.jsonl ../03-evals/decontam/*.jsonl ../03-evals/tasks/*/items.jsonl"

# fixed-size sample so every config sees identical input
$PY - "$INPUT" "$MAXB" <<'EOF'
import sys
limit = float(sys.argv[2]); n = 0
with open("/tmp/bench_pipe_sample.jsonl", "w") as out:
    for line in open(sys.argv[1], encoding="utf-8"):
        out.write(line); n += len(line)
        if n >= limit: break
print(f"sample: {n/1e6:.0f}MB")
EOF
MB=$(du -m $SAMPLE | cut -f1)
DOCS=$(wc -l < $SAMPLE)

echo "config,workers,chunk_bytes,secs,mb_s,docs_s,peak_rss_mb" | tee bench_pipeline.csv
for W in 4 8 12 16; do
  for CB in 4e6 8e6 16e6; do
    rm -f /tmp/bench_pipe_state.bin
    /usr/bin/time -v $PY pipeline/parallel_clean.py \
        --inputs "$SAMPLE" --evals "$EVALS" \
        --dedup-state /tmp/bench_pipe_state.bin \
        --out /tmp/bench_pipe_out.jsonl \
        --workers $W --chunk-bytes $CB 2> /tmp/bench_time.txt || true
    SECS=$(grep "Elapsed (wall" /tmp/bench_time.txt | awk -F: '{m=$(NF-1); s=$NF; print m*60+s}')
    RSS=$(grep "Maximum resident" /tmp/bench_time.txt | awk '{print int($NF/1024)}')
    echo "run,$W,$CB,$SECS,$(echo "$MB $SECS" | awk '{printf "%.1f", $1/$2}'),$(echo "$DOCS $SECS" | awk '{printf "%.0f", $1/$2}'),$RSS" | tee -a bench_pipeline.csv
  done
done
echo "NOTE: /usr/bin/time -v RSS covers the parent; sample worker RSS via 'ps' during a run for the full picture."
