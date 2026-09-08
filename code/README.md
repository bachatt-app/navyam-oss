# Phase 0 — Foundations (months 0–6)

Everything in this folder exists to de-risk the Alpha (7–8B) run. Per the
[execution plan](../execution-plan.tex), Phase 0 ends when the exit gate below is
green — not when the calendar says so.

```
PHASE 0 -- FOUNDATIONS                          months 0--6
  |-- 01-tokenizer         Tokenizer v1 (13 languages, code-mix aware)
  |-- 02-data-pipeline     Data pipeline v1 (crawl, filter, dedup, mix)
  |-- 03-evals             Eval harness + BachattBench v0
  |-- 04-training-stack    Training stack v1 (proven at 1B scale)
  `-- 05-research-models   Research models 300M--1B (many runs)
```

## Exit gate (Phase 0 → Alpha)

- [ ] Tokenizer frozen: fertility within 15% of English for all 13 languages; no
      regression on English/code vs a strong open tokenizer
- [ ] Data pipeline has processed 1T+ tokens end-to-end; quality classifier beats
      heuristics on downstream 1B evals
- [ ] Scaling fit (300M/500M/1B) predicts a held-out 1B run within noise
- [ ] 1B run survives a deliberately killed node with <10 min lost; MFU ≥ 35%
- [ ] Eval harness runs any checkpoint in <2h; BachattBench v0 live with 500+
      expert-reviewed items
- [ ] Scenario factory has produced 10K expert-audited samples
- [ ] Data-source legal register exists and covers every corpus source

## Quickstart

```bash
cd phase-0
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Train a smoke tokenizer and measure fertility
python 01-tokenizer/train_tokenizer.py --input "01-tokenizer/eval_sets/*.txt" \
       --vocab-size 4096 --out 01-tokenizer/tokenizer-smoke.json
python 01-tokenizer/eval_fertility.py --tokenizer 01-tokenizer/tokenizer-smoke.json

# 2. Prepare data and smoke-train a tiny model on CPU
python 04-training-stack/prepare_data.py \
       --input "01-tokenizer/eval_sets/*.txt" \
       --tokenizer 01-tokenizer/tokenizer-smoke.json \
       --out 04-training-stack/data/smoke
python 04-training-stack/train.py --config 04-training-stack/configs/smoke.json

# 3. Run the eval harness (echo backend = mechanics check only)
python 03-evals/harness/run_eval.py --items 03-evals/bachattbench/seed_v0.jsonl \
       --backend echo
```

The smoke runs prove the *mechanics*. Real work replaces the sample corpus with
crawled data (02), the smoke tokenizer with the 128K–256K sweep (01), and the tiny
config with `300m.json` / `1b.json` on a rented GPU node (04, 05).

## Operating rules (from the execution plan)

1. **No run without a question.** Every experiment is registered in
   `05-research-models/registry.md` before launch, with its eval and decision rule.
2. **Corpus snapshots are immutable and versioned.** A run must be reproducible
   from its snapshot ID alone.
3. **The tokenizer freezes last and freezes carefully** — it is the one component
   that cannot change after Alpha pretraining starts.
4. **Decisions default on deadline** — see the decision log in the execution plan.
