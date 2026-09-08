# Navya baseline ladder — who each stage is measured against

Two kinds of comparison, never conflated:

1. **Matched-token baselines** (fairness test of OUR stack): Pythia
   intermediate checkpoints at the same token budget — `run_matched_pythia.sh`.
   This is the gate for "is our data/loader/training sound?"
2. **Product bars** (this file): the best modern open models of similar SIZE,
   trained on 100-1000x our tokens by frontier labs. Navya stages should
   close the gap on these as our token budgets and data quality grow —
   reaching parity on English tasks is a long-horizon goal; BEATING them on
   Indic, Hinglish and Indian-finance tasks is the near-term differentiation
   (none of them target India).

| Navya stage | Params | Product bars (HF ids) | Their train tokens |
|---|---|---|---|
| navya-1a | 151M | `HuggingFaceTB/SmolLM2-135M` | ~2T |
| navya-1b | 338M | `LiquidAI/LFM2-350M`, `Qwen/Qwen2.5-0.5B`, `google/gemma-3-270m` | ~2-18T |
| navya-2 | 1.3B | `google/gemma-3-1b-pt`, `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base` | ~14-36T |
| navya-2.5 | 1.5-2B | `Qwen/Qwen3-1.7B-Base`, `HuggingFaceTB/SmolLM2-1.7B` | ~11-36T |
| navya-3 | ~2.7B | `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3-4B-Base` (goal of record 2026-08-21; tiered victory conditions in TARGETS_3.md) | ~11T / ~36T |

Practical notes:
- Use BASE (`-pt`/`-Base`) variants against navya base checkpoints; compare
  instruct variants only against navya SFT checkpoints.
- `google/gemma-3-*` are licence-gated on HF — accept the licence once and
  export HF_TOKEN before running.
- Suite: the Tier-1 tasks + hinglish_finance + BachattBench + per-domain val
  loss (`run_baselines.sh <stage>` runs the right set).
- Expectation management per stage: at navya-1a (4B tokens) SmolLM2-135M
  will win the English tasks decisively — the tracking metric is the GAP
  trend across stages, plus outright wins on hinglish_finance/BachattBench
  where their India-blindness shows.

## Measured (2026-08-20) — the 1b targets

| task | navya-1a (4B) | LFM2-350M (~10T) | Qwen2.5-0.5B (18T) | Gemma-3-270m (~6T) |
|---|---:|---:|---:|---:|
| arc_easy | 44.5 | 69.4 | 64.6 | 59.1 |
| sciq | 66.3 | 91.9 | 93.1 | 88.7 |
| piqa | 59.4 | 69.0 | 70.2 | 67.7 |
| lambada | 20.4 | 40.2 | 52.5 | 43.3 |
| hellaswag | 27.7 | 38.4 | 40.6 | 34.3 |
| winogrande | 50.7 | 55.7 | 56.4 | 53.8 |
| hinglish_finance (acc_norm) | 26.7 | 26.7 (=chance) | 31.7 | **36.7** |

Key finding: the India lane is DEFENDED — by Gemma-3-270m (36.7% hinglish),
not by the bigger models. LFM2 is India-blind. 1b win condition: beat 36.7%
decisively on hinglish_finance (+ BachattBench/date-aware), halve the
English Tier-1 gaps. English parity at these baselines' 6-18T budgets is a
navya-2/Alpha objective, not a 1b one.
