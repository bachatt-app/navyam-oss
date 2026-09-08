# 05 — Research models (300M–1B)

Many small runs, each answering one question. The entire Phase-0 research
programme costs less than 1% of one Beta run — every mistake caught here saves
its cost ×1000 at 35B.

## Discipline

> **No run without a question. No question without an eval.**

1. Before launch, an experiment gets a card in `experiments/` (copy
   `EXP-000-template.md`) and a row in `registry.md`.
2. The card states the **question**, the **eval that answers it**, and the
   **decision rule** — written *before* results exist.
3. Results land in the registry whether they win or lose. Dead ends are data.
4. Weekly ablation review walks the registry (execution plan, Operating cadence).

## The Phase-0 experiment queue (initial)

| ID | Question | Decides |
| -- | -------- | ------- |
| E001 | Does the smoke stack train, checkpoint, and resume deterministically? | stack v1 accepted |
| E002 | Vocab 128K vs 160K vs 256K: downstream loss at 300M on identical data? | tokenizer freeze |
| E003 | Digit-split vs no digit-split tokenizer: GSM8K-style probes at 300M? | tokenizer freeze |
| E004 | Charter mix vs −10pp global-English mix: eval deltas at 1B? | corpus mix v1 |
| E005 | Code at 10% vs 15%: reasoning + code evals at 1B? | corpus mix v1 |
| E006 | 300M/500M/1B scaling fit — does it predict a held-out 1B run? | Alpha sizing |
| E007 | FP8 matmuls at 1B: loss parity + speedup vs BF16? | Alpha precision (default: no) |
| E008 | Kill-a-node drill on a 1B run: lost work <10 min? | infra exit criterion |

## Scaling study (E006) reminder

Fit \(L(N,D) \approx E + A/N^\alpha + B/D^\beta\) on our data. We are not
rediscovering the literature's constants — we are confirming *our mix* behaves,
and locating where it saturates. Plan for over-trained, inference-optimal
models (150–250 tokens/param), not compute-optimal ~20.
