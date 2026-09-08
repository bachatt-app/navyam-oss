# 03 — Eval harness + BachattBench v0

> "No run without a question. No question without an eval."

## Three tiers (execution plan, Workstream 5)

| Tier | Question | Contents |
| ---- | -------- | -------- |
| 1 | Are we globally competitive? | MMLU/MMLU-Pro, GSM8K+MATH, HumanEval/MBPP, ARC, HellaSwag, IFEval, long-context |
| 2 | Are we better for India? | IndicGLUE-class, per-language translation, cross-script consistency, fertility-adjusted efficiency |
| 3 | Are we the best at Indian finance? | **BachattBench** — built here, in-house |

Tier 1/2 run through an adopted harness (lm-evaluation-harness-class) — do not
rebuild those. This folder builds what cannot be adopted: **BachattBench** and
the glue that runs everything on any checkpoint in under 2 hours.

## BachattBench item format (`bachattbench/seed_v0.jsonl`)

One JSON object per line:

```json
{"id": "BB-C-001", "section": "concepts", "type": "mcq",
 "prompt": "...", "options": {"A": "...", "B": "..."}, "answer": "A"}

{"id": "BB-N-001", "section": "calculation", "type": "numeric",
 "prompt": "...", "answer": 8997.26, "rel_tol": 0.01}

{"id": "BB-J-001", "section": "compliance", "type": "judge",
 "prompt": "...", "rubric": ["must not ...", "must ..."]}
```

Sections: `concepts`, `calculation`, `scenarios`, `compliance`, `codemix`.
Types: `mcq` and `numeric` grade automatically; `judge` items are graded by
judge models with periodic human expert audit (harness v1 records them as
ungraded).

## Ground rules

1. **Every item is expert-reviewed before it counts.** Seed items are drafts
   until a finance professional signs off (track in `bachattbench/review_log.md`).
2. **Anything rule- or rate-dependent (tax slabs, LTCG rates, current repo rate)
   is banned from the answer key.** Those belong to the tool layer; testing them
   as static facts would train the model to fabricate live data. Time-invariant
   math and concepts only.
3. **Eval sets feed decontamination** (02) — every new item automatically
   becomes banned n-grams for the corpus.
4. Target: 500+ items for the Phase-0 exit gate; seed_v0 ships 18 to prove the
   format and harness.

## Run

```bash
python harness/run_eval.py --items bachattbench/seed_v0.jsonl --backend oracle  # grading self-test (expect 100%)
python harness/run_eval.py --items bachattbench/seed_v0.jsonl --backend echo    # mechanics check
python harness/run_eval.py --items bachattbench/seed_v0.jsonl \
       --backend openai --model <name>      # real model via OPENAI_BASE_URL/OPENAI_API_KEY
```
