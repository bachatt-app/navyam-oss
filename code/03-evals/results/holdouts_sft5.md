# S061-S066 holdout scoreboard — navya-1a-sft5 (2026-08-20)

Generative eval on never-trained holdout questions (paraphrase siblings of
trained intents), scored by the shared 08-trl reward functions. Each question
asked twice: single-turn and with a greeting exchange prepended (the
historical multi-turn blending failure mode).

| Bank | n | Single | Greeting | Lang match | Overlap (single) |
|---|---:|---:|---:|---:|---:|
| income_tax (S065) | 18 | **+0.618** | +0.589 | 83% | +0.331 |
| insurance (S064) | 35 | +0.507 | +0.439 | 69% | +0.284 |
| loans (S062) | 55 | +0.485 | **+0.504** | 71% | +0.142 |
| credit_card (S063) | 16 | +0.481 | +0.423 | 75% | +0.038 |
| gst (S066) | 24 | +0.443 | +0.398 | 62% | +0.160 |
| mf_sip (S061) | 54 | +0.426 | +0.401 | 60% | +0.174 |

Readings:
- Multi-turn degradation is gone as a systematic failure — greeting-mode is
  within noise of single-turn everywhere, and BEATS it on loans.
- MF/SIP recovered from the sft3 capacity dip (+0.421 → +0.426) despite the
  corpus doubling again: thin-intent augmentation holds.
- Weakest cell: credit_card greeting-mode overlap (−0.158, n=16) — CC has the
  fewest intents (16) and holdouts; tranche-2 authoring should hit CC first.
- Language match tracks how English-technical each bank's questions are
  (income_tax 83% vs mf_sip 60%) — partly detector noise on code-switched
  finance jargon, partly real. Revisit the lang heuristic before using it as
  a gate.

Baseline history: sft2 (MF only) +0.44; sft3 (+loans) mf dipped to +0.42;
sft5 (six banks, 645K sup tokens) holds ~+0.43-0.62 across all six.
Next levers: tranche-2/3 authoring (~1,700 unmapped questions), first real
DPO round on unmapped-question failures, navya-1b capacity (338M).
