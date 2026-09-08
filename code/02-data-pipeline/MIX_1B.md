# navya-1b data mix — adopted starting point (2026-08-19)

Decision: adopt the reviewed reweighting for 1b. Rationale and caveats:

| Domain | 1a (actual) | 1b main phase | Why |
|---|---:|---:|---|
| Global English | 50% | 48% | slight trim, still the backbone |
| Indian English (incl. Hinglish) | 17.5% | 17% | unchanged in spirit |
| Indic languages | 12.5% | 10% | CONCENTRATED, not spread (below) |
| Code | 10% | 5% | 1b is not a code product; REVISIT at navya-2/Alpha where coding is product phase 2 |
| Finance/econ/law | 7.5% | 8% | plus the growing S047 Indian slice |
| Math/STEM/reasoning | 2.5% | 12% | the big correction: reasoning transfer needs volume |

**Math sourcing note:** 12% of a 10B-token run = 1.2B math tokens; the
current pool holds 188M. The 1b ingest must pull open-web-math far deeper
(source has ~15B tokens) — raise the cap in bulk_ingest v3.

## Indic: per-language token targets (replaces equal byte caps)

1b priority tier (Hindi + 4 largest for Bachatt's user base — final pick of
the 3-4 companions is a product call):

| Language | Share of total mix | Bin |
|---|---:|---|
| Hindi (hin) | 4.0% | `indic_hin` |
| Bengali (ben) | 1.5% | `indic_ben` |
| Tamil (tam) | 1.5% | `indic_tam` |
| Marathi (mar) | 1.5% | `indic_mar` |
| Telugu (tel) | 1.5% | `indic_tel` |

Remaining six (guj/kan/mal/pan/ori/asm): stay in pools, 0% in the 1b mix;
each enters when it has enough unique high-quality tokens to matter
(gate: pool tokens >= 2x its intended allocation) — per-language fertility
and val loss are measured from each language's own bin val-tail.

Mechanics now in place: `tokenize_pool.py --lang <code>` cuts per-language
bins from the pooled clean.jsonl (docs carry `lang` from langid);
`build_stream.py --weights <json>` takes the mix from a file.

## Cooldown phase (last 15% of steps, LR already decaying)

Qwen3-style curriculum: `train.py` now supports `cooldown_data_dir` +
`cooldown_start_frac` (default 0.85) — a second stream takes over while the
cosine schedule is in its tail. Cooldown mix (build as a separate stream
with `--weights configs/mix-navya1b-cooldown.json`):

global_english 20%, indian_english/Hinglish 20%, math 20%, finance 20%,
hin+priority Indic 15%, code 5%.

Caveat recorded honestly: "carefully reviewed Indic material" does not exist
yet as a curated subset — 1b cooldown uses Sangraha-verified as the proxy;
building a reviewed Indic/finance cooldown set is an open corpus task.

### S061 + S062 in the cooldown finance slice

`make_s061_pretrain.py` (now generic: --pairs/--holdout/--source/--out)
renders the first-party question banks as cooldown Q/A corpora — exactly the
"carefully reviewed finance" material the caveat above asks for. Include
both inside the finance 20% of the cooldown stream:

- `s061_mf_sip_qa.jsonl` — 605 MF/SIP Q/A docs (54-q holdout excluded)
- `s062_loans_qa.jsonl` — 606 loans/credit Q/A docs (55-q holdout excluded)
- `s063_cc_qa.jsonl` — 182 credit-card Q/A docs (16-q holdout excluded)
- `s064_insurance_qa.jsonl` — 385 insurance Q/A docs (35-q holdout excluded)
- `s065_income_tax_qa.jsonl` — 208 income-tax Q/A docs (18-q holdout excluded)
- `s066_gst_qa.jsonl` — 268 GST Q/A docs (24-q holdout excluded)

The eval holdouts (`07-sft/mf_sip_eval_holdout.jsonl`,
`07-sft/loans_eval_holdout.jsonl`) are excluded at generation time and must
stay in the decontamination ban list. Unanswered tranches:
`mf_sip_tranche3_worksheet.md` (341 q) and `loans_tranche2_worksheet.md`
(339 q).
