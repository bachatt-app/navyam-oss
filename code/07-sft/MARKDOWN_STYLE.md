# Canonical answers: markdown authoring style (from tranche 3)

The consumer app now renders Navya's replies as markdown (escape-first
renderer in `apps/consumer/ui/chat.html`: headers, bold/italic, inline
code, fenced code, tables, lists, hr, http(s) links). Formatting habits
come from SFT demonstrations — so **canonical answers written from
tranche 3 onward should use markdown where structure helps**. The model
learns to emit what we author; the renderer is already waiting for it.

## When to use what

| Structure | Use for | Example intents |
|---|---|---|
| **Table** | 2-way comparisons, option grids | sip_vs_lumpsum, regime comparison, TP-vs-OD |
| **Numbered list** | processes and sequences | folio consolidation steps, CAS download |
| **Bulleted list** | checklists, document lists | claim documents, KYC papers |
| **Bold** | the one number/term that matters | **₹1.5L limit**, **50% NCB**, deadlines |
| `inline code` | exact field names, portal labels | `Statement > CAS`, password format |
| Fenced code | only for actual formulas/calc steps | EMI arithmetic (rare) |

## Rules

1. **Voice unchanged.** Hinglish stays Hinglish; markdown is layout, not
   register. A table's cells are written exactly like prose was.
2. **Short answers stay prose.** If the answer is 2–3 sentences, add at
   most a bold key term. No headers on short answers; use `##`/`###` only
   when an answer genuinely has 2+ sections.
3. **Tables max ~4 columns**, header row required (the renderer needs the
   `|---|` separator line). Keep cells short — one clause, not sentences.
4. **Still no URLs, no fund names** — the existing rules and
   `reward_no_fund_naming` apply unchanged; markdown links are for future
   citation work, not authoring.
5. **Plain-text degradation must hold.** An answer must remain readable if
   shown raw (WhatsApp copy-paste, API consumers): prefer `-` bullets and
   simple tables over deep nesting.
6. **JSONL escaping**: responses live in one JSON string — newlines are
   `\n` inside the `response` field. Author in a scratch .md, then escape.

## Worked example (sip_vs_lumpsum, before → after)

Before (prose): "Dono se paisa usi fund mein jaata hai — farak sirf
timing ka hai. Lumpsum pehle din se poora invested..."

After:

```
Dono se paisa usi fund mein jaata hai — farak **timing** ka hai:

| | SIP | Lumpsum |
|---|---|---|
| Invest hota hai | Har month thoda | Pehle din poora |
| Market timing risk | Average ho jaata hai | Poora ek din pe |
| Best jab | Salary income | Bonus/windfall |

Beech ka raasta: windfall ka 20-30% abhi, baaki 6-12 month ki SIP se.
Final faisla apni risk capacity dekh kar ya SEBI-registered adviser se
baat karke lein.
```

## Pipeline impact (checked)

- `build_chat_data.py` / chat template: text passes through verbatim — no
  change needed.
- `rewards.py`: `reward_canonical_overlap` tokenizes `\w+`, so pipes and
  asterisks are invisible to it; `reward_length` counts words — table
  answers run slightly longer, within existing bounds. No reward changes
  needed, but spot-check holdout rewards after the first markdown tranche.
- Serving: live renderer deployed with the markdown UI change; older
  clients see readable plain text (rule 5).
