# 01 — Tokenizer v1

Byte-level BPE covering: English, Hindi, Hinglish, Bengali, Marathi, Telugu,
Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese — plus code and
numbers. Urdu and others later.

## Why this is the first workstream

The tokenizer cannot change after Alpha pretraining starts. Every corpus snapshot
is tokenized with it; every model inherits it. It freezes **last** in Phase 0
(after the vocab sweep), but work starts **first** because the sweep needs the
data pipeline's output.

## Decisions (defaults from the execution plan's decision log)

| Decision       | Default                          | Resolved by          |
| -------------- | -------------------------------- | -------------------- |
| Scheme         | byte-level BPE                   | —                    |
| Vocab size     | 160K (sweep 128K / 160K / 256K)  | 1B-scale ablation    |
| Digits         | split into individual digits     | keep unless ablation says otherwise |
| Whitespace     | preserved (code-friendly)        | —                    |
| Special tokens | bos/eos/pad + 64 reserved slots  | before freeze        |

## The metric: fertility

```
fertility = tokens / words        (lower is better)
```

**Targets (exit criteria):**
- every language within **15% of English-level fertility** on held-out text
- Devanagari/Indic scripts ≤ **1.8 tokens per word**
- no regression on English or code vs a strong open tokenizer baseline

## Files

| File                 | Purpose                                                    |
| -------------------- | ---------------------------------------------------------- |
| `train_tokenizer.py` | trains a byte-level BPE from text files                    |
| `eval_fertility.py`  | per-language fertility table + target check                |
| `eval_sets/`         | held-out per-language samples. **Placeholders for now** — replace with real held-out data (never training data) before any real sweep |

## Workflow

1. Data pipeline (02) produces a *tokenizer training sample*: a mix-weighted,
   deduped ~10–50GB sample across all 13 languages + code.
2. Train candidates at 128K / 160K / 256K.
3. `eval_fertility.py` on real held-out sets; compare against open baselines.
4. Winning candidates go into 300M-model ablations (05) — downstream loss on
   identical data decides, not fertility alone.
5. Freeze, version, and record the freeze in the execution plan's decision log.
