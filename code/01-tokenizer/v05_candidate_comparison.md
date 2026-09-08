# v0.5 tokenizer candidates — comparison and recommendation

This is the engineering-foundation deliverable for Navya's v0.5 tokenizer:
normalization + pre-tokenization fixes, three trained candidates, a rigorous
metric suite, and a recommendation. It retires the flat "≤1.8 tokens/word for
every Indic script" gate (see `PER_SCRIPT_TARGETS.md`) in favor of per-script
targets plus a fuller distributional/compression/fidelity view.

**Nothing here touches `tokenizer-v0.3-64k.json` (production) or any running
job.** This is measurement + candidates + tooling for a future freeze
decision, same as `PER_SCRIPT_TARGETS.md` and `fertility_report_v4.json`
were.

## TL;DR recommendation

**Ship `tokenizer-v0.5-B96k.json`** (byte-BPE, 96k vocab, script-aware
corpus, grouped digits, NFC + ZWJ/ZWNJ-aware pre-tokenization) as the lead
candidate for the Navya-1.31B run, pending the GPU small-model ablation this
task could not run (see Limitations). `tokenizer-v0.5-A80k.json` is the safe
fallback if the 1.31B's embedding-table budget can't absorb 96k rows.
**Do not ship `tokenizer-v0.5-C96k-unigram.json`** as trained here — it
underperforms even the smaller v0.3-64k on nearly every metric (see below);
Unigram is not ruled out in principle, but this run does not support it.

## What changed vs v0.3/v0.4

| Area | v0.3/v0.4 (`train_tokenizer.py`) | v0.5 (`train_tokenizer_v05.py`) |
|---|---|---|
| Normalization | none | NFC (composition) + CRLF→LF + BOM strip, applied in the tokenizer's own normalizer graph; a shared `normalize_text.py` additionally collapses repeated invisible/format characters and **must run at corpus-prep, training, and inference** (see contract below) |
| ZWJ/ZWNJ | fell into the punctuation catch-all, splitting conjuncts/chillus from their base letter | attached to `\p{L}\p{M}` in the split regex, per spec pattern |
| Digits | always individual (`2026`→`2 0 2 6`) | flag: `individual` (unchanged) or `grouped` (`\p{N}{1,3}`, `2026`→`202 6`) — v0.5 candidates use `grouped` |
| Vocab | 64k | 80k (A) / 96k (B, C) |
| Corpus | `corpus_balanced/` only (~35MB, all duplicated/sampled, no fresh Dravidian text) | `corpus_balanced/` **merged** with a fresh, non-duplicated AI4Bharat Sangraha sample (see Corpus section) |
| Model type | BPE only | BPE (A, B) + Unigram w/ byte fallback (C) |

### Normalization contract (read before touching any of this)

`normalize_text.normalize_corpus_text()` is the single source of truth and
must run at **three** points or the vocabulary silently stops matching its
input — this is documented in the module's own docstring, repeated here
because it is the easiest way to quietly break a shipped tokenizer:

1. **Corpus prep** — `build_corpus_v05.py` normalizes every file before
   writing it to `corpus_v05_balanced/` / `corpus_v05_scriptaware/`.
2. **Training** — `train_tokenizer_v05.py` additionally installs
   `normalizers.NFC()` (+ CRLF/BOM `Replace` rules) directly into the saved
   `tokenizer.json`, so any consumer of the file gets NFC automatically
   inside `tokenizer.encode()`.
3. **Inference/serving** — whatever calls `.encode()` on raw text (chat
   handler, data-pipeline retokenization, eval harness) **must** run
   `normalize_corpus_text()` first. The in-tokenizer NFC normalizer is a
   safety net for direct `tokenizer.json` consumers; it does **not** do the
   repeated-invisible-character collapse (that needs a regex backreference
   the `tokenizers` Rust `Replace` normalizer doesn't support), so skipping
   step 3 upstream is a real, silent correctness gap.

NFC (not NFD/NFKC/NFKD) was chosen because real-world Indic text in the wild
is overwhelmingly already NFC (input methods and most renderers assume
precomposed forms), and NFKC/NFKD's compatibility folding can lossily merge
visually- or semantically-distinct characters.

**One concrete, measured effect of normalization**: Devanagari nukta letters
(क़/ज़/ड़/...) are Unicode canonical-decomposition targets that are on the
composition exclusion list — `unicodedata.normalize('NFC', ...)` actually
**decomposes** them (base consonant + combining nukta), and both the
precomposed and hand-decomposed spelling of the same word converge to that
one decomposed form under NFC. Verified directly:

Both spellings render as the identical word "क़र्ज़" (karz/"loan") but are
different codepoint sequences on disk — verified with explicit codepoints,
since the two forms are visually indistinguishable in text:

```
precomposed input:  U+0958 U+0930 U+094D U+095B   (क़, र, ्, ज़ — QA/ZA precomposed)
decomposed input:   U+0915 U+093C U+0930 U+094D U+091C U+093C  (क, ़, र, ्, ज, ़)

NFC(precomposed) -> U+0915 U+093C U+0930 U+094D U+091C U+093C  (decomposed!)
NFC(decomposed)  -> U+0915 U+093C U+0930 U+094D U+091C U+093C  (unchanged)

NFC(precomposed) == NFC(decomposed)  ->  True
```

This means text that arrives spelled either way collapses to one canonical
form before the tokenizer ever sees it — exactly the property that makes
`tokenizer_metrics.py`'s round-trip suite able to test both spellings as
independent, meaningful cases.

## Corpus: what's real and what's a sample

`sample_corpus.py`'s `corpus_balanced/` (~35MB) is the same mix used for
v0.3/v0.4 — no new text. `PER_SCRIPT_TARGETS.md`'s #2 recommendation for a
real v0.5 was explicit: **"Real additional Dravidian corpus, not
duplication."** This task did that, partially:

- `fetch_sangraha_sample.py` pulls real, non-synthetic text from
  **AI4Bharat Sangraha** (`config=verified`, real web/document text, not
  the `synthetic` LLM-generated config) via HuggingFace's
  `datasets-server /rows` API — a paginated JSON endpoint that serves
  slices of the underlying parquet shards **without** a bulk
  `datasets.load_dataset()` pull (which is 10GB+ per language and not
  feasible here). Hit HTTP 429 rate limits partway through the first run;
  added exponential backoff/retry, second run completed cleanly.
- Result: **~3.0-3.6MB of fresh Sangraha text per language**, 12 languages
  (11 Indic + English), ~40MB total, `corpus_sangraha/MANIFEST.json` records
  exact doc counts and byte counts per split.
- `build_corpus_v05.py` **merges** (not duplicates) this into
  `corpus_balanced/`'s per-language files, normalizing everything through
  `normalize_text.py` on the way, producing two training corpora:
  - `corpus_v05_balanced/` (72MB) — straight merge, no reweighting. Trains
    candidates A and C.
  - `corpus_v05_scriptaware/` (82MB) — same merge, plus a **modest 1.5x**
    weight on Tamil/Telugu/Kannada/Malayalam (not 3x — `fertility_report_v4.json`
    already showed 3x oversampling measurably hurts every non-Dravidian
    Indic script at a fixed vocab budget; 1.5x is a deliberately lighter
    touch, paired with the extra 16k of vocab budget rather than relying on
    either alone). Trains candidate B.

**What this is not**: this is still a CPU-smoke-scale corpus (72-82MB) next
to the project's 10-50GB production-sweep target, and it covers only the 11
scheduled Indic languages + English that `corpus_balanced/` already had —
not the full 22 scheduled languages (Urdu, Sanskrit, Nepali, Konkani,
Bodo, Dogri, Kashmiri, Maithili, Manipuri, Santali, Sindhi are absent from
both corpus and eval), and not romanized/Latin-script Indic text at any
scale (Sangraha does have `*_Latn` splits under its `synthetic` config;
they were not pulled — see Limitations).

## Candidates trained

| Candidate | Model | Vocab | Corpus | Digits |
|---|---|---|---|---|
| `tokenizer-v0.5-A80k.json` | byte-BPE | 80,000 | `corpus_v05_balanced` | grouped |
| `tokenizer-v0.5-B96k.json` | byte-BPE | 96,000 | `corpus_v05_scriptaware` (1.5x Dravidian) | grouped |
| `tokenizer-v0.5-C96k-unigram.json` | Unigram + byte fallback | 96,000 | `corpus_v05_balanced` (same as A, for a clean model-type comparison) | grouped |

All three trained on CPU in well under a minute (BPE) to ~3.5 minutes
(Unigram); see the training commands in each file's header comment
(`train_tokenizer_v05.py`).

## Metric suite (`tokenizer_metrics.py`)

Per language, over FLORES-200 dev+devtest (`eval_sets_flores200/`, same
corpus as `fertility_report_v3/v4.json`, this time run through
`normalize_corpus_text()` before encoding — see the methodology note below
for why that alone shifts some v0.3 numbers slightly from the old reports):

- **fertility mean + p50/p95/p99** — per-*sentence* tokens/whitespace-word
  ratio, distribution over every line, not just one corpus-wide average
  that a handful of long-tail sentences can't move.
- **tokens/word** (aggregate total_tokens/total_words) — kept for
  continuity with `fertility_report_v3/v4.json`'s single number.
- **tokens/grapheme-cluster** — tokens per approximate UAX#29 extended
  grapheme cluster (own lightweight implementation, not a full UAX#29
  library — see the docstring in `tokenizer_metrics.py` for exactly what's
  simplified: Hangul composition and some extended-pictographic edge cases
  aren't specially handled, but Indic conjunct clusters, ZWJ emoji
  sequences, and variation selectors are).
- **bytes/token** — UTF-8 bytes of source text per emitted token (higher =
  more compression = fewer tokens the model has to spend attention/compute
  on per unit of information).
- **byte-fragment rate** — fraction of emitted tokens that are a single raw
  byte ≥0x80, i.e. a byte that is part of a multi-byte UTF-8 character with
  **no merge covering it** — the tokenizer had nothing better than a lone
  fallback byte. This is the sharpest single "is this script underserved"
  signal: reconstructed by inverting the standard GPT-2
  `bytes_to_unicode()` byte-level alphabet mapping back to raw bytes for
  each token string.
- **Rényi efficiency (α=2.5)** + **vocabulary utilization** — per Zouhar et
  al. 2023 ("Tokenization and the Noiseless Channel"): the Rényi entropy of
  the observed token-ID frequency distribution, normalized by
  log₂(vocab size); bounded in (0,1], higher means the vocabulary budget is
  spent more evenly rather than a handful of tokens dominating. Vocabulary
  utilization is the plain fraction of the tokenizer's vocab actually used
  on this eval set (note: computed only over the ~2,000-sentence FLORES set
  per language, not the full training corpus — see Limitations, these
  utilization numbers are small (5-14%) because of that, not because most
  of the vocab is dead weight in production).
- **Round-trip fidelity** — `decode(encode(normalize(x))) == normalize(x)`
  over a 30-case adversarial suite: nukta letters written two ways (precomposed
  vs base+combining-nukta, verified distinct on disk via explicit `chr()`
  construction — see above), Sanskrit/Hindi conjuncts (क्ष, ज्ञ, त्र, राष्ट्र),
  Malayalam chillu letters written two ways (atomic U+0D7B/U+0D7D vs legacy
  consonant+virama+ZWJ), emoji ZWJ sequences (family, profession, skin-tone
  modifier, flag), WhatsApp-forward-style noise (repeated ZWSP, word joiners,
  mixed CRLF), OCR-style extra spacing + soft hyphens, Devanagari-digit
  currency amounts, Bidi control marks (LRM/RLM) around an invoice number,
  repeated-ZWNJ garbage, Bengali/Tamil/Malayalam agglutination samples, an
  Urdu (Arabic-script, RTL) sentence, and edge cases (empty string,
  pure whitespace).

## Results

### Fertility mean (tokens/word), per language

| language | v0.3-64k | v0.5-A80k | v0.5-B96k | v0.5-C96k-uni |
|---|---:|---:|---:|---:|
| english   | 1.357 | 1.341 | **1.330** | 1.982 |
| assamese  | 2.066 | 1.907 | **1.883** | 2.174 |
| bengali   | 1.945 | 1.819 | **1.794** | 2.129 |
| gujarati  | 2.048 | 1.889 | **1.861** | 2.135 |
| hindi     | 1.622 | 1.513 | **1.495** | 1.825 |
| kannada   | 2.700 | 2.459 | **2.310** | 2.772 |
| malayalam | 3.102 | 2.775 | **2.592** | 3.174 |
| marathi   | 2.113 | 1.924 | **1.896** | 2.285 |
| odia      | 2.186 | 1.948 | **1.916** | 2.291 |
| punjabi   | 1.698 | 1.566 | **1.546** | 2.013 |
| tamil     | 2.463 | 2.269 | **2.125** | 2.559 |
| telugu    | 2.557 | 2.288 | **2.144** | 2.559 |

**Every language improves monotonically v0.3 → A → B.** B (96k, modest
Dravidian weighting) beats A on every single language, not just the four it
targets — confirming the earlier "more vocab helps everyone, no losers"
finding at 80k extends cleanly to 96k+modest-oversample. Candidate C
(Unigram) is **worse than v0.3-64k on every language despite 1.5x the
vocab** — see the dedicated section below.

### Per-script gate (retiring the flat 1.8, per `PER_SCRIPT_TARGETS.md`)

Indo-Aryan target ≤1.8 tokens/word (7 languages: hindi, bengali, marathi,
gujarati, odia, punjabi, assamese), Dravidian target ≤2.3 (4 languages:
tamil, telugu, kannada, malayalam):

| candidate | Indo-Aryan pass | Dravidian pass |
|---|---:|---:|
| v0.3-64k | 2/7 | 0/4 |
| v0.5-A80k | 2/7 | 2/4 |
| v0.5-B96k | **3/7** | **2/4** |
| v0.5-C96k-unigram | 0/7 | 0/4 |

Still not a clean sweep at either 80k or 96k — hitting every Indo-Aryan
target needs the production-scale corpus and/or a further vocab increase,
per `PER_SCRIPT_TARGETS.md`'s own roadmap (recommendation #3 there:
production-scale training data; this task's corpus is 72-82MB, an order of
magnitude below the 10-50GB target).

### Compression, fragment rate, information-theoretic (Indic-language averages)

| candidate | avg byte-fragment rate | avg bytes/token | avg Rényi eff. (α=2.5) | avg vocab util. |
|---|---:|---:|---:|---:|
| v0.3-64k | 0.42% | 8.69 | 0.482 | 5.9% |
| v0.5-A80k | 0.25% | 9.50 | 0.468 | 6.4% |
| v0.5-B96k | **0.22%** | **9.83** | 0.457 | 6.0% |
| v0.5-C96k-unigram | **4.85%** | 8.12 | 0.390 | 6.4% |

A and B both **compress better** (more bytes per token = fewer tokens per
unit of text) and **fragment less** (fewer raw-byte fallbacks) than v0.3,
consistent with the fertility gains being real vocabulary-coverage
improvements, not an artifact. Rényi efficiency drifts down slightly as
vocab grows (expected — the same eval text now spreads its token-ID mass
over a larger space) and is not, by itself, a reason to prefer a smaller
vocab; it is one input among several, not a gate.

### Candidate C (Unigram, 96k) — do not ship as trained

Unigram with byte fallback, same corpus as A, same 96k budget as B,
underperforms **v0.3-64k** — a smaller, older, worse-corpus BPE tokenizer —
on every language and every metric measured: fertility, bytes/token, and
especially byte-fragment rate (4.85% average vs 0.22-0.42% for the BPE
candidates; individual languages spike to 6-7%, e.g. Punjabi 7.39%,
Malayalam 6.17%, Gujarati 6.52%). Spot-checking actual token output
(`नमस्ते, आप कैसे हैं?` → `['नम', 'स्ते', ',', ' आप', ' कै', 'से', ' हैं', '?']`)
confirms the model *is* using the byte-level alphabet correctly and
composing multi-byte tokens most of the time — the elevated fragment rate
is a real training-quality gap, not a measurement bug in
`tokenizer_metrics.py`.

This does **not** mean Unigram is categorically worse for Indic
morphology — the literature's claimed advantage (better alignment to
morpheme boundaries) is a real, separately-documented phenomenon this run
simply didn't reproduce. The likely cause is untuned `UnigramTrainer`
defaults (seed-vocab size, shrinking factor) meeting a small, multi-script
corpus where Unigram's EM-based pruning may need more data or a larger seed
vocabulary to retain enough Indic multi-byte pieces before pruning them
away. **Recommendation: re-attempt Unigram with tuned trainer
hyperparameters and/or the full production corpus before ruling it out —
this result rules out shipping *this specific* Candidate C, not Unigram as
a model family.**

### Digit grouping — concrete spot check

Trained two small (32k-vocab) BPE tokenizers, identical except
`--digit-mode`, on `corpus_v05_balanced/`, and encoded finance/date-heavy
examples:

| example | individual (tokens) | grouped (tokens) |
|---|---:|---:|
| `₹1,23,456.78` | 13 | **10** |
| `₹१,२३,४५६.७८` | 13 | 13 (tie at this small vocab) |
| `12.5%` | 5 | **4** |
| `FY 2026-27` | 10 | **6** |
| `PAN/IFSC` | 5 | 5 (no digits, tie) |
| `01/04/2026` | 10 | **6** |
| `"Account balance is ₹45,000.00 as of 06/09/2026."` | 28 | **20** |

Grouped digits win on every case with digits except the Devanagari-numeral
one, where the 32k smoke-scale vocab hasn't yet learned any Devanagari
2-3-digit merges (plausible corpus-frequency issue at this tiny scale, not
a property of grouping itself) — worth re-checking at production vocab/corpus
scale. All three shipped v0.5 candidates use `--digit-mode grouped` based on
this result; `--digit-mode individual` remains a supported flag in
`train_tokenizer_v05.py` if arithmetic-transparency needs later outweigh
the compression win.

### Round-trip fidelity

**All four tokenizers (v0.3-64k, v0.5-A80k, v0.5-B96k, v0.5-C96k-unigram)
pass 30/30 (100%)** on the adversarial suite — nukta forms, conjuncts,
chillus, emoji ZWJ, OCR/WhatsApp noise, bidi marks, Urdu, and edge cases all
round-trip exactly. Byte-level BPE/Unigram with `byte_fallback` guarantees
no `<unk>` by construction, so this result is expected structurally — but
it was not previously *tested* this rigorously (no round-trip suite existed
before this task), and it is a real, load-bearing property to keep
regression-testing as the tokenizer changes: run
`python tokenizer_metrics.py --tokenizer <file> --out <report>` on any
future candidate before shipping and check the `roundtrip_fidelity.pass_rate`
field is still 1.0.

### Methodology note: why some v0.3 numbers here differ slightly from `fertility_report_v3.json`

`tokenizer_metrics.py` normalizes FLORES eval text through
`normalize_corpus_text()` (NFC + CRLF/BOM cleanup) before encoding, which
`measure_flores_fertility.py` did not do. The **tokenizer file is
identical** (`tokenizer-v0.3-64k.json`, untouched) — only the eval-text
preprocessing differs between the two measurement scripts, and normalization
alone measurably changes the numbers (e.g. Punjabi's aggregate tokens/word
drops from 1.8075 — a narrow *fail* — in the old, un-normalized measurement
to 1.6747 in this normalized one). This is not a like-for-like methodology
regression: normalizing eval text before encoding is the *correct* thing to
do (it's what production serving must do too, per the normalization
contract above), so the new v0.3 numbers in this report are more
representative of real-world v0.3 behavior than the old ones were — but it
means don't diff v0.3's row in this table against `fertility_report_v3.json`
expecting an exact match.

## Honest limitations (what this task could not do)

1. **Corpus scale**: 72-82MB training corpora vs the project's 10-50GB
   production-sweep target. All fertility/compression numbers here are
   directionally reliable (they isolate the variables — vocab size,
   oversample weight, model type, digit mode — cleanly) but the absolute
   numbers would shift with the real training-scale corpus, same caveat
   `fertility_report_v4.json` carried.
2. **Language coverage**: 11 scheduled Indic languages + English, not the
   full 22 scheduled languages. Urdu, Sanskrit, Nepali, Konkani, Bodo,
   Dogri, Kashmiri, Maithili, Manipuri, Santali, and Sindhi have neither
   training corpus nor eval coverage here.
3. **Romanized/Latin-script Indic text**: not tested at all. AI4Bharat
   Sangraha has `*_Latn` splits (romanized Hindi, Bengali, etc.) under its
   `synthetic` config that were not pulled — this is a real gap for
   Hinglish-heavy real-world traffic (the round-trip suite's
   `hinglish_codemix` case is Latin-script English+transliterated-terms
   mixed prose, not systematic romanized-Indic coverage).
4. **AI4Bharat data volume**: pulled via HuggingFace's `datasets-server
   /rows` API specifically to avoid a 10GB+/language bulk download in this
   CPU-only, time-boxed environment — got ~3-3.6MB/language of real,
   fresh, non-synthetic text (~40MB total), which is real progress against
   `PER_SCRIPT_TARGETS.md`'s "not duplication" ask but is nowhere near
   IndicCorp v2/Sangraha's full scale. Hit HTTP 429 rate limiting even at
   this modest volume; a production pull needs the full parquet files via
   `datasets.load_dataset()`, ideally from a proper data-pipeline job with
   its own retry/rate-limit budget, not an interactive research script.
5. **No GPU ablation**: the definitive tokenizer selection, per
   `README.md`'s own workflow step 4 and `PER_SCRIPT_TARGETS.md`'s
   recommendation #4, requires training small (150-300M) models under
   matched compute on identical data and comparing downstream loss —
   fertility/compression/fidelity metrics are useful screening signals but
   are explicitly *not* the final arbiter. This is out of scope for a
   CPU-only task by construction and is the single most important
   next step before any freeze decision.
6. **Candidate C needs another pass**: as detailed above, ship-blocking as
   trained, but Unigram as a model family is not eliminated — retry with
   tuned `UnigramTrainer` hyperparameters before concluding BPE wins on
   morphology, which is the opposite of Unigram's usual claimed strength.
7. **Grapheme-cluster segmentation** in `tokenizer_metrics.py` is an
   approximate UAX#29 implementation (GB9 extend/ZWJ rule + simple regional-
   indicator pairing), not a full spec implementation — documented as such
   in the module, sufficient for this metric's purpose but not a
   general-purpose grapheme library.
8. **Vocabulary utilization** numbers (5-14%) are computed only over the
   ~2,000-sentence-per-language FLORES eval set, not the full training
   corpus or production traffic — low numbers here reflect eval-set size,
   not necessarily vocabulary bloat.

## Cost reminder (unchanged from `PER_SCRIPT_TARGETS.md`/`fertility_report_v4.json`)

None of this is a drop-in swap. Shipping any v0.5 candidate means
retokenizing the full corpus, retraining embeddings, and abandoning
`navya-1c-final`'s tokenizer contract. This report is measurement +
candidates + tooling only — nothing here touched any running job,
checkpoint, or served model.

## Files produced by this task

- `normalize_text.py` — shared normalization (NFC + CRLF/BOM + repeated-
  invisible-char collapse); the single source of truth for corpus-prep,
  training, and (required) inference-time text cleanup.
- `train_tokenizer_v05.py` — v0.5 training script (NFC normalizer,
  ZWJ/ZWNJ-aware pre-tokenization, `--digit-mode {individual,grouped}`,
  `--model-type {bpe,unigram}`).
- `fetch_sangraha_sample.py` — pulls a bounded, real (non-synthetic)
  AI4Bharat Sangraha sample per language via the HF `datasets-server /rows`
  API, no bulk download.
- `build_corpus_v05.py` — merges `corpus_balanced/` + `corpus_sangraha/`
  into `corpus_v05_balanced/` and `corpus_v05_scriptaware/` (1.5x modest
  Dravidian weighting), normalizing every file on the way.
- `tokenizer_metrics.py` — the metric suite (fertility distribution,
  tokens/grapheme, bytes/token, byte-fragment rate, Rényi efficiency +
  vocab utilization, round-trip fidelity adversarial suite).
- `tokenizer-v0.5-A80k.json`, `tokenizer-v0.5-B96k.json`,
  `tokenizer-v0.5-C96k-unigram.json` — the three trained candidates.
- `metrics_tokenizer-v0.3-64k.json`, `metrics_tokenizer-v0.5-A80k.json`,
  `metrics_tokenizer-v0.5-B96k.json`, `metrics_tokenizer-v0.5-C96k-unigram.json`
  — full per-language metric-suite output backing every table above.
- `corpus_sangraha/`, `corpus_v05_balanced/`, `corpus_v05_scriptaware/` —
  gitignored (regenerable via the two scripts above), same convention as
  `corpus_balanced/`/`corpus_v04/`.
