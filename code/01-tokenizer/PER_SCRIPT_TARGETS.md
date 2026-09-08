# Per-script fertility targets (proposal)

The tokenizer's exit criteria (see `README.md`) set one flat number for every
Indic script: **≤1.8 tokens/word**. The definitive re-measure on FLORES-200
(`fertility_report_v3.json`, 2009 real parallel sentences/language, 29k-50k
words/language) confirms what the earlier small-sample check indicated
(`fertility_report_v2.json`): 10 of 11 Indic scripts fail that flat gate, on
both `tokenizer-v0.3-64k.json` and the Dravidian-oversampled
`tokenizer-v0.4-64k.json` (`fertility_report_v4.json`). Only Hindi passes on
both. This document proposes replacing the flat gate with per-script targets,
and states what a real (not CPU-smoke-scale) v0.4 would need to hit them.

## Why one number doesn't work

`tokens/word` as measured here uses **whitespace splitting** to define
"word" for every language. That is a fair proxy for space-delimited
Indo-Aryan text (Hindi, Marathi, Bengali, Assamese, Gujarati, Odia, Punjabi —
all Devanagari/Bengali-Assamese/Gujarati/Odia/Gurmukhi, all descended from a
broadly similar Middle Indo-Aryan morphology: mostly analytic, postpositions
as separate words, moderate inflection). It is not a fair proxy for the
Dravidian scripts (Tamil, Telugu, Kannada, Malayalam):

- **Agglutination.** Dravidian languages build a single orthographic word by
  chaining case markers, postpositions, tense/aspect/mood suffixes, and
  clitics onto a root — where Hindi would use 2-4 separate whitespace-delimited
  words, Tamil or Malayalam commonly express the same content as one
  unbroken string. A BPE tokenizer counts subword pieces of that one long
  string; the "word" denominator undercounts the actual content by not
  splitting where Hindi's spaces would.
- **Longer orthographic syllables (akshara).** All Brahmic scripts encode
  consonant+vowel (and consonant clusters) as a single visual/Unicode
  akshara, but Dravidian scripts average more independent aksharas per
  morpheme than Devanagari does for equivalent meaning, compounding the
  effect above.
- **Compounding (sandhi).** Especially in Malayalam and Tamil, compound
  nouns and verb chains are conventionally written without spaces, further
  inflating whitespace-word length relative to Indo-Aryan orthographic
  convention.

The measured data is consistent with this: the FLORES corpus is the *same*
2009 sentences translated into every language (apples-to-apples content), yet
Dravidian whitespace-word counts are 30-45% lower than Indo-Aryan for the
same content (e.g. Malayalam 29,307 words vs Hindi 50,250 words over the
identical sentence set) — Dravidian is doing more work per whitespace-word,
so it necessarily needs more tokens per whitespace-word to encode that work,
independent of tokenizer quality.

## Proposed targets

| Family | Scripts | Target | Rationale |
|---|---|---|---|
| Indo-Aryan | Devanagari (Hindi, Marathi), Bengali-Assamese, Gujarati, Odia, Gurmukhi (Punjabi) | **≤ 1.8 tokens/word** (unchanged) | Whitespace word ≈ morphological word; the original target's assumption holds. |
| Dravidian | Tamil, Telugu, Kannada, Malayalam | **≤ 2.3 tokens/word** | Agglutination inflates true information-per-whitespace-word by roughly the same 25-30% the FLORES word-count gap shows; scaling 1.8 by that factor lands at ~2.3-2.4. Set at 2.3 as an achievable-but-real bar, not a rubber stamp for whatever the tokenizer currently does (current v0.3/v0.4 range 2.2-3.1, still above this on 3-4 of the 4 scripts). |
| English / code | Latin | ≤ baseline (informational, current gate) | Unchanged; reference point, not renegotiated. |

This is a proposal, not a re-freeze — it should go through the same
decision-log process as the original targets before the tokenizer freezes.

## What a real v0.4 needs to actually hit these

The `tokenizer-v0.4-64k.json` built for this task changed only the training
**mix weight** (3x duplication of existing Dravidian text, same 64k vocab,
same recipe) and confirmed the mechanism works — Tamil/Telugu/Kannada/Malayalam
all improved 0.2-0.29 tokens/word — but at a fixed 64k budget that gain came
directly out of every other language's merges (Punjabi flipped from a
narrow pass to a fail; see `fertility_report_v4.json`). Duplication also adds
frequency weight, not lexical diversity — the same word-forms just count
more, so merge coverage doesn't extend to Dravidian vocabulary the current
sample never saw. A production v0.4 needs, in order of expected impact:

1. **More vocab, not just more weight.** The supplementary experiment in
   `fertility_report_v4.json` found that raising vocab 64k→80k on the
   *unmodified* corpus mix improved every single language with zero losers
   (English 1.391→1.372, all 11 Indic scripts down, Punjabi 1.808→1.753
   flipped pass). Budget, not reallocation, was the cheaper win in this test.
   Combine a vocab increase with modest (not 3x) Dravidian oversampling next.
2. **Real additional Dravidian corpus**, not duplication: AI4Bharat
   IndicCorp v2 or Sangraha (both have large Tamil/Telugu/Kannada/Malayalam
   web-text pools) to give the BPE trainer new word-forms and compounds to
   merge on, not just more copies of the ~2-4MB/language already in
   `corpus_balanced/`.
3. **Production-scale training data.** This task's tokenizers were trained
   on ~35-50MB total (CPU, <1 minute); the project plan calls for a
   10-50GB mix-weighted sample for the real sweep. Merge quality — especially
   for a morphologically rich family like Dravidian — is sensitive to corpus
   scale in a way a CPU smoke test cannot validate. Numbers here are directional.
4. **Re-run the ablation at the 300M-model stage** (per `README.md`
   workflow step 4) — fertility alone should not decide the freeze;
   downstream loss on identical data is the actual arbiter.

## Cost reminder

Any tokenizer change — reweighted mix, larger vocab, or both — invalidates
every prior tokenized corpus snapshot and every checkpoint trained against
the old vocab (including `navya-1c-final`'s ~103B-token checkpoint). This
document and `tokenizer-v0.4-64k.json` are a measurement/proposal, not a
drop-in; nothing here was applied to any running job or served model.
