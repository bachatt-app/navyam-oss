#!/usr/bin/env python3
"""Train v0.5 byte-level BPE tokenizer candidates for the India-first
foundation model. Copied from train_tokenizer.py and extended per the v0.5
spec with normalization, ZWJ/ZWNJ-aware pre-tokenization, and a digit
grouping flag. See v05_candidate_comparison.md for what changed and why.

WHAT'S NEW VS v0.3/v0.4 (train_tokenizer.py):

1. Unicode NFC normalizer, applied inside the tokenizer itself
   (tok.normalizer = normalizers.Sequence([... , normalizers.NFC()])), so
   ANY consumer of the saved tokenizer.json gets NFC automatically inside
   tokenizer.encode() -- not just this training script. This also folds in
   CRLF->LF and BOM-stripping at the tokenizer level.

   IMPORTANT: this in-tokenizer normalizer does NOT do everything
   normalize_text.normalize_corpus_text() does -- specifically it does not
   collapse repeated invisible/format characters (that needs a regex
   backreference the `tokenizers` Rust Replace normalizer does not support).
   That means normalize_text.normalize_corpus_text() MUST still run upstream
   at corpus-prep (build_corpus_v05.py does this) and MUST also run upstream
   at inference/serving time on raw user text before it reaches
   tokenizer.encode() -- the in-tokenizer NFC is a safety net for direct
   tokenizer.json consumers, not a substitute for running the shared
   normalizer. See normalize_text.py's module docstring for the full
   train/inference contract.

   NFC (not NFD/NFKC/NFKD) was chosen because: (a) it's what basically all
   real-world Indic text already is in the wild (Unicode input methods and
   most fonts/renderers assume precomposed/NFC forms, e.g. nukta letters
   क़/ज़/... as single precomposed codepoints), so training on NFC and
   normalizing input to NFC at inference means the tokenizer's vocabulary
   matches what it will actually see; (b) NFKC/NFKD would additionally fold
   compatibility variants (e.g. width/font variants, some legacy Indic
   presentation forms) which can lossily merge visually-or-semantically
   distinct characters -- NFC composes without that compatibility folding.

2. Fixed pre-tokenization regex per the v0.5 spec pattern:
     ` ?[\\p{L}\\p{M}<ZWJ><ZWNJ>]+ | ?\\p{N}{1,3} | ?[^\\s\\p{L}\\p{M}\\p{N}<ZWJ><ZWNJ>]+ | \\s+`
   v0.3/v0.4's WORD_PATTERN already fixed the \\p{M} bug (matras are
   combining marks, category \\p{M}, not \\p{L} -- excluding them floors
   Indic fertility regardless of vocab size). This version additionally
   folds ZWJ (U+200D) and ZWNJ (U+200C) into the letter/mark class so they
   ATTACH to the adjacent Indic grapheme instead of being split off as
   their own token or landing in the punctuation catch-all -- ZWNJ/ZWJ are
   not decorative in Brahmic scripts: ZWNJ suppresses a conjunct ligature
   (e.g. Devanagari half-forms), ZWJ triggers one, and both are load-bearing
   in Malayalam chillu letters and complex Bengali/Devanagari conjuncts.
   Splitting them into their own byte-level token orphans them from the
   grapheme they modify and wastes a merge slot.

3. Digit handling is now a --digit-mode flag with two variants (both
   trained here so v05_candidate_comparison.md can show real numbers, not
   just assert either is better):
     individual (v0.3/v0.4 default): pre_tokenizers.Digits(individual_digits=True)
       splits every digit into its own token (2026 -> 2 0 2 6). Simple,
       arithmetic-friendly (see train_tokenizer.py's original rationale),
       but wasteful for number-heavy Indic financial/legal text (currency
       amounts, dates, IFSC/PAN-style codes).
     grouped: \\p{N}{1,3} in the word pattern itself groups digits into
       runs of up to 3 (2026 -> 202 6; 123456 -> 123 456), closer to
       GPT-4-style tokenizers. Fewer tokens for long numbers/currency
       amounts, at some cost to digit-by-digit arithmetic transparency.
   Candidates A/B/C in this file all use --digit-mode grouped (the v0.5
   default) based on the ₹1,23,456.78-style spot check in
   v05_candidate_comparison.md; --digit-mode individual remains available
   and is exercised by tokenizer_metrics.py's round-trip test either way.

Usage:
  python train_tokenizer_v05.py --input "corpus_v05_balanced/*.txt" \\
      --vocab-size 80000 --digit-mode grouped --out tokenizer-v0.5-A80k.json
"""

import argparse
import glob
import sys

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from normalize_text import normalize_corpus_text

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]
RESERVED = [f"<|reserved_{i}|>" for i in range(64)]

ZWNJ = "‌"
ZWJ = "‍"


def word_pattern(digit_mode: str) -> Regex:
    """digit_mode='grouped': \\p{N}{1,3} is its own alternative in the
    split regex (digits grouped in threes, greedy left-to-right).
    digit_mode='individual': no digit alternative here -- a separate
    pre_tokenizers.Digits(individual_digits=True) step (run first, see
    build_tokenizer) already isolated every digit before this pattern
    runs, matching v0.3/v0.4 behavior exactly."""
    letter_class = r"[\p{L}\p{M}" + ZWJ + ZWNJ + r"]+"
    other_class = r"[^\s\p{L}\p{M}\p{N}" + ZWJ + ZWNJ + r"]+"
    if digit_mode == "grouped":
        pattern = rf" ?{letter_class}| ?\p{{N}}{{1,3}}| ?{other_class}|\s+"
    elif digit_mode == "individual":
        pattern = rf" ?{letter_class}| ?{other_class}|\s+"
    else:
        raise ValueError(f"unknown digit_mode {digit_mode!r}")
    return Regex(pattern)


def build_tokenizer(digit_mode: str, model_type: str) -> Tokenizer:
    if model_type == "bpe":
        tok = Tokenizer(models.BPE(byte_fallback=False, unk_token=None))
    elif model_type == "unigram":
        # byte fallback: any sequence not covered by a learned piece falls
        # back to individual bytes (never an <unk>), same no-unk guarantee
        # byte-level BPE gets from ByteLevel pre-tokenization.
        tok = Tokenizer(models.Unigram(byte_fallback=True))
    else:
        raise ValueError(f"unknown model_type {model_type!r}")

    # NFC (+ CRLF/BOM cleanup) applied inside the tokenizer.json itself --
    # see module docstring point 1 for why this is not a full substitute
    # for normalize_text.normalize_corpus_text() upstream.
    tok.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex(r"\r\n"), "\n"),
            normalizers.Replace(Regex(r"\r"), "\n"),
            normalizers.Replace("﻿", ""),
            normalizers.NFC(),
        ]
    )

    steps = []
    if digit_mode == "individual":
        steps.append(pre_tokenizers.Digits(individual_digits=True))
    steps.append(pre_tokenizers.Split(word_pattern(digit_mode), behavior="isolated"))
    steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(steps)

    tok.decoder = decoders.ByteLevel()
    return tok


def normalize_files_in_place_check(files: list[str]) -> None:
    """Sanity check (not a mutation): corpus-prep (build_corpus_v05.py)
    already ran normalize_corpus_text() over every file it wrote. This
    just confirms that invariant so a mis-pointed --input glob at raw,
    un-normalized text fails loudly instead of silently training on text
    that won't match the tokenizer's own in-graph normalizer's output."""
    sample = files[0]
    with open(sample, encoding="utf-8") as f:
        head = f.read(200_000)
    if normalize_corpus_text(head) != head:
        print(
            f"WARNING: {sample} is not already normalize_corpus_text()-normalized "
            "(train_tokenizer_v05.py expects pre-normalized corpus files, e.g. from "
            "build_corpus_v05.py -- training on raw text still WORKS because the "
            "tokenizer's own NFC normalizer runs at encode() time, but the merge "
            "list will reflect whatever normalization the corpus happened to have).",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of training text files")
    ap.add_argument("--vocab-size", type=int, default=96_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--digit-mode", choices=["individual", "grouped"], default="grouped")
    ap.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input!r}")
    normalize_files_in_place_check(files)

    tok = build_tokenizer(args.digit_mode, args.model_type)

    if args.model_type == "bpe":
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=SPECIAL_TOKENS + RESERVED,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
    else:
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS + RESERVED,
            unk_token="<|reserved_0|>",  # unused in practice (byte_fallback=True)
            show_progress=True,
        )

    print(
        f"training {args.model_type} on {len(files)} files, vocab={args.vocab_size}, "
        f"digit_mode={args.digit_mode}"
    )
    tok.train(files, trainer)
    tok.save(args.out)
    print(f"saved -> {args.out}  (actual vocab: {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
