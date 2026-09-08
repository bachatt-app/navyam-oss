#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer for the India-first foundation model.

Design choices (see 01-tokenizer/README.md and the execution plan):
  - byte-level: no unknown tokens, every script representable
  - digits split individually: helps arithmetic
  - whitespace preserved: code-friendly
  - special tokens + reserved slots fixed up front so the vocab layout
    never shifts between candidates

Usage:
  python train_tokenizer.py --input "path/to/*.txt" --vocab-size 160000 \
         --out tokenizer-160k.json
"""

import argparse
import glob
import sys

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]
# Reserved slots for future use (tool-call markers, FIM, voice/vision later)
# so the vocab layout is stable across generations.
RESERVED = [f"<|reserved_{i}|>" for i in range(64)]


# Word-splitting pattern. The stock GPT-2/ByteLevel regex uses \p{L}+ alone,
# which SPLITS Indic syllables: vowel signs (matras) are category \p{M}, not
# \p{L}, so merges can never span consonant+matra and Indic fertility is
# floored at the syllable count regardless of vocab size (measured: Hindi
# stuck at ~3.3 tokens/word from 2K to 64K vocab). Including \p{M} fixes it.
WORD_PATTERN = Regex(r" ?[\p{L}\p{M}]+| ?[^\s\p{L}\p{M}\p{N}]+|\s+")


def build_tokenizer() -> Tokenizer:
    # creates a tokenizer using the BPE model.
    # byte_fallback=False => not relying on BPE's separate byte_fallback mechanism.
    tok = Tokenizer(models.BPE(byte_fallback=False))

    # Run these preprocessing steps in order
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # each digit is its own token: 2026 -> 2 0 2 6
            pre_tokenizers.Digits(individual_digits=True),
            # creates linguistically useful word-like segments for BPE training
            pre_tokenizers.Split(WORD_PATTERN, behavior="isolated"),
            # all remaining characters to byte-level symbols, to avoid unknown-token failure
            # Some byte-level tokenizers automatically insert a space at the beginning, so set add_prefix_space=False to avoid this.
            # ByteLevel itself can perform its own regex splitting. so set use_regex=False to avoid this.
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )

    # explicitly use
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of training text files")
    ap.add_argument("--vocab-size", type=int, default=160_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", required=True, help="output tokenizer.json path")
    args = ap.parse_args()

    # find training text files
    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input!r}")

    tok = build_tokenizer()

    # create a BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS + RESERVED,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"training on {len(files)} files, vocab={args.vocab_size}")
    tok.train(files, trainer)
    tok.save(args.out)
    print(f"saved -> {args.out}  (actual vocab: {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
