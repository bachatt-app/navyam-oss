#!/usr/bin/env python3
"""Tokenize text files into flat binary token arrays for training.

Documents are joined with <|eos|>. Output: train.bin / val.bin (last VAL_FRAC
of tokens) + meta.json recording tokenizer, vocab size and dtype — train.py
reads vocab_size from meta.json so config files never drift from the data.

Usage:
  python prepare_data.py --input "texts/*.txt" --tokenizer tok.json --out data/run1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

VAL_FRAC = 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of text files")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"no files match {args.input!r}")

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|eos|>")
    if eos is None:
        sys.exit("tokenizer has no <|eos|> token")

    ids: list[int] = []
    for path in files:
        text = open(path, encoding="utf-8").read()
        ids.extend(tok.encode(text).ids)
        ids.append(eos)

    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 2**16 else np.uint32  # 65,536 ids (0..65535) fit uint16
    arr = np.array(ids, dtype=dtype)
    n_val = max(1, int(len(arr) * VAL_FRAC))

    os.makedirs(args.out, exist_ok=True)
    arr[:-n_val].tofile(os.path.join(args.out, "train.bin"))
    arr[-n_val:].tofile(os.path.join(args.out, "val.bin"))
    meta = {
        "tokenizer": os.path.abspath(args.tokenizer),
        "vocab_size": vocab,
        "dtype": np.dtype(dtype).name,
        "train_tokens": int(len(arr) - n_val),
        "val_tokens": int(n_val),
        "sources": files,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(arr):,} tokens -> {args.out} "
          f"(train {meta['train_tokens']:,} / val {meta['val_tokens']:,}, "
          f"vocab {vocab}, {meta['dtype']})")


if __name__ == "__main__":
    main()
