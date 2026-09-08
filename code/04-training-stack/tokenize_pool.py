#!/usr/bin/env python3
"""Stream-tokenize a cleaned pool (JSONL) into a flat binary token file.

Unlike prepare_data.py (which accumulates every id in RAM), this streams
batches of documents through encode_batch — the Rust tokenizers library
parallelizes a batch across all cores — and appends to disk with constant
memory. Documents are joined with <|eos|>.

Usage:
  python tokenize_pool.py --pool ../02-data-pipeline/pools/global_english/clean.jsonl \
      --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json \
      --out data/domains/global_english.bin
"""

import argparse
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", help="keep only docs whose 'lang' field "
                    "matches (per-language bins for the 1b Indic targets)")
    ap.add_argument("--batch-docs", type=int, default=512,
                    help="docs per encode_batch call")
    ap.add_argument("--batch-bytes", type=float, default=16e6,
                    help="flush the batch early past this many bytes of text")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|eos|>")
    if eos is None:
        sys.exit("tokenizer has no <|eos|> token")
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 2**16 else np.uint32  # 65,536 ids (0..65535) fit uint16

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    total = docs = 0
    texts: list[str] = []
    text_bytes = 0
    buf: list[int] = []

    def flush_texts(out):
        nonlocal docs, total, buf, texts, text_bytes
        for enc in tok.encode_batch(texts):
            buf.extend(enc.ids)
            buf.append(eos)
            docs += 1
        texts = []
        text_bytes = 0
        if len(buf) >= 1_000_000:
            np.array(buf, dtype=dtype).tofile(out)
            total += len(buf)
            buf = []
            print(f"  {docs} docs, {total/1e6:.0f}M tokens",
                  end="\r", file=sys.stderr)

    with open(args.out + ".part", "wb") as out:
        for line in open(args.pool, encoding="utf-8"):
            doc = json.loads(line)
            if args.lang and doc.get("lang") != args.lang:
                continue
            text = doc.get("text", "")
            if not text.strip():
                continue
            texts.append(text)
            text_bytes += len(text)
            if len(texts) >= args.batch_docs or text_bytes >= args.batch_bytes:
                flush_texts(out)
        if texts:
            flush_texts(out)
        if buf:
            np.array(buf, dtype=dtype).tofile(out)
            total += len(buf)
    os.replace(args.out + ".part", args.out)
    meta = {"pool": os.path.abspath(args.pool),
            "tokenizer": os.path.abspath(args.tokenizer),
            "vocab_size": vocab, "dtype": np.dtype(dtype).name,
            "docs": docs, "tokens": total}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=1)
    print(f"{args.pool} -> {total:,} BPE tokens ({docs:,} docs) -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
