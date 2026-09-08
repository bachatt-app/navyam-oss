#!/usr/bin/env python3
"""Build the navya-0-sft v2 dataset — 100% first-party authored content (S046).

v1 used dolly-15k + oasst2 and taught the model three bad habits: inventing
URLs (open-data answers are full of links), replying in English to Hinglish
questions (open data is all-English), and citing non-Indian references.
v2 replaces all of it with content we wrote ourselves: corpus_v2/*.jsonl
(topic files) + seed_finance_instructions.jsonl — EN + Hinglish, language-
matched, India-specific institutions, zero URLs, reason-don't-recommend.

Template (matches the serving shim):   Q: {instruction}\nA: {response}<|eos|>

Small corpus by design (LIMA-style): repetition comes from training epochs,
not data-level oversampling. Still decontaminated against BachattBench.

Usage:  python build_sft_data_v2.py --out ../04-training-stack/data/sft_v2
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "02-data-pipeline", "pipeline"))
import decontaminate as qdecon  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def load_pairs():
    files = sorted(glob.glob(os.path.join(HERE, "corpus_v2", "*.jsonl")))
    files.append(os.path.join(HERE, "seed_finance_instructions.jsonl"))
    pairs, ids = [], set()
    for path in files:
        n = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d["instruction"].strip(), d["response"].strip()
            assert d["id"] not in ids, f"duplicate id {d['id']}"
            ids.add(d["id"])
            assert "http" not in a and "www." not in a, \
                f"URL in {d['id']} — corpus must be link-free"
            pairs.append((d["id"], q, a))
            n += 1
        print(f"  {os.path.basename(path)}: {n} pairs")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(
        CODE, "01-tokenizer", "tokenizer-v0.3-64k.json"))
    args = ap.parse_args()

    pairs = load_pairs()
    rng = random.Random(20260818)
    rng.shuffle(pairs)

    banned = set()
    for text in qdecon.eval_texts(os.path.join(
            CODE, "03-evals", "bachattbench", "seed_v0.jsonl")):
        banned.update(qdecon.ngrams(text))
    kept, dropped = [], 0
    for pid, q, a in pairs:
        text = f"Q: {q}\nA: {a}"
        if any(g in banned for g in qdecon.ngrams(text)):
            dropped += 1
            print(f"  decontaminated out: {pid}")
            continue
        kept.append(text)
    print(f"decontamination: kept={len(kept)} dropped={dropped}")

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|eos|>")
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 2**16 else np.uint32  # 65,536 ids (0..65535) fit uint16

    # Behaviour-cloning mode: EVERY pair goes into train — an unseen holdout
    # pair is a user-facing question the model answers badly. val is a small
    # overlapping monitoring slice (so "VAL loss" tracks memorisation, not
    # generalisation — documented, deliberate).
    n_val = max(12, len(kept) // 20)
    splits = {"train": kept, "val": kept[-n_val:]}
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    # Block alignment: every doc is padded with <|eos|> to a multiple of
    # ALIGN tokens, and train.py samples windows only at block starts. So the
    # model regularly sees "Q: ..." at position 0 — the exact inference shape —
    # and learns answer-end -> <|eos|> hard (padding runs reinforce it).
    ALIGN = 128
    for name, docs in splits.items():
        buf = []
        for text in docs:
            ids = tok.encode(text).ids + [eos]
            ids += [eos] * (-len(ids) % ALIGN)
            buf.extend(ids)
        np.array(buf, dtype=dtype).tofile(os.path.join(args.out, f"{name}.bin"))
        counts[name] = len(buf)
        print(f"  {name}: {len(docs)} docs, {len(buf):,} tokens (eos-padded)")

    json.dump({"vocab_size": vocab, "dtype": np.dtype(dtype).name,
               "train_tokens": counts["train"], "val_tokens": counts["val"],
               "source": "S046 first-party authored (corpus_v2 + seed)",
               "n_pairs": len(kept),
               "template": "Q: {q}\\nA: {a}<|eos|>",
               "align_block": 128,
               "decontaminated_dropped": dropped},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"sft v2 data → {args.out}")


if __name__ == "__main__":
    main()
