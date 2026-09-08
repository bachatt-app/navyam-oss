#!/usr/bin/env python3
"""Build the navya-0-sft dataset: dolly-15k (S043) + oasst2 English pairs
(S044) + our handwritten finance seed (S045, oversampled), rendered in the
serving shim's exact template:

    Q: {instruction}
    A: {response}<|eos|>

then decontaminated against BachattBench 13-grams and written as
train.bin/val.bin/meta.json compatible with train.py.

v0 tradeoff (documented): plain next-token loss over the whole Q+A text —
no prompt-loss masking. Masking lands with navya-1.

Usage:  python build_sft_data.py --out ../04-training-stack/data/sft_v1
"""

import argparse
import io
import json
import os
import random
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "02-data-pipeline", "pipeline"))
import decontaminate as qdecon  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

UA = "navyam-gpt-research/0.1 (sft data; contact: bachattapp@gmail.com)"
SEED_OVERSAMPLE = 15


def fetch_parquet_urls(dataset, config, split):
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def read_parquet(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        return pq.read_table(io.BytesIO(r.read()))


def load_dolly():
    pairs = []
    for url in fetch_parquet_urls("databricks/databricks-dolly-15k",
                                  "default", "train"):
        t = read_parquet(url).to_pylist()
        for row in t:
            q = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip()
            a = (row.get("response") or "").strip()
            if not (5 <= len(q) <= 600 and 20 <= len(a) <= 2500):
                continue
            if ctx and len(ctx) < 1500:
                q = q + "\n" + ctx
            pairs.append(("S043", q, a))
    return pairs


def load_oasst2():
    msgs = {}
    for url in fetch_parquet_urls("OpenAssistant/oasst2", "default", "train"):
        for row in read_parquet(url).to_pylist():
            msgs[row["message_id"]] = row
    pairs = []
    for m in msgs.values():
        if m.get("role") != "assistant" or m.get("lang") != "en":
            continue
        rank = m.get("rank")
        if rank is not None and rank != 0:
            continue                      # keep only top-ranked replies
        parent = msgs.get(m.get("parent_id"))
        if not parent or parent.get("role") != "prompter" \
                or parent.get("lang") != "en":
            continue
        q, a = parent["text"].strip(), m["text"].strip()
        if 5 <= len(q) <= 600 and 20 <= len(a) <= 2500:
            pairs.append(("S044", q, a))
    return pairs


def load_seed():
    pairs = []
    for line in open(os.path.join(HERE, "seed_finance_instructions.jsonl"),
                     encoding="utf-8"):
        d = json.loads(line)
        pairs.append(("S045", d["instruction"].strip(), d["response"].strip()))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(
        CODE, "01-tokenizer", "tokenizer-v0.3-64k.json"))
    args = ap.parse_args()

    print("loading dolly-15k …")
    dolly = load_dolly()
    print(f"  {len(dolly)} pairs")
    print("loading oasst2 (EN, top-ranked) …")
    oasst = load_oasst2()
    print(f"  {len(oasst)} pairs")
    seed = load_seed()
    print(f"  {len(seed)} seed pairs ×{SEED_OVERSAMPLE}")

    pairs = dolly + oasst + seed * SEED_OVERSAMPLE
    rng = random.Random(20260818)
    rng.shuffle(pairs)

    # decontaminate vs BachattBench
    banned = set()
    for text in qdecon.eval_texts(os.path.join(
            CODE, "03-evals", "bachattbench", "seed_v0.jsonl")):
        banned.update(qdecon.ngrams(text))
    kept, dropped = [], 0
    for src, q, a in pairs:
        text = f"Q: {q}\nA: {a}"
        if any(g in banned for g in qdecon.ngrams(text)):
            dropped += 1
            continue
        kept.append((src, text))
    print(f"decontamination: kept={len(kept)} dropped={dropped}")

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|eos|>")
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab <= 2**16 else np.uint32  # 65,536 ids (0..65535) fit uint16

    n_val = max(50, len(kept) // 50)
    splits = {"train": kept[:-n_val], "val": kept[-n_val:]}
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for name, docs in splits.items():
        buf, total = [], 0
        with open(os.path.join(args.out, f"{name}.bin"), "wb") as f:
            for _, text in docs:
                buf.extend(tok.encode(text).ids)
                buf.append(eos)
                if len(buf) >= 1_000_000:
                    np.array(buf, dtype=dtype).tofile(f)
                    total += len(buf)
                    buf = []
            if buf:
                np.array(buf, dtype=dtype).tofile(f)
                total += len(buf)
        counts[name] = total
        print(f"  {name}: {len(docs)} docs, {total:,} tokens")

    json.dump({"vocab_size": vocab, "dtype": np.dtype(dtype).name,
               "train_tokens": counts["train"], "val_tokens": counts["val"],
               "sources": {"S043_dolly": len(dolly), "S044_oasst2": len(oasst),
                           "S045_seed": len(seed),
                           "seed_oversample": SEED_OVERSAMPLE},
               "template": "Q: {q}\\nA: {a}<|eos|>",
               "decontaminated_dropped": dropped},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"sft data → {args.out}")


if __name__ == "__main__":
    main()
