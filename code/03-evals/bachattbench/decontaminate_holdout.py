#!/usr/bin/env python3
"""Decontaminate bench_v1_holdout.jsonl against the SFT training corpus.

Direction matters: build_chat_data.py's decontaminate() bans training docs
that overlap the eval set (protects held-out scores from *train* leakage).
This script does the complementary check the eval author is responsible for:
build the banned n-gram set FROM the TRAINING data, then drop any authored
eval item whose prompt overlaps it -- i.e. make sure the eval doesn't
(coincidentally or by reuse) restate text the model was trained on, which
would make that item measure memorization instead of ability.

Two checks (matching build_chat_data.decontaminate's method):
  (1) shares >=1 13-gram with any training doc (long-passage overlap)
  (2) shares >=1 7-gram with any training doc (short MCQ-stem overlap)

Training sources (per the task spec -- do not widen without instruction):
  code/07-sft/corpus_v2/*.jsonl
  code/07-sft/conversations/*.jsonl
  code/07-sft/seed_finance_instructions.jsonl

Usage:
  python3 decontaminate_holdout.py                 # report only
  python3 decontaminate_holdout.py --write          # overwrite the holdout file, dropping hits
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DIR = os.path.join(HERE, "..", "..", "07-sft")
sys.path.insert(0, os.path.join(HERE, "..", "..", "02-data-pipeline", "pipeline"))
import decontaminate as d  # noqa: E402

SHORT_NGRAM = 7
HOLDOUT = os.path.join(HERE, "bench_v1_holdout.jsonl")

TRAIN_GLOBS = [
    os.path.join(SFT_DIR, "corpus_v2", "*.jsonl"),
    os.path.join(SFT_DIR, "conversations", "*.jsonl"),
    os.path.join(SFT_DIR, "seed_finance_instructions.jsonl"),
]


def train_texts(path):
    """Yield the human-readable text of one training doc: instruction+response,
    or the joined content of a `messages` conversation, or a bare `text`."""
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        if "messages" in doc:
            yield " ".join(m.get("content", "") for m in doc["messages"])
        else:
            parts = [doc.get(k, "") for k in ("instruction", "response", "text")]
            yield " ".join(p for p in parts if p)


def build_banned():
    files = sorted(f for pat in TRAIN_GLOBS for f in glob.glob(pat))
    if not files:
        sys.exit(f"no training files matched {TRAIN_GLOBS!r}")
    banned13, banned7 = set(), set()
    for path in files:
        for text in train_texts(path):
            banned13.update(d.ngrams(text))
            banned7.update(d.ngrams(text, SHORT_NGRAM))
    return banned13, banned7, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                     help="overwrite bench_v1_holdout.jsonl, dropping contaminated items")
    args = ap.parse_args()

    banned13, banned7, files = build_banned()
    print(f"decontaminate_holdout: {len(banned13)} banned 13-grams, "
          f"{len(banned7)} banned 7-grams from {len(files)} training files", file=sys.stderr)

    items = [json.loads(line) for line in open(HOLDOUT, encoding="utf-8") if line.strip()]
    kept, dropped = [], []
    for it in items:
        text = it["prompt"]
        hit13 = any(g in banned13 for g in d.ngrams(text))
        hit7 = any(g in banned7 for g in d.ngrams(text, SHORT_NGRAM))
        if hit13 or hit7:
            dropped.append((it["id"], "13-gram" if hit13 else "7-gram"))
        else:
            kept.append(it)

    print(f"decontaminate_holdout: kept={len(kept)} dropped={len(dropped)} "
          f"of {len(items)} authored items", file=sys.stderr)
    for item_id, reason in dropped:
        print(f"  DROPPED {item_id} ({reason} overlap with training data)", file=sys.stderr)

    if args.write:
        with open(HOLDOUT, "w", encoding="utf-8") as f:
            for it in kept:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"wrote {len(kept)} decontaminated items -> {HOLDOUT}", file=sys.stderr)

    return len(items), len(kept), len(dropped)


if __name__ == "__main__":
    main()
