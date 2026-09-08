#!/usr/bin/env python3
"""Decontamination: remove training docs that overlap eval sets.

Non-negotiable stage. A single leaked eval item silently corrupts every
measurement built on it.

Method: build the set of word 13-grams appearing in any eval item; drop any
training doc sharing >= 1 such n-gram (strict on purpose — at trillion-token
scale the cost of over-dropping is negligible, the cost of leakage is not).

Eval inputs: JSONL files (uses `prompt`/`answer`/`text` fields) or plain .txt.

Usage:
  python decontaminate.py --evals "path/to/evals/*" < in.jsonl > out.jsonl
"""

import argparse
import glob
import json
import sys
from hashlib import blake2b

NGRAM = 13


def _h(s: str) -> int:
    # stable across processes (builtin hash() is salted: it worked under
    # fork but silently matched nothing under spawn)
    return int.from_bytes(blake2b(s.encode(), digest_size=8).digest(), "big")


def ngrams(text: str, n: int = NGRAM, feats=None):
    words = feats.lower_words if feats is not None else text.lower().split()
    for i in range(len(words) - n + 1):
        yield _h(" ".join(words[i:i + n]))


def eval_texts(path: str):
    """Yield every human-readable string of an eval item. Beyond prompt/answer/
    text this now also yields MCQ `options` (dict or list), the `rationale`, and
    the judge `rubric` — a training doc that reproduces an option or a rationale
    is contamination just as much as one that reproduces the question stem."""
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for field in ("prompt", "answer", "text", "rationale", "rubric"):
                v = item.get(field)
                if isinstance(v, str):
                    yield v
            opts = item.get("options")
            if isinstance(opts, dict):
                for v in opts.values():
                    if isinstance(v, str):
                        yield v
            elif isinstance(opts, list):
                for v in opts:
                    if isinstance(v, str):
                        yield v
    else:
        yield open(path, encoding="utf-8").read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True, help="glob of eval files")
    args = ap.parse_args()

    banned: set[int] = set()
    files = sorted(f for pat in args.evals.split() for f in glob.glob(pat))
    if not files:
        sys.exit(f"no eval files match {args.evals!r}")
    for path in files:
        for text in eval_texts(path):
            banned.update(ngrams(text))
    print(f"decontaminate: {len(banned)} banned {NGRAM}-grams "
          f"from {len(files)} eval files", file=sys.stderr)

    kept = dropped = 0
    for line in sys.stdin:
        doc = json.loads(line)
        if any(g in banned for g in ngrams(doc["text"])):
            dropped += 1
            continue
        kept += 1
        sys.stdout.write(line)
    print(f"decontaminate: kept={kept} dropped={dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
