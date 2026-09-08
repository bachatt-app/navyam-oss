#!/usr/bin/env python3
"""Exact-duplicate removal for bulk corpora — streaming, constant-ish memory.

MinHash near-dup (dedup_minhash.py) keeps every document's shingle set in RAM
and cannot survive millions of documents on one machine. At bulk scale we do
exact dedup on a normalized-text hash (16 bytes/doc), and rely on upstream
dedup in the curated sources (FineWeb-Edu, Sangraha are deduplicated at
origin). Distributed MinHash returns at the 1T-token stage.

Usage: python dedup_exact.py < in.jsonl > out.jsonl
"""

import hashlib
import json
import sys


def key(text: str, feats=None) -> bytes:
    normalized = feats.norm_text if feats is not None \
        else " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).digest()


def main() -> None:
    seen: set[bytes] = set()
    kept = dropped = 0
    for line in sys.stdin:
        doc = json.loads(line)
        k = key(doc["text"])
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        kept += 1
        sys.stdout.write(line)
    print(f"dedup_exact: kept={kept} dropped={dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
