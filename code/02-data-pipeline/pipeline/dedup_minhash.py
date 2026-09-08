#!/usr/bin/env python3
"""Near-duplicate removal: MinHash + LSH banding, v1.

In-memory reference implementation for the first tens of GB. The distributed
version (same shingling, same signature scheme) replaces it at scale.

Method:
  1. shingle each doc into word 5-grams
  2. 128 MinHash values per doc (one per hash seed)
  3. LSH: split the signature into 16 bands of 8; docs sharing any band bucket
     are candidates
  4. verify candidates with exact Jaccard on shingle sets; >= 0.8 = duplicate
  5. keep the first-seen doc of each duplicate cluster

Usage: python dedup_minhash.py < in.jsonl > out.jsonl
"""

import json
import sys

import numpy as np

SHINGLE = 5
NUM_HASHES = 128
BANDS = 16          # rows per band = NUM_HASHES // BANDS = 8
JACCARD_THRESHOLD = 0.8
SEED = 20260814     # fixed: dedup must be reproducible

_rng = np.random.default_rng(SEED)
_MERSENNE = (1 << 61) - 1
_A = _rng.integers(1, _MERSENNE, size=NUM_HASHES, dtype=np.int64)
_B = _rng.integers(0, _MERSENNE, size=NUM_HASHES, dtype=np.int64)


def shingles(text: str) -> set[int]:
    words = text.lower().split()
    if len(words) < SHINGLE:
        return {hash(" ".join(words)) & 0xFFFFFFFFFFFF}
    return {hash(" ".join(words[i:i + SHINGLE])) & 0xFFFFFFFFFFFF
            for i in range(len(words) - SHINGLE + 1)}


def minhash(sh: set[int]) -> np.ndarray:
    x = np.fromiter(sh, dtype=np.int64)
    # (a*x + b) mod p for each hash function; min over shingles
    vals = (_A[:, None] * x[None, :] + _B[:, None]) % _MERSENNE
    return vals.min(axis=1)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def main() -> None:
    buckets: dict[tuple, list[int]] = {}
    shingle_sets: list[set[int]] = []
    kept = dropped = 0
    rows = NUM_HASHES // BANDS

    for line in sys.stdin:
        doc = json.loads(line)
        sh = shingles(doc["text"])
        sig = minhash(sh)
        idx = len(shingle_sets)

        candidates = set()
        keys = []
        for b in range(BANDS):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            keys.append(key)
            candidates.update(buckets.get(key, ()))

        is_dup = any(jaccard(sh, shingle_sets[c]) >= JACCARD_THRESHOLD
                     for c in candidates)
        shingle_sets.append(sh)
        if is_dup:
            dropped += 1
            continue
        for key in keys:
            buckets.setdefault(key, []).append(idx)
        kept += 1
        sys.stdout.write(line)

    print(f"dedup: kept={kept} dropped={dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
