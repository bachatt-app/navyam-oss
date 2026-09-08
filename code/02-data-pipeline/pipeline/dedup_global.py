#!/usr/bin/env python3
"""GLOBAL near-duplicate removal across domains — streaming, bounded memory.

dedup_exact.py catches byte-identical docs within one domain. This pass
catches (a) near-duplicates (boilerplate variants, re-crawls) and (b) copies
that landed in DIFFERENT domains (a finance article in both web-en and
finance pools), by sharing one signature store across sequential runs.

Method: 5-word shingles → 16 MinHash values → 4 LSH bands of 4 rows. Two docs
sharing any band signature are near-dup candidates → the later one is dropped.
With 8-byte band keys this is ~32B/doc of state: 10M docs ≈ 320MB — fits the
ingest VM. (Band collisions are probabilistic: ~equivalent to Jaccard ≳ 0.7
matching; tune BANDS/ROWS if too aggressive.)

Usage (state persists across domain runs — that's the cross-domain part):
  python dedup_global.py --state /tmp/dedup_state.bin < a.jsonl > a_out.jsonl
  python dedup_global.py --state /tmp/dedup_state.bin < b.jsonl > b_out.jsonl
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import zlib

try:
    import numpy as _np
except ImportError:          # numpy-less boxes fall back to pure python
    _np = None

BANDS, ROWS = 4, 4          # 16 minhashes total
NUM_HASHES = BANDS * ROWS
WORD = re.compile(r"\w+", re.UNICODE)
MASK = (1 << 61) - 1        # Mersenne prime modulus for cheap universal hashing
# fixed hash parameters — deterministic across runs
AB = [(int.from_bytes(hashlib.md5(f"a{i}".encode()).digest()[:8], "big") | 1,
       int.from_bytes(hashlib.md5(f"b{i}".encode()).digest()[:8], "big"))
      for i in range(NUM_HASHES)]


# Vectorized minhash (PERF item 3). (a*s+b) % (2^61-1) exceeds uint64 for
# 61-bit a times 32-bit s, so the product is reduced with exact Mersenne
# identities (2^61 === 1 mod M) in uint64 pieces — results are BIT-IDENTICAL
# to the python path (verified by --self-test), just ~an order of magnitude
# faster on the min-reduction that dominates cleaning time.
_M = MASK
_A_MOD = None
_B_MOD = None
if _np is not None:
    _A_MOD = _np.array([a % _M for a, _ in AB], dtype=_np.uint64)
    _B_MOD = _np.array([b % _M for _, b in AB], dtype=_np.uint64)


def _reduce61(x):
    # x < 2^63: one fold + conditional subtract lands in [0, M)
    x = (x & _np.uint64(_M)) + (x >> _np.uint64(61))
    return _np.where(x >= _M, x - _np.uint64(_M), x)


def _minhash_np(shingle_arr):
    s = shingle_arr[None, :].astype(_np.uint64)          # (1, S) < 2^32
    a = _A_MOD[:, None]                                  # (16, 1) < 2^61
    b = _B_MOD[:, None]
    a_hi, a_lo = a >> _np.uint64(32), a & _np.uint64(0xFFFFFFFF)
    r1 = _reduce61(a_lo * s)                             # a_lo*s < 2^64
    v = a_hi * s                                         # < 2^61
    v_hi, v_lo = v >> _np.uint64(29), v & _np.uint64((1 << 29) - 1)
    term2 = _reduce61(v_hi + (v_lo << _np.uint64(32)))   # v*2^32 mod M
    total = _reduce61(_reduce61(r1 + term2) + b)
    return total.min(axis=1)


def band_keys(text: str, feats=None):
    lower = feats.lower_text if feats is not None else text.lower()
    words = WORD.findall(lower)
    if len(words) < 5:      # tiny docs: fall back to exact hash of the text
        return [hashlib.md5(text.lower().encode()).digest()[:8]]
    # zlib.crc32 is stable across processes (builtin hash() is salted)
    shingles = {zlib.crc32(" ".join(words[i:i + 5]).encode())
                for i in range(len(words) - 4)}
    if _np is not None:
        arr = _np.fromiter(shingles, dtype=_np.uint64, count=len(shingles))
        mins = [int(m) for m in _minhash_np(arr)]
    else:
        mins = [min((a * s + b) % MASK for s in shingles) for a, b in AB]
    keys = []
    for bi in range(BANDS):
        h = hashlib.md5(repr(mins[bi * ROWS:(bi + 1) * ROWS]).encode())
        keys.append(h.digest()[:8])
    return keys


def _self_test():
    """Bit-identical check: numpy kernel vs pure-python on random docs."""
    import random
    import time as _t
    rng = random.Random(7)
    docs = [" ".join(rng.choice(["paisa", "loan", "sip", "fund", "tax",
                                 "credit", "bank", "emi", "gold", "nav"])
                     for _ in range(rng.randint(5, 400)))
            for _ in range(400)]
    global _np
    np_saved = _np
    t0 = _t.time()
    fast = [band_keys(d) for d in docs]
    t_np = _t.time() - t0
    _np = None
    t0 = _t.time()
    slow = [band_keys(d) for d in docs]
    t_py = _t.time() - t0
    _np = np_saved
    assert fast == slow, "MISMATCH: numpy kernel differs from python path"
    print(f"self-test OK: {len(docs)} docs bit-identical | "
          f"python {t_py:.2f}s vs numpy {t_np:.2f}s "
          f"({t_py / max(t_np, 1e-9):.1f}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=False,
                    help="pickle file shared across domain runs")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.state:
        ap.error("--state is required")

    seen: set[bytes] = set()
    if os.path.exists(args.state):
        seen = pickle.load(open(args.state, "rb"))

    kept = dropped = 0
    for line in sys.stdin:
        doc = json.loads(line)
        keys = band_keys(doc["text"])
        if any(k in seen for k in keys):
            dropped += 1
            continue
        seen.update(keys)
        kept += 1
        sys.stdout.write(line if line.endswith("\n") else line + "\n")

    tmp = args.state + ".part"
    pickle.dump(seen, open(tmp, "wb"), protocol=4)
    os.replace(tmp, args.state)
    print(f"dedup_global: kept={kept} dropped={dropped} "
          f"state={len(seen)} band-keys", file=sys.stderr)


if __name__ == "__main__":
    main()
