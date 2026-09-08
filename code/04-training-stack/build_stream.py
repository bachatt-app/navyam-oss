#!/usr/bin/env python3
"""Build the final train/val streams from per-domain token bins + mix weights.

Realizes the snapshot mix at the token level: each domain contributes
weight × total tokens, repeating its bin up to the epoch cap (4) and
recording a shortfall beyond it — the same semantics as mix.py, but over
real BPE tokens instead of word counts (the doc's words-vs-tokens caveat,
resolved). Validation is document-stream-disjoint per domain: the LAST
val_frac of every domain bin is held out before any training slice is cut.

Usage:
  python build_stream.py --domains data/domains --out data/navya0_v1 \
      --total-tokens 1.3e9
"""

import argparse
import glob
import re
import hashlib
import json
import os

import numpy as np

WEIGHTS = {
    "global_english": 0.50, "indian_english": 0.175, "indian_languages": 0.125,
    "code": 0.10, "finance_econ_law": 0.075, "math_reasoning": 0.025,
}
MAX_EPOCHS = 4.0
VAL_FRAC = 0.005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True, help="dir of <domain>.bin(+meta)")
    ap.add_argument("--virtual", action="store_true",
                    help="PERF item 9: skip writing train.bin; emit a segment "
                         "manifest the trainer reads windows through. The "
                         "train-stream sha256 is computed over the SAME byte "
                         "sequence a physical build would write, so stream_id "
                         "is identical for identical data. val.bin is always "
                         "physical (tiny).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-tokens", type=float, required=True)
    ap.add_argument("--weights", help="JSON file {domain: weight} overriding "
                    "the built-in mix (weights must sum to 1.0)")
    ap.add_argument("--shards", help="local dir of downloaded flywheel shard "
                    "bins (shards/<domain>/<dataset>/NNNNN.bin); suffixed "
                    "shard domains fold into their mix domain "
                    "(global_english_dclm -> global_english)")
    ap.add_argument("--bin-include", default=None,
                    help="regex on shard bin paths; only matching bins enter "
                         "the mix (pool selection from int_score-partitioned "
                         "shards, e.g. '\\.s[45]\\.bin$' for int_score>=4). "
                         "Recorded in the stream manifest.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    weights = WEIGHTS
    if args.weights:
        weights = {k: v for k, v in json.load(open(args.weights)).items()
                   if not k.startswith("_")}     # drop _comment etc.
        total_w = sum(weights.values())
        assert abs(total_w - 1.0) < 1e-6, f"weights sum to {total_w}"

    # domain -> ordered list of bin files (legacy single-bin + shard bins)
    files: dict[str, list] = {}
    metas = {}
    for mp in glob.glob(os.path.join(args.domains, "*.meta.json")):
        m = json.load(open(mp))
        name = os.path.basename(mp).replace(".bin.meta.json", "")
        metas[name] = m
        files.setdefault(name, []).append(mp.replace(".meta.json", ""))
    if args.shards:
        inc = re.compile(args.bin_include) if args.bin_include else None
        for b in sorted(glob.glob(os.path.join(args.shards, "*", "*",
                                               "*.bin"))):
            if inc and not inc.search(b):
                continue
            sdom = b.split(os.sep)[-3]
            # fold suffixed shard domains into their mix domain
            dom = sdom
            while dom not in weights and "_" in dom:
                dom = dom.rsplit("_", 1)[0]
            if dom in weights:
                files.setdefault(dom, []).append(b)
                metas.setdefault(dom, {"vocab_size": 65536,
                                       "dtype": "uint16"})
    missing = set(weights) - set(files)
    if missing:
        raise SystemExit(f"missing domain bins: {sorted(missing)}")
    vocab = {m["vocab_size"] for m in metas.values()}
    dtype = {m["dtype"] for m in metas.values()}
    assert len(vocab) == 1, "mixed tokenizers!"
    assert dtype <= {"uint16"}, f"mixed dtypes: {dtype} — retokenize legacy " \
        "uint32 bins before mixing with shards"
    dtype_np = np.dtype("uint16")

    manifest = {"total_tokens_requested": args.total_tokens,
                "bin_include": args.bin_include,
                "weights_source": args.weights or "built-in",
                "max_epochs": MAX_EPOCHS, "val_frac": VAL_FRAC,
                "vocab_size": vocab.pop(), "dtype": dtype_np.name,
                "domains": {}}
    train_f = None if args.virtual else \
        open(os.path.join(args.out, "train.bin"), "wb")
    val_f = open(os.path.join(args.out, "val.bin"), "wb")
    train_total = val_total = 0
    val_spans = {}   # domain -> [start, end) token offsets inside val.bin
    segments = []    # virtual mode: ordered domain blocks for the trainer
    h_train, h_val = hashlib.sha256(), hashlib.sha256()

    def write_hashed(pool, f, h, chunk=1 << 22):
        for i in range(0, len(pool), chunk):
            b = pool[i:i + chunk].tobytes()
            h.update(b)
            if f is not None:
                f.write(b)

    def stream_range(parts, start, end, chunk=1 << 24):
        # yield chunks of the logical concatenation over [start, end) WITHOUT
        # materializing the pool — chunked reads straight off the memmaps, so
        # peak RAM stays ~chunk (matters at 100B+ token pools).
        off = 0
        for pm in parts:
            pl = len(pm)
            if off + pl <= start:
                off += pl; continue
            if off >= end:
                break
            lo = max(0, start - off); hi = min(pl, end - off)
            i = lo
            while i < hi:
                j = min(hi, i + chunk)
                yield np.asarray(pm[i:j])
                i = j
            off += pl

    for name, w in weights.items():
        part_paths = sorted(files[name])
        parts = [np.memmap(p, dtype=dtype_np, mode="r") for p in part_paths]
        total_len = sum(len(pm) for pm in parts)     # no load
        n_val = max(1, int(total_len * VAL_FRAC))
        train_len = total_len - n_val
        target = w * args.total_tokens
        epochs = min(target / train_len, MAX_EPOCHS)
        got = int(epochs * train_len)
        full, rem = divmod(got, train_len)
        for _ in range(full):
            for ch in stream_range(parts, 0, train_len):
                write_hashed(ch, train_f, h_train)
        if rem:
            for ch in stream_range(parts, 0, rem):
                write_hashed(ch, train_f, h_train)
        val_spans[name] = [val_total, val_total + n_val]
        for ch in stream_range(parts, train_len, total_len):
            write_hashed(ch, val_f, h_val)
        if args.virtual:
            segments.append({
                "domain": name, "start": train_total,
                "pool_tokens": int(train_len),
                "full": int(full), "rem": int(rem),
                "parts": [{"path": os.path.abspath(pp), "tokens": int(len(pm))}
                          for pp, pm in zip(part_paths, parts)],
            })
        train_total += got
        val_total += n_val
        manifest["domains"][name] = {
            "weight": w, "pool_tokens": int(train_len),
            "target_tokens": int(target), "epochs": round(epochs, 4),
            "sampled_tokens": got,
            "shortfall_tokens": int(max(0, target - got)),
        }
        print(f"  {name:<18} pool={train_len/1e6:8.1f}M "
              f"epochs={epochs:5.2f} sampled={got/1e6:8.1f}M"
              f"{'  SHORTFALL' if target - got > 1 else ''}")
    if train_f is not None:
        train_f.close()
    val_f.close()

    manifest["train_tokens"] = train_total
    manifest["val_tokens"] = val_total
    manifest["val_spans"] = val_spans      # domain -> [start, end) in val.bin
    # stream_id derives from the ACTUAL binary content, not just the manifest:
    # identical config over changed bins now yields a different id
    manifest["sha256"] = {"train.bin": h_train.hexdigest(),
                          "val.bin": h_val.hexdigest()}
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest["stream_id"] = "stream-" + hashlib.sha256(blob).hexdigest()[:16]
    # Representation detail is added AFTER the id: a virtual and a physical
    # build of the same data must carry the SAME stream_id (the id names the
    # byte sequence, which sha256[train.bin] already pins either way).
    if args.virtual:
        manifest["virtual"] = True
        manifest["segments"] = segments
    # single write: full manifest IS the meta train.py reads (it has
    # vocab_size/dtype/train_tokens/val_tokens plus provenance)
    json.dump(manifest, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"\n{manifest['stream_id']}: train {train_total/1e9:.3f}B tokens, "
          f"val {val_total/1e6:.1f}M -> {args.out}")


if __name__ == "__main__":
    main()
