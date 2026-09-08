#!/usr/bin/env python3
"""Parallel cleaning driver — PIPELINE ORDER v2 (deliberate change from v1).

v1 (sequential pipe): langid | filters | exact-dedup | near-dedup | pii | decontam
v2 (this driver):     langid → filters → pii → decontam   [parallel workers]
                      exact-dedup → near-dedup             [serial parent]

The reordering is intentional, not an accident of parallelization:
  - rejected/contaminated/PII-dense documents no longer enter dedup state,
    so a bad document can never suppress a good near-duplicate later;
  - dedup keys are computed on PII-scrubbed text, so documents differing only
    in emails/phones correctly collapse to one.
Outputs therefore differ from v1 in edge cases. Streams built with this
driver record pipeline_order=v2 in their stderr summary; treat v1/v2 corpora
as different data versions.

Scaling design: workers do ALL per-doc compute — the four pure stages plus
the exact-dedup hash and the MinHash band signatures (shingling + 16 hash
scans per doc, the expensive part). The parent only does ordered set
membership checks and updates, so throughput scales with worker count.

Failure safety: accounting is validated BEFORE anything commits; then the
dedup state commits, then the output. The commit order is load-bearing: if a
crash lands between the two, clean.jsonl is older than its raw inputs, so the
next bulk_ingest run declares a fresh sweep, resets the dedup state, and
re-cleans everything — the inconsistent window self-heals. Missing eval files
fail closed (exit 1), never an empty contamination set.

Memory: state is 4 x 8-byte keys/doc but Python set/bytes overhead makes the
realistic cost ~250-350 bytes/doc — measure RSS before sizing the 1B-token VM.

Usage:
  python parallel_clean.py --inputs "pools/d/raw.jsonl pools/d/raw_bulk*.jsonl" \
      --evals "../03-evals/bachattbench/*.jsonl ../03-evals/decontam/*.jsonl" \
      --dedup-state pools/.dedup_global_state.bin \
      --out pools/d/clean.jsonl [--skip-filters] [--workers N]
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys

import decontaminate
import dedup_exact
import dedup_global
import filters
import langid
import pii_scrub
from docfeat import DocFeatures

PIPELINE_ORDER = "v3"
AUDIT_EVERY = 1024   # full-reason filter evaluation on 1/N of inputs
_G = {}   # per-worker globals — set by _init


def _init(banned, skip_filters):
    _G["banned"] = banned
    _G["skip_filters"] = skip_filters


def _work(args):
    """All per-doc compute: pure stages + dedup signatures, driven by ONE
    DocFeatures per doc (rebuilt only if PII scrubbing changed the text).
    Returns ([(json_line, exact_key, band_keys)...], stats)."""
    chunk_idx, lines = args
    out = []
    stats = {"in": len(lines), "filters": 0, "decontam": 0,
             "pii_dense": 0, "pii_scrubbed": 0, "audit": {}}
    for i, line in enumerate(lines):
        doc = json.loads(line)
        ft = DocFeatures(doc["text"])
        doc.update(langid.classify(doc["text"], feats=ft))
        if not _G["skip_filters"]:
            # audit sample keeps the all-reasons evaluation for tuning;
            # production path fails fast on the first rejection
            audit = (chunk_idx * 100003 + i) % AUDIT_EVERY == 0
            keep, reasons = filters.judge(doc, feats=ft, fail_fast=not audit)
            if audit:
                for r in reasons:
                    key = r.split(":")[0]
                    stats["audit"][key] = stats["audit"].get(key, 0) + 1
            if not keep:
                stats["filters"] += 1
                continue
        text, hits = pii_scrub.scrub(doc["text"])
        if hits > 0 and hits / max(1, len(text) / 1000) > 5:
            stats["pii_dense"] += 1
            continue
        if hits:
            doc["text"] = text
            ft = DocFeatures(text)         # text changed: rebuild features
            stats["pii_scrubbed"] += 1     # non-terminal: doc continues
        if any(g in _G["banned"]
               for g in decontaminate.ngrams(doc["text"], feats=ft)):
            stats["decontam"] += 1
            continue
        vkey = doc.get("version_key")
        ekey = dedup_exact.key(doc["text"] + ("\x00" + vkey if vkey else ""),
                               feats=None if vkey else ft)
        out.append((json.dumps(doc, ensure_ascii=False), ekey,
                    dedup_global.band_keys(doc["text"], feats=ft),
                    bool(vkey), ft.n_words, len(ft.text)))
    return out, stats


def read_chunks(paths, chunk_lines, chunk_bytes):
    buf, size = [], 0
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                buf.append(line)
                size += len(line)
            if len(buf) >= chunk_lines or size >= chunk_bytes:
                yield buf
                buf, size = [], 0
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="space-separated globs of input JSONL files")
    ap.add_argument("--evals", required=True)
    ap.add_argument("--dedup-state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-filters", action="store_true",
                    help="code domain: Gopher text filters do not apply")
    ap.add_argument("--skip-near-dedup", action="store_true",
                    help="sources deduplicated at origin: skip per-shard "
                         "MinHash (exact-dedup, PII, decontam still run)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--chunk-lines", type=int, default=8000)
    ap.add_argument("--chunk-bytes", type=float, default=8e6)
    args = ap.parse_args()

    paths = [f for pat in args.inputs.split() for f in sorted(glob.glob(pat))]
    if not paths:
        sys.exit(f"parallel_clean: no inputs match {args.inputs!r}")

    # fail closed: decontamination with an empty banned set is a silent hole
    efiles = sorted(f for pat in args.evals.split() for f in glob.glob(pat))
    if not efiles:
        sys.exit(f"parallel_clean: no eval files match {args.evals!r} — "
                 "refusing to run without decontamination")
    banned: set = set()
    for path in efiles:
        for text in decontaminate.eval_texts(path):
            banned.update(decontaminate.ngrams(text))
    if not banned:
        sys.exit("parallel_clean: eval files produced 0 banned n-grams — "
                 "refusing to run without decontamination")
    print(f"parallel_clean[{PIPELINE_ORDER}]: {len(banned)} banned n-grams "
          f"from {len(efiles)} files; {args.workers} workers", file=sys.stderr)

    dd_state: set = set()
    if os.path.exists(args.dedup_state):
        dd_state = pickle.load(open(args.dedup_state, "rb"))

    totals = {"in": 0, "filters": 0, "decontam": 0, "pii_dense": 0,
              "pii_scrubbed": 0, "exact_dup": 0, "near_dup": 0, "kept": 0,
              "kept_words": 0, "kept_chars": 0}
    audit_totals = {}
    exact_seen: set = set()
    out_part = args.out + ".part"
    ctx = mp.get_context("fork" if sys.platform != "darwin" else "spawn")
    try:
        with ctx.Pool(args.workers, initializer=_init,
                      initargs=(banned, args.skip_filters)) as pool, \
                open(out_part, "w", encoding="utf-8") as out:
            for surv, stats in pool.imap(
                    _work, enumerate(read_chunks(paths, args.chunk_lines,
                                                 int(args.chunk_bytes))),
                    chunksize=1):
                audit = stats.pop("audit")
                for k, v in audit.items():
                    audit_totals[k] = audit_totals.get(k, 0) + v
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0) + v
                # serial, order-dependent: membership checks + updates ONLY
                for line, ekey, bkeys, versioned, n_words, n_chars in surv:
                    if ekey in exact_seen:
                        totals["exact_dup"] += 1
                        continue
                    exact_seen.add(ekey)
                    if not args.skip_near_dedup and \
                            any(k in dd_state for k in bkeys):
                        totals["near_dup"] += 1
                        continue
                    if not versioned and not args.skip_near_dedup:
                        # versioned statute/judgment revisions must not be
                        # near-collapsed against their own later versions
                        dd_state.update(bkeys)
                    totals["kept"] += 1
                    totals["kept_words"] += n_words
                    totals["kept_chars"] += n_chars
                    out.write(line + "\n")
    except Exception:
        # leave previous clean.jsonl and dedup state untouched
        if os.path.exists(out_part):
            os.remove(out_part)
        raise
    # validate BEFORE committing anything: every input line has one fate
    kept_words = totals.pop("kept_words")
    kept_chars = totals.pop("kept_chars")
    terminal = (totals["filters"] + totals["pii_dense"] + totals["decontam"]
                + totals["exact_dup"] + totals["near_dup"] + totals["kept"])
    if terminal != totals["in"]:
        os.remove(out_part)
        sys.exit(f"parallel_clean: accounting mismatch {terminal} != "
                 f"{totals['in']} — nothing committed ({totals})")
    # commit order matters: state -> sidecar -> OUTPUT LAST (see docstring).
    # clean.jsonl is the commit point bulk_ingest's staleness check reads, so
    # everything it implies (state, sidecar) must already be on disk when it
    # appears; any earlier crash leaves output stale -> fresh sweep re-does all.
    tmp = args.dedup_state + ".part"
    with open(tmp, "wb") as f:
        pickle.dump(dd_state, f, protocol=4)
    os.replace(tmp, args.dedup_state)
    sidecar = {"pipeline_order": PIPELINE_ORDER,
               "stages": "langid,filters,pii_scrub,decontaminate | "
                         "dedup_exact,dedup_global (serial)",
               "skip_filters": args.skip_filters,
               "skip_near_dedup": args.skip_near_dedup,
               "workers": args.workers,
               "banned_ngrams": len(banned),
               "eval_files": efiles,
               "inputs": paths,
               "totals": totals,
               "kept_words": kept_words,
               "kept_chars": kept_chars,
               "filter_audit_sample": {"every": AUDIT_EVERY,
                                       "reasons": audit_totals}}
    meta_part = args.out + ".meta.json.part"
    with open(meta_part, "w") as f:
        json.dump(sidecar, f, indent=1)
    os.replace(meta_part, args.out + ".meta.json")
    os.replace(out_part, args.out)     # output commits LAST
    print(f"parallel_clean[{PIPELINE_ORDER}]: {totals}", file=sys.stderr)


if __name__ == "__main__":
    main()
