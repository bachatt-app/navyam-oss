# Pipeline speed plan — land BEFORE the navya-1b ingest (8–15B tokens)

Status (2026-08-19, second pass): items 1, 2, 5, 6, 7 LANDED and verified
(shared DocFeatures, fail-fast filters with 1/1024 audit sample, stats from
cleaning sidecars, bounded shard prefetch, per-host crawl queues — pipeline
order v3, output byte-identical to v2 on the test pool, ~11% on a
MinHash-dominated synthetic; real pools with filter rejections gain more).
Item 8's harness exists (bench_pipeline.sh — run on the ingest VM before
sizing the 1b box). 2026-08-20 third pass: 3, 4, 9 LANDED.
Item 3 — numpy minhash kernel in dedup_global (exact Mersenne modmul in
uint64 pieces; BIT-IDENTICAL band keys, verified by `--self-test`; 4.9x on
800-word docs). A Rust kernel remains optional headroom on top.
Item 4 — shard_ingest tokenize loop now collects per-doc uint16 arrays and
writes one concatenate per flush (the giant python-int buffer is gone).
Item 9 — build_stream --virtual: no train.bin write; trainer reads windows
through a segment manifest (_VirtualTrain). stream_id is representation-
independent (hash of the identical byte sequence) and equivalence is a
permanent 13th check in tests/test_stack.py.
Also fixed in passing: decontaminate n-gram hashing used the salted builtin
hash() — correct under fork, silently matched NOTHING under spawn; now
stable blake2b-64. The navya-1a
corpus was built on the current stack; at 1b scale (~3× the data) these
become the difference between an overnight run and a multi-day one.
Training-side optimizations (chunked CE, GPU-side loss accumulation, native
GQA, fused AdamW, compile) already landed in 04-training-stack.

Ordered by expected ROI:

1. **Shared per-document features.** A surviving doc is currently split/
   scanned ~5× (langid, filters, exact-dedup normalize, 13-gram decontam,
   MinHash shingles). Build one feature object (normalized text, lowercase
   words, lines, counts, reusable hashes) in the worker and pass it to every
   stage. Acceptance: identical clean.jsonl output, measured docs/s gain.
2. **Fast-fail filtering.** filters.judge evaluates all 8 Gopher filters to
   collect reasons; production cleaning should run cheap/high-rejection
   filters first and return on first failure, keeping full-reason evaluation
   only for an audit sample (e.g. 1 in 1000). Output-equivalent.
3. **Native MinHash + decontam hashing (Rust).** dedup_global builds every
   5-word shingle string and scans the shingle set 16×. A native kernel:
   tokenize once, rolling hashes (no joined strings), all 16 minima in one
   traversal, uint64 band keys, cache-friendly table. PRESERVE current hash
   semantics initially so outputs can be diffed exactly.
4. **Native tokenize-and-write.** tokenize_pool crosses the Rust→Python
   boundary per token (billions of PyObjects). A small native writer:
   JSON text → tokenizer → +EOS → uint16 file, no Python ints.
5. **Stats during cleaning.** Pool statistics currently re-read and re-split
   every cleaned doc; accumulate them in the cleaning pass (sidecar already
   carries totals — extend with words/chars).
6. **Bounded prefetch I/O.** ingest_hf_parquet downloads shard N, parses,
   then downloads N+1; prefetch next shard during parse, and download
   independent sources concurrently (small concurrency cap). Keep domain
   CLEANING order deterministic for global dedup.
7. **Per-host crawl queues.** ingest_gov_in is one global queue; per-host
   queues let different hosts overlap while preserving per-host delay and
   robots policy.
8. **Worker/chunk sweep with RSS.** Before choosing the 1B ingest VM, sweep
   workers {4,8,12,16} × chunk-bytes settings; record MB/s, docs/s, parent
   CPU, worker CPU, peak RSS, queue idle. Parent idle → bigger VM helps;
   parent saturated → fix state/IPC first.
9. **Virtual stream manifest.** Stream construction physically copies domain
   bins into a 16–30GB mixed file; let the trainer consume weighted domain
   shards through a manifest (deterministic per-domain window schedule)
   instead. Removes a full-data write+read per corpus build.

Also open (from earlier reviews, same cycle): varlen/Flash attention for SFT
packing vs dense B×T×T masks; auditable reviewer records for SFT provenance;
structured tool-call training data; larger decorrelated SFT holdout; DDP
(2–4 GPUs, no_sync accumulation) for 1b wall-clock; tokenizer 32/48/64/96K
study before Alpha.

## Pre-1b hyperparameter gate — CLOSED 2026-08-20

LR sweep complete on the A100 (six points, 400M tokens each, multi-signal
selection): 2e-4 3.7397 / 4e-4 3.4636 / 6e-4 3.4171 / 8e-4 3.3476 /
**1e-3 3.3440 (adopted)** / 1.2e-3 3.4182 — optimum bracketed both sides;
1e-3 also won slope, stability, and per-domain (Indic decisively).
Launch chain updated to lr=1e-3. A batch-size axis (262K/524K/1M) remains
optional if budget allows.
