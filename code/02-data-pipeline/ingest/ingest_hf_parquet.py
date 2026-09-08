#!/usr/bin/env python3
"""Bulk-ingest a HuggingFace dataset via its auto-converted parquet files.

Replaces the polite rows-API fetcher at scale: downloads whole parquet shards
and streams record batches, stopping at --max-bytes of emitted text. Optional
--group-field concatenates consecutive rows sharing a key into one document
(for sentence-level datasets like financial-reports-sec).

Usage:
  python ingest_hf_parquet.py --dataset HuggingFaceFW/fineweb-edu \
      --config sample-10BT --split train --max-bytes 3200e6 \
      --source-id S010 --out pools/global_english/raw_bulk.jsonl
"""

import argparse
import threading
import json
import os
import sys
import tempfile
import time
import urllib.request

import pyarrow.parquet as pq

UA = "navyam-gpt-research/0.1 (bulk ingest; contact: bachattapp@gmail.com)"


def fetch_json(url):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise SystemExit(f"giving up on {url}: {e}")
            time.sleep(5 * (attempt + 1))


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--group-field", help="concat consecutive rows sharing this key")
    ap.add_argument("--meta-fields", default="", help="comma-separated extra fields to keep")
    ap.add_argument("--filter-field", help="keep only rows whose field is in --filter-values")
    ap.add_argument("--filter-values", default="", help="comma-separated allowed values")
    ap.add_argument("--max-bytes", type=float, required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="bulk_cache")
    args = ap.parse_args()

    urls = fetch_json("https://huggingface.co/api/datasets/"
                      f"{args.dataset}/parquet/{args.config}/{args.split}")
    if isinstance(urls, dict):
        raise SystemExit(f"unexpected listing: {str(urls)[:200]}")
    meta_fields = [m for m in args.meta_fields.split(",") if m]
    os.makedirs(args.cache, exist_ok=True)

    written = docs = 0
    group_key, group_parts = None, []

    def emit(f, text, extra=None):
        nonlocal written, docs
        text = text.strip()
        if not text:
            return
        doc = {"id": f"{args.dataset}/{args.config}/{args.split}/{docs}",
               "source": args.source_id, "text": text}
        if extra:
            doc.update(extra)
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        written += len(text.encode())
        docs += 1

    def shard_path(i):
        return os.path.join(args.cache,
                            f"{args.dataset.replace('/', '_')}_{args.config}_{i}.parquet")

    def fetch_shard(i):
        shard = shard_path(i)
        if not os.path.exists(shard):
            print(f"  downloading shard {i+1}/{len(urls)} ...", file=sys.stderr)
            tmp = shard + f".{os.getpid()}.part"
            download(urls[i], tmp)
            os.replace(tmp, shard)
        return shard

    # bounded prefetch: download shard i+1 while shard i parses. One thread,
    # one shard ahead — enough to hide download latency without hammering HF.
    prefetch: dict[int, threading.Thread] = {}

    def start_prefetch(i):
        if i < len(urls) and i not in prefetch:
            t = threading.Thread(target=lambda: fetch_shard(i), daemon=True)
            t.start()
            prefetch[i] = t

    with open(args.out, "w", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            if written >= args.max_bytes:
                break
            if i in prefetch:
                prefetch.pop(i).join()
            shard = fetch_shard(i)
            start_prefetch(i + 1)
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=2048):
                if written >= args.max_bytes:
                    break
                cols = batch.to_pydict()
                texts = cols[args.text_field]
                n = len(texts)
                allowed = set(v.strip() for v in args.filter_values.split(",") if v)
                for j in range(n):
                    if written >= args.max_bytes:
                        break
                    if args.filter_field and str(cols.get(args.filter_field, [None]*n)[j]) not in allowed:
                        continue
                    extra = {m: cols[m][j] for m in meta_fields if m in cols}
                    if args.group_field:
                        key = cols[args.group_field][j]
                        if key != group_key and group_parts:
                            emit(f, " ".join(group_parts))
                            group_parts = []
                        group_key = key
                        group_parts.append(str(texts[j]))
                    else:
                        emit(f, str(texts[j]), extra)
            print(f"  shard {i+1}: {docs} docs, {written/1e6:.0f} MB",
                  file=sys.stderr)
            os.remove(shard)   # keep the cache dir lean; parquet is re-downloadable
        if group_parts:
            emit(f, " ".join(group_parts))
    print(f"{args.dataset}:{args.config} -> {docs} docs, "
          f"{written/1e6:.0f} MB -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
