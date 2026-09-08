#!/usr/bin/env python3
"""Ingest rows from a Hugging Face dataset via the datasets-server rows API.

Laptop-scale ingestion for research pools — no auth, no parquet tooling,
polite pagination. At real scale this is replaced by bulk parquet downloads
on the distributed runner; the output document format is identical.

Usage:
  python ingest_hf_rows.py --dataset HuggingFaceFW/fineweb-edu \
         --config sample-10BT --split train --rows 3000 \
         --source-id S010 --out pools/global_english/raw.jsonl
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100          # API maximum
RETRIES = 8
PACE = 1.5          # seconds between page requests — the API rate-limits
UA = "bachatt-phase0-research/0.1 (data pipeline; contact: bachattapp@gmail.com)"


def fetch(url: str) -> dict:
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                wait = (float(retry_after) if retry_after
                        else min(120, 10 * (attempt + 1)))
                print(f"\n  429 rate-limited; waiting {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            last = e
            time.sleep(2 ** attempt)
    raise SystemExit(f"giving up on {url}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--source-id", required=True, help="legal register row ID")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for off in range(args.offset, args.offset + args.rows, PAGE):
            n = min(PAGE, args.offset + args.rows - off)
            q = urllib.parse.urlencode({
                "dataset": args.dataset, "config": args.config,
                "split": args.split, "offset": off, "length": n})
            data = fetch(f"{API}?{q}")
            rows = data.get("rows", [])
            if not rows:
                break
            for r in rows:
                row = r["row"]
                text = row.get(args.text_field) or ""
                if not text.strip():
                    continue
                doc = {
                    "id": f"{args.dataset}/{args.config}/{args.split}/{r['row_idx']}",
                    "source": args.source_id,
                    "url": row.get("url"),
                    "text": text,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {args.split}: {written} docs", end="\r", file=sys.stderr)
            time.sleep(PACE)
    print(f"{args.dataset}:{args.config}:{args.split} -> {written} docs "
          f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
