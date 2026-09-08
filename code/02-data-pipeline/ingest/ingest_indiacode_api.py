#!/usr/bin/env python3
"""S050: India Code bare acts via the official DSpace REST API.

The 2026 India Code migration replaced the server-rendered site with an
Angular SPA (HTML crawling gets an empty shell) backed by DSpace 9.1's
documented public REST API at /server/api — the machine interface the
repository itself publishes. Items carry a TEXT bundle with extracted
plain text, so no PDF stage is needed for most acts.

Output rows: {id, handle, title, year, status, text, source} — version-aware
dedup downstream keys on (handle, year, status), never on title alone.

Usage:
  python ingest_indiacode_api.py --out pools/india_tax_authoritative/raw_indiacode_api.jsonl \
      [--max-items 2000] [--delay 0.5] [--query ""]
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

BASE = "https://indiacode.gov.in/server/api"
UA = {"User-Agent": "navyam-research/0.1 (contact: bachattapp@gmail.com)",
      "Accept": "application/json"}


def get(url, timeout=40, raw=False, tries=4):
    """GET with retries; the server drops chunked responses mid-stream."""
    import http.client
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            return data if raw else json.loads(
                data.decode("utf-8", "replace"))
        except http.client.IncompleteRead as e:
            # partial body: fine for raw text, retry for JSON
            if raw and len(e.partial) > 0:
                return e.partial
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def meta_first(md, key):
    v = md.get(key, [])
    return v[0]["value"] if v else ""


def item_text(uuid, delay):
    """Fetch the TEXT bundle's first bitstream content, if any."""
    bundles = get(f"{BASE}/core/items/{uuid}/bundles")
    for b in bundles.get("_embedded", {}).get("bundles", []):
        if b.get("name") != "TEXT":
            continue
        time.sleep(delay)
        bits = get(b["_links"]["bitstreams"]["href"])
        for bit in bits.get("_embedded", {}).get("bitstreams", []):
            time.sleep(delay)
            raw = get(f"{BASE}/core/bitstreams/{bit['uuid']}/content",
                      raw=True, timeout=90)
            text = raw.decode("utf-8", "replace").strip()
            if len(text) > 200:
                return text
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=2000)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--query", default="")
    ap.add_argument("--min-chars", type=int, default=600)
    args = ap.parse_args()

    seen = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            seen.add(json.loads(line)["handle"])
    out = open(args.out, "a", encoding="utf-8")

    page, kept, scanned = 0, 0, 0
    while scanned < args.max_items:
        q = urllib.parse.quote(args.query) if args.query else "*"
        url = (f"{BASE}/discover/search/objects?query={q}&dsoType=item"
               f"&size=20&page={page}")
        try:
            d = get(url)
        except Exception as e:  # noqa: BLE001
            print(f"page {page} failed: {e}; stopping")
            break
        objs = (d.get("_embedded", {}).get("searchResult", {})
                 .get("_embedded", {}).get("objects", []))
        if not objs:
            break
        for o in objs:
            scanned += 1
            if scanned > args.max_items:
                break
            it = o.get("_embedded", {}).get("indexableObject", {})
            handle = it.get("handle", "")
            if not handle or handle in seen:
                continue
            md = it.get("metadata", {})
            try:
                time.sleep(args.delay)
                text = item_text(it["uuid"], args.delay)
            except Exception as e:  # noqa: BLE001
                print(f"  {handle}: text fetch failed ({e})")
                continue
            if len(text) < args.min_chars:
                continue
            out.write(json.dumps({
                "id": f"indiacode-{handle.replace('/', '-')}",
                "handle": handle,
                "title": it.get("name", ""),
                "year": meta_first(md, "dc.date.issued")
                        or meta_first(md, "dspace.entity.year"),
                "status": meta_first(md, "dc.description.status"),
                "text": text,
                "source": "S050-indiacode-api",
            }, ensure_ascii=False) + "\n")
            out.flush()
            seen.add(handle)
            kept += 1
            if kept % 25 == 0:
                print(f"  kept {kept} / scanned {scanned}")
        page += 1
    print(f"done: kept {kept}, scanned {scanned} -> {args.out}")


if __name__ == "__main__":
    main()
