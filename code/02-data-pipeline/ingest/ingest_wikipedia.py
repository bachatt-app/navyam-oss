#!/usr/bin/env python3
"""Ingest plaintext article extracts from a Wikipedia via the MediaWiki API.

Two modes:
  --titles file.txt   curated article list (finance, math), one title per line
  --random N          N random main-namespace articles (bulk language text)

Wikipedia text is CC BY-SA 4.0 — attribution and share-alike recorded in the
legal register row given by --source-id.

Usage:
  python ingest_wikipedia.py --lang en --titles titles_finance_en.txt \
         --source-id S020 --out pools/finance_econ_law/wiki_en.jsonl
  python ingest_wikipedia.py --lang hi --random 200 \
         --source-id S021 --out pools/indian_languages/wiki_hi.jsonl
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

UA = "bachatt-phase0-research/0.1 (data pipeline; contact: bachattapp@gmail.com)"
# TextExtracts returns only ONE full-text extract per request (exlimit is
# effectively 1 unless exintro is set), so fetch page-by-page.
MIN_CHARS = 300       # drop stubs


def api(lang: str, params: dict) -> dict:
    base = {"action": "query", "format": "json", "formatversion": 2,
            "prop": "extracts", "explaintext": 1, "exlimit": "max"}
    base.update(params)
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(base)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise SystemExit(f"giving up on {lang}.wikipedia: {e}")
            time.sleep(2 ** attempt)
    return {}


def emit(pages, lang, source_id, f, seen: set) -> int:
    n = 0
    for p in pages:
        text = p.get("extract") or ""
        title = p.get("title", "")
        if len(text) < MIN_CHARS or title in seen:
            continue
        seen.add(title)
        f.write(json.dumps({
            "id": f"wikipedia/{lang}/{p.get('pageid')}",
            "source": source_id,
            "url": f"https://{lang}.wikipedia.org/wiki/"
                   + urllib.parse.quote(title.replace(" ", "_")),
            "title": title,
            "text": text,
        }, ensure_ascii=False) + "\n")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--titles", help="file of article titles, one per line")
    ap.add_argument("--random", type=int, help="number of random articles")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.titles and not args.random:
        ap.error("need --titles or --random")

    seen: set = set()
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        if args.titles:
            titles = [t.strip() for t in open(args.titles, encoding="utf-8")
                      if t.strip() and not t.startswith("#")]
            for title in titles:
                data = api(args.lang, {"titles": title, "redirects": 1})
                written += emit(data.get("query", {}).get("pages", []),
                                args.lang, args.source_id, f, seen)
                time.sleep(0.2)
        else:
            while written < args.random:
                data = api(args.lang, {
                    "generator": "random", "grnnamespace": 0,
                    "grnlimit": 1})
                written += emit(data.get("query", {}).get("pages", []),
                                args.lang, args.source_id, f, seen)
                print(f"  {args.lang}wiki random: {written}", end="\r",
                      file=sys.stderr)
                time.sleep(0.3)
    print(f"{args.lang}.wikipedia -> {written} docs -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
