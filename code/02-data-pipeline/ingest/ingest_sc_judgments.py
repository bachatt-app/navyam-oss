#!/usr/bin/env python3
"""Ingest Supreme Court of India judgments (S054) from the AWS Open Data set.

Official open-data release — synced with `aws s3 sync --no-sign-request`,
never scraped. Original judgment text only (no publisher headnotes). Triage
by metadata: keep finance/tax/corporate matters; EXCLUDE sensitive
categories (family disputes, victim-identifying matters) by case-type
before any text enters a pool.

Each doc carries the versioning metadata (S05x schema): authority, doc_id,
date, status — plus version_key so version-aware dedup keeps distinct
revisions (see parallel_clean).

Two-step usage on the ingest VM:
  aws s3 sync --no-sign-request s3://indian-supreme-court-judgments/  \
      bulk_cache/sc_judgments/ --exclude "*" --include "*.json"       # or per-year prefixes
  python ingest_sc_judgments.py --src bulk_cache/sc_judgments \
      --out pools/india_case_law/raw_sc.jsonl --max-docs 2000
"""

import argparse
import glob
import json
import os
import re
import sys

# case-type keywords for the finance/tax/corporate pilot slice
KEEP = re.compile(r"income.?tax|gst|excise|customs|sebi|company|companies|"
                  r"banking|negotiable|insolvency|ibc|arbitration.*commercial|"
                  r"rbi|insurance|provident fund|stamp duty|money.?laundering",
                  re.I)
# sensitive-category exclusion happens BEFORE any keep logic
EXCLUDE = re.compile(r"matrimonial|custody|guardianship|adoption|juvenile|"
                     r"rape|pocso|sexual|domestic violence|maintenance.*wife|"
                     r"divorce", re.I)


def doc_meta_text(rec):
    """Best-effort extraction across the dataset's record shapes."""
    text = rec.get("judgment_text") or rec.get("text") or ""
    title = rec.get("case_title") or rec.get("title") or ""
    cat = " ".join(str(rec.get(k, "")) for k in
                   ("case_type", "subject", "bench", "act", "category"))
    date = rec.get("judgment_date") or rec.get("date") or ""
    doc_id = rec.get("case_id") or rec.get("id") or title[:80]
    return text, title, cat, date, doc_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="synced open-data directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-id", default="S054")
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--min-chars", type=int, default=2000)
    args = ap.parse_args()

    kept = skipped_sensitive = skipped_offtopic = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(os.path.join(args.src, "**", "*.json"),
                                     recursive=True)):
            if kept >= args.max_docs:
                break
            try:
                rec = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            recs = rec if isinstance(rec, list) else [rec]
            for r in recs:
                if kept >= args.max_docs:
                    break
                text, title, cat, date, doc_id = doc_meta_text(r)
                probe = f"{title} {cat}"
                if EXCLUDE.search(probe):
                    skipped_sensitive += 1
                    continue
                if not KEEP.search(probe + " " + text[:2000]):
                    skipped_offtopic += 1
                    continue
                if len(text) < args.min_chars:
                    continue
                header = (f"[Supreme Court of India | {title} | "
                          f"decided {date} | {doc_id}]\n\n")
                out.write(json.dumps({
                    "text": header + text.strip(),
                    "source": args.source_id, "authority": "Supreme Court",
                    "doc_id": str(doc_id), "date": str(date),
                    "status": "judgment",
                    "version_key": f"{doc_id}|{date}",
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"sc_judgments: kept={kept} sensitive_excluded={skipped_sensitive} "
          f"offtopic={skipped_offtopic} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
