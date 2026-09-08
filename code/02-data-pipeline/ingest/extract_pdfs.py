#!/usr/bin/env python3
"""Extract Supreme Court judgment text from the open-data PDF tree (S054).

Joins each PDF with the metadata parquet (case title, decision date,
categories) and emits pipeline-ready JSONL with a visible metadata header
and version_key. Parallel across processes; text-layer extraction only —
scanned/image-only PDFs are SKIPPED and counted (OCR is a listed backlog
item, not a silent gap).

Sensitive-category exclusion happens at the metadata join, BEFORE any
extraction; finance/tax/corporate filtering is optional (--all keeps every
non-excluded case — for the full-corpus build).

Usage (on the ingest VM):
  python extract_pdfs.py --pdf-root /mnt/data/bulk_cache/sc_judgments/pdf \
      --meta bulk_cache/sc_judgments/metadata_parquet \
      --out pools/india_case_law/raw_sc.jsonl [--all] [--workers 3]
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys

import pyarrow.parquet as pq

KEEP = re.compile(r"income.?tax|gst|excise|customs|sebi|company|companies|"
                  r"banking|negotiable|insolvency|ibc|arbitration|rbi|"
                  r"insurance|provident fund|stamp|money.?laundering|"
                  r"contract|tax", re.I)
EXCLUDE = re.compile(r"matrimonial|custody|guardianship|adoption|juvenile|"
                     r"rape|pocso|sexual|domestic violence|divorce", re.I)

_META = {}


def load_meta(meta_dir):
    """Join key: basename of the metadata `path` column (the PDF path)."""
    meta = {}
    cols = ["title", "petitioner", "respondent", "description", "judge",
            "citation", "case_id", "decision_date", "disposal_nature",
            "path", "year"]
    for p in glob.glob(os.path.join(meta_dir, "**", "*.parquet"),
                       recursive=True):
        t = pq.read_table(p, columns=cols).to_pylist()
        for r in t:
            if r.get("path"):
                meta[os.path.basename(str(r["path"]))] = r
    return meta


def extract_one(pdf_path):
    import fitz  # PyMuPDF — imported in worker
    base = os.path.basename(pdf_path)
    # metadata path is "1950_1_806_821"; PDF is "1950_1_806_821_EN.pdf"
    stem = re.sub(r"_[A-Z]{2,3}\.pdf$", "", base)
    rec = _META.get(stem, {})
    probe = " ".join(str(rec.get(k) or "") for k in
                     ("title", "petitioner", "respondent", "description",
                      "disposal_nature"))
    if EXCLUDE.search(probe):
        return ("sensitive", None)
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
    except Exception:
        return ("corrupt", None)
    text = "\n".join(pages).strip()
    # scanned PDFs yield near-empty text layers — skip, never OCR silently
    if len(text) < 200 * max(1, len(pages)) * 0.2 or len(text) < 1500:
        return ("scanned_or_short", None)
    title = rec.get("title") or base
    date = str(rec.get("decision_date") or "")
    cite = str(rec.get("citation") or "")
    case_id = str(rec.get("case_id") or base)
    header = (f"[Supreme Court of India | {title} | {cite} | "
              f"decided {date} | {case_id}]\n\n")
    return ("ok", {"text": header + text, "source": "S054",
                   "authority": "Supreme Court", "doc_id": case_id,
                   "date": date, "status": "judgment", "citation": cite,
                   "title": str(title), "probe": probe,
                   "version_key": f"{case_id}|{date}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true",
                    help="keep every non-excluded case (default: finance/"
                         "tax/corporate KEEP filter)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    global _META
    print("loading metadata ...", file=sys.stderr)
    _META = load_meta(args.meta)
    print(f"{len(_META)} metadata records", file=sys.stderr)

    # english/ subdirs only — regional/ holds translated duplicates
    pdfs = sorted(glob.glob(os.path.join(args.pdf_root, "**", "english",
                                         "*.pdf"), recursive=True))
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f"{len(pdfs)} PDFs to extract", file=sys.stderr)

    counts = {"ok": 0, "sensitive": 0, "scanned_or_short": 0,
              "corrupt": 0, "offtopic": 0}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with mp.Pool(args.workers) as pool, \
            open(args.out + ".part", "w", encoding="utf-8") as out:
        for status, doc in pool.imap_unordered(extract_one, pdfs,
                                               chunksize=16):
            if status == "ok":
                if not args.all and not KEEP.search(
                        doc["probe"] + " " + doc["text"][:3000]):
                    counts["offtopic"] += 1
                    continue
                doc.pop("probe", None)
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                counts["ok"] += 1
                if counts["ok"] % 500 == 0:
                    print(f"  {counts}", file=sys.stderr)
            else:
                counts[status] += 1
    os.replace(args.out + ".part", args.out)
    print(f"extract_pdfs: {counts} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
