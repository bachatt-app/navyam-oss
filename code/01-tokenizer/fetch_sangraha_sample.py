#!/usr/bin/env python3
"""Pull a fresh, real (non-duplicated) Indic-text sample from AI4Bharat
Sangraha via the HuggingFace datasets-server rows API, WITHOUT downloading
the full multi-hundred-GB dataset.

Why this exists: PER_SCRIPT_TARGETS.md's #2 recommendation for a real v0.5
is "real additional Dravidian corpus... not duplication" (AI4Bharat
IndicCorp/Sangraha). A full `datasets.load_dataset("ai4bharat/sangraha")`
pull is 10GB+ per language and not feasible on this CPU-only box in this
session. The datasets-server `/rows` endpoint instead serves paginated JSON
slices of the same underlying parquet shards without a bulk download --
this script uses it to grab a bounded-size (`--target-mb` per language)
sample of REAL Sangraha "verified" (non-synthetic) web/document text for
each language in `LANGS`, so BPE training sees new word-forms/compounds,
not more copies of what's already in corpus_balanced/.

This is still a small sample relative to the 10-50GB production sweep the
project plan calls for -- see the honesty notes in
v05_candidate_comparison.md. It IS real, freshly-sourced, non-duplicated
text, which is the specific gap PER_SCRIPT_TARGETS.md flagged.

Usage:
  python fetch_sangraha_sample.py --out corpus_sangraha --target-mb 3
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "ai4bharat/sangraha"
CONFIG = "verified"
PAGE = 100

# Sangraha "verified" split code -> our corpus filename tag (matches
# corpus_balanced/indic_<tag>.txt naming where practical).
LANGS = {
    "hin": "hin", "ben": "ben", "mar": "mar", "guj": "guj", "pan": "pan",
    "ori": "ori", "asm": "asm", "tam": "tam", "tel": "tel", "kan": "kan",
    "mal": "mal", "eng": "eng",
}


def fetch_split(split: str, target_bytes: int, out_path: str) -> tuple[int, int]:
    total_bytes = 0
    n_docs = 0
    offset = 0
    with open(out_path, "w", encoding="utf-8") as out:
        while total_bytes < target_bytes:
            url = f"{API}?dataset={DATASET}&config={CONFIG}&split={split}&offset={offset}&length={PAGE}"
            req = urllib.request.Request(url, headers={"User-Agent": "navyam-tokenizer-research/0.5"})
            data = None
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.load(resp)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait = 5 * (attempt + 1)
                        print(f"  [{split}] 429 at offset={offset}, backing off {wait}s (attempt {attempt+1}/6)")
                        time.sleep(wait)
                        continue
                    print(f"  [{split}] stopped at offset={offset}: {e}")
                    data = None
                    break
                except (urllib.error.URLError, TimeoutError) as e:
                    print(f"  [{split}] stopped at offset={offset}: {e}")
                    data = None
                    break
            if data is None:
                break
            rows = data.get("rows", [])
            if not rows:
                print(f"  [{split}] exhausted at offset={offset}")
                break
            for r in rows:
                text = r["row"].get("text", "")
                if not text:
                    continue
                out.write(text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n")
                total_bytes += len(text.encode("utf-8"))
                n_docs += 1
            offset += PAGE
            if total_bytes >= target_bytes:
                break
            time.sleep(0.6)  # be polite to the shared API
    return n_docs, total_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_sangraha")
    ap.add_argument("--target-mb", type=float, default=3.0,
                     help="approx MB of raw text to pull per language")
    ap.add_argument("--langs", nargs="*", default=list(LANGS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target_bytes = int(args.target_mb * 1_000_000)

    manifest = {}
    for split in args.langs:
        tag = LANGS[split]
        out_path = os.path.join(args.out, f"sangraha_{tag}.txt")
        print(f"fetching {DATASET}/{CONFIG}/{split} -> {out_path} (target {args.target_mb}MB)")
        n_docs, n_bytes = fetch_split(split, target_bytes, out_path)
        manifest[split] = {"docs": n_docs, "bytes": n_bytes, "path": out_path}
        print(f"  [{split}] {n_docs} docs, {n_bytes/1e6:.2f}MB")

    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "https://huggingface.co/datasets/ai4bharat/sangraha "
                       "(config=verified, real non-synthetic web/document text), "
                       "fetched via datasets-server /rows API (no bulk download)",
            "per_split": manifest,
        }, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}/MANIFEST.json")


if __name__ == "__main__":
    main()
