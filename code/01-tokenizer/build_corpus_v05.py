#!/usr/bin/env python3
"""Build the v0.5 training corpora from corpus_balanced/ (existing
mix-weighted sample) + corpus_sangraha/ (fresh, non-duplicated AI4Bharat
Sangraha "verified" text pulled by fetch_sangraha_sample.py).

PER_SCRIPT_TARGETS.md's #2 recommendation for a real v0.4/v0.5 was: "Real
additional Dravidian corpus, not duplication... to give the BPE trainer new
word-forms and compounds to merge on, not just more copies of the ~2-4MB
already in corpus_balanced/." This script does exactly that: it MERGES (not
duplicates) corpus_balanced/indic_<lang>.txt with corpus_sangraha/sangraha_
<lang>.txt for every language both cover, so BPE sees new lexical material,
not more frequency weight on the same text.

Produces two output directories, matching PER_SCRIPT_TARGETS.md's own
finding that "vocab size increase alone... has no measured downside" while
oversampling should be "modest (not 3x)":

  corpus_v05_balanced/    every file merged 1x -- candidate A (80k) and
                           candidate C (96k Unigram) train on this.
  corpus_v05_scriptaware/ same merge, but Dravidian files (tam/tel/kan/mal)
                           get a further MODEST 1.5x weight (not 3x, per the
                           v0.4 finding that 3x measurably hurts every other
                           Indic script at a fixed vocab budget) -- candidate
                           B (96k) trains on this, pairing extra vocab budget
                           with a light thumb on the scale rather than
                           relying on either alone.

All output is run through the same normalization used at tokenizer training
time (NFC, CRLF->LF, BOM strip) via normalize_text.py, so the corpus on disk
already matches what train_tokenizer_v05.py's own normalizer will produce
-- keeping corpus-prep and training normalization consistent, as required
by the v0.5 spec.

Usage:
  python build_corpus_v05.py --balanced corpus_balanced --sangraha corpus_sangraha \
      --out-balanced corpus_v05_balanced --out-scriptaware corpus_v05_scriptaware
"""
import argparse
import glob
import os

from normalize_text import normalize_corpus_text

DRAVIDIAN = {"tam", "tel", "kan", "mal"}
MODEST_OVERSAMPLE = 1.5  # PER_SCRIPT_TARGETS.md: "modest (not 3x)"

# corpus_balanced/indic_<tag>.txt <-> corpus_sangraha/sangraha_<tag>.txt
LANG_TAGS = ["hin", "ben", "mar", "guj", "pan", "ori", "asm", "tam", "tel", "kan", "mal"]


def read_norm(path: str) -> str:
    return normalize_corpus_text(open(path, encoding="utf-8").read())


def write_weighted(text: str, weight: float, out_f) -> None:
    """Write `text` `weight` times (fractional part = that prefix fraction
    of the text, so 1.5x = full text + first half again)."""
    whole = int(weight)
    frac = weight - whole
    for _ in range(whole):
        out_f.write(text)
        if not text.endswith("\n"):
            out_f.write("\n")
    if frac > 0:
        cut = int(len(text) * frac)
        out_f.write(text[:cut])
        out_f.write("\n")


def build(balanced_dir: str, sangraha_dir: str, out_dir: str, oversample_dravidian: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Non-Indic domain files (code, math, english, finance) copied
    #    through unchanged (normalized) -- Sangraha has no equivalent.
    for path in sorted(glob.glob(os.path.join(balanced_dir, "*.txt"))):
        fname = os.path.basename(path)
        tag = fname.replace("indic_", "").replace(".txt", "")
        if tag in LANG_TAGS:
            continue  # handled below (merged with Sangraha)
        text = read_norm(path)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as out:
            out.write(text)

    # 1b. English gets the Sangraha English sample merged in too (real,
    #     fresh web text, same treatment as the Indic languages).
    eng_sangraha_path = os.path.join(sangraha_dir, "sangraha_eng.txt")
    global_eng_path = os.path.join(balanced_dir, "global_english.txt")
    if os.path.exists(eng_sangraha_path) and os.path.exists(global_eng_path):
        merged = read_norm(global_eng_path) + "\n" + read_norm(eng_sangraha_path)
        with open(os.path.join(out_dir, "global_english.txt"), "w", encoding="utf-8") as out:
            out.write(merged)

    # 2. Indic languages: MERGE corpus_balanced + fresh Sangraha (not
    #    duplication), then apply the modest Dravidian oversample if asked.
    for tag in LANG_TAGS:
        bal_path = os.path.join(balanced_dir, f"indic_{tag}.txt")
        san_path = os.path.join(sangraha_dir, f"sangraha_{tag}.txt")
        parts = []
        if os.path.exists(bal_path):
            parts.append(read_norm(bal_path))
        if os.path.exists(san_path):
            parts.append(read_norm(san_path))
        if not parts:
            continue
        merged = "\n".join(parts)
        out_path = os.path.join(out_dir, f"indic_{tag}.txt")
        weight = MODEST_OVERSAMPLE if (oversample_dravidian and tag in DRAVIDIAN) else 1.0
        with open(out_path, "w", encoding="utf-8") as out:
            write_weighted(merged, weight, out)
        print(f"{tag:<5} balanced={os.path.exists(bal_path)} sangraha={os.path.exists(san_path)} "
              f"weight={weight} -> {out_path} ({os.path.getsize(out_path)/1e6:.2f}MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", default="corpus_balanced")
    ap.add_argument("--sangraha", default="corpus_sangraha")
    ap.add_argument("--out-balanced", default="corpus_v05_balanced")
    ap.add_argument("--out-scriptaware", default="corpus_v05_scriptaware")
    args = ap.parse_args()

    if not os.path.isdir(args.balanced):
        raise SystemExit(f"{args.balanced!r} not found")

    print(f"== building {args.out_balanced} (merged, no oversample) ==")
    build(args.balanced, args.sangraha, args.out_balanced, oversample_dravidian=False)

    print(f"\n== building {args.out_scriptaware} (merged + {MODEST_OVERSAMPLE}x Dravidian) ==")
    build(args.balanced, args.sangraha, args.out_scriptaware, oversample_dravidian=True)


if __name__ == "__main__":
    main()
