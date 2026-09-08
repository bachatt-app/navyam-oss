#!/usr/bin/env python3
"""Build corpus_v04/: the corpus_balanced/ mix with Dravidian scripts
oversampled, for training tokenizer-v0.4-64k.json.

Every file in corpus_balanced/ is copied through unchanged EXCEPT
indic_tam.txt, indic_tel.txt, indic_mal.txt, indic_kan.txt (Tamil, Telugu,
Malayalam, Kannada), which are each concatenated OVERSAMPLE times to give
Dravidian text more frequency weight in BPE merge selection -- see
fertility_report_v4.json and PER_SCRIPT_TARGETS.md for why (agglutination
means the shared byte-level BPE budget under-serves these scripts at their
"fair share" of the corpus).

corpus_balanced/ itself is gitignored (regenerate via sample_corpus.py from
02-data-pipeline/pools); corpus_v04/ is derived from it the same way and is
also gitignored -- this script is the reproducible record of how.

Usage:
  python build_corpus_v04.py --src corpus_balanced --out corpus_v04
"""
import argparse
import os
import shutil

DRAVIDIAN_FILES = ["indic_tam.txt", "indic_tel.txt", "indic_mal.txt", "indic_kan.txt"]
OVERSAMPLE = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="corpus_balanced")
    ap.add_argument("--out", default="corpus_v04")
    ap.add_argument("--oversample", type=int, default=OVERSAMPLE)
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(
            f"{args.src!r} not found -- regenerate it first "
            "(python sample_corpus.py --pools ../02-data-pipeline/pools --out corpus_balanced)"
        )

    os.makedirs(args.out, exist_ok=True)
    for fname in sorted(os.listdir(args.src)):
        if not fname.endswith(".txt"):
            continue
        src_path = os.path.join(args.src, fname)
        dst_path = os.path.join(args.out, fname)
        if fname in DRAVIDIAN_FILES:
            text = open(src_path, encoding="utf-8").read()
            with open(dst_path, "w", encoding="utf-8") as f:
                f.write(text * args.oversample)
            print(f"{fname:<20} oversampled {args.oversample}x")
        else:
            shutil.copyfile(src_path, dst_path)
            print(f"{fname:<20} copied 1x")


if __name__ == "__main__":
    main()
