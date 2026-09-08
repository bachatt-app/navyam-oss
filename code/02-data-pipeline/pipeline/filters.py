#!/usr/bin/env python3
"""Heuristic quality filters (Gopher/FineWeb-style), v1.

Reference implementation: develops the logic that later gets ported to the
distributed runner and eventually replaced by a trained quality classifier
(exit criterion: classifier beats these heuristics on downstream 1B evals).

Each filter returns None if the doc passes, or a short reason string.
`judge(doc)` returns (keep: bool, reasons: list[str]).

Usage as a stage:
  python filters.py < in.jsonl > out.jsonl        # keeps passing docs
  python filters.py --rejects rej.jsonl < in.jsonl > out.jsonl
"""

import argparse
import json
import sys

from docfeat import DocFeatures

# Tunable thresholds — every change is an experiment in 05-research-models.
MIN_WORDS = 50
MAX_WORDS = 200_000
MIN_MEAN_WORD_LEN = 2.0
MAX_MEAN_WORD_LEN = 12.0
MAX_SYMBOL_WORD_RATIO = 0.10      # '#' and '...' per word
MAX_BULLET_LINE_FRAC = 0.90
MAX_ELLIPSIS_LINE_FRAC = 0.30
MAX_DUP_LINE_FRAC = 0.30
MAX_TOP_BIGRAM_FRAC = 0.18
MIN_ALPHA_WORD_FRAC = 0.60        # words containing at least one letter

# Small stopword lists: a doc claiming to be in language X should contain some.
STOPWORDS = {
    "english": {"the", "and", "of", "to", "in", "is", "that", "for", "with"},
    # Devanagari maps to "hindi" in langid but covers Marathi (and Nepali,
    # Sanskrit-register text) too — the set is a Hindi+Marathi UNION, else
    # the stopword filter wrongly kills 90%+ of Marathi documents
    # (observed: sangraha-mar shard 0, 819K/873K rejected).
    "hindi": {"है", "और", "के", "में", "से", "का", "की", "पर", "यह",
              "आणि", "आहे", "या", "ते", "हे", "व", "मध्ये", "एक", "आहेत"},
    "hinglish": {"hai", "aur", "ke", "mein", "se", "ka", "ki", "par", "kya"},
}
MIN_STOPWORD_HITS = 2


def f_length(doc, ft):
    n = ft.n_words
    if n < MIN_WORDS:
        return f"too_short:{n}"
    if n > MAX_WORDS:
        return f"too_long:{n}"


def f_mean_word_length(doc, ft):
    m = sum(len(w) for w in ft.words) / ft.n_words
    if not (MIN_MEAN_WORD_LEN <= m <= MAX_MEAN_WORD_LEN):
        return f"mean_word_len:{m:.1f}"


def f_symbol_ratio(doc, ft):
    symbols = ft.text.count("#") + ft.text.count("...")
    if symbols / ft.n_words > MAX_SYMBOL_WORD_RATIO:
        return "symbol_ratio"


def f_bullet_ellipsis(doc, ft):
    lines = ft.lines
    if not lines:
        return "empty"
    bullets = sum(l.startswith(("-", "*", "•")) for l in lines) / len(lines)
    ellipsis = sum(l.endswith(("...", "…")) for l in lines) / len(lines)
    if bullets > MAX_BULLET_LINE_FRAC:
        return "bullet_lines"
    if ellipsis > MAX_ELLIPSIS_LINE_FRAC:
        return "ellipsis_lines"


def f_dup_lines(doc, ft):
    # character-weighted (Gopher-style): repeated SHORT lines (headings,
    # formula fragments in math articles) shouldn't sink a document the way
    # repeated paragraphs should
    lines = ft.lines
    if not lines:
        return None
    seen: dict[str, int] = {}
    dup_chars = 0
    for l in lines:
        if l in seen:
            dup_chars += len(l)
        seen[l] = 1
    total = sum(len(l) for l in lines)
    if total and dup_chars / total > MAX_DUP_LINE_FRAC:
        return "dup_lines"


def f_top_bigram(doc, ft):
    words = ft.words
    if len(words) < 20:
        return None
    bigrams: dict[tuple, int] = {}
    for a, b in zip(words, words[1:]):
        bigrams[(a, b)] = bigrams.get((a, b), 0) + 1
    if max(bigrams.values()) * 2 / len(words) > MAX_TOP_BIGRAM_FRAC:
        return "repeated_bigram"


def f_alpha_words(doc, ft):
    alpha = sum(any(c.isalpha() for c in w) for w in ft.words) / ft.n_words
    if alpha < MIN_ALPHA_WORD_FRAC:
        return f"alpha_frac:{alpha:.2f}"


def f_stopwords(doc, ft):
    sw = STOPWORDS.get(doc.get("lang", ""))
    if not sw:
        return None  # no list for this language yet
    if len(set(ft.lower_words) & sw) < MIN_STOPWORD_HITS:
        return "few_stopwords"


# fail-fast order: cheapest / highest-rejection first. The original audit
# order (all filters, all reasons) is preserved for --rejects mode and the
# parallel cleaner's audit sample.
FILTERS = [f_length, f_mean_word_length, f_symbol_ratio, f_alpha_words,
           f_stopwords, f_bullet_ellipsis, f_dup_lines, f_top_bigram]


def judge(doc, feats=None, fail_fast=False) -> tuple[bool, list[str]]:
    ft = feats or DocFeatures(doc["text"])
    if not ft.n_words:
        return (False, ["too_short:0"])
    if fail_fast:
        for f in FILTERS:
            r = f(doc, ft)
            if r:
                return (False, [r])
        return (True, [])
    reasons = [r for f in FILTERS if (r := f(doc, ft))]
    return (not reasons, reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejects", help="optional JSONL for rejected docs+reasons")
    args = ap.parse_args()
    rej = open(args.rejects, "w", encoding="utf-8") if args.rejects else None

    kept = dropped = 0
    for line in sys.stdin:
        doc = json.loads(line)
        keep, reasons = judge(doc)
        if keep:
            kept += 1
            sys.stdout.write(line)
        else:
            dropped += 1
            if rej:
                doc["reject_reasons"] = reasons
                rej.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"filters: kept={kept} dropped={dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
