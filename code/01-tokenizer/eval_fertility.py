#!/usr/bin/env python3
"""Fertility evaluation: tokens per word, per language.

    fertility = #tokens / #words          (lower is better)

Exit criteria (execution plan, Workstream 2):
  - every language within 15% of English fertility (relative gap)
  - Indic scripts <= 1.8 tokens/word absolute

Eval sets live in eval_sets/<lang>.txt, one document per file. These must be
HELD-OUT text (never in tokenizer training data) for real measurements — the
bundled samples are smoke-test placeholders.

Usage:
  python eval_fertility.py --tokenizer tokenizer-160k.json
  python eval_fertility.py --tokenizer a.json --baseline b.json   # compare two
"""

import argparse
import glob
import os
import sys

from tokenizers import Tokenizer

RELATIVE_GAP_TARGET = 0.15   # within 15% of English
INDIC_ABS_TARGET = 1.8       # tokens/word for Indic-script languages

INDIC = {
    "hindi", "bengali", "marathi", "telugu", "tamil", "gujarati",
    "kannada", "malayalam", "punjabi", "odia", "assamese",
}
# Reported but not pass/fail-checked against the language targets
# (code fertility is judged against open-tokenizer baselines instead).
INFORMATIONAL = {"code"}


def fertility(tok: Tokenizer, text: str) -> tuple[float, int, int]:
    words = text.split()
    if not words:
        return 0.0, 0, 0
    n_tokens = len(tok.encode(text).ids)
    return n_tokens / len(words), n_tokens, len(words)


def measure(tok_path: str, eval_dir: str) -> dict[str, float]:
    tok = Tokenizer.from_file(tok_path)
    out = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.txt"))):
        lang = os.path.splitext(os.path.basename(path))[0]
        text = open(path, encoding="utf-8").read()
        f, _, _ = fertility(tok, text)
        out[lang] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--baseline", help="optional second tokenizer.json to compare")
    ap.add_argument("--json-out", help="also write results as JSON (for the dashboard)")
    ap.add_argument(
        "--eval-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_sets"),
    )
    args = ap.parse_args()

    ours = measure(args.tokenizer, args.eval_dir)
    base = measure(args.baseline, args.eval_dir) if args.baseline else None
    if "english" not in ours:
        sys.exit("eval_sets must include english.txt (the reference language)")

    if args.json_out:
        import json as _json
        _json.dump({"tokenizer": args.tokenizer,
                    "baseline": args.baseline,
                    "fertility": ours,
                    "baseline_fertility": base,
                    "targets": {"relative_gap": RELATIVE_GAP_TARGET,
                                "indic_abs": INDIC_ABS_TARGET},
                    "indic": sorted(INDIC),
                    "note": "eval_sets are smoke placeholders until real "
                            "held-out data lands (week-1 sprint item)"},
                   open(args.json_out, "w"), indent=1)

    eng = ours["english"]
    print(f"\n{'language':<12} {'fertility':>9} {'vs eng':>8} "
          f"{'baseline':>9} {'target':>8}  status")
    print("-" * 62)
    failures = 0
    for lang in sorted(ours, key=lambda l: (l != "english", l)):
        f = ours[lang]
        gap = (f - eng) / eng if eng else 0.0
        if lang in INFORMATIONAL:
            status, tgt = "info", "-"
        else:
            ok = gap <= RELATIVE_GAP_TARGET
            if lang in INDIC:
                ok = ok and f <= INDIC_ABS_TARGET
            status = "ok" if ok else "FAIL"
            failures += 0 if ok else 1
            tgt = (f"<={INDIC_ABS_TARGET}" if lang in INDIC
                   else f"+{RELATIVE_GAP_TARGET:.0%}")
        b = f"{base[lang]:9.2f}" if base else "        -"
        print(f"{lang:<12} {f:9.2f} {gap:+8.0%} {b} {tgt:>8}  {status}")

    print("-" * 62)
    print("NOTE: bundled eval_sets are smoke placeholders; conclusions require "
          "real held-out data.")
    if failures:
        print(f"{failures} language(s) miss targets")
        sys.exit(1)
    print("all languages within targets")


if __name__ == "__main__":
    main()
