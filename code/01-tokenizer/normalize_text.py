#!/usr/bin/env python3
"""Shared text normalization for the v0.5 tokenizer.

IMPORTANT -- this exact function must run at THREE points, in this order,
or the tokenizer's vocabulary silently stops matching its input:

  1. corpus prep      (build_corpus_v05.py calls this before writing
                        corpus_v05_*/*.txt, so what BPE/Unigram trains on
                        is already normalized)
  2. tokenizer training (train_tokenizer_v05.py additionally installs
                        normalizers.NFC() as tok.normalizer, which the
                        `tokenizers` library then re-applies internally --
                        see the note in that file for why both matter)
  3. inference / serving (whatever calls tokenizer.encode() on raw user
                        text -- chat handler, data pipeline re-tokenizing
                        new corpus, eval harness -- MUST run raw text
                        through normalize_corpus_text() first, or through
                        an equivalent NFC + CRLF/BOM pass, before encoding.
                        A tokenizer.json with normalizers.NFC() set DOES
                        apply NFC automatically inside tok.encode(), but it
                        does NOT do the CRLF/BOM/repeated-invisible-char
                        cleanup below -- that part must run before encode()
                        wherever raw text enters the system.)

Skipping this at inference is the single most common way a "trained fine,
serves wrong" tokenizer bug shows up: text that was NFC-normalized at
training time (e.g. pre-composed nukta forms) meets un-normalized user text
(the same visual character, NFD-decomposed) at serving time, and every
merge trained on the composed form misses.

What this does, in order:
  1. Strip BOM (U+FEFF) wherever it appears, not just at file start --
     concatenated files can carry a BOM mid-stream.
  2. CRLF -> LF, lone CR -> LF.
  3. Unicode NFC normalization (canonical composition) -- this is the
     normalization form; see NFC_FORM below and the module docstring in
     train_tokenizer_v05.py for why NFC (not NFD/NFKC/NFKD) was chosen.
  4. Collapse runs of 2+ identical invisible/format characters down to a
     single occurrence -- garbage from OCR exports and WhatsApp-forwarded
     text often pads with repeated ZWSP/ZWJ/word-joiner/variation-selector
     characters that carry no meaning repeated, but a SINGLE ZWJ/ZWNJ next
     to Indic letters is semantically load-bearing (conjunct formation,
     chillu letters) and must NOT be touched -- see KEEP_SINGLE below.
"""
import re
import unicodedata

NFC_FORM = "NFC"

# Invisible / zero-width format characters that are legitimate ONE-AT-A-TIME
# (ZWJ/ZWNJ for Indic conjuncts and chillus, emoji ZWJ sequences, variation
# selectors, directional marks) but meaningless -- and a sign of copy-paste
# corruption -- when repeated back-to-back. Collapsing repeats to 1 never
# removes a legitimate single use.
_INVISIBLE_FORMAT_CODEPOINTS = [
    0x00AD,  # soft hyphen
    0x180E,  # Mongolian vowel separator
    0x200B,  # zero width space
    0x200C,  # ZWNJ (single = meaningful for Indic conjuncts; repeats = junk)
    0x200D,  # ZWJ   (single = meaningful for Indic conjuncts/emoji; repeats = junk)
    0x200E,  # LRM
    0x200F,  # RLM
    0x2060,  # word joiner
    0x2061, 0x2062, 0x2063, 0x2064,  # invisible times/plus/separator/function-apply
    0x061C,  # Arabic letter mark
    0xFEFF,  # BOM / zero width no-break space (mid-string occurrences)
    0xFE0E, 0xFE0F,  # text/emoji variation selectors
]
_INVISIBLE_CLASS = "".join(chr(c) for c in _INVISIBLE_FORMAT_CODEPOINTS)
_REPEATED_INVISIBLE_RE = re.compile(f"([{re.escape(_INVISIBLE_CLASS)}])\\1+")


def normalize_corpus_text(text: str) -> str:
    """Canonical normalization applied at corpus-prep, training, and
    (required) inference. See module docstring for why all three matter."""
    if not text:
        return text
    # 1. BOM anywhere in the stream, not just position 0.
    text = text.replace("﻿", "")
    # 2. Line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 3. NFC.
    text = unicodedata.normalize(NFC_FORM, text)
    # 4. Collapse repeated invisible/format characters (single occurrences
    #    of ZWJ/ZWNJ etc. are left untouched -- they are meaningful).
    text = _REPEATED_INVISIBLE_RE.sub(r"\1", text)
    return text


if __name__ == "__main__":
    import sys

    # quick self-check / CLI: normalize a file or stdin
    data = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    sys.stdout.write(normalize_corpus_text(data))
