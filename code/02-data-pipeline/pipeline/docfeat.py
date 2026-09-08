#!/usr/bin/env python3
"""Shared per-document features — compute the expensive derivations ONCE.

A surviving document used to be split/scanned ~6 times across the cleaning
stages (langid, 8 filters, exact-dedup normalize, 13-gram decontamination,
MinHash shingling). Every stage now accepts an optional DocFeatures and
derives its inputs from it; standalone/CLI use still works without one.

Only derivations shared by 2+ stages live here; stage-specific work
(script profiling, shingle hashing) stays in the stage.
"""


class DocFeatures:
    __slots__ = ("text", "words", "n_words", "lower_text", "lower_words",
                 "lines", "norm_text")

    def __init__(self, text: str):
        self.text = text
        self.words = text.split()
        self.n_words = len(self.words)
        self.lower_text = text.lower()
        self.lower_words = self.lower_text.split()
        self.lines = [l.strip() for l in text.splitlines() if l.strip()]
        # dedup_exact's normalization: whitespace-collapsed lowercase
        self.norm_text = " ".join(self.lower_words)
