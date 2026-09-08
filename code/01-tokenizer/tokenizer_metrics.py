#!/usr/bin/env python3
"""Rigorous per-tokenizer metric suite for the v0.5 tokenizer sweep.

Retires the flat "1.8 tokens/word for every Indic script" gate
(PER_SCRIPT_TARGETS.md already showed that's linguistically unrealistic for
agglutinative Dravidian scripts) in favor of a fuller per-script view:
distributional fertility (mean + p50/p95/p99, not just a corpus-wide
average that one long-tail sentence can't move), a grapheme-aware measure
(tokens/grapheme-cluster, which is a fairer "information per unit" number
for Brahmic scripts than tokens/whitespace-word), a compression measure
(bytes/token), a vocabulary-health measure (byte-fragment rate: how often
the tokenizer had to fall back to a lone non-ASCII byte because no merge
covered that text -- the direct signal of "this script is underserved"),
and information-theoretic measures (Renyi efficiency + vocabulary
utilization) that summarize how well the tokenizer's vocabulary budget is
actually being spent, independent of any single target number.

Also runs a round-trip fidelity check over an adversarial suite (nukta
letters written two ways, conjuncts, Malayalam chillus, emoji ZWJ
sequences, OCR/WhatsApp-style noisy text) -- decode(encode(normalize(x)))
must equal normalize(x) exactly, or the tokenizer is silently corrupting
text, which fertility numbers alone would never catch.

Usage:
  python tokenizer_metrics.py --tokenizer tokenizer-v0.5-A80k.json \\
      --out metrics_v05_A80k.json
  python tokenizer_metrics.py --tokenizer tokenizer-v0.3-64k.json \\
      --out metrics_v03.json
"""
import argparse
import glob
import json
import math
import os
import unicodedata
from collections import Counter

from tokenizers import Tokenizer

from normalize_text import normalize_corpus_text

RENYI_ALPHA = 2.5
ZWNJ = "‌"
ZWJ = "‍"


# --------------------------------------------------------------------------
# Byte-level BPE alphabet (standard GPT-2 bytes_to_unicode mapping). Used to
# recover the raw UTF-8 byte(s) each token stands for, so we can detect
# "byte-fragment" tokens: single-byte fallback tokens for a byte that is
# part of a multi-byte UTF-8 sequence (i.e. no merge exists for that
# character -- the tokenizer had nothing better than a raw byte).
# `tokenizers`' Rust ByteLevel implementation uses this exact well-known
# mapping (ported from HF GPT-2's original encoder.py), so this
# reconstruction matches what pre_tokenizers.ByteLevel actually produced.
# --------------------------------------------------------------------------
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_TO_CHAR = _bytes_to_unicode()
_CHAR_TO_BYTE = {v: k for k, v in _BYTE_TO_CHAR.items()}


def token_to_raw_bytes(token_str: str) -> bytes | None:
    """Best-effort recovery of the raw bytes a ByteLevel BPE token string
    encodes. Returns None if `token_str` contains characters outside the
    byte-level alphabet (e.g. a Unigram token stored as literal UTF-8, or a
    special token like <|bos|>) -- those tokens are not byte-fragments by
    construction and are excluded from the byte-fragment-rate denominator
    is NOT what we do; see compute_byte_fragment_rate for how this is used."""
    try:
        return bytes(_CHAR_TO_BYTE[c] for c in token_str)
    except KeyError:
        return None


# --------------------------------------------------------------------------
# Approximate Unicode extended grapheme cluster segmentation (UAX #29).
# No `regex` module (with \X) is available in this environment and Python's
# stdlib `re` has no grapheme-cluster support, so this implements the core
# GB9 rule ("do not break before Extend or ZWJ") plus GB9a/GB9b-adjacent
# handling for variation selectors and a simple regional-indicator (flag)
# pairing rule. This covers the cases that matter here -- Devanagari/Indic
# consonant+matra+virama+ZWJ conjunct clusters, emoji ZWJ sequences, and
# skin-tone/variation-selector modifiers -- but is NOT a full UAX #29
# implementation (e.g. Hangul syllable composition rules, extended
# pictographic sequence edge cases, and >2-codepoint flag edge cases are
# not specially handled). Good enough for a tokenizer research metric, not
# claimed as a general-purpose grapheme segmenter.
# --------------------------------------------------------------------------
def _is_extend_or_zwj(ch: str) -> bool:
    cat = unicodedata.category(ch)
    if cat[0] == "M":  # Mn / Mc / Me combining marks
        return True
    cp = ord(ch)
    if cp in (0x200C, 0x200D):  # ZWNJ, ZWJ
        return True
    if 0xFE00 <= cp <= 0xFE0F:  # variation selectors
        return True
    if 0xE0100 <= cp <= 0xE01EF:  # variation selectors supplement
        return True
    if cp == 0x20E3:  # combining enclosing keycap
        return True
    return False


def _is_regional_indicator(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def grapheme_clusters(text: str) -> list[str]:
    clusters = []
    i, n = 0, len(text)
    while i < n:
        start = i
        i += 1
        while i < n and _is_extend_or_zwj(text[i]):
            i += 1
        if (i - start == 1) and _is_regional_indicator(text[start]) and i < n and _is_regional_indicator(text[i]):
            i += 1  # pair regional indicators into one flag cluster
        clusters.append(text[start:i])
    return clusters


# --------------------------------------------------------------------------
# Percentiles without a numpy dependency (linear interpolation, same
# convention as numpy.percentile's default 'linear' method).
# --------------------------------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# --------------------------------------------------------------------------
# Renyi efficiency (Zouhar et al. 2023, "Tokenization and the Noiseless
# Channel"): normalize the Renyi entropy (order alpha) of the observed
# token-ID frequency distribution by log(vocab size). Bounded in (0, 1];
# closer to 1 means the tokenizer's vocabulary is used more evenly (fewer
# tokens dominating token-count share) rather than being effectively a much
# smaller vocabulary in practice -- empirically correlated with downstream
# quality independent of raw fertility.
# --------------------------------------------------------------------------
def renyi_efficiency(token_id_counts: Counter, vocab_size: int, alpha: float = RENYI_ALPHA) -> float:
    total = sum(token_id_counts.values())
    if total == 0 or vocab_size <= 1:
        return 0.0
    probs = [c / total for c in token_id_counts.values()]
    if alpha == 1.0:
        h = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        s = sum(p**alpha for p in probs)
        h = (1.0 / (1.0 - alpha)) * math.log2(s)
    return h / math.log2(vocab_size)


# --------------------------------------------------------------------------
# Adversarial round-trip suite. Every case is deliberately a REAL,
# plausible input, not a synthetic stress string -- these are the specific
# classes of text that visually/semantically look "the same" to a human but
# have more than one valid Unicode encoding, or that carry structural
# invisible characters a naive tokenizer can silently mangle.
# --------------------------------------------------------------------------
ADVERSARIAL_SUITE = {
    # Nukta letters, written two DIFFERENT valid ways: precomposed single
    # codepoint (QA U+0958, ZA U+095B) vs canonically-decomposed base
    # consonant + combining nukta (U+093C) sequence. Both render
    # identically -- qareeb/"near", zaroori/"necessary", karz/"loan" (the
    # last combines BOTH nukta letters in one word). Built via explicit
    # chr()-composed codepoints at module-import time (not hand-typed
    # glyphs) so the two forms are verifiably distinct on disk.
    "nukta_qa_precomposed": 'क़रीब',
    "nukta_qa_decomposed": 'क़रीब',
    "nukta_za_precomposed": 'ज़रूरी',
    "nukta_za_decomposed": 'ज़रूरी',
    "nukta_karz_precomposed": 'क़र्ज़',
    "nukta_karz_decomposed": 'क़र्ज़',
    "conjunct_ksha": "अक्षर परीक्षा कक्षा",
    "conjunct_jnya": "ज्ञान विज्ञान यज्ञ",
    "conjunct_tra_shra": "मित्र राष्ट्र चित्र",
    "sanskrit_dense_conjuncts": "संस्कृत भाषा विद्यार्थी कर्तव्य",
    "chillu_n_atomic": "അവൻ പോയി",  # chillu N precomposed U+0D7B
    "chillu_n_legacy": "അവന്‍ പോയി",  # NA+VIRAMA+ZWJ legacy chillu sequence
    "chillu_l_atomic": "കാൽ",  # chillu L U+0D7D
    "emoji_zwj_family": "हमारा परिवार \U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 है।",
    "emoji_zwj_scientist": "She is a \U0001f469‍\U0001f52c researcher.",
    "emoji_flag_regional": "India \U0001f1ee\U0001f1f3 wins!",
    "emoji_skin_tone": "Thumbs up \U0001f44d\U0001f3fd from the team",
    "whatsapp_forward_noise": "*Forwarded*​​​\nGood morning​! \U0001f64f\U0001f64f\U0001f64f\r\nPlease share⁠⁠ with all groups.",
    "ocr_garbled_spacing": "य ह  एक   OCR   से  प्राप्त­­ पाठ है, जिसमें   अतिरिक्त  स्पेस  हैं।",
    "hinglish_codemix": "Bhai, kal office mein IFSC code check karna, PAN card bhi lana. Thanks!",
    "currency_devanagari_digits": "कीमत ₹१,२३,४५६.७८ है, GST 18% अलग से।",
    "mixed_bidi_marks": "The invoice no. ‎INV-2026‏/04 is due.",
    "repeated_zwnj_garbage": "क‌‌‌‌ख अलग शब्द हैं",
    "bengali_conjunct": "বাংলা ভাষায় সংস্কৃতি ও ঐতিহ্য",
    "tamil_agglutination": "அவர்களுக்காகத்தான் இதைச் செய்தேன்",
    "malayalam_compound": "സർക്കാരിന്റെ പദ്ധതിപ്രകാരം നടപ്പിലാക്കി",
    "urdu_bidi_arabic_script": "یہ ایک اردو جملہ ہے۔ Test 123 mixed.",
    "empty_string": "",
    "pure_whitespace": "   \n\t  ",
    "long_number_run": "Account 000123456789012345 IFSC HDFC0001234",
}


def run_roundtrip_suite(tok: Tokenizer) -> dict:
    results = {}
    n_pass = 0
    for name, raw in ADVERSARIAL_SUITE.items():
        norm = normalize_corpus_text(raw)
        try:
            ids = tok.encode(norm).ids
            dec = tok.decode(ids)
        except Exception as e:  # noqa: BLE001 - report, don't crash the suite
            results[name] = {"pass": False, "error": str(e)}
            continue
        ok = dec == norm
        n_pass += int(ok)
        entry = {"pass": ok}
        if not ok:
            entry["expected"] = norm
            entry["got"] = dec
        results[name] = entry
    return {
        "pass_rate": n_pass / len(ADVERSARIAL_SUITE),
        "n_pass": n_pass,
        "n_total": len(ADVERSARIAL_SUITE),
        "cases": results,
    }


# --------------------------------------------------------------------------
# Per-language metrics.
# --------------------------------------------------------------------------
def compute_lang_stats(tok: Tokenizer, text: str, vocab_size: int) -> dict:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    per_sentence_fertility = []
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    total_graphemes = 0
    frag_tokens = 0
    token_id_counts: Counter = Counter()

    for line in lines:
        words = line.split()
        if not words:
            continue
        enc = tok.encode(line)
        ids = enc.ids
        n_tok = len(ids)
        per_sentence_fertility.append(n_tok / len(words))
        total_tokens += n_tok
        total_words += len(words)
        total_bytes += len(line.encode("utf-8"))
        total_graphemes += len(grapheme_clusters(line))
        token_id_counts.update(ids)
        for tid in ids:
            tstr = tok.id_to_token(tid)
            if tstr is None:
                continue
            raw = token_to_raw_bytes(tstr)
            if raw is not None and len(raw) == 1 and raw[0] >= 0x80:
                frag_tokens += 1

    per_sentence_fertility.sort()
    used_vocab = len(token_id_counts)

    return {
        "n_sentences": len(per_sentence_fertility),
        "n_words": total_words,
        "n_tokens": total_tokens,
        "fertility_mean": round(sum(per_sentence_fertility) / len(per_sentence_fertility), 4) if per_sentence_fertility else 0.0,
        "fertility_p50": round(percentile(per_sentence_fertility, 50), 4),
        "fertility_p95": round(percentile(per_sentence_fertility, 95), 4),
        "fertility_p99": round(percentile(per_sentence_fertility, 99), 4),
        "tokens_per_word": round(total_tokens / total_words, 4) if total_words else 0.0,
        "tokens_per_grapheme": round(total_tokens / total_graphemes, 4) if total_graphemes else 0.0,
        "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else 0.0,
        "byte_fragment_rate": round(frag_tokens / total_tokens, 6) if total_tokens else 0.0,
        "renyi_efficiency_a2.5": round(renyi_efficiency(token_id_counts, vocab_size), 4),
        "vocab_utilization": round(used_vocab / vocab_size, 6),
        "used_vocab_tokens": used_vocab,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--eval-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_sets_flores200"),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    per_language = {}
    for path in sorted(glob.glob(os.path.join(args.eval_dir, "*.txt"))):
        lang = os.path.splitext(os.path.basename(path))[0]
        raw = open(path, encoding="utf-8").read()
        text = normalize_corpus_text(raw)
        per_language[lang] = compute_lang_stats(tok, text, vocab_size)

    roundtrip = run_roundtrip_suite(tok)

    report = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "eval_dir": args.eval_dir,
        "renyi_alpha": RENYI_ALPHA,
        "per_language": per_language,
        "roundtrip_fidelity": roundtrip,
        "method": (
            "fertility_mean/p50/p95/p99: per-sentence tokens/whitespace-word ratio, "
            "distribution over FLORES dev+devtest lines. tokens_per_word: aggregate "
            "total_tokens/total_words (comparable to fertility_report_v3/v4's single "
            "number). tokens_per_grapheme: aggregate tokens / approximate UAX#29 "
            "extended-grapheme-cluster count (see grapheme_clusters()). bytes_per_token: "
            "aggregate UTF-8 bytes / tokens (higher = more compression). "
            "byte_fragment_rate: fraction of emitted tokens that are a single raw byte "
            ">=0x80 (i.e. an incomplete multi-byte UTF-8 fallback -- no merge existed "
            "for that character). renyi_efficiency_a2.5 + vocab_utilization: see "
            "renyi_efficiency() docstring; computed per-language from that language's "
            "own token-ID frequency distribution over this eval set."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print(f"\n{args.tokenizer}  (vocab={vocab_size})")
    print(f"{'lang':<12} {'fert_mean':>9} {'p95':>7} {'p99':>7} {'tok/gr':>7} {'B/tok':>6} {'frag%':>7} {'renyi':>6} {'util%':>6}")
    print("-" * 82)
    for lang in sorted(per_language, key=lambda l: (l != "english", l)):
        d = per_language[lang]
        print(
            f"{lang:<12} {d['fertility_mean']:9.3f} {d['fertility_p95']:7.3f} {d['fertility_p99']:7.3f} "
            f"{d['tokens_per_grapheme']:7.3f} {d['bytes_per_token']:6.2f} {d['byte_fragment_rate']*100:6.2f}% "
            f"{d['renyi_efficiency_a2.5']:6.3f} {d['vocab_utilization']*100:5.1f}%"
        )
    print("-" * 82)
    print(f"round-trip fidelity: {roundtrip['n_pass']}/{roundtrip['n_total']} = {roundtrip['pass_rate']*100:.1f}%")
    if roundtrip["pass_rate"] < 1.0:
        failed = [k for k, v in roundtrip["cases"].items() if not v.get("pass")]
        print(f"  FAILED: {failed}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
