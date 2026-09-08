#!/usr/bin/env python3
"""Language / script / code-mix identification, v1.

Script detection is reliable via Unicode ranges. The hard problem — and the one
that matters most to us — is Latin-script text: English vs romanized Hindi
(Hinglish) vs other romanized Indic. v1 uses a common-word heuristic; the
planned upgrade is fastText LID + a trained romanized-Indic classifier.

Adds to each doc: lang, script, code_mix (bool).

Usage: python langid.py < in.jsonl > out.jsonl
"""

import json
import sys

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),   # hindi, marathi
    "bengali": (0x0980, 0x09FF),      # bengali, assamese
    "gurmukhi": (0x0A00, 0x0A7F),     # punjabi
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}
SCRIPT_TO_LANG = {  # coarse; devanagari/bengali need a real classifier
    "devanagari": "hindi", "bengali": "bengali", "gurmukhi": "punjabi",
    "gujarati": "gujarati", "odia": "odia", "tamil": "tamil",
    "telugu": "telugu", "kannada": "kannada", "malayalam": "malayalam",
}

# High-frequency romanized-Hindi words that are rare in English.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "kya", "kaise", "karna", "karoon", "karun",
    "chahiye", "mera", "meri", "mujhe", "aur", "lekin", "bhi", "wala", "wale",
    "wali", "hoga", "hogi", "raha", "rahi", "rahe", "kab", "kaun", "kitna",
    "kitni", "milega", "milegi", "sakta", "sakte", "sakti", "kar", "karke",
    "karein", "karo", "hota", "hoti", "hote", "tha", "thi", "the", "ho", "ka",
    "ki", "ke", "ko", "se", "mein", "par", "paisa", "paise", "kaha", "kahan",
    "kyun", "kyu", "isliye", "matlab", "zyada", "kam", "accha", "sahi", "galat",
    "chahta", "batao", "samajh", "abhi", "phir", "yaha", "waha", "mere", "tere",
    "apna", "apne", "kuch", "koi", "jyada", "thoda", "bahut", "nivesh", "byaj",
}

# Marathi (Devanagari) markers — distinguish from Hindi. The letter ळ (LLA) and
# these function words are common in Marathi and rare/absent in Hindi.
MARATHI_MARKERS = (
    "आहे", "नाही", "माझ", "तुम्ही", "मला", "आणि", "होते", "करण्या", "मध्ये",
    "काय", "कसे", "त्या", "पण", "किंवा", "यांनी", "आम्ही", "तुला", "झाले",
    "पाहिजे", "करावे", "असे", "याची", "ची", "चा", "ला", "ने",
)
MARATHI_CHAR = "ळ"          # U+0933, frequent in Marathi, rare in Hindi

# Assamese uses ৰ (U+09F0) and ৱ (U+09F1) where Bengali uses র / ব — the single
# most reliable script-level signal separating the two.
ASSAMESE_CHARS = "ৰৱ"
ASSAMESE_MARKERS = ("অসম", "কৰি", "হৈছে", "নাই", "মই", "আৰু", "কৰা", "হয়")


def script_profile(text: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        cp = ord(ch)
        if ch.isalpha():
            letters += 1
            if cp < 0x0250:
                counts["latin"] = counts.get("latin", 0) + 1
                continue
            for name, (lo, hi) in SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    counts[name] = counts.get(name, 0) + 1
                    break
    if not letters:
        return {}
    return {k: v / letters for k, v in counts.items()}


def classify(text: str, feats=None) -> dict:
    prof = script_profile(text)
    if not prof:
        return {"lang": "unknown", "script": "none", "code_mix": False}
    top = max(prof, key=prof.get)

    if top != "latin":
        lang = SCRIPT_TO_LANG.get(top, top)
        # Disambiguate the two scripts shared by two languages each.
        if top == "devanagari":
            # ळ (LLA) is the reliable Marathi-vs-Hindi signal. The function-word
            # markers are SHARED with Hindi (ने/ला/चा/ची are common in both), so
            # a single short match over-flags Marathi — require the char OR >=2
            # DISTINCT multi-char (>=3) markers.
            strong = sum(1 for m in MARATHI_MARKERS if len(m) >= 3 and m in text)
            if MARATHI_CHAR in text or strong >= 2:
                lang = "marathi"
        elif top == "bengali":
            # ৰ/ৱ are Assamese-only; the word markers over-flag (নাই etc. shared),
            # so require the char OR >=2 distinct multi-char markers.
            strong = sum(1 for m in ASSAMESE_MARKERS if len(m) >= 3 and m in text)
            if any(c in text for c in ASSAMESE_CHARS) or strong >= 2:
                lang = "assamese"
        # native script with substantial Latin admixture = code-mixed
        return {"lang": lang, "script": top,
                "code_mix": prof.get("latin", 0.0) > 0.15}

    # Latin-dominant: english vs hinglish via marker density
    base = feats.words if feats is not None else text.split()
    words = [w.strip(".,!?").lower() for w in base]
    if not words:
        return {"lang": "unknown", "script": "latin", "code_mix": False}
    marker_frac = sum(w in HINGLISH_MARKERS for w in words) / len(words)
    if marker_frac > 0.05:
        return {"lang": "hinglish", "script": "latin", "code_mix": True}
    return {"lang": "english", "script": "latin", "code_mix": False}


def main() -> None:
    for line in sys.stdin:
        doc = json.loads(line)
        doc.update(classify(doc["text"]))
        sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
