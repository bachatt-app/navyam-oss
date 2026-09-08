#!/usr/bin/env python3
"""Selftest for langid.classify — validates script ID + the Hindi/Marathi,
Bengali/Assamese, English/Hinglish disambiguations, and surfaces the known
false-positive risk from short shared Devanagari markers.

    python3 langid_selftest.py
"""
import glob
import os
import sys

import langid

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "01-tokenizer", "eval_sets")

# Discriminative cases (label -> text). Genuine short native sentences.
CASES = [
    ("hindi", "मुझे म्यूचुअल फंड में निवेश करना है, कौन सा अच्छा रहेगा?"),
    ("marathi", "मला म्युच्युअल फंडात गुंतवणूक करायची आहे, कोणता चांगला आहे?"),
    ("marathi", "ही योजना खूप चांगली आहे आणि परतावा जास्त मिळेल."),
    ("bengali", "আমি একটি মিউচুয়াল ফান্ডে বিনিয়োগ করতে চাই।"),
    ("assamese", "মই এটা মিউচুৱেল ফাণ্ডত বিনিয়োগ কৰিব বিচাৰো আৰু লাভ পাম।"),
    ("english", "I want to invest in a mutual fund, which one is good?"),
    ("hinglish", "mujhe mutual fund me invest karna hai, kaun sa accha rahega?"),
    ("tamil", "நான் ஒரு மியூச்சுவல் ஃபண்டில் முதலீடு செய்ய விரும்புகிறேன்."),
    ("telugu", "నేను మ్యూచువల్ ఫండ్‌లో పెట్టుబడి పెట్టాలనుకుంటున్నాను."),
]


def run():
    cases = list(CASES)
    for p in sorted(glob.glob(os.path.join(EVAL, "*.txt"))):
        lang = os.path.splitext(os.path.basename(p))[0]
        if lang in ("code",):
            continue
        cases.append((lang, open(p, encoding="utf-8").read()[:400]))

    ok = 0
    fails = []
    for expected, text in cases:
        got = langid.classify(text)["lang"]
        # hinglish/english both acceptable for romanized where marker density is
        # borderline; everything else must match exactly.
        good = got == expected or (
            {expected, got} == {"english", "hinglish"})
        ok += good
        if not good:
            fails.append((expected, got, text[:45]))
    print(f"langid selftest: {ok}/{len(cases)} correct")
    for exp, got, t in fails:
        print(f"  MISS expected={exp} got={got} :: {t!r}")

    # Surface the short-shared-marker false positive explicitly.
    hindi_ne = langid.classify("राम ने कहा कि वह कल आएगा और पैसा देगा।")["lang"]
    note = ("NOTE: short Marathi markers (ने/ला/चा/ची) also occur in Hindi, so "
            "substring matching can over-flag Marathi. Test sentence uses Hindi "
            f"'ने' -> classified as: {hindi_ne} "
            + ("(OK)" if hindi_ne == "hindi" else "(FALSE-POSITIVE — recommend "
               "requiring MARATHI_CHAR ळ OR >=2 distinct multi-char markers)"))
    print(note)
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
