#!/usr/bin/env python3
"""PII scrubbing for pretraining corpora — streaming, stdin→stdout JSONL.

Replaces personally identifiable spans with typed placeholder tokens rather
than dropping documents (the surrounding text is usually fine training data).
Documents that are PII-dense (many hits per KB — dumps, leaked lists) are
dropped entirely.

India-aware patterns:
  emails, phone numbers (+91 / 10-digit starting 6-9), Aadhaar-shaped 12-digit
  groups, PAN codes, card-shaped 13-16 digit runs that pass Luhn, IFSC codes
  paired with long account numbers.

Usage: python pii_scrub.py < in.jsonl > out.jsonl
"""

import json
import re
import sys

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
AADHAAR = re.compile(r"(?<!\d)\d{4}[\s-]\d{4}[\s-]\d{4}(?!\d)")
PAN = re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])")
CARD = re.compile(r"(?<!\d)\d(?:[\s-]?\d){12,15}(?!\d)")
IFSC_ACCT = re.compile(r"(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])")


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def scrub(text: str):
    hits = 0

    def sub(pattern, repl, t, guard=None):
        nonlocal hits
        out, last = [], 0
        for m in pattern.finditer(t):
            if guard and not guard(m.group()):
                continue
            out.append(t[last:m.start()])
            out.append(repl)
            last = m.end()
            hits += 1
        out.append(t[last:])
        return "".join(out)

    text = sub(EMAIL, "<email>", text)
    text = sub(CARD, "<card>", text,
               guard=lambda s: luhn_ok(re.sub(r"\D", "", s)))
    text = sub(AADHAAR, "<id-number>", text)
    text = sub(PAN, "<pan>", text)
    text = sub(PHONE, "<phone>", text)
    text = sub(IFSC_ACCT, "<ifsc>", text)
    return text, hits


def main() -> None:
    kept = dropped = scrubbed = 0
    for line in sys.stdin:
        doc = json.loads(line)
        text, hits = scrub(doc["text"])
        # PII-dense documents (contact dumps, leaked lists) are not worth keeping
        if hits > 0 and hits / max(1, len(text) / 1000) > 5:
            dropped += 1
            continue
        if hits:
            doc["text"] = text
            scrubbed += 1
        kept += 1
        sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"pii_scrub: kept={kept} scrubbed={scrubbed} dropped_dense={dropped}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
