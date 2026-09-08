#!/usr/bin/env python3
"""Calculator router: deterministic finance math in front of the model.

A 1.3B model cannot do amortization / compound-interest arithmetic reliably (it
hallucinates numbers). So when the user asks for a specific COMPUTED figure — EMI,
SIP maturity, income tax, credit-card payoff, ... — we parse the inputs, compute
the exact answer with calc_tools, and present it. The model never does the math.

Returns None (fall through to the model) when the intent or the inputs are
unclear, so concept questions still reach the model. Every reply states that the
figure is deterministically computed, and any assumed govt rate carries its
as-of date (calc_tools stamps rate_asof) — a number without provenance is
misinformation waiting to happen.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "07-sft"))
import calc_tools as C  # noqa: E402

_UNIT = {"crore": 1e7, "cr": 1e7, "karod": 1e7, "lakh": 1e5, "lac": 1e5,
         "lakhs": 1e5, "lacs": 1e5, "l": 1e5, "k": 1e3, "thousand": 1e3,
         "hazaar": 1e3, "hazar": 1e3}


def amounts(text):
    """Rupee amounts, honouring lakh/crore/k suffixes. Skips bare %/year numbers."""
    out = []
    for m in re.finditer(
            r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*"
            r"(crore|cr|karod|lakhs?|lacs?|thousand|hazaar|hazar|k|l)?", text, re.I):
        tail = text[m.end():m.end() + 2].lower()
        unit = (m.group(2) or "").lower()
        if not unit and (tail.startswith("%") or "year" in text[m.end():m.end() + 7].lower()):
            continue  # a rate or tenure, not a money amount
        num = float(m.group(1).replace(",", ""))
        out.append(num * _UNIT.get(unit, 1))
    return out


def _rate(text):
    m = re.search(r"([\d.]+)\s*(?:%|percent|pct|fisad|fisadi)", text, re.I)
    return float(m.group(1)) if m else None


def _years(text):
    m = re.search(r"([\d.]+)\s*(years?|yrs?|saal|varsh|barse?)", text, re.I)
    return float(m.group(1)) if m else None


def _months(text):
    m = re.search(r"([\d.]+)\s*(months?|mahin[ae]|maah)", text, re.I)
    return float(m.group(1)) if m else None


def _inr(x):
    return C._fmt_inr(x)


def _tenure_years(q):
    y = _years(q)
    if y:
        return y
    mo = _months(q)
    return mo / 12 if mo else None


# Intent detection — ordered; first match wins. Each handler returns a string
# (answer) or None (missing inputs / not really this intent → fall through).
def _emi(q, amt):
    if not re.search(r"\bemi\b|installment|kisht|monthly.{0,10}(payment|pay)", q, re.I):
        return None
    r, y = _rate(q), _tenure_years(q)
    if not amt or r is None or not y:
        return None
    d = C.emi(amt[0], r, y)
    return (f"**EMI ≈ {_inr(d['emi'])}/month** for a {_inr(amt[0])} loan at {r}% "
            f"over {int(y)} years.\nTotal payment {_inr(d['total_payment'])} "
            f"(interest {_inr(d['total_interest'])}). Computed exactly, not estimated.")


def _sip(q, amt):
    if not re.search(r"\bsip\b|systematic|monthly.{0,10}invest", q, re.I):
        return None
    r, y = _rate(q), _tenure_years(q)
    if not amt or r is None or not y:
        return None
    d = C.sip_future_value(amt[0], r, y)
    return (f"A {_inr(amt[0])}/month SIP for {int(y)} years at {r}% p.a. grows to "
            f"**≈ {_inr(d['future_value'])}** (you invest {_inr(d['invested'])}, "
            f"gain {_inr(d['gain'])}). Assumed constant return — actual market "
            "returns vary. Computed exactly.")


def _tax(q, amt):
    if not re.search(r"income tax|\btax\b.{0,12}(on|kitna|liab)|kitna tax", q, re.I):
        return None
    if not amt:
        return None
    regime = "old" if re.search(r"old regime|purani", q, re.I) else "new"
    d = C.income_tax(amt[0], regime)
    if d["total_tax"] == 0:
        return (f"On {_inr(amt[0])} income ({regime} regime, FY{d['fy']}), tax is "
                f"**₹0** — the 87A rebate covers it. {d['note']}.")
    return (f"On {_inr(amt[0])} income ({regime} regime, FY{d['fy']}): "
            f"**tax ≈ {_inr(d['total_tax'])}** (incl. 4% cess), effective "
            f"{d['effective_rate_pct']}%. {d['note']}.")


def _cc(q, amt):
    if not re.search(r"credit card|\bcc\b|card.{0,12}(due|balance|outstanding|bill)", q, re.I):
        return None
    r = _rate(q) or 42
    if not amt:
        return None
    bal = max(amt)
    pays = [a for a in amt if a < bal]
    if pays:
        d = C.credit_card_payoff(bal, r, min(pays))
        if not d["clears"]:
            return (f"Paying {_inr(min(pays))}/month on a {_inr(bal)} balance at "
                    f"{r}% p.a. **never clears it** — that barely covers interest "
                    f"({_inr(d['monthly_interest'])}/mo). Pay more.")
        return (f"Clearing a {_inr(bal)} card balance at {r}% p.a. paying "
                f"{_inr(min(pays))}/month takes **{d['months']} months** "
                f"({d['years']} yrs); total interest {_inr(d['total_interest'])}. "
                "Computed exactly.")
    d = C.credit_card_min_trap(bal, r)
    return (f"Paying only the ~5% minimum on a {_inr(bal)} balance at {r}% p.a. "
            f"takes **{d['years']} years** and {_inr(d['total_interest'])} in "
            "interest — the minimum-payment trap. Computed exactly.")


def _ppf(q, amt):
    if not re.search(r"\bppf\b|public provident", q, re.I) or not amt:
        return None
    y = _years(q) or 15
    d = C.ppf_maturity(amt[0], years=y)
    note = f" (assumed {d['rate_pct']}% as of {d['rate_asof']} — verify current rate)"
    return (f"A {_inr(amt[0])}/year PPF for {int(y)} years matures to "
            f"**≈ {_inr(d['maturity'])}**{note}. Computed exactly.")


def _fd(q, amt):
    if not re.search(r"\bfd\b|fixed deposit|term deposit", q, re.I) or not amt:
        return None
    r, y = _rate(q), _tenure_years(q)
    if r is None or not y:
        return None
    d = C.fd_maturity(amt[0], r, y)
    return (f"A {_inr(amt[0])} FD at {r}% for {int(y)} years matures to "
            f"**≈ {_inr(d['maturity'])}** (interest {_inr(d['interest'])}). "
            "TDS/tax on interest applies. Computed exactly.")


def _gst(q, amt):
    if not re.search(r"\bgst\b", q, re.I) or not amt:
        return None
    r = _rate(q)
    if r is None:
        return None
    incl = bool(re.search(r"inclusive|including|shaamil", q, re.I))
    d = C.gst(amt[0], r, inclusive=incl)
    return (f"GST at {r}%: base {_inr(d['base'])}, GST {_inr(d['gst'])}, "
            f"total **{_inr(d['total'])}**. Computed exactly.")


_HANDLERS = [_emi, _tax, _cc, _sip, _ppf, _fd, _gst]


def route(messages):
    """Return a computed-answer string if a calculator owns this query, else None."""
    q = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            q = m.get("content", "")
            break
    if not q:
        return None
    amt = amounts(q)
    for h in _HANDLERS:
        try:
            out = h(q, amt)
        except Exception:
            out = None
        if out:
            return out
    return None


if __name__ == "__main__":
    tests = [
        "₹40 lakh home loan at 9% for 20 years, EMI kitni?",
        "10000 monthly SIP for 15 years at 12%",
        "income tax on 15 lakh",
        "tax on 12 lakh new regime",
        "1 lakh credit card outstanding at 42%, paying 5000 per month",
        "1.5 lakh PPF for 15 years",
        "5 lakh FD at 7% for 5 years",
        "18% GST on 1000",
        "what is a mutual fund?",   # should fall through
    ]
    for t in tests:
        r = route([{"role": "user", "content": t}])
        print(f"\nQ: {t}\nA: {r}")
