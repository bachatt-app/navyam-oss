#!/usr/bin/env python3
"""Tool router: detect live-data questions, answer them deterministically.

Sits in front of the model in serve.py. When the last user message asks for
a LIVE number (NAV, exchange rate), the router fetches it and formats the
reply from a template — the model never sees or copies the number, because
a small model WILL corrupt injected digits. Everything else falls through
to the model (concepts live in weights, numbers live in tools).

Formatted replies always carry source + as-of date: a number without a date
is misinformation waiting to happen.
"""

import re

import tools

try:
    import calc_router
except Exception:  # noqa: BLE001
    calc_router = None

# Up-to-date info that lives NOWHERE in the model's weights and isn't a NAV/FX/
# calc query: repo rate, small-savings/loan/FD rates, gold/silver, budget & tax
# changes, RBI/SEBI news, IPOs, fuel prices. Routed to web search and presented
# as sourced snippets (never model-synthesised) with a "verify" caveat.
LIVE_INFO_RX = re.compile(
    r"repo rate|reverse repo|\bmclr\b|home ?loan rate|fd rate|loan rate|"
    r"gold (rate|price|bhav)|silver (rate|price|bhav)|small savings|ppf rate|"
    r"nsc rate|ssy rate|scss rate|budget 20|union budget|"
    r"new tax (rule|slab|regime|change)|rbi (rate|policy|news|circular)|"
    r"sebi (rule|circular|news)|\bipo\b|petrol price|diesel price|"
    r"inflation rate|current .*rate|latest .*rate|aaj ka rate", re.I)

NAV_RX = re.compile(
    r"\bnav\b|\bnaav\b|(fund|scheme).{0,24}(price|value|nav)|"
    r"nav (kya|kitna|batao)", re.I)
FX_RX = re.compile(
    r"(dollar|usd|euro|eur|pound|gbp|yen|jpy|dirham|aed).{0,24}"
    r"(rate|bhav|price|kitna|kya chal)|"
    r"(rate|bhav).{0,16}(dollar|usd|euro)|exchange rate", re.I)
FX_CODE = {"dollar": "USD", "usd": "USD", "euro": "EUR", "eur": "EUR",
           "pound": "GBP", "gbp": "GBP", "yen": "JPY", "jpy": "JPY",
           "dirham": "AED", "aed": "AED"}
# Words that mean the user wants TODAY's number, not the concept.
LIVE_RX = re.compile(r"aaj|abhi|current|latest|today|is (waqt|samay)|"
                     r"kitna (hai|chal)|kya (hai|chal|rate)", re.I)

# Scope guard: a 151M finance model answers ANY question with finance
# fragments (observed: a cervical-cancer question got an insurance blend).
# Explicit off-domain markers + zero finance vocabulary => deterministic
# polite refusal, in the question's language. Finance vocabulary always
# wins: "health insurance me cancer cover hota hai?" must reach the model.
FINANCE_RX = re.compile(
    r"\bsip\b|lumpsum|mutual|fund|\bnav\b|invest|loan|\bemi\b|prepay|"
    r"tax|\bgst\b|\bitr\b|cibil|credit|card|bank|insurance|polic[yi]|"
    r"premium|claim|rider|cover|nominee|maturity|surrender|\bkyc\b|"
    r"paisa|paise|salary|\bfd\b|\brd\b|\bppf\b|\bnps\b|\bepf\b|"
    r"\bupi\b|share|stock|gold|bachat|savings?|interest|refund|pension|"
    r"demat|nifty|sensex|rupee|rupay|money|financ|elss|portfolio|"
    r"budget|kharcha|udhaar|karz|byaj|nivesh", re.I)
OOD_RX = re.compile(
    r"vaccine|cancer|disease|doctor|medicine|symptom|treatment|surgery|"
    r"pregnan|diabetes treatment|bimari ka ilaj|dawai|"
    r"cricket|football|match|movie|film|song|gaana|actor|"
    r"recipe|cooking|banane ki vidhi|"
    r"python|javascript|code likh|program likh|"
    r"capital of|prime minister|president|election result|"
    r"weather|mausam|temperature|"
    r"homework|essay|poem|kavita|story likh|joke|shayari", re.I)

OOS_HI = ("Main Navya hoon — sirf personal finance ke liye banayi gayi hoon: "
          "SIP, mutual funds, loans, credit cards, insurance, tax, GST. Yeh "
          "sawaal mere dayre se bahar hai, isliye main iska galat jawab dene "
          "ke bajaye seedha bata rahi hoon ki main iski expert nahi hoon. "
          "Medical sawaal ke liye doctor se zaroor milein. Paise se juda "
          "kuch bhi poochhiye — main yahin hoon!")
OOS_EN = ("I'm Navya — built only for personal finance: SIP, mutual funds, "
          "loans, credit cards, insurance, tax, GST. That question is "
          "outside my scope, and I'd rather say so directly than give you a "
          "wrong answer. For medical questions please consult a doctor. Ask "
          "me anything about money — that's where I can actually help!")
_EN_HINT = re.compile(r"\b(what|which|how|why|the|are|is|of|explain)\b", re.I)
_HI_HINT = re.compile(r"\b(kya|kaise|kaun|kitna|hai|karo|batao|likh|mujhe|do)\b", re.I)


# Identity: a 1.3B model hallucinates wildly on "who are you / what's your name"
# (observed: "My first name is ___ … GPA is 3. Apply today…"). Answer it
# deterministically — the model must never improvise Navya's identity.
IDENTITY_RX = re.compile(
    r"(?:\b(?:who|what)\s+(?:are\s+you|is\s+your\s+name)\b|"
    r"\b(?:who\s+(?:made|built|created)\s+you|introduce\s+yourself|"
    r"tell\s+me\s+(?:about\s+yourself|your\s+name)|"
    r"what\s+model\s+are\s+you|are\s+you\s+(?:an?\s+)?(?:ai|bot|model))\b|"
    r"\b(?:tum|tu|aap)\s+(?:ho\s+)?kaun\s+(?:ho|hai|hain)\b|"
    r"\b(?:tumhara|tera|aapka)\s+naam\s+kya\s+(?:hai|hain)\b|"
    r"\b(?:tumhe|aapko)\s+kisne\s+banaya\b|"
    r"\bkhud\s+ke\s+baare\s+mein\s+batao\b|"
    r"(?:तुम|तू|आप)\s+कौन\s+(?:हो|है|हैं)|"
    r"(?:तुम्हारा|तेरा|आपका)\s+नाम\s+क्या\s+(?:है|हैं))", re.I)
IDENTITY_EN = (
    "I'm **Navya** — an India-first personal-finance AI, built from scratch by "
    "Navyam AI (Bachatt) (not fine-tuned from another model). I help with "
    "savings, SIPs & mutual funds, loans & EMIs, credit scores, insurance, and "
    "tax/GST, in English and Hinglish. I'm a research model, so I'm not a "
    "substitute for a licensed advisor — but ask me anything about money!")
IDENTITY_HI = (
    "Main **Navya** hoon — India ke liye scratch se bani ek personal-finance AI, "
    "Navyam AI (Bachatt) dwara banayi gayi (kisi aur model se fine-tune nahi ki "
    "gayi). Main savings, SIP, mutual funds, loan/EMI, credit score, insurance, "
    "aur tax/GST me madad karti hoon. Main ek research model hoon, isliye "
    "licensed advisor ka vikalp nahi — par paison se juda kuch bhi poochhiye!")


def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def route(messages):
    """Return a formatted reply string if a tool owns this query, else None."""
    q = _last_user(messages)
    if not q:
        return None

    if IDENTITY_RX.search(q):
        hi = len(_HI_HINT.findall(q)) >= max(1, len(_EN_HINT.findall(q)))
        return IDENTITY_HI if hi else IDENTITY_EN

    if OOD_RX.search(q) and not FINANCE_RX.search(q):
        hi = len(_HI_HINT.findall(q)) >= max(1, len(_EN_HINT.findall(q)))
        return OOS_HI if hi else OOS_EN

    # Deterministic calculators (EMI/SIP/tax/…) — the model must never do math.
    if calc_router is not None:
        c = calc_router.route(messages)
        if c:
            return c

    if NAV_RX.search(q) and LIVE_RX.search(q):
        scheme = re.sub(NAV_RX, " ", q)
        scheme = re.sub(r"aaj|abhi|kitna|kya|hai|ka|ki|batao|current|latest",
                        " ", scheme, flags=re.I)
        scheme = " ".join(scheme.split())
        if len(scheme) < 4:
            return ("Kis fund ka NAV chahiye? Scheme ka naam likhiye (jaise "
                    "'UTI Nifty 50 Index Direct Growth') — main AMFI ke "
                    "official data se bata doongi.")
        r = tools.mf_nav(scheme)
        if not r.get("ok"):
            return ("Is naam se AMFI data me scheme match nahi hui. Poora "
                    "scheme naam likhiye — plan (Direct/Regular) aur option "
                    "(Growth/IDCW) ke saath.")
        lines = [f"- {d['name']}: ₹{d['nav']}" for d in r["data"]]
        return (f"AMFI ke official data ({r['asof']}) ke hisaab se:\n"
                + "\n".join(lines)
                + "\n\nNAV har trading day ke end me update hota hai. "
                  "(Source: amfiindia.com)")

    if FX_RX.search(q) and LIVE_RX.search(q):
        base = "USD"
        for word, code in FX_CODE.items():
            if re.search(rf"\b{word}\b", q, re.I):
                base = code
                break
        r = tools.fx_rate(base, "INR")
        if not r.get("ok"):
            return None  # model can still answer conceptually
        rate = r["data"]["rate"]
        return (f"{base}/INR reference rate: ₹{rate} "
                f"(as of {r['asof']}, ECB reference via frankfurter.app). "
                "Bank/card se convert karte waqt iske upar 0.5-3.5% ka markup "
                "lagta hai, isliye aapka asli rate thoda alag hoga.")

    # Up-to-date info via web search — sourced snippets, never model-synthesised.
    # Merely mentioning a live-data topic is not a request for today's number.
    # For example, "if repo rate rises, why does my EMI rise?" is conceptual
    # and must reach the model. Route only when the question also asks for a
    # current/latest value.
    if FINANCE_RX.search(q) and LIVE_RX.search(q) and LIVE_INFO_RX.search(q):
        r = tools.web_search(q, count=3)
        snips = [h.get("snippet", "").strip()
                 for h in (r.get("data") or []) if h.get("snippet")]
        if r.get("ok") and snips:
            src = (r["data"][0].get("url") or "").strip() or r.get("source", "web")
            body = "\n".join(f"- {s}" for s in snips[:2])
            return ("I don't keep live rates/news in my memory, so here's what a "
                    "web search returned — please verify on the official source, "
                    "these change often:\n" + body
                    + f"\n\nSource: {src}. For anything time-sensitive, confirm on "
                      "the official site (rbi.org.in, incometax.gov.in, "
                      "amfiindia.com, etc.).")
        # search empty/keyless — say so honestly instead of guessing a number.
        return ("That needs an up-to-date figure I don't hold in my weights, and "
                "my web lookup came up empty right now. Please check the official "
                "source directly — e.g. rbi.org.in for the repo rate, "
                "incometax.gov.in for tax, or the AMC/bank site for current rates.")

    return None
