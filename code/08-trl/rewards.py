#!/usr/bin/env python3
"""Programmatic rewards for Navya post-training (DPO pair scoring + GRPO).

Every reward is a pure function of (prompt messages, completion text,
optional canonical answer) returning a float in [-1, 1]. No model-based
judges at this stage — the S061 canonical answers plus hard rules cover the
failure modes we actually observe (language mismatch, topic blending,
repetition, naming specific funds).

`total_reward` is the weighted sum used everywhere, so DPO's notion of
"bad sample" and GRPO's optimization target never drift apart.
"""

import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH_HINTS = re.compile(
    r"\b(kya|kaise|karun|karoon|chahiye|hai|hota|hoti|karna|kaun|kitna|"
    r"kab|kyu|kyun|mujhe|mera|meri|paisa|paise|sahi|galat|wala|wale|"
    r"lein|dein|banega|milega|lagta|nahi|nahin|zyada|accha|behtar|"
    r"mahine|saal|abhi|pehle|baad|karein|daalein|sakta|sakte)\b", re.I)
ENGLISH_HINTS = re.compile(
    r"\b(what|should|how|which|invest|better|the|is|are|can|will|my)\b", re.I)

# Naming a specific fund/AMC is the one thing Navya must never do.
FUND_NAMES = re.compile(
    r"\b(hdfc|icici|sbi|axis|kotak|nippon|uti|aditya birla|absl|dsp|tata|"
    r"mirae|parag parikh|ppfas|quant|motilal|franklin|invesco|edelweiss|"
    r"canara|bandhan|whiteoak|bluechip fund|prudential)\b", re.I)
RECOMMEND_CONTEXT = re.compile(
    r"\b(best|recommend|suggest|le lo|kharid|invest kar)\b", re.I)


def _lang(text):
    if DEVANAGARI.search(text):
        return "hi"
    h = len(HINGLISH_HINTS.findall(text))
    e = len(ENGLISH_HINTS.findall(text))
    # Short queries carry few hint words — one Hinglish marker with no
    # English-majority signal is already decisive ("SIP karun ya lumpsum?").
    if h >= 1 and h >= max(1, e):
        return "hinglish"
    return "en"


def reward_language_match(messages, completion, canonical=None):
    """Reply in the language of the last user message (the '₹25k SIP karoon
    ya loan prepay?' answered in English failure)."""
    user = next((m["content"] for m in reversed(messages)
                 if m["role"] == "user"), "")
    want, got = _lang(user), _lang(completion)
    if want == got:
        return 1.0
    if {want, got} == {"hi", "hinglish"}:
        return 0.5
    return -1.0


def reward_no_repetition(messages, completion, canonical=None):
    words = completion.lower().split()
    if len(words) < 12:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    dup = 1.0 - len(set(trigrams)) / len(trigrams)
    return 1.0 - 4.0 * min(dup, 0.5)


def reward_length(messages, completion, canonical=None):
    """30-140 words is the product voice; walls of text and one-liners on
    substantive questions both score down."""
    n = len(completion.split())
    if 30 <= n <= 140:
        return 1.0
    if n < 8:
        return -1.0
    if n < 30:
        return 0.3
    return max(-1.0, 1.0 - (n - 140) / 100.0)


def reward_no_fund_naming(messages, completion, canonical=None):
    """Never name a specific AMC/fund in a recommending register."""
    if FUND_NAMES.search(completion) and RECOMMEND_CONTEXT.search(completion):
        return -1.0
    return 0.2 if not FUND_NAMES.search(completion) else 0.0


def reward_canonical_overlap(messages, completion, canonical=None):
    """Token-F1 against the S061 canonical answer when the prompt has one.
    This is the anti-blending signal: a reply stitched from unrelated
    corpus fragments shares few content words with the intended answer."""
    if not canonical:
        return 0.0
    stop = {"hai", "ke", "ka", "ki", "me", "aur", "to", "par", "se", "the",
            "a", "of", "in", "is", "ya", "na", "ho", "kar", "bhi", "aap"}
    a = {w for w in re.findall(r"\w+", completion.lower()) if w not in stop}
    b = {w for w in re.findall(r"\w+", canonical.lower()) if w not in stop}
    if not a or not b:
        return -1.0
    p, r = len(a & b) / len(a), len(a & b) / len(b)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return 2.0 * f1 - 1.0


WEIGHTS = {
    reward_language_match: 0.30,
    reward_canonical_overlap: 0.30,
    reward_no_repetition: 0.15,
    reward_length: 0.15,
    reward_no_fund_naming: 0.10,
}


def total_reward(messages, completion, canonical=None):
    return sum(w * fn(messages, completion, canonical)
               for fn, w in WEIGHTS.items())


def grpo_reward_funcs():
    """Adapt to TRL's GRPOTrainer reward_funcs signature: each takes
    (prompts, completions, **kwargs) and returns a list of floats. The
    dataset must carry a `canonical` column (empty string when absent)."""
    def make(fn):
        def rf(prompts, completions, canonical=None, **kwargs):
            out = []
            for i, comp in enumerate(completions):
                msgs = prompts[i] if isinstance(prompts[i], list) else \
                    [{"role": "user", "content": str(prompts[i])}]
                text = comp if isinstance(comp, str) else comp[0]["content"]
                canon = (canonical[i] or None) if canonical else None
                out.append(WEIGHTS[fn] * fn(msgs, text, canon))
            return out
        rf.__name__ = fn.__name__
        return rf
    return [make(fn) for fn in WEIGHTS]
