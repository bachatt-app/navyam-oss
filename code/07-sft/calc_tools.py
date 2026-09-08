#!/usr/bin/env python3
"""Deterministic finance calculators + tool-call SFT for Navya.

WHY tool-calling (not direct-answer SFT): a 337M–1.3B model cannot do
compound-interest / amortization arithmetic reliably. Training it to emit the
NUMBER teaches it to hallucinate numbers (v9 invented "EMI = Loan ÷ interest").
Instead we teach it to EMIT A TOOL CALL; a deterministic calculator (this module)
computes the exact figure; the model then PRESENTS it. Accuracy becomes 100% and
independent of model size.

The tokenizer already has a `tool` role (reserved_3, id 6) and chat_format
supports a tool turn, so the four-turn shape below trains cleanly:

    user      : "₹40 lakh home loan at 9% for 20 years, EMI kitni?"
    assistant : {"tool":"emi","args":{"principal":4000000,"annual_rate":9,"years":20}}
    tool      : {"emi":35989,"total_interest":4637360,"total_payment":8637360}
    assistant : "Aapki EMI ~**₹35,989/month** hogi. 20 saal mein total ..."

Serving integration (serve.py): when the assistant turn is a single JSON tool
call, execute dispatch(), inject the result as a `tool` turn, and continue
generation for the final presentation turn. The model never does math.

    python3 calc_tools.py selftest          # verify the math
    python3 calc_tools.py gen-sft --n 60 --out training_data/corpus_v2/calc_tools_v1.jsonl
"""
import argparse
import json
import math
import sys

# ---------------------------------------------------------------- calculators
# All rates are ANNUAL percent (e.g. 9 == 9%). Money in rupees. Returns rounded
# ints for money and a small breakdown dict; the presentation layer formats.

def emi(principal, annual_rate, years):
    r = annual_rate / 12 / 100
    n = int(round(years * 12))
    if r == 0:
        e = principal / n
    else:
        e = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total = e * n
    return {"emi": round(e), "months": n,
            "total_payment": round(total), "total_interest": round(total - principal)}


def sip_future_value(monthly, annual_rate, years):
    i = annual_rate / 12 / 100
    n = int(round(years * 12))
    fv = monthly * (((1 + i) ** n - 1) / i) * (1 + i) if i else monthly * n
    invested = monthly * n
    return {"future_value": round(fv), "invested": round(invested),
            "gain": round(fv - invested), "months": n}


def step_up_sip(monthly, annual_rate, years, step_up_pct):
    i = annual_rate / 12 / 100
    fv = 0.0
    invested = 0.0
    m = monthly
    for yr in range(int(years)):
        for _ in range(12):
            fv = fv * (1 + i) + m
            invested += m
        m *= (1 + step_up_pct / 100)
    return {"future_value": round(fv), "invested": round(invested),
            "gain": round(fv - invested)}


def goal_sip(target, annual_rate, years):
    """Monthly SIP needed to reach `target`."""
    i = annual_rate / 12 / 100
    n = int(round(years * 12))
    if i == 0:
        m = target / n
    else:
        m = target / ((((1 + i) ** n - 1) / i) * (1 + i))
    return {"monthly_sip": round(m), "target": round(target), "months": n}


def lumpsum_future_value(principal, annual_rate, years):
    fv = principal * (1 + annual_rate / 100) ** years
    return {"future_value": round(fv), "invested": round(principal),
            "gain": round(fv - principal)}


def fd_maturity(principal, annual_rate, years, comp_per_year=4):
    fv = principal * (1 + annual_rate / 100 / comp_per_year) ** (comp_per_year * years)
    return {"maturity": round(fv), "invested": round(principal),
            "interest": round(fv - principal)}


def rd_maturity(monthly, annual_rate, months):
    i = annual_rate / 4 / 100  # RD compounds quarterly in India
    fv = 0.0
    for k in range(months):
        # each installment compounds for the remaining (months-k) months
        fv += monthly * (1 + i) ** ((months - k) / 3)
    invested = monthly * months
    return {"maturity": round(fv), "invested": round(invested),
            "interest": round(fv - invested)}


# PPF interest is reset by the Ministry of Finance EVERY QUARTER (Small Savings).
# Hardcoding it as a silent default is misinformation waiting to happen, so when
# no rate is supplied we use this documented assumption AND stamp the result with
# the rate + its as-of date + a "verify current rate" note. The serving layer
# should override annual_rate from a live small-savings-rates source.
PPF_RATE_DEFAULT = 7.1
PPF_RATE_ASOF = "2025-07-01"   # MoF small-savings notification, FY2025-26 Q2


def ppf_maturity(annual_deposit, annual_rate=None, years=15, rate_asof=None):
    assumed = annual_rate is None
    rate = PPF_RATE_DEFAULT if assumed else annual_rate
    years = int(years)
    fv = 0.0
    for _ in range(years):
        fv = (fv + annual_deposit) * (1 + rate / 100)
    invested = annual_deposit * years
    out = {"maturity": round(fv), "invested": round(invested),
           "interest": round(fv - invested), "rate_pct": rate,
           "rate_asof": rate_asof or (PPF_RATE_ASOF if assumed else None)}
    if assumed:
        out["rate_note"] = ("assumed PPF rate — the govt revises it quarterly; "
                            "confirm the current small-savings rate")
    return out


def swp(corpus, withdrawal_monthly, annual_rate, years):
    i = annual_rate / 12 / 100
    bal = corpus
    n = int(round(years * 12))
    for _ in range(n):
        bal = bal * (1 + i) - withdrawal_monthly
        if bal <= 0:
            # month _ (0-indexed) IS a month the corpus paid out, so it lasted
            # _+1 months — the previous code reported _ (off-by-one: a corpus
            # that depletes on the very first withdrawal reported 0 months).
            return {"remaining": 0, "months_lasted": _ + 1, "depleted": True}
    return {"remaining": round(bal), "months_lasted": n, "depleted": False,
            "total_withdrawn": round(withdrawal_monthly * n)}


def cagr(begin, end, years):
    return {"cagr_pct": round(((end / begin) ** (1 / years) - 1) * 100, 2)}


def compound_interest(principal, annual_rate, years, comp_per_year=1):
    fv = principal * (1 + annual_rate / 100 / comp_per_year) ** (comp_per_year * years)
    return {"amount": round(fv), "interest": round(fv - principal)}


def inflation_adjusted(amount, inflation_pct, years):
    real = amount / (1 + inflation_pct / 100) ** years
    return {"today_value": round(real), "future_amount": round(amount)}


# Small-savings / statutory rates are set by the govt and revised periodically;
# each calculator that assumes one stamps the result with the rate + as-of date.
SMALL_SAVINGS_ASOF = "2025-07-01"   # MoF small-savings notification, FY2025-26 Q2


def _rated(out, rate, assumed, asof=SMALL_SAVINGS_ASOF, what="rate"):
    out["rate_pct"] = rate
    if assumed:
        out["rate_asof"] = asof
        out["rate_note"] = (f"assumed {what} — govt revises it periodically; "
                            "confirm the current notified rate")
    return out


def simple_interest(principal, annual_rate, years):
    si = principal * annual_rate * years / 100
    return {"interest": round(si), "amount": round(principal + si)}


def credit_card_payoff(balance, annual_rate, monthly_payment):
    """Months + interest to clear a card balance at a fixed monthly payment."""
    i = annual_rate / 12 / 100
    if monthly_payment <= balance * i:
        return {"clears": False, "monthly_interest": round(balance * i),
                "note": "payment barely covers interest — balance never clears; "
                        "pay more than the monthly interest"}
    bal, paid, months = balance, 0.0, 0
    while bal > 0 and months < 1200:
        bal += bal * i
        step = min(monthly_payment, bal)
        bal -= step
        paid += step
        months += 1
    return {"clears": True, "months": months, "years": round(months / 12, 1),
            "total_paid": round(paid), "total_interest": round(paid - balance)}


def credit_card_min_trap(balance, annual_rate, min_pct=5, min_floor=200):
    """The minimum-payment trap: pay only min(min_pct% of outstanding, floor)."""
    i = annual_rate / 12 / 100
    bal, paid, months = balance, 0.0, 0
    while bal > 0.5 and months < 2400:
        bal += bal * i
        pay = min(max(bal * min_pct / 100, min_floor), bal)
        bal -= pay
        paid += pay
        months += 1
    return {"months": months, "years": round(months / 12, 1),
            "total_paid": round(paid), "total_interest": round(paid - balance),
            "min_pct": min_pct}


def loan_prepayment(principal, annual_rate, years, prepay_amount,
                    after_months=12):
    """Interest + tenure saved by a one-time prepayment, keeping the EMI same."""
    e = emi(principal, annual_rate, years)
    emi_amt, i = e["emi"], annual_rate / 12 / 100
    n_orig = e["months"]
    orig_interest = e["total_interest"]
    bal, interest_paid, months = principal, 0.0, 0
    for m in range(1, 6000):
        interest_paid += bal * i
        bal = bal * (1 + i) - emi_amt
        months += 1
        if m == int(after_months):
            bal -= prepay_amount
        if bal <= 0:
            break
    return {"emi": emi_amt, "new_tenure_months": months,
            "months_saved": max(0, n_orig - months),
            "interest_saved": max(0, round(orig_interest - interest_paid)),
            "prepay_amount": round(prepay_amount)}


def xirr(cashflows):
    """Annualised return for dated, irregular cashflows. cashflows: list of
    {"days": <days from the first flow>, "amount": <-out / +in>}. Solved by
    bisection on NPV=0 (robust; no derivative)."""
    cf = [(int(c["days"]), float(c["amount"])) for c in cashflows]
    if not any(a < 0 for _, a in cf) or not any(a > 0 for _, a in cf):
        return {"error": "need at least one negative (invest) and one positive (return) flow"}

    def npv(rate):
        return sum(a / (1 + rate) ** (d / 365.0) for d, a in cf)

    lo, hi = -0.9999, 100.0
    if npv(lo) * npv(hi) > 0:
        return {"error": "no sign change — XIRR not bracketed for these flows"}
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return {"xirr_pct": round((lo + hi) / 2 * 100, 2)}


def _slab_tax(taxable, slabs):
    """slabs: list of (upper_bound, rate_pct); last upper_bound is math.inf."""
    tax, prev = 0.0, 0
    for upper, rate in slabs:
        if taxable > prev:
            tax += (min(taxable, upper) - prev) * rate / 100
            prev = upper
        else:
            break
    return tax


# Income-tax slabs are effective-dated. FY2025-26 (AY2026-27), post Budget 2025.
_TAX_FY = "2025-26"
_NEW_SLABS = [(400000, 0), (800000, 5), (1200000, 10), (1600000, 15),
              (2000000, 20), (2400000, 25), (math.inf, 30)]
_NEW_STD_DED = 75000
_NEW_REBATE_LIMIT = 1200000       # 87A: nil tax up to 12L taxable (new regime)
_OLD_SLABS_BASE = [(500000, 5), (1000000, 20), (math.inf, 30)]
_OLD_STD_DED = 50000
_OLD_REBATE_LIMIT = 500000


def income_tax(income, regime="new", fy=_TAX_FY, age=30, salaried=True):
    """Simplified India income tax. Applies standard deduction (salaried),
    slab tax, 87A rebate with marginal relief, and 4% cess. Excludes surcharge
    (very high incomes), chapter-VIA deductions beyond std ded, and special-rate
    income — an estimate, not a filing. Always returns regime + fy for provenance."""
    regime = regime.lower()
    if regime == "new":
        std = _NEW_STD_DED if salaried else 0
        taxable = max(0, income - std)
        tax = _slab_tax(taxable, _NEW_SLABS)
        if taxable <= _NEW_REBATE_LIMIT:
            tax = 0.0
        else:                                    # marginal relief above 12L
            tax = min(tax, taxable - _NEW_REBATE_LIMIT)
    else:
        exempt = 250000 if age < 60 else (300000 if age < 80 else 500000)
        std = _OLD_STD_DED if salaried else 0
        taxable = max(0, income - std)
        slabs = [(exempt, 0)] + _OLD_SLABS_BASE
        tax = _slab_tax(taxable, slabs)
        if taxable <= _OLD_REBATE_LIMIT:
            tax = 0.0
    cess = tax * 0.04
    total = tax + cess
    return {"regime": regime, "fy": fy, "taxable_income": round(taxable),
            "tax_before_cess": round(tax), "cess": round(cess),
            "total_tax": round(total),
            "effective_rate_pct": round(total / income * 100, 2) if income else 0,
            "note": "estimate — excludes surcharge, 80C/80D etc.; not a filing"}


def hra_exemption(basic, hra_received, rent_paid, metro=True):
    """Section 10(13A) HRA exemption = least of the three limbs (old regime)."""
    limbs = [hra_received,
             max(0, rent_paid - 0.10 * basic),
             (0.50 if metro else 0.40) * basic]
    exempt = max(0, min(limbs))
    return {"exempt": round(exempt), "taxable_hra": round(hra_received - exempt),
            "metro": metro}


def gratuity(monthly_basic_da, years):
    """Payment of Gratuity Act: 15/26 x last basic+DA x years (>=6 months rounds
    up), capped at ₹20 lakh."""
    yrs = int(years) + (1 if (years - int(years)) * 12 >= 6 else 0)
    g = 15 / 26 * monthly_basic_da * yrs
    capped = min(g, 2000000)
    return {"gratuity": round(capped), "years_counted": yrs,
            "capped": g > 2000000}


def nps_maturity(monthly, years, annual_rate=10, annuity_pct=40, annuity_rate=6):
    """NPS corpus at 60; >=40% must buy an annuity, up to 60% is tax-free lump."""
    corpus = sip_future_value(monthly, annual_rate, years)["future_value"]
    annuity_corpus = corpus * annuity_pct / 100
    lump = corpus - annuity_corpus
    pension = annuity_corpus * annuity_rate / 100 / 12
    return {"corpus": round(corpus), "lumpsum": round(lump),
            "annuity_corpus": round(annuity_corpus),
            "monthly_pension": round(pension), "annuity_pct": annuity_pct}


def epf_maturity(monthly_basic, years, annual_rate=8.25, growth_pct=5):
    """EPF corpus: employee 12% + employer 3.67% to EPF, salary growing yearly,
    balance compounding at the notified EPF rate."""
    contrib = monthly_basic * (12 + 3.67) / 100
    bal = 0.0
    for _ in range(int(years)):
        for _m in range(12):
            bal = bal * (1 + annual_rate / 100 / 12) + contrib
        contrib *= (1 + growth_pct / 100)
    return _rated({"corpus": round(bal)}, annual_rate,
                  assumed=annual_rate == 8.25, what="EPF rate (FY2024-25)")


def nsc_maturity(principal, annual_rate=7.7, years=5):
    fv = principal * (1 + annual_rate / 100) ** years
    return _rated({"maturity": round(fv), "invested": round(principal),
                   "interest": round(fv - principal)}, annual_rate,
                  assumed=annual_rate == 7.7, what="NSC rate")


def ssy_maturity(annual_deposit, annual_rate=8.2, deposit_years=15,
                 maturity_years=21):
    """Sukanya Samriddhi: deposits for 15 years, matures at 21."""
    fv = 0.0
    for yr in range(int(maturity_years)):
        if yr < int(deposit_years):
            fv += annual_deposit
        fv *= (1 + annual_rate / 100)
    invested = annual_deposit * int(deposit_years)
    return _rated({"maturity": round(fv), "invested": round(invested),
                   "interest": round(fv - invested)}, annual_rate,
                  assumed=annual_rate == 8.2, what="SSY rate")


def scss_maturity(principal, annual_rate=8.2, years=5):
    """Senior Citizens' Savings Scheme: interest paid out quarterly (not
    compounded)."""
    total_interest = principal * annual_rate / 100 * years
    return _rated({"principal": round(principal),
                   "quarterly_payout": round(principal * annual_rate / 100 / 4),
                   "total_interest": round(total_interest),
                   "maturity_value": round(principal)}, annual_rate,
                  assumed=annual_rate == 8.2, what="SCSS rate")


def gst(amount, rate, inclusive=False):
    if inclusive:
        base = amount * 100 / (100 + rate)
        tax = amount - base
        return {"base": round(base), "gst": round(tax), "total": round(amount)}
    tax = amount * rate / 100
    return {"base": round(amount), "gst": round(tax), "total": round(amount + tax)}


def rule_of_72(annual_rate):
    return {"years_to_double": round(72 / annual_rate, 1)}


def retirement_corpus(monthly_expense_today, years_to_retire,
                      years_in_retirement, inflation_pct=6, return_pct=8):
    """Corpus needed at retirement to fund an inflation-growing expense stream."""
    fut_monthly = monthly_expense_today * (1 + inflation_pct / 100) ** years_to_retire
    annual_at_retire = fut_monthly * 12
    real = (1 + return_pct / 100) / (1 + inflation_pct / 100) - 1
    n = years_in_retirement
    if abs(real) < 1e-9:
        corpus = annual_at_retire * n
    else:
        corpus = annual_at_retire * (1 - (1 + real) ** -n) / real
    return {"corpus_needed": round(corpus),
            "monthly_expense_at_retirement": round(fut_monthly),
            "real_return_pct": round(real * 100, 2)}


def term_cover(annual_income, years_to_retire, existing_cover=0, liabilities=0):
    """Income-replacement term-insurance gap (human life value, simple)."""
    replacement = annual_income * years_to_retire
    recommended = max(0, replacement + liabilities - existing_cover)
    return {"recommended_cover": round(recommended),
            "income_replacement": round(replacement),
            "method": "income x working-years + liabilities - existing cover"}


DISPATCH = {
    "emi": emi, "sip": sip_future_value, "sip_future_value": sip_future_value,
    "step_up_sip": step_up_sip, "goal_sip": goal_sip,
    "lumpsum": lumpsum_future_value, "fd": fd_maturity, "rd": rd_maturity,
    "ppf": ppf_maturity, "swp": swp, "cagr": cagr,
    "compound_interest": compound_interest, "inflation": inflation_adjusted,
    "simple_interest": simple_interest,
    "credit_card_payoff": credit_card_payoff, "cc_payoff": credit_card_payoff,
    "credit_card_min_trap": credit_card_min_trap, "cc_min_trap": credit_card_min_trap,
    "loan_prepayment": loan_prepayment, "prepayment": loan_prepayment,
    "xirr": xirr, "income_tax": income_tax, "tax": income_tax,
    "hra": hra_exemption, "hra_exemption": hra_exemption,
    "gratuity": gratuity, "nps": nps_maturity, "nps_maturity": nps_maturity,
    "epf": epf_maturity, "epf_maturity": epf_maturity,
    "nsc": nsc_maturity, "ssy": ssy_maturity, "sukanya": ssy_maturity,
    "scss": scss_maturity, "gst": gst, "rule_of_72": rule_of_72,
    "retirement_corpus": retirement_corpus, "retirement": retirement_corpus,
    "term_cover": term_cover, "term_insurance": term_cover,
}


def dispatch(tool, args):
    """Execute a parsed tool call. Serving calls this on the assistant tool turn."""
    fn = DISPATCH.get(tool)
    if not fn:
        return {"error": f"unknown tool {tool}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad args for {tool}: {e}"}


# ------------------------------------------------------------- SFT generation
# Four-turn conversations teaching: recognise intent -> emit tool call -> (tool
# result) -> present with markdown. lang alternates en / hinglish.

def _fmt_inr(x):
    """Indian digit grouping: 8637369 -> ₹86,37,369 (exact, for EMIs/small sums)."""
    x = round(x)
    sign = "-" if x < 0 else ""
    s = str(abs(x))
    if len(s) <= 3:
        grouped = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}₹{grouped}"


def _inr_compact(x):
    """Lakh/crore words for large sums: 8637369 -> ₹86.37 lakh (reads naturally)."""
    x = round(x)
    a = abs(x)
    if a >= 10 ** 7:
        return f"₹{x / 10 ** 7:.2f}".rstrip("0").rstrip(".") + " crore"
    if a >= 10 ** 5:
        return f"₹{x / 10 ** 5:.2f}".rstrip("0").rstrip(".") + " lakh"
    return _fmt_inr(x)


def _make_example(idx, spec):
    tool, args, q_en, q_hi, present = spec
    result = dispatch(tool, args)
    lang = "hi" if idx % 3 else "en"          # ~2/3 hinglish, like the corpus
    q = q_hi if lang == "hi" else q_en
    assistant_call = json.dumps({"tool": tool, "args": args}, ensure_ascii=False)
    tool_turn = json.dumps(result, ensure_ascii=False)
    final = present(result, lang)
    return {
        "id": f"calc-{idx:04d}", "topic": "calculators", "lang": lang,
        "source": "calc_tools:deterministic", "reviewer": "deterministic",
        "safety": "tool-call-authoring", "licence": "first-party-project-content",
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_call},
            {"role": "tool", "content": tool_turn},
            {"role": "assistant", "content": final},
        ],
    }


def _emi_present(r, lang):
    if lang == "hi":
        return (f"Aapki EMI **{_fmt_inr(r['emi'])}/month** hogi. Poore {r['months']//12} "
                f"saal mein total {_inr_compact(r['total_payment'])} bharoge — usme "
                f"**{_inr_compact(r['total_interest'])} interest** hai. Tenure kam karoge to "
                f"EMI badhegi par total interest kaafi ghatega.")
    return (f"Your EMI is **{_fmt_inr(r['emi'])}/month**. Over {r['months']//12} years you "
            f"repay {_inr_compact(r['total_payment'])} in all, of which "
            f"**{_inr_compact(r['total_interest'])} is interest**. A shorter tenure raises the "
            f"EMI but cuts total interest sharply.")


def _sip_present(r, lang):
    if lang == "hi":
        return (f"Is SIP se maturity par ~**{_inr_compact(r['future_value'])}** ban sakta hai. "
                f"Aapne total {_inr_compact(r['invested'])} invest kiya, "
                f"aur **{_inr_compact(r['gain'])}** growth (assumed return). Returns guaranteed "
                f"nahi hain; market ke hisaab se kam-zyada ho sakte hain.")
    return (f"This SIP could grow to ~**{_inr_compact(r['future_value'])}** at maturity — you "
            f"invest {_inr_compact(r['invested'])} in total, with **{_inr_compact(r['gain'])}** of "
            f"assumed growth. Returns aren't guaranteed and vary with the market.")


def build_specs():
    """A small seed grid; scale by widening the value lists. Realistic India params."""
    specs = []
    for p, rate, yrs in [(4000000, 9, 20), (2500000, 8.5, 15), (500000, 14, 3),
                         (800000, 10.5, 5), (10000000, 8.75, 25)]:
        specs.append(("emi", {"principal": p, "annual_rate": rate, "years": yrs},
                      f"A {_fmt_inr(p)} loan at {rate}% for {yrs} years — what's the EMI?",
                      f"{_fmt_inr(p)} ka loan {rate}% pe {yrs} saal ke liye — EMI kitni hogi?",
                      _emi_present))
    for m, rate, yrs in [(5000, 12, 15), (10000, 11, 20), (25000, 12, 10),
                         (2000, 12, 25), (15000, 10, 7)]:
        specs.append(("sip", {"monthly": m, "annual_rate": rate, "years": yrs},
                      f"If I invest {_fmt_inr(m)}/month in an SIP at {rate}% for {yrs} years, "
                      f"what will it grow to?",
                      f"{_fmt_inr(m)}/month SIP {rate}% return pe {yrs} saal — kitna banega?",
                      _sip_present))
    return specs


def gen_sft(n, out):
    specs = build_specs()
    rows = [_make_example(i, specs[i % len(specs)]) for i in range(n)]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tool-call examples -> {out}")
    print("NOTE: widen build_specs() value lists for more coverage; add goal_sip/"
          "fd/ppf/rd/swp presenters the same way. Keep params realistic.")


def selftest():
    assert emi(4000000, 9, 20)["emi"] == 35989, emi(4000000, 9, 20)
    assert lumpsum_future_value(100000, 10, 10)["future_value"] == 259374
    assert goal_sip(10000000, 12, 15)["monthly_sip"] > 0
    assert sip_future_value(10000, 12, 20)["future_value"] > sip_future_value(10000, 12, 20)["invested"]
    print("selftest OK — emi(40L,9%,20y)=₹35,989, lumpsum(1L,10%,10y)=₹2,59,374")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    g = sub.add_parser("gen-sft")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        gen_sft(a.n, a.out)


if __name__ == "__main__":
    main()
