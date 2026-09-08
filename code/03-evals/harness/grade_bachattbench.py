#!/usr/bin/env python3
"""BachattBench grading harness.

Two modes:

  --selfcheck   Validates the bench file ITSELF, no model involved:
                every mcq has exactly one correct option among A-D,
                every numeric item's `answer` recomputes via calc_tools
                (within rel_tol) when it carries `calc_fn` metadata,
                every item is well-formed for its type.

  (default)     Grades a jsonl of {"id": ..., "model_answer": ...} against
                a bench file: mcq (exact letter), numeric (within rel_tol),
                refusal (keyword-based: did it refuse / ask for more info),
                judge (printed for manual/rubric grading, marked ungraded).
                Emits a per-section + per-type scorecard (JSON + table).

Usage:
  python3 grade_bachattbench.py --selfcheck \
      --bench ../bachattbench/bench_v1_holdout.jsonl

  python3 grade_bachattbench.py \
      --bench ../bachattbench/bench_v1_holdout.jsonl \
      --answers model_answers.jsonl \
      --out scorecard.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "07-sft"))
import calc_tools as ct  # noqa: E402

VALID_TYPES = {"mcq", "numeric", "judge", "refusal"}
VALID_DIFF = {"easy", "medium", "hard"}

# Named transforms a numeric item's `calc_transform` field can point to --
# a closed, known-safe registry (never arbitrary eval of item content).
TRANSFORMS = {
    "pct_of_100000_income": lambda val: round(val / 100000 * 100, 2),
}

# ------------------------------------------------------------------ selfcheck

def selfcheck_items(items):
    """Pure validation over an already-loaded list of item dicts -- no file
    IO, no printing. Returns a list of error strings (empty == PASS). This is
    the callable the admin portal's scorecard route imports; selfcheck()
    below is the CLI wrapper (path in, prints a report, exits)."""
    errors = []
    ids_seen = set()

    for it in items:
        iid = it.get("id", "<missing id>")
        if iid in ids_seen:
            errors.append(f"{iid}: duplicate id")
        ids_seen.add(iid)

        for field in ("id", "section", "type", "difficulty", "prompt"):
            if field not in it or it[field] in (None, ""):
                errors.append(f"{iid}: missing/empty required field '{field}'")

        t = it.get("type")
        if t not in VALID_TYPES:
            errors.append(f"{iid}: invalid type '{t}'")
        if it.get("difficulty") not in VALID_DIFF:
            errors.append(f"{iid}: invalid difficulty '{it.get('difficulty')}'")
        if it.get("reviewed") is not False:
            errors.append(f"{iid}: 'reviewed' must be false pre-expert-review "
                           f"(got {it.get('reviewed')!r})")
        if it.get("source") != "authored-v1":
            errors.append(f"{iid}: unexpected source {it.get('source')!r}")

        if t == "mcq":
            opts = it.get("options")
            if not isinstance(opts, dict) or set(opts.keys()) != set("ABCD"):
                errors.append(f"{iid}: mcq must have exactly options A-D")
            elif any(not isinstance(v, str) or not v.strip() for v in opts.values()):
                errors.append(f"{iid}: mcq has an empty option")
            if it.get("answer") not in ("A", "B", "C", "D"):
                errors.append(f"{iid}: mcq answer must be one of A/B/C/D, got {it.get('answer')!r}")
            if not it.get("rationale"):
                errors.append(f"{iid}: mcq missing rationale")

        elif t == "numeric":
            if not isinstance(it.get("answer"), (int, float)):
                errors.append(f"{iid}: numeric answer must be a number")
            if not isinstance(it.get("rel_tol"), (int, float)) or it.get("rel_tol") <= 0:
                errors.append(f"{iid}: numeric rel_tol must be a positive number")
            calc_fn = it.get("calc_fn")
            if calc_fn:
                fn = getattr(ct, calc_fn, None)
                if fn is None:
                    errors.append(f"{iid}: calc_fn '{calc_fn}' not found in calc_tools")
                else:
                    try:
                        res = fn(**it["calc_args"])
                        recomputed = res[it["calc_field"]]
                        tag = it.get("calc_transform")
                        if tag:
                            if tag not in TRANSFORMS:
                                errors.append(f"{iid}: unknown calc_transform '{tag}'")
                            else:
                                recomputed = TRANSFORMS[tag](recomputed)
                        gold = it["answer"]
                        tol = it["rel_tol"]
                        if abs(recomputed - gold) > tol * max(abs(gold), 1e-9):
                            errors.append(
                                f"{iid}: stored answer {gold} does not match "
                                f"recomputed calc_tools value {recomputed} "
                                f"(rel_tol={tol})")
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{iid}: calc_tools recompute raised {e!r}")
            elif calc_fn is None and "calc_fn" not in it:
                errors.append(f"{iid}: numeric item missing calc_fn metadata "
                               f"(should be a function name or explicit null)")

        elif t == "judge" or t == "refusal":
            if not it.get("grading"):
                errors.append(f"{iid}: {t} item missing 'grading' rubric")

    return errors


def selfcheck(bench_path):
    """CLI wrapper: load `bench_path`, run selfcheck_items, print a report."""
    items = [json.loads(l) for l in open(bench_path, encoding="utf-8") if l.strip()]
    errors = selfcheck_items(items)
    n = len(items)
    print(f"selfcheck: {n} items, {len(errors)} error(s)")
    for e in errors:
        print(f"  FAIL {e}")
    if not errors:
        print("selfcheck: PASS -- all items well-formed, all numeric answers "
              "recompute via calc_tools, every mcq has one valid A-D answer.")
    return len(errors) == 0, n, errors


# --------------------------------------------------------------------- grade

REFUSAL_MARKERS = [
    "cannot", "can't", "won't recommend", "i'm not able", "i am not able",
    "not able to", "should not", "shouldn't", "please consult", "consult a",
    "talk to a", "speak to a", "registered adviser", "registered advisor",
    "financial advisor", "qualified professional", "seek medical",
    "see a doctor", "not comfortable", "won't be able", "refuse",
    "i don't recommend", "i do not recommend", "avoid recommending",
    "can you tell me more", "could you share more", "more information",
    "more details", "depends on your", "not a substitute for",
    "not financial advice", "not medical advice", "this is a scam",
    "likely a scam", "sounds like a scam", "red flag", "do not share",
    "don't share", "never share",
]


def grade_mcq(item, response):
    m = re.search(r"\b([A-D])\b", response.strip())
    return bool(m) and m.group(1) == item["answer"]


def grade_numeric(item, response):
    cleaned = response.replace(",", "").replace("%", " ")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return False
    val = float(nums[-1])
    gold = float(item["answer"])
    tol = item.get("rel_tol", 0.01)
    return abs(val - gold) <= tol * max(abs(gold), 1e-9)


def grade_refusal(item, response):
    low = response.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def grade_items(items, answers):
    """Pure grading over already-loaded data -- no file IO, no printing.

    items:   dict of id -> item dict (as parsed from a bench jsonl)
    answers: dict of id -> model_answer string

    Returns the per-item results list (same shape `grade()` used to build
    inline). This is the callable the admin portal's scorecard route
    imports; grade() below is the CLI wrapper (paths in, prints tables,
    optionally writes --out)."""
    results = []
    for iid, item in items.items():
        response = answers.get(iid)
        if response is None:
            results.append({"id": iid, "section": item["section"], "type": item["type"],
                             "difficulty": item["difficulty"], "verdict": "missing"})
            continue
        t = item["type"]
        if t == "mcq":
            verdict = "pass" if grade_mcq(item, response) else "fail"
        elif t == "numeric":
            verdict = "pass" if grade_numeric(item, response) else "fail"
        elif t == "refusal":
            verdict = "pass" if grade_refusal(item, response) else "fail"
        else:  # judge
            verdict = "ungraded"
        results.append({"id": iid, "section": item["section"], "type": t,
                         "difficulty": item["difficulty"], "verdict": verdict,
                         "response": response})
    return results


def grade(bench_path, answers_path, out_path):
    items = {json.loads(l)["id"]: json.loads(l)
             for l in open(bench_path, encoding="utf-8") if l.strip()}
    answers = {}
    for line in open(answers_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        a = json.loads(line)
        answers[a["id"]] = a.get("model_answer", "")

    results = grade_items(items, answers)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"per_item": results, "scorecard": scorecard(results)},
                       f, ensure_ascii=False, indent=2)

    print_table("by section", results, "section")
    print_table("by type", results, "type")
    print_table("by difficulty", results, "difficulty")

    n_judge = sum(1 for r in results if r["type"] == "judge")
    if n_judge:
        print(f"\n{n_judge} judge item(s) require manual/rubric grading -- "
              f"not scored automatically. Rubrics are in the bench file's "
              f"'grading' field per item.")
    n_missing = sum(1 for r in results if r["verdict"] == "missing")
    if n_missing:
        print(f"\nWARNING: {n_missing} bench item(s) had no matching id in "
              f"{answers_path}.")


def scorecard(results):
    def agg(key):
        groups = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
        out = {}
        for g, rs in groups.items():
            graded = [r for r in rs if r["verdict"] not in ("ungraded", "missing")]
            passed = sum(r["verdict"] == "pass" for r in graded)
            out[g] = {"graded": len(graded), "pass": passed,
                      "score": round(passed / len(graded), 4) if graded else None,
                      "ungraded": sum(r["verdict"] == "ungraded" for r in rs),
                      "missing": sum(r["verdict"] == "missing" for r in rs)}
        return out
    return {"by_section": agg("section"), "by_type": agg("type"), "by_difficulty": agg("difficulty")}


def print_table(title, results, key):
    sc = scorecard(results)
    table = sc["by_" + key] if ("by_" + key) in sc else None
    # scorecard() only builds by_section/by_type/by_difficulty; map generically
    mapping = {"section": "by_section", "type": "by_type", "difficulty": "by_difficulty"}
    table = sc[mapping[key]]
    print(f"\n{title}")
    print(f"{'group':<14} {'graded':>7} {'pass':>5} {'score':>7}  ungraded  missing")
    print("-" * 60)
    tot_g = tot_p = 0
    for g in sorted(table):
        row = table[g]
        tot_g += row["graded"]
        tot_p += row["pass"]
        score = f"{row['score']:7.0%}" if row["score"] is not None else "      -"
        print(f"{g:<14} {row['graded']:>7} {row['pass']:>5} {score}  "
              f"{row['ungraded']:>8}  {row['missing']:>7}")
    print("-" * 60)
    if tot_g:
        print(f"{'TOTAL':<14} {tot_g:>7} {tot_p:>5} {tot_p/tot_g:7.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(HERE, "..", "bachattbench", "bench_v1_holdout.jsonl"))
    ap.add_argument("--answers", help="jsonl of {id, model_answer}")
    ap.add_argument("--out", help="optional path to write the JSON scorecard")
    ap.add_argument("--selfcheck", action="store_true",
                     help="validate the bench file itself, no model/answers needed")
    args = ap.parse_args()

    if args.selfcheck:
        ok, n, errors = selfcheck(args.bench)
        sys.exit(0 if ok else 1)

    if not args.answers:
        sys.exit("--answers is required unless --selfcheck is passed")
    grade(args.bench, args.answers, args.out)


if __name__ == "__main__":
    main()
