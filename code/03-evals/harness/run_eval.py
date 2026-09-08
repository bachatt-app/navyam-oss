#!/usr/bin/env python3
"""BachattBench harness v0.

Grades `mcq` and `numeric` items automatically; records `judge` items as
ungraded (judge-model grading + human audit is harness v1).

Backends:
  oracle  - answers with the gold answer (self-test of the grading path; expect 100%)
  echo    - answers with the empty string (mechanics check; expect 0%)
  openai  - any OpenAI-compatible endpoint (vLLM, SGLang, hosted).
            Env: OPENAI_BASE_URL, OPENAI_API_KEY. Flag: --model.

Usage:
  python run_eval.py --items ../bachattbench/seed_v0.jsonl --backend oracle
"""

import argparse
import json
import os
import re
import sys
import urllib.request

MCQ_INSTRUCTION = "\n\nAnswer with the single letter of the correct option."
NUM_INSTRUCTION = "\n\nAnswer with the final number only."


def backend_oracle(item, _model):
    if item["type"] == "mcq":
        return item["answer"]
    if item["type"] == "numeric":
        return str(item["answer"])
    return "(judge item; oracle has no gold text)"


def backend_echo(item, _model):
    return ""


def backend_openai(item, model):
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY", "none")
    if not base or not model:
        sys.exit("openai backend needs OPENAI_BASE_URL env and --model")
    prompt = item["prompt"]
    if item["type"] == "mcq":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        prompt = f"{prompt}\n\n{opts}{MCQ_INSTRUCTION}"
    elif item["type"] == "numeric":
        prompt += NUM_INSTRUCTION
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


BACKENDS = {"oracle": backend_oracle, "echo": backend_echo,
            "openai": backend_openai}


def grade_mcq(item, response: str):
    m = re.search(r"\b([A-D])\b", response.strip())
    return bool(m) and m.group(1) == item["answer"]


def grade_numeric(item, response: str):
    # last number in the response, tolerant of Rs / commas / % / lakh-crore words
    cleaned = response.replace(",", "").replace("%", " ")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return False
    val = float(nums[-1])
    gold = float(item["answer"])
    tol = item.get("rel_tol", 0.01)
    return abs(val - gold) <= tol * abs(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--backend", choices=BACKENDS, default="oracle")
    ap.add_argument("--model")
    ap.add_argument("--out", help="optional JSONL of per-item results")
    args = ap.parse_args()

    ask = BACKENDS[args.backend]
    results = []
    for line in open(args.items, encoding="utf-8"):
        item = json.loads(line)
        response = ask(item, args.model)
        if item["type"] == "mcq":
            verdict = "pass" if grade_mcq(item, response) else "fail"
        elif item["type"] == "numeric":
            verdict = "pass" if grade_numeric(item, response) else "fail"
        else:
            verdict = "ungraded"
        results.append({"id": item["id"], "section": item["section"],
                        "type": item["type"], "verdict": verdict,
                        "response": response})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sections: dict[str, list] = {}
    for r in results:
        sections.setdefault(r["section"], []).append(r)
    print(f"\n{'section':<14} {'graded':>7} {'pass':>5} {'score':>7}  ungraded")
    print("-" * 50)
    tot_g = tot_p = 0
    for sec in sorted(sections):
        rs = sections[sec]
        graded = [r for r in rs if r["verdict"] != "ungraded"]
        passed = sum(r["verdict"] == "pass" for r in graded)
        tot_g += len(graded)
        tot_p += passed
        score = f"{passed/len(graded):7.0%}" if graded else "      -"
        print(f"{sec:<14} {len(graded):>7} {passed:>5} {score}  "
              f"{len(rs) - len(graded)}")
    print("-" * 50)
    if tot_g:
        print(f"{'TOTAL':<14} {tot_g:>7} {tot_p:>5} {tot_p/tot_g:7.0%}")


if __name__ == "__main__":
    main()
