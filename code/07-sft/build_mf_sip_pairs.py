#!/usr/bin/env python3
"""Map the MF/SIP question bank (S061) onto canonical intent answers.

Many phrasings -> one answer: the paraphrase-expansion training that binds
answers to question MEANING rather than exact wording (the anti-blending
fix for small models). First matching intent wins; unmatched questions land
in mf_sip_unmapped.txt for the next authoring tranche. A held-out sample of
mapped questions is reserved for generative eval (never trained).

Usage:  python build_mf_sip_pairs.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "..", "04-training-stack", "MF_SIP_Questions.md")
HOLDOUT_EVERY = 12          # every 12th mapped question -> eval holdout


def main():
    intents = []
    for line in open(os.path.join(HERE, "mf_sip_intents.jsonl"),
                     encoding="utf-8"):
        d = json.loads(line)
        d["rx"] = [re.compile(p, re.I) for p in d["patterns"]]
        intents.append(d)

    questions = []
    for line in open(BANK, encoding="utf-8"):
        m = re.match(r"^(\d+)\. (.+)$", line.strip())
        if m:
            questions.append((int(m.group(1)), m.group(2)))

    mapped, unmapped, holdout = [], [], []
    counts = {}
    for n, q in questions:
        hit = None
        for it in intents:
            if any(r.search(q) for r in it["rx"]):
                hit = it
                break
        if not hit:
            unmapped.append(f"{n}. {q}")
            continue
        counts[hit["intent"]] = counts.get(hit["intent"], 0) + 1
        row = {"id": f"ms-{n:04d}", "source": "S061-mf-sip-question-bank",
               "topic": "sip" if "sip" in hit["intent"] else "mf",
               "lang": "hi", "intent": hit["intent"],
               "instruction": q, "response": hit["response"],
               "reviewer": "unreviewed-auto-intent-mapping",
               "safety": "unreviewed-auto-intent-mapping",
               "licence": "first-party-project-content"}
        total_mapped = len(mapped) + len(holdout)
        if total_mapped % HOLDOUT_EVERY == HOLDOUT_EVERY - 1:
            holdout.append(row)
        else:
            mapped.append(row)

    out = os.path.join(HERE, "corpus_v2", "mf_sip_paraphrase.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in mapped:
            r = dict(r)
            r.pop("intent")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "mf_sip_eval_holdout.jsonl"), "w",
              encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "mf_sip_unmapped.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(unmapped) + "\n")

    print(f"{len(questions)} questions | mapped {len(mapped)} train + "
          f"{len(holdout)} eval-holdout | unmapped {len(unmapped)} "
          f"(-> next tranche)")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
