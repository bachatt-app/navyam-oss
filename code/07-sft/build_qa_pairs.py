#!/usr/bin/env python3
"""Generic question-bank -> intent-answer pair builder (S061/S062/... pattern).

Same design as build_mf_sip_pairs.py but parameterized: numbered-markdown
bank + intents jsonl in, paraphrase pairs + eval holdout + unmapped list out.
Many phrasings -> one canonical answer binds responses to question MEANING
(the anti-blending recipe for small models). First matching intent wins.

Usage:
  python build_qa_pairs.py --bank ../04-training-stack/Loans.md \
      --intents loans_intents.jsonl --prefix loans --source S062-loans-bank \
      [--id-prefix ln] [--holdout-every 12]
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--intents", required=True)
    ap.add_argument("--prefix", required=True,
                    help="basename for outputs (corpus_v2/<prefix>_paraphrase"
                         ".jsonl, <prefix>_eval_holdout.jsonl, "
                         "<prefix>_unmapped.txt)")
    ap.add_argument("--source", required=True,
                    help="provenance tag, e.g. S062-loans-bank")
    ap.add_argument("--id-prefix", default=None)
    ap.add_argument("--holdout-every", type=int, default=12)
    args = ap.parse_args()
    idp = args.id_prefix or args.prefix[:2]

    intents = []
    for line in open(args.intents, encoding="utf-8"):
        d = json.loads(line)
        d["rx"] = [re.compile(p, re.I) for p in d["patterns"]]
        intents.append(d)

    questions = []
    for line in open(args.bank, encoding="utf-8"):
        m = re.match(r"^(\d+)\. (.+)$", line.strip())
        if m:
            questions.append((int(m.group(1)), m.group(2)))

    mapped, unmapped, holdout = [], [], []
    counts = {}
    for n, q in questions:
        hit = next((it for it in intents
                    if any(r.search(q) for r in it["rx"])), None)
        if not hit:
            unmapped.append(f"{n}. {q}")
            continue
        counts[hit["intent"]] = counts.get(hit["intent"], 0) + 1
        row = {"id": f"{idp}-{n:04d}", "source": args.source,
               "topic": args.prefix, "lang": "hi", "intent": hit["intent"],
               "instruction": q, "response": hit["response"],
               "reviewer": "unreviewed-auto-intent-mapping",
               "safety": "unreviewed-auto-intent-mapping",
               "licence": "first-party-project-content"}
        total_mapped = len(mapped) + len(holdout)
        if total_mapped % args.holdout_every == args.holdout_every - 1:
            holdout.append(row)
        else:
            mapped.append(row)

    out = os.path.join(HERE, "corpus_v2", f"{args.prefix}_paraphrase.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in mapped:
            r = dict(r)
            r.pop("intent")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, f"{args.prefix}_eval_holdout.jsonl"), "w",
              encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, f"{args.prefix}_unmapped.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(unmapped) + "\n")

    print(f"{len(questions)} questions | mapped {len(mapped)} train + "
          f"{len(holdout)} eval-holdout | unmapped {len(unmapped)}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
