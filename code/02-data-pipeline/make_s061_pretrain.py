#!/usr/bin/env python3
"""Render the S061 MF/SIP bank as a pre-training text corpus (1b cooldown).

Each mapped question + canonical answer becomes one plain-prose Q/A document.
The 54-question eval holdout is NEVER included — decontamination for the 1b
generative eval starts here, at authoring time, not at the filter stage.
Unmapped questions are also excluded (no answers exist yet).

Output: jsonl with {id, text, source} rows, the shape shard tokenization
expects. Weight it inside the cooldown mix's finance_qa slice.

Usage:  python make_s061_pretrain.py [--out s061_mf_sip_qa.jsonl]
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(HERE, "..", "07-sft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "s061_mf_sip_qa.jsonl"))
    ap.add_argument("--pairs", default=os.path.join(
        SFT, "corpus_v2", "mf_sip_paraphrase.jsonl"))
    ap.add_argument("--holdout", default=os.path.join(
        SFT, "mf_sip_eval_holdout.jsonl"))
    ap.add_argument("--source", default="S061-mf-sip-question-bank")
    args = ap.parse_args()

    holdout_ids = set()
    with open(args.holdout, encoding="utf-8") as f:
        for line in f:
            holdout_ids.add(json.loads(line)["id"])

    n = 0
    with open(args.pairs, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if row["id"] in holdout_ids:
                raise SystemExit(f"holdout id {row['id']} in train pairs — "
                                 "rebuild pairs before packaging")
            text = (f"Sawaal: {row['instruction']}\n\n"
                    f"Jawaab: {row['response']}")
            fout.write(json.dumps(
                {"id": f"{args.source.split(chr(45))[0].lower()}-{row['id']}", "text": text,
                 "source": args.source},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} Q/A docs -> {args.out} (holdout {len(holdout_ids)} excluded)")


if __name__ == "__main__":
    main()
