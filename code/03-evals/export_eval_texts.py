#!/usr/bin/env python3
"""Export Tier-1 benchmark texts for pipeline decontamination.

Dumps the test/validation texts of every Tier-1 task into one JSONL that
decontaminate.py consumes alongside BachattBench. Run once (and re-run when a
task is added); the output is committed to blob so the ingest VM can pull it.

Usage:  python export_eval_texts.py --out decontam/tier1_eval_texts.jsonl
"""

import argparse
import json
import os

import io
import urllib.request

import pyarrow.parquet as pq
from datasets import load_dataset


def load_rows(name, config, split):
    """load_dataset, falling back to HF's server-side parquet conversion for
    legacy script-based repos (e.g. piqa)."""
    try:
        return load_dataset(name, config, split=split, trust_remote_code=True)
    except Exception:
        api = (f"https://huggingface.co/api/datasets/{name}/parquet/"
               f"{config or 'plain_text'}/{split}")
        req = urllib.request.Request(api, headers={"User-Agent": "navyam-gpt"})
        urls = json.load(urllib.request.urlopen(req, timeout=120))
        rows = []
        for u in urls:
            r = urllib.request.Request(u, headers={"User-Agent": "navyam-gpt"})
            with urllib.request.urlopen(r, timeout=600) as resp:
                rows.extend(pq.read_table(io.BytesIO(resp.read())).to_pylist())
        return rows

TASKS = [
    ("Rowan/hellaswag", None, "validation",
     lambda r: r["ctx"] + " " + " ".join(r["endings"])),
    ("allenai/ai2_arc", "ARC-Easy", "test",
     lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    ("baber/piqa", None, "validation",
     lambda r: r["goal"] + " " + r["sol1"] + " " + r["sol2"]),
    ("allenai/sciq", None, "test",
     lambda r: r["question"] + " " + r["correct_answer"] + " " +
               r["support"][:500]),
    ("allenai/winogrande", "winogrande_xl", "validation",
     lambda r: r["sentence"] + " " + r["option1"] + " " + r["option2"]),
    ("EleutherAI/lambada_openai", "en", "test",
     lambda r: r["text"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "decontam", "tier1_eval_texts.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name, config, split, render in TASKS:
            ds = load_rows(name, config, split)
            for row in ds:
                f.write(json.dumps({"task": name, "text": render(row)},
                                   ensure_ascii=False) + "\n")
                n += 1
            print(f"  {name}: {len(ds)} rows")
    print(f"{n} eval texts → {args.out}")


if __name__ == "__main__":
    main()
