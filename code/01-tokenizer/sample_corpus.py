#!/usr/bin/env python3
"""Build a tokenizer-training corpus from cleaned pools.

Takes up to --mb megabytes of text per domain from
02-data-pipeline/pools/<domain>/clean.jsonl and writes one .txt per domain.
Capping per-domain keeps the tokenizer from being dominated by whichever pool
happens to be largest — the vocab should serve the target mix, not the
accidental one.

Usage:
  python sample_corpus.py --pools ../02-data-pipeline/pools --out corpus/ --mb 20
"""

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=20.0, help="cap per domain")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = int(args.mb * 1e6)
    for domain in sorted(os.listdir(args.pools)):
        src = os.path.join(args.pools, domain, "clean.jsonl")
        if not os.path.exists(src):
            continue
        written = 0
        dst = os.path.join(args.out, f"{domain}.txt")
        with open(dst, "w", encoding="utf-8") as out:
            for line in open(src, encoding="utf-8"):
                text = json.loads(line)["text"]
                out.write(text + "\n\n")
                written += len(text.encode("utf-8"))
                if written >= cap:
                    break
        print(f"{domain:<18} {written/1e6:6.1f} MB -> {dst}")


if __name__ == "__main__":
    main()
