#!/usr/bin/env python3
"""Generative eval on the S061 MF/SIP holdout (54 never-trained questions).

Queries an OpenAI-compatible endpoint (the A10 serve.py, or any HF serve)
with each held-out question — single-turn AND with a greeting prefix (the
multi-turn blending failure mode) — and scores replies with the shared
reward functions from 08-trl/rewards.py. Because every holdout question has
a canonical intent answer, canonical-overlap measures whether the model
learned the INTENT rather than memorizing phrasings; the trained paraphrase
siblings of these questions were in SFT, the questions themselves never.

Usage:
  python eval_mf_sip_holdout.py [--endpoint http://YOUR_A10_IP:8000/v1]
      [--limit 0] [--json-out results/mf_sip_holdout.json]
"""

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08-trl"))
import rewards  # noqa: E402

HOLDOUT = os.path.join(HERE, "..", "07-sft", "mf_sip_eval_holdout.jsonl")
GREETING = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hi! I'm Navya. Ask me anything "
             "about money — savings, investments, loans, insurance or tax."}]


def ask(endpoint, messages, timeout=90):
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": "navya", "messages": messages,
                         "max_tokens": 300}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://YOUR_A10_IP:8000/v1")
    ap.add_argument("--holdout", default=HOLDOUT,
                    help="holdout jsonl (default: S061 MF/SIP; pass "
                         "../07-sft/loans_eval_holdout.jsonl for S062)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    results = []
    for i, row in enumerate(rows):
        for mode, prefix in (("single", []), ("greeting", GREETING)):
            messages = prefix + [{"role": "user",
                                  "content": row["instruction"]}]
            try:
                reply = ask(args.endpoint, messages)
            except Exception as exc:  # noqa: BLE001
                print(f"endpoint error on {row['id']}: {exc}")
                sys.exit(1)
            results.append({
                "id": row["id"], "mode": mode, "intent": row["intent"],
                "question": row["instruction"], "reply": reply,
                "reward": rewards.total_reward(messages, reply,
                                               row["response"]),
                "lang_ok": rewards.reward_language_match(
                    messages, reply) > 0,
                "overlap": rewards.reward_canonical_overlap(
                    messages, reply, row["response"]),
            })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)} questions …")

    for mode in ("single", "greeting"):
        sub = [r for r in results if r["mode"] == mode]
        mean_r = sum(r["reward"] for r in sub) / len(sub)
        lang = sum(r["lang_ok"] for r in sub) / len(sub)
        ovl = sum(r["overlap"] for r in sub) / len(sub)
        print(f"{mode:>8}: mean reward {mean_r:+.3f} | language match "
              f"{lang:.0%} | canonical overlap {ovl:+.3f}  (n={len(sub)})")

    worst = sorted(results, key=lambda r: r["reward"])[:5]
    print("\nworst 5:")
    for r in worst:
        print(f"  [{r['reward']:+.2f} {r['mode']}] {r['question'][:60]}")
        print(f"      -> {r['reply'][:100]}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
