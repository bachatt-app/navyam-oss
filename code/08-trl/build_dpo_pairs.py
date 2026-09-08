#!/usr/bin/env python3
"""Automated DPO preference-pair engine.

Samples the CURRENT policy on S061 question-bank prompts (plus multi-turn
probes that reproduce the observed blending failure), scores every sample
with rewards.total_reward, and emits pairs only where the policy is actually
wrong:

    chosen   = the canonical intent answer (reward-scored ceiling)
    rejected = the policy's own worst sample, if its reward trails the
               canonical answer's by --margin

Prompts the model already handles are dropped — DPO on already-correct
behavior just sharpens memorization. Output is TRL DPOTrainer's
conversational format: {"prompt": [messages], "chosen": [...], "rejected": [...]}.

Usage (on the GPU box):
  python build_dpo_pairs.py --hf-dir .../navya-1a-sft2/hf \
      --pairs ../07-sft/corpus_v2/mf_sip_paraphrase.jsonl \
      --out dpo_pairs.jsonl [--samples 4] [--margin 0.15] [--limit 0]
"""

import argparse
import json
import random

import rewards

GREETING_PREFIXES = [
    [{"role": "user", "content": "hi"},
     {"role": "assistant", "content": "Hi! I'm Navya. Ask me anything about "
      "money — savings, investments, loans, insurance or tax."}],
    [],  # plain single-turn
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pairs", required=True,
                    help="jsonl with instruction/response (canonical) rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap prompts for a smoke run (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    assert tok.chat_template, \
        "no chat template on export — run hf_chat_setup.py first (parity gate)"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, torch_dtype=torch.bfloat16).to(device).eval()

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    end_turn = tok.convert_tokens_to_ids("<|reserved_4|>")
    kept, skipped_good, n = [], 0, 0
    for i, row in enumerate(rows):
        prefix = GREETING_PREFIXES[i % len(GREETING_PREFIXES)]
        messages = prefix + [{"role": "user", "content": row["instruction"]}]
        canonical = row["response"]

        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt")
        if not torch.is_tensor(ids):        # BatchEncoding in newer HF
            ids = ids["input_ids"]
        ids = ids.to(device)
        with torch.no_grad():
            gen = model.generate(
                ids, do_sample=True, temperature=0.8, top_p=0.95,
                num_return_sequences=args.samples, max_new_tokens=220,
                eos_token_id=end_turn, pad_token_id=tok.pad_token_id or 2)
        samples = [tok.decode(g[ids.shape[1]:], skip_special_tokens=True)
                   .strip() for g in gen]

        scored = sorted(
            ((rewards.total_reward(messages, s, canonical), s)
             for s in samples), key=lambda x: x[0])
        worst_r, worst = scored[0]
        best_r = scored[-1][0]
        canon_r = rewards.total_reward(messages, canonical, canonical)

        if canon_r - best_r < args.margin:
            skipped_good += 1          # policy already answers this well
            continue
        kept.append({
            "prompt": messages,
            "chosen": [{"role": "assistant", "content": canonical}],
            "rejected": [{"role": "assistant", "content": worst}],
            "meta": {"id": row.get("id", f"row-{i}"), "reward_chosen": canon_r,
                     "reward_rejected": worst_r},
        })
        n += 1
        if n % 50 == 0:
            print(f"  {n} pairs from {i + 1} prompts …")

    with open(args.out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts -> {len(kept)} DPO pairs "
          f"({skipped_good} already-good skipped) -> {args.out}")


if __name__ == "__main__":
    main()
