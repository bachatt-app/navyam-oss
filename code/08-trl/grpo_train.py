#!/usr/bin/env python3
"""GRPO on the HF export — on-policy stage, run after DPO plateaus.

Same reward functions as the DPO pair engine (rewards.py), so the two stages
optimize one target. The dataset is prompts only; TRL generates groups of
completions during training and advantages are computed within each group.
More GPU-hungry than DPO (generation in the loop) — prefer the A100.

Usage:
  python grpo_train.py --hf-dir .../navya-1a-dpo --prompts \
      ../07-sft/corpus_v2/mf_sip_paraphrase.jsonl --out .../navya-1a-grpo
"""

import argparse
import json

import rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--prompts", required=True,
                    help="jsonl rows with instruction (+optional response "
                         "used as the canonical reward anchor)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    assert tok.chat_template, "run hf_chat_setup.py first (parity gate)"

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    data = Dataset.from_list([{
        "prompt": [{"role": "user", "content": r["instruction"]}],
        "canonical": r.get("response", ""),
    } for r in rows])
    print(f"{len(data)} prompts, group size {args.group_size}")

    cfg = GRPOConfig(
        output_dir=args.out, num_generations=args.group_size,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        max_completion_length=220, max_prompt_length=1536,
        temperature=0.8, logging_steps=5, save_strategy="epoch",
        bf16=True, report_to=[], seed=1337)
    trainer = GRPOTrainer(
        model=args.hf_dir, args=cfg, processing_class=tok,
        reward_funcs=rewards.grpo_reward_funcs(), train_dataset=data)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"GRPO checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
