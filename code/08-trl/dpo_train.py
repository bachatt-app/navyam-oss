#!/usr/bin/env python3
"""DPO on the HF export with TRL — the first automated post-training stage.

Consumes pairs from build_dpo_pairs.py (conversational format; the chat
template installed by hf_chat_setup.py renders them, so tokenization is
byte-identical to the from-scratch stack). Reference model is a frozen copy
of the policy — at 151-338M both fit any of our GPUs.

Usage:
  python dpo_train.py --hf-dir .../navya-1a-sft2/hf --pairs dpo_pairs.jsonl \
      --out .../navya-1a-dpo [--beta 0.1] [--epochs 2] [--lr 5e-7]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    assert tok.chat_template, "run hf_chat_setup.py first (parity gate)"

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    data = Dataset.from_list(
        [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
    print(f"{len(data)} preference pairs")

    model = AutoModelForCausalLM.from_pretrained(args.hf_dir)
    cfg = DPOConfig(
        output_dir=args.out, beta=args.beta,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=10, save_strategy="epoch", bf16=True,
        max_length=2048, max_prompt_length=1536,
        report_to=[], seed=1337)
    trainer = DPOTrainer(model=model, args=cfg,
                         processing_class=tok, train_dataset=data)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"DPO checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
