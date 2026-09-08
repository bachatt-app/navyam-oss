# 08-trl — automated post-training engine (TRL on HF exports)

The custom stack stays the source of truth for PRE-training. Post-training
(making the pre-trained model *useful*) runs on TRL against our HF exports —
`export_hf.py` already produces `LlamaForCausalLM` + `PreTrainedTokenizerFast`
with 0.00e+00 logit parity, so SFT / DPO / GRPO from TRL apply directly.

## The contract that makes it safe

Everything hinges on ONE thing: the HF chat template must produce byte-for-byte
the same token ids as `07-sft/chat_format.py` (navya-chat-v1 on reserved slots
3-7). `hf_chat_setup.py` installs the template onto an exported tokenizer and
hard-fails unless `apply_chat_template` matches `encode_conversation` /
`encode_prompt` on a battery of conversations. Never run a TRL trainer on an
export that hasn't passed this gate.

## The automated preference engine

The S061 question bank gives us canonical answers per intent. That turns
preference-data creation into a loop with no manual labeling:

1. `build_dpo_pairs.py` samples the CURRENT policy on question-bank prompts
   (plus multi-turn probes for the blending failure mode).
2. Every sample is scored by `rewards.py` (language match, canonical-answer
   overlap, refusal correctness, repetition).
3. chosen = canonical answer, rejected = the model's own low-scoring sample.
   Pairs where the model already answers well are dropped — DPO only trains
   where the policy is actually wrong.
4. `dpo_train.py` (TRL `DPOTrainer`) trains on the pairs.
5. `grpo_train.py` (TRL `GRPOTrainer`) is the on-policy alternative: same
   reward functions, no pair building, more GPU-hungry (needs generation
   during training) — use once DPO plateaus.

`run_trl_pipeline.sh` chains: export → template gate → sample → pairs → DPO →
regression gate. Each stage refuses to continue if the previous gate failed.

## Where it runs

- navya-1a (151M): the A10 (24 GB) fits policy+reference for DPO easily.
- navya-1b (338M): A10 still fine for DPO; GRPO prefer the A100.

## Install (on the GPU box, inside the training venv)

    pip install "trl>=0.9" "transformers>=4.44" datasets accelerate
