# 04 — Training stack v1

The executable version of `main.tex`, upgraded to the frozen starting recipe
(execution plan, Workstream 3): RoPE, GQA, RMSNorm pre-norm, SwiGLU, no biases,
tied embeddings, next-token objective.

## Role of this code

| Scale | What runs it |
| ----- | ------------ |
| smoke (~3M) | this trainer, CPU/MPS — CI-style sanity check |
| 300M–1B research runs | this trainer + DDP/FSDP increment, rented GPU nodes |
| 8B+ (Alpha and beyond) | adopted distributed framework (Megatron-class — decision log, month 2) |

`model.py` is the **executable spec**: whatever framework we adopt must
reproduce its forward pass bit-for-bit at fp32 on a fixed batch. That test is
the acceptance gate for the framework decision.

## Phase-0 requirements built in

- **Deterministic sampling:** the batch at step *k* depends only on
  `(seed, k)` — a resumed run consumes exactly what an unbroken run would.
- **Exact-step resume:** `--resume` restores model + optimizer + step from
  `ckpt_last.pt`. The node-failure drill (exit criterion: <10 min lost work)
  is: kill the process mid-run, `--resume`, verify the loss curve continues.
- **MFU logging:** set `peak_flops` in the config (H100 BF16 dense ≈ 989e12).
  Exit criterion: ≥35% MFU at 1B on the target cluster.

## Files

| File | Purpose |
| ---- | ------- |
| `model.py` | the architecture (maps 1:1 onto main.tex; substitutions documented in its docstring) |
| `prepare_data.py` | text + tokenizer → `train.bin`/`val.bin`/`meta.json` |
| `train.py` | training loop: AdamW, warmup+cosine, clip, bf16, checkpoint/resume |
| `../07-sft/build_chat_data.py` | SFT-v2 role encoding, response masks, provenance, and whole-example packing |
| `../07-sft/sft_train.py` | block-isolated packed chat training with RoPE reset and exact resume |
| `configs/navya-1a-sft-v2.json` | gated production SFT config (minimum 10,000 reviewed conversations) |
| `configs/smoke.json` | ~3M params, CPU, 60 steps — proves the mechanics |
| `configs/300m.json` | Research-S shape (24 × 1024, GQA 16/4) |
| `configs/1b.json` | Research-L shape (26 × 2048, GQA 16/4) |

## Next increments (in order)

1. `torch.compile` + DDP across one node's GPUs
2. FSDP for the 1B runs; gradient-checkpointing flag
3. Per-domain loss logging (needs domain tags in the data shards)
4. Loss-spike detector: auto-halt + rewind to last checkpoint on >4σ spikes
5. The framework bake-off for 8B+ (acceptance gate above)
