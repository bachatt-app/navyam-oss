# 06 — Inference (research preview)

`serve.py` — thin OpenAI-compatible server over a train.py checkpoint:
`/v1/completions` (honest base-model API), `/v1/chat/completions` (renders
base/legacy checkpoints to Q:/A:, but automatically uses the checkpoint's
`navya-chat-v1` role-token template for SFT-v2), `/health`, `/v1/models`.
SFT-v2 generation stops on its structural end-turn token or EOS, and serving
uses the same encoder as the data builder. One generation at a time,
no KV cache, no batching — vLLM takes over at navya-1 (see MODEL_HOSTING.md).

Local test (needs a checkpoint, e.g. mini or navya-0):
    ../.venv/bin/python serve.py --ckpt ../04-training-stack/out/mini/ckpt_last.pt
    curl -s localhost:8000/v1/completions -d '{"prompt":"Mutual funds are"}'

`serve.py` is an OpenAI-compatible server: `MODEL_ID`, `NAVYA_TEMPLATE=chat|qa`,
`NAVYA_TEMP`, `--host`, `--port` (default :8000). Point `--ckpt` at any checkpoint
and `--tokenizer` at the matching tokenizer JSON.
