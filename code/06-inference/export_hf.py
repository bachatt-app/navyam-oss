#!/usr/bin/env python3
"""Export a train.py checkpoint to HuggingFace format (LlamaForCausalLM).

Our architecture IS the Llama recipe (RoPE half-split rotation, GQA, RMSNorm
pre-norm, SwiGLU, no biases, tied embeddings), so export is a weight-renaming
exercise. This script refuses to ship silently-wrong weights: it ends with a
logit-parity test between the custom model and the HF model on a fixed batch
(fp32, CPU) and hard-fails beyond --tolerance.

Usage:
  python export_hf.py --ckpt ../04-training-stack/out/navya-0/ckpt_last.pt \
      --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json \
      --out ../04-training-stack/out/navya-0/hf
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    print("loading custom checkpoint …")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    ours = GPT(cfg)
    ours.load_state_dict(ckpt["model"])
    ours.eval().float()
    sd = ours.state_dict()

    intermediate = sd["blocks.0.ffn.w_gate.weight"].shape[0]
    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.dim,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, intermediate_size=intermediate,
        max_position_embeddings=cfg.max_seq_len, rope_theta=cfg.rope_base,
        rms_norm_eps=cfg.norm_eps, tie_word_embeddings=True,
        attention_bias=False, mlp_bias=False, hidden_act="silu",
        bos_token_id=0, eos_token_id=1, pad_token_id=2)
    print(f"LlamaConfig: dim={cfg.dim} layers={cfg.n_layers} "
          f"heads={cfg.n_heads}/{cfg.n_kv_heads} inter={intermediate}")

    hf = LlamaForCausalLM(hf_cfg)
    new_sd = {"model.embed_tokens.weight": sd["tok_emb.weight"],
              "model.norm.weight": sd["final_norm.weight"]}
    for i in range(cfg.n_layers):
        b, l = f"blocks.{i}", f"model.layers.{i}"
        new_sd[f"{l}.input_layernorm.weight"] = sd[f"{b}.attn_norm.weight"]
        new_sd[f"{l}.self_attn.q_proj.weight"] = sd[f"{b}.attn.wq.weight"]
        new_sd[f"{l}.self_attn.k_proj.weight"] = sd[f"{b}.attn.wk.weight"]
        new_sd[f"{l}.self_attn.v_proj.weight"] = sd[f"{b}.attn.wv.weight"]
        new_sd[f"{l}.self_attn.o_proj.weight"] = sd[f"{b}.attn.wo.weight"]
        new_sd[f"{l}.post_attention_layernorm.weight"] = sd[f"{b}.ffn_norm.weight"]
        new_sd[f"{l}.mlp.gate_proj.weight"] = sd[f"{b}.ffn.w_gate.weight"]
        new_sd[f"{l}.mlp.up_proj.weight"] = sd[f"{b}.ffn.w_up.weight"]
        new_sd[f"{l}.mlp.down_proj.weight"] = sd[f"{b}.ffn.w_down.weight"]
    missing, unexpected = hf.load_state_dict(new_sd, strict=False)
    # lm_head.weight is tied to embed_tokens — anything else missing is a bug
    real_missing = [m for m in missing if m != "lm_head.weight"]
    assert not real_missing and not unexpected, (real_missing, unexpected)
    hf.tie_weights()
    hf.eval().float()

    print("parity test (fp32, cpu) …")
    torch.manual_seed(7)
    x = torch.randint(3, cfg.vocab_size, (2, 128))
    with torch.no_grad():
        # pass targets so the custom model returns logits for ALL positions
        # (without targets it returns only the last position)
        ours_logits, _ = ours(x, x)
        hf_logits = hf(x).logits
    diff = (ours_logits - hf_logits).abs().max().item()
    print(f"max |Δlogit| = {diff:.2e}  (tolerance {args.tolerance:.0e})")
    if diff > args.tolerance:
        sys.exit(f"PARITY FAILED: {diff} > {args.tolerance} — export aborted")

    os.makedirs(args.out, exist_ok=True)
    hf.save_pretrained(args.out, safe_serialization=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer,
                                  bos_token="<|bos|>", eos_token="<|eos|>",
                                  pad_token="<|pad|>")
    tok.save_pretrained(args.out)
    print(f"exported → {args.out}  (step {ckpt.get('step')}, parity OK)")


if __name__ == "__main__":
    main()
