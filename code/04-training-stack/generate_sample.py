#!/usr/bin/env python3
"""Standalone sampler for a train.py-format checkpoint (no server deps).

Minimal generation harness for sanity-checking a checkpoint (e.g. the
embedding-transfer proxy experiment) without pulling in serve.py's HTTP
server / chat-template / tool-router machinery. Same sampling logic as
serve.py's generate() (temperature, top-k, top-p, repetition penalty).

Usage:
  python generate_sample.py --ckpt out/proxy-v05-transfer/ckpt_last.pt \
      --tokenizer ../01-tokenizer/tokenizer-v0.5-B96k.json \
      --prompt "GST applies to" --max-new-tokens 80
"""

import argparse

import torch
import torch.nn.functional as F
from model import GPT, ModelConfig
from tokenizers import Tokenizer


def load(ckpt_path: str, tokenizer_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model = (model.to(device=device, dtype=torch.bfloat16)
             if device == "cuda" else model.to(device))
    model.eval()
    tok = Tokenizer.from_file(tokenizer_path)
    eos = tok.token_to_id("<|eos|>")
    return model, cfg, tok, eos, ckpt


@torch.no_grad()
def generate(model, cfg, tok, eos, device, prompt, max_new_tokens=100,
             temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.1):
    ids = tok.encode(prompt).ids
    ids = ids[-max(1, cfg.max_seq_len - max_new_tokens - 1):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -cfg.max_seq_len:])
        logits = logits[:, -1, :].float()
        if repetition_penalty > 1.0:
            seen = torch.unique(x[0, -256:])
            sel = logits[0, seen]
            logits[0, seen] = torch.where(sel > 0, sel / repetition_penalty,
                                          sel * repetition_penalty)
        logits = logits / max(temperature, 1e-4)
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        mask = sp.cumsum(-1) - sp > top_p
        sp[mask] = 0.0
        sp /= sp.sum(-1, keepdim=True)
        next_id = int(si[0, torch.multinomial(sp[0], 1)])
        if next_id == eos:
            break
        out_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(out_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", action="append", required=True,
                    help="repeatable; one generation per --prompt")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, tok, eos, ckpt = load(args.ckpt, args.tokenizer, device)
    print(f"loaded {args.ckpt} step={ckpt.get('step')} vocab={cfg.vocab_size} "
          f"device={device}")
    for p in args.prompt:
        out = generate(model, cfg, tok, eos, device, p,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature)
        print(f"\n--- PROMPT: {p!r}\n{out}")


if __name__ == "__main__":
    main()
