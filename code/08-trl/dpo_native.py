#!/usr/bin/env python3
"""Native DPO for Navya — pure torch on our own model.py, NO trl/transformers/peft.

Two TRL attempts died on the GCP DLVM's tangled transformers/jinja env; our SFT
(sft_train.py) trained the 1.31B cleanly on that same box with just torch + our
GPT. This reuses that exact stack.

Policy (trainable, bf16, act_ckpt) + frozen Reference (bf16) both init from the
SFT checkpoint. For each preference pair we encode  prompt + assistant(text)  with
chat_format, take the response-token log-prob SUM under policy & ref (computed via
model.forward's chunked-CE so the 64k logits are never materialized: logp_sum =
-CE_mean * n_response_tokens, gradient-carrying for the policy), and minimise
  L = -logsigmoid( beta * ((lp_chosen_pol - lp_chosen_ref) - (lp_rej_pol - lp_rej_ref)) ).

Pairs jsonl: {"prompt": [messages], "chosen": "...", "rejected": "..."} per line.

  python dpo_native.py --config configs/navya-1c-dpo.json \
      --init-from <navya-1c-sft ckpt> --pairs dpo_pairs.jsonl
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "04-training-stack"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "07-sft"))
from model import GPT, ModelConfig                       # noqa: E402
import chat_format                                        # noqa: E402


def device_of():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gpt(source, device, trainable, act_ckpt):
    m = GPT(ModelConfig(**source["model_config"])).to(device).to(torch.bfloat16)
    m.load_state_dict(source["model"])
    if trainable:
        m.train(); m.act_ckpt = act_ckpt
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def encode_pair_side(tok, prompt_messages, text):
    """(ids, loss_mask) for prompt + one assistant response, via chat_format."""
    msgs = list(prompt_messages) + [{"role": "assistant", "content": text}]
    return chat_format.encode_conversation(tok, msgs)


def logp_sum(model, ids, mask, device, ce_chunk):
    """Sum of log P over the response tokens (mask==1). Uses the chunked-CE path
    so no full-vocab logits are held; gradient flows for a trainable model."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    m = torch.tensor([mask], dtype=torch.bool, device=device)
    targets = torch.full_like(x, -100)
    targets[:, :-1] = torch.where(m[:, 1:], x[:, 1:], -100)
    n = (targets != -100).sum().clamp(min=1)
    _, loss = model(x, targets, ce_chunk=ce_chunk)        # loss = CE mean (fp32)
    return -loss * n.float()                               # = sum log P(response)


def lr_at(step, c):
    if step < c["warmup_steps"]:
        return c["lr"] * (step + 1) / max(1, c["warmup_steps"])
    prog = (step - c["warmup_steps"]) / max(1, c["max_steps"] - c["warmup_steps"])
    lo = c["lr"] * c.get("min_lr_frac", 0.1)
    return lo + 0.5 * (c["lr"] - lo) * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--pairs", required=True)
    args = ap.parse_args()
    c = {k: v for k, v in json.load(open(args.config)).items()
         if not k.startswith("_")}
    device = device_of()
    torch.manual_seed(int(c.get("seed", 1337)))

    src = torch.load(args.init_from, map_location=device, weights_only=False)
    tok = Tokenizer.from_file(c["tokenizer"])
    chat_format.token_ids(tok)   # validate the chat aliases up front
    policy = load_gpt(src, device, trainable=True, act_ckpt=c.get("act_ckpt", True))
    ref = load_gpt(src, device, trainable=False, act_ckpt=False)
    print(f"device={device} params={policy.num_params()/1e6:.1f}M "
          f"init={os.path.basename(args.init_from)} step={src.get('step')}")

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    if len(pairs) < c.get("min_pairs", 20):
        sys.exit(f"only {len(pairs)} pairs (<{c.get('min_pairs',20)}) — policy "
                 f"already aligned on this prompt set; not running DPO.")
    print(f"pairs={len(pairs)} beta={c['beta']} lr={c['lr']} "
          f"steps={c['max_steps']} grad_accum={c['grad_accum']}")

    decay = [p for p in policy.parameters() if p.requires_grad and p.dim() >= 2]
    nod = [p for p in policy.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": c.get("weight_decay", 0.0)},
              {"params": nod, "weight_decay": 0.0}]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=c["lr"], betas=(0.9, 0.95))
        print("optimizer: 8-bit AdamW")
    except Exception as e:
        opt = torch.optim.AdamW(groups, lr=c["lr"], betas=(0.9, 0.95))
        print(f"optimizer: fp32 AdamW ({e})")

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.autocast("cpu", enabled=False))
    ce = int(c.get("ce_chunk", 1024))
    ga = int(c["grad_accum"])
    rng = torch.Generator().manual_seed(int(c.get("seed", 1337)))
    order = torch.randperm(len(pairs), generator=rng).tolist()
    cur = 0
    t0 = time.time()
    for step in range(c["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, c)
        opt.zero_grad(set_to_none=True)
        acc_loss = acc_acc = 0.0
        for _ in range(ga):
            if cur >= len(order):
                order = torch.randperm(len(pairs), generator=rng).tolist(); cur = 0
            pr = pairs[order[cur]]; cur += 1
            pm = pr["prompt"]
            ch_ids, ch_m = encode_pair_side(tok, pm, pr["chosen"])
            rj_ids, rj_m = encode_pair_side(tok, pm, pr["rejected"])
            with autocast:
                lp_ch = logp_sum(policy, ch_ids, ch_m, device, ce)
                lp_rj = logp_sum(policy, rj_ids, rj_m, device, ce)
                with torch.no_grad():
                    rf_ch = logp_sum(ref, ch_ids, ch_m, device, ce)
                    rf_rj = logp_sum(ref, rj_ids, rj_m, device, ce)
                margin = (lp_ch - rf_ch) - (lp_rj - rf_rj)
                loss = -F.logsigmoid(c["beta"] * margin)
            (loss / ga).backward()
            acc_loss += loss.item() / ga
            acc_acc += float(margin.item() > 0) / ga
        torch.nn.utils.clip_grad_norm_(policy.parameters(), c.get("grad_clip", 1.0))
        opt.step()
        if step % c.get("log_interval", 10) == 0 or step == c["max_steps"] - 1:
            print(f"step {step:>5} | dpo_loss {acc_loss:.4f} | pref_acc "
                  f"{acc_acc:.2f} | lr {lr_at(step,c):.2e} | "
                  f"{(step+1)*ga/max(time.time()-t0,1e-9):.1f} pairs/s")

    os.makedirs(c["out_dir"], exist_ok=True)
    payload = {"model": policy.state_dict(),
               "model_config": src["model_config"],
               "step": int(src.get("step", 0)),
               "chat_template": src.get("chat_template"),
               "dpo": {"beta": c["beta"], "lr": c["lr"], "steps": c["max_steps"],
                       "pairs": len(pairs), "base": os.path.basename(args.init_from)}}
    out = os.path.join(c["out_dir"], "ckpt_last.pt")
    torch.save(payload, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"done. DPO checkpoint: {out}")


if __name__ == "__main__":
    main()
