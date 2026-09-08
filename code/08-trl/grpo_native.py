#!/usr/bin/env python3
"""Native GRPO for Navya — pure torch on our own model.py, NO trl/transformers.

Mirrors dpo_native.py's proven stack (SFT + DPO both trained the 1.31B cleanly on
the GCP DLVM with just torch + our GPT). Group-Relative Policy Optimization
(DeepSeekMath): for each prompt we sample K on-policy completions, score each with
the SAME programmatic reward used to mine DPO pairs (rewards.total_reward — so the
two objectives never drift), turn the K rewards into group-normalized advantages
  A_i = (r_i - mean_k r) / (std_k r + eps)
and take a length-normalized policy-gradient step with a KL leash to the frozen
reference (the DPO checkpoint):
  L = mean_i [ -A_i * logp_mean_i(policy)  +  beta_kl * (logp_mean_i(policy) - logp_mean_i(ref)) ]

logp_mean_i is the per-response-token mean log-prob = -CE_mean, taken from
model.forward's chunked-CE path (the 64k logits are never materialized; gradient
flows for the policy). The KL term is the k1 log-ratio estimator on samples drawn
from the policy — a light, low-variance leash for the small beta_kl used here; it
keeps this a *gentle* alignment step, not a reward-hacking free-for-all.

Prompts + canonical answers come from the SFT corpus (corpus_v2 rows
{instruction, response}); the canonical drives rewards.reward_canonical_overlap.

  python grpo_native.py --config configs/navya-1c-grpo.json \
      --init-from <navya-1c-sft-dpo ckpt>
"""
import argparse, glob, json, math, os, sys, time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "04-training-stack"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "07-sft"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "06-inference"))
from model import GPT, ModelConfig                       # noqa: E402
import chat_format                                        # noqa: E402
import serve                                              # noqa: E402
import rewards                                            # noqa: E402

CORPUS_DIRS = [
    os.path.join(HERE, "..", "07-sft", "training_data", "corpus_v2"),
    os.path.join(HERE, "..", "07-sft", "training_data"),
]


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


def load_corpus():
    """Canonical {instruction, response} rows, deduped by instruction — the GRPO
    prompt set (same source build_pairs_native mines DPO pairs from)."""
    files, rows, seen = [], [], set()
    for d in CORPUS_DIRS:
        if os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "*.jsonl")))
    for path in files:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            instr = (r.get("instruction") or "").strip()
            resp = (r.get("response") or "").strip()
            if not instr or not resp or instr in seen:
                continue
            seen.add(instr)
            rows.append({"instruction": instr, "response": resp})
    return rows


def logp_mean(model, ids, mask, device, ce_chunk):
    """Mean log P over the response tokens (mask==1) = -CE_mean, via the chunked-CE
    path (no full-vocab logits held; gradient flows for a trainable model)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    m = torch.tensor([mask], dtype=torch.bool, device=device)
    targets = torch.full_like(x, -100)
    targets[:, :-1] = torch.where(m[:, 1:], x[:, 1:], -100)
    _, loss = model(x, targets, ce_chunk=ce_chunk)         # loss = CE mean (fp32)
    return -loss                                           # mean log P(response)


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
    args = ap.parse_args()
    c = {k: v for k, v in json.load(open(args.config)).items()
         if not k.startswith("_")}
    device = device_of()
    torch.manual_seed(int(c.get("seed", 1337)))

    src = torch.load(args.init_from, map_location=device, weights_only=False)
    tok = Tokenizer.from_file(c["tokenizer"])
    special = chat_format.token_ids(tok)          # validate chat aliases up front
    eos = special["eos"]
    stop_ids = (special["end_turn"], special["eos"])
    cfg = ModelConfig(**src["model_config"])

    policy = load_gpt(src, device, trainable=True, act_ckpt=c.get("act_ckpt", True))
    ref = load_gpt(src, device, trainable=False, act_ckpt=False)
    print(f"device={device} params={policy.num_params()/1e6:.1f}M "
          f"init={os.path.basename(args.init_from)} step={src.get('step')}")

    corpus = load_corpus()
    if len(corpus) < c.get("min_prompts", 20):
        sys.exit(f"only {len(corpus)} corpus prompts (<{c.get('min_prompts',20)})")
    K = int(c["group_size"])
    pps = int(c["prompts_per_step"])
    mnt = int(c.get("max_new_tokens", 160))
    print(f"prompts={len(corpus)} group_size={K} prompts_per_step={pps} "
          f"steps={c['max_steps']} beta_kl={c['beta_kl']} lr={c['lr']} "
          f"max_new_tokens={mnt}")

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
    temp = float(c.get("temperature", 0.9))
    top_k = int(c.get("top_k", 50))
    top_p = float(c.get("top_p", 0.95))
    rep = float(c.get("repetition_penalty", 1.1))
    beta_kl = float(c["beta_kl"])
    rng = torch.Generator().manual_seed(int(c.get("seed", 1337)))
    order = torch.randperm(len(corpus), generator=rng).tolist()
    cur = 0
    t0 = time.time()

    for step in range(c["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, c)
        opt.zero_grad(set_to_none=True)
        denom = max(1, pps * K)
        s_loss = s_pg = s_kl = s_rmean = s_rstd = 0.0
        gen_toks = 0
        gen_time = 1e-9
        used_prompts = 0
        for _ in range(pps):
            if cur >= len(order):
                order = torch.randperm(len(corpus), generator=rng).tolist(); cur = 0
            row = corpus[order[cur]]; cur += 1
            msgs = [{"role": "user", "content": row["instruction"]}]
            canonical = row["response"]
            prompt_ids = chat_format.encode_prompt(tok, msgs)

            # --- sample K on-policy completions (no grad) ---
            comps = []
            policy.eval()
            with torch.no_grad():
                for _k in range(K):
                    text, _pl, glen, dt = serve.generate(
                        policy, cfg, tok, eos, device, prompt_ids,
                        max_new_tokens=mnt, temperature=temp, top_k=top_k,
                        top_p=top_p, repetition_penalty=rep, stop_ids=stop_ids)
                    comps.append(text.strip())
                    gen_toks += glen; gen_time += dt
            policy.train()

            rs = [rewards.total_reward(msgs, t, canonical) if len(t) >= 1 else -1.0
                  for t in comps]
            rt = torch.tensor(rs, dtype=torch.float32)
            if float(rt.std()) < 1e-6:
                continue                     # no intra-group signal — skip prompt
            adv = (rt - rt.mean()) / (rt.std() + 1e-4)
            s_rmean += float(rt.mean()); s_rstd += float(rt.std()); used_prompts += 1

            # --- policy-gradient + KL update over the K samples ---
            for k, text in enumerate(comps):
                if len(text) < 1:
                    continue
                ids, mask = chat_format.encode_conversation(
                    tok, msgs + [{"role": "assistant", "content": text}])
                if sum(mask) < 2:
                    continue
                with autocast:
                    lp_pol = logp_mean(policy, ids, mask, device, ce)
                    with torch.no_grad():
                        lp_ref = logp_mean(ref, ids, mask, device, ce)
                    kl = lp_pol - lp_ref
                    pg = -adv[k].to(lp_pol.device) * lp_pol
                    loss = (pg + beta_kl * kl) / denom
                (loss).backward()
                s_loss += float(loss.item()) * denom / K
                s_pg += float(pg.item()) / K
                s_kl += float(kl.item()) / K

        torch.nn.utils.clip_grad_norm_(policy.parameters(), c.get("grad_clip", 1.0))
        opt.step()
        up = max(1, used_prompts)
        if step % c.get("log_interval", 1) == 0 or step == c["max_steps"] - 1:
            print(f"step {step:>4} | loss {s_loss/up:+.4f} | pg {s_pg/up:+.4f} "
                  f"| kl {s_kl/up:+.4f} | reward {s_rmean/up:+.3f}±{s_rstd/up:.3f} "
                  f"| used {used_prompts}/{pps} | gen {gen_toks/gen_time:.1f} tok/s "
                  f"| lr {lr_at(step,c):.2e} | {time.time()-t0:.0f}s", flush=True)

    os.makedirs(c["out_dir"], exist_ok=True)
    payload = {"model": policy.state_dict(),
               "model_config": src["model_config"],
               "step": int(src.get("step", 0)),
               "chat_template": src.get("chat_template"),
               "grpo": {"beta_kl": beta_kl, "lr": c["lr"], "steps": c["max_steps"],
                        "group_size": K, "prompts_per_step": pps,
                        "base": os.path.basename(args.init_from)}}
    out = os.path.join(c["out_dir"], "ckpt_last.pt")
    torch.save(payload, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"done. GRPO checkpoint: {out}")


if __name__ == "__main__":
    main()
