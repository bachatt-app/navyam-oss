"""navya-1b-av M1 trainer: audio CPT of a text checkpoint on one GPU (A10).

Research-scale single-GPU loop, deliberately independent of train.py's
DDP/virtual-stream machinery: M1 mixes two sources per step draw —

  text replay   p = 1 - p_audio   random windows from uint16 token .bin files
                                  (a prepared replay dir; protects the base)
  audio ASR     p = p_audio       ASRShardDataset batches (ffmpeg-decoded
                                  16 kHz windows spliced via forward_av)

Init from a text checkpoint (train.py format, {"model": ...}) with
--init-from: weights load into .gpt strict=True; the audio projector starts
fresh. Checkpoints save the full MultimodalGPT ({"model", "opt", "step"}),
resume with --resume. Gates tracked externally: eval_asr_cer.py for CER,
text val loss vs the base for regression (<2%, doc 14 §3).

  ../.venv/bin/python train_av.py --config configs/navya-1b-av.json \
      --init-from out/navya-1b/ckpt.pt
"""
import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch

from model import ModelConfig
from multimodal import MultimodalGPT
from multimodal_data import ASRShardDataset, collate_av, forward_av

MODEL_KEYS = {f.name for f in ModelConfig.__dataclass_fields__.values()}


class TextReplay:
    """Random fixed-length windows from a directory of uint16 token bins."""

    def __init__(self, bin_dir, seq_len, seed=0):
        self.arrs = [np.memmap(p, dtype=np.uint16, mode="r") for p in
                     sorted(glob.glob(os.path.join(bin_dir, "**", "*.bin"),
                                      recursive=True))]
        self.arrs = [a for a in self.arrs if a.size > seq_len + 1]
        if not self.arrs:
            raise FileNotFoundError(f"no usable .bin > {seq_len} tokens "
                                    f"in {bin_dir}")
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def batch(self, bs):
        xs, ys = [], []
        for _ in range(bs):
            a = self.rng.choice(self.arrs)
            i = self.rng.randrange(a.size - self.seq_len - 1)
            w = torch.from_numpy(a[i:i + self.seq_len + 1].astype(np.int64))
            xs.append(w[:-1])
            ys.append(w[1:])
        return torch.stack(xs), torch.stack(ys)


def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(
        cfg["max_steps"] - cfg["warmup_steps"], 1)
    return cfg["lr"] * (0.1 + 0.45 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", default=None,
                    help="text checkpoint (train.py format); loads ['model'] "
                         "into .gpt strict=True")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    mcfg = ModelConfig(**{k: v for k, v in cfg.items() if k in MODEL_KEYS})

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.autocast("cpu", enabled=False))
    torch.manual_seed(cfg.get("seed", 0))

    mm = MultimodalGPT(mcfg, audio=True).to(device)
    opt = torch.optim.AdamW(mm.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=cfg.get("weight_decay", 0.1))
    step = 0
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt_av.pt")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        mm.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}")
    elif args.init_from:
        base = torch.load(args.init_from, map_location=device,
                          weights_only=False)
        mm.gpt.load_state_dict(base["model"], strict=True)
        print(f"initialized gpt from {args.init_from}")

    asr = ASRShardDataset(cfg["asr_shards"])
    asr_loader = torch.utils.data.DataLoader(
        asr, batch_size=cfg["audio_batch_size"], shuffle=True,
        collate_fn=collate_av, num_workers=cfg.get("loader_workers", 2),
        drop_last=True, persistent_workers=cfg.get("loader_workers", 2) > 0)
    asr_iter = iter(asr_loader)
    replay = TextReplay(cfg["text_replay_dir"], cfg["replay_seq_len"],
                        seed=cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0) + 1)
    ce_chunk = cfg.get("ce_chunk", 1024)
    ema = {"text": None, "audio": None}

    while step < cfg["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.get("grad_accum", 1)):
            if rng.random() < cfg["p_audio"]:
                try:
                    batch = next(asr_iter)
                except StopIteration:
                    asr_iter = iter(asr_loader)
                    batch = next(asr_iter)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with autocast:
                    _, loss = forward_av(mm, batch, ce_chunk=ce_chunk)
                kind = "audio"
            else:
                x, y = replay.batch(cfg["text_batch_size"])
                with autocast:
                    _, loss = mm.gpt(x.to(device), y.to(device),
                                     ce_chunk=ce_chunk)
                kind = "text"
            (loss / cfg.get("grad_accum", 1)).backward()
            v = float(loss)
            ema[kind] = v if ema[kind] is None else 0.98 * ema[kind] + 0.02 * v
        torch.nn.utils.clip_grad_norm_(mm.parameters(),
                                       cfg.get("grad_clip", 1.0))
        opt.step()
        step += 1
        if step % cfg.get("log_interval", 20) == 0:
            print(f"step {step} lr {opt.param_groups[0]['lr']:.2e} "
                  f"ema_text {ema['text'] and round(ema['text'], 4)} "
                  f"ema_audio {ema['audio'] and round(ema['audio'], 4)}",
                  flush=True)
        if step % cfg.get("ckpt_interval", 500) == 0 or step == cfg["max_steps"]:
            torch.save({"model": mm.state_dict(), "opt": opt.state_dict(),
                        "step": step, "config": cfg}, ckpt_path)
            print(f"ckpt -> {ckpt_path} @ {step}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
