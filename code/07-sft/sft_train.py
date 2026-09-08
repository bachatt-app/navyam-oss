#!/usr/bin/env python3
"""SFT trainer v2: role-aware, response-only, and packing-correct.

Compared with the base-model trainer, this path consumes three aligned arrays:
tokens, assistant loss masks, and packed-example ids. It masks non-assistant
targets with -100, prevents attention across packed conversations, resets RoPE
positions at each conversation boundary, and shuffles every row exactly once
per deterministic epoch. Resume needs no loader state because every batch is a
pure function of (seed, step, micro).

Usage:
  python sft_train.py --config ../04-training-stack/configs/navya-1a-sft-v2.json \
      --init-from ../04-training-stack/out/navya-1a/ckpt_last.pt
  python sft_train.py --config ../04-training-stack/configs/navya-1a-sft-v2.json \
      --resume
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChatData:
    """Memory-mapped SFT rows with deterministic full-coverage sampling."""

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as handle:
            self.meta = json.load(handle)
        if self.meta.get("format") != "navya-sft-v2":
            raise ValueError("data is not navya-sft-v2; run build_chat_data.py")
        self.seq_len = int(self.meta["seq_len"])
        self.pad_example_id = int(self.meta["pad_example_id"])
        token_dtype = np.dtype(self.meta.get("token_dtype", "uint16"))
        self.rows = {}
        for split in ("train", "val"):
            n_rows = int(self.meta[split]["rows"])
            if n_rows < 1:
                raise ValueError(f"{split} split has no rows")
            specs = (("x", token_dtype), ("m", np.uint8), ("ex", np.uint16))
            self.rows[split] = tuple(
                np.memmap(os.path.join(data_dir, f"{split}_{suffix}.bin"),
                          dtype=dtype, mode="r", shape=(n_rows, self.seq_len))
                for suffix, dtype in specs
            )
        self._perm_cache = {}

    def n_rows(self, split):
        return int(self.meta[split]["rows"])

    def _permutation(self, split, epoch, seed):
        key = (split, epoch)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 3:
                self._perm_cache.clear()
            rng = np.random.default_rng((seed, split == "val", epoch, 11))
            self._perm_cache[key] = rng.permutation(self.n_rows(split))
        return self._perm_cache[key]

    def supervised_count(self, split, batch_size, seed, step, micro,
                         grad_accum):
        """Supervised-token count of this micro-batch — numpy only, no GPU.
        Used to token-weight the loss across gradient accumulation."""
        _x, mask_all, _ex = self.rows[split]
        n_rows = self.n_rows(split)
        base = (step * grad_accum + micro) * batch_size
        count = 0
        for counter in base + np.arange(batch_size):
            epoch, position = divmod(int(counter), n_rows)
            row = self._permutation(split, epoch, seed)[position]
            count += int(mask_all[row][1:].sum())
        return count

    def batch(self, split, batch_size, seed, step, micro, grad_accum, device):
        x_all, mask_all, example_all = self.rows[split]
        n_rows = self.n_rows(split)
        base = (step * grad_accum + micro) * batch_size
        indexes = []
        for counter in base + np.arange(batch_size):
            epoch, position = divmod(int(counter), n_rows)
            indexes.append(self._permutation(split, epoch, seed)[position])

        x_np = np.stack([x_all[index] for index in indexes]).astype(np.int64)
        mask_np = np.stack([mask_all[index] for index in indexes]).astype(np.bool_)
        example_np = np.stack([example_all[index]
                               for index in indexes]).astype(np.int64)
        x = torch.from_numpy(x_np).to(device)
        response_mask = torch.from_numpy(mask_np).to(device)
        example = torch.from_numpy(example_np).to(device)

        targets = torch.full_like(x, -100)
        targets[:, :-1] = torch.where(response_mask[:, 1:], x[:, 1:], -100)
        if not torch.any(targets != -100):
            raise ValueError("SFT batch has no supervised response tokens")

        # Conversation isolation: causal attention only inside one packed id.
        valid = example != self.pad_example_id
        same = example.unsqueeze(2) == example.unsqueeze(1)
        same = same & valid.unsqueeze(2) & valid.unsqueeze(1)
        causal = torch.tril(torch.ones(self.seq_len, self.seq_len,
                                       dtype=torch.bool, device=device))
        attention = (same & causal).unsqueeze(1)
        # Padding queries attend only to self; all-False SDPA rows are unsafe.
        diagonal = torch.eye(self.seq_len, dtype=torch.bool, device=device)
        attention = attention | diagonal.view(1, 1, self.seq_len, self.seq_len)

        # Reset RoPE to zero at each packed conversation boundary.
        change = np.ones_like(example_np, dtype=np.bool_)
        change[:, 1:] = example_np[:, 1:] != example_np[:, :-1]
        offsets = np.where(change, np.arange(self.seq_len), 0)
        starts = np.maximum.accumulate(offsets, axis=1)
        positions_np = np.arange(self.seq_len) - starts
        positions_np[example_np == self.pad_example_id] = 0
        positions = torch.from_numpy(positions_np.astype(np.int64)).to(device)
        return x, targets, attention, positions


def lr_at(step, config):
    if step < config["warmup_steps"]:
        return config["lr"] * (step + 1) / max(1, config["warmup_steps"])
    progress = ((step - config["warmup_steps"])
                / max(1, config["max_steps"] - config["warmup_steps"]))
    minimum = config["lr"] * config.get("min_lr_frac", 0.1)
    return minimum + 0.5 * (config["lr"] - minimum) * (
        1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from",
                        help="base checkpoint; required for a new SFT run")
    parser.add_argument("--resume", action="store_true",
                        help="resume model, optimizer, and exact next batch")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = {key: value for key, value in json.load(handle).items()
                  if not key.startswith("_")}
    required = ("data_dir", "out_dir", "batch_size", "grad_accum", "max_steps",
                "lr", "warmup_steps", "eval_interval", "eval_iters",
                "ckpt_interval", "log_interval")
    missing = [key for key in required if key not in config]
    if missing:
        parser.error(f"config missing keys: {missing}")

    seed = int(config.get("seed", 1337))
    device = pick_device()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = ChatData(config["data_dir"])
    minimum_conversations = int(config.get("min_train_conversations", 0))
    available_conversations = int(
        data.meta["train"].get("conversations", data.n_rows("train")))
    if available_conversations < minimum_conversations:
        raise ValueError(
            f"SFT promotion gate: {available_conversations} training conversations "
            f"< required {minimum_conversations}; expand and review the corpus")
    checkpoint_path = os.path.join(config["out_dir"], "ckpt_last.pt")
    os.makedirs(config["out_dir"], exist_ok=True)

    def fingerprint():
        # hash the actual token/mask/example files, not just metadata —
        # in-place bin corruption or swaps must break resume
        file_hashes = {}
        for split in ("train", "val"):
            for suffix in ("x", "m", "ex"):
                p = os.path.join(config["data_dir"], f"{split}_{suffix}.bin")
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        h.update(block)
                file_hashes[f"{split}_{suffix}"] = h.hexdigest()
        return {
            "data_files": file_hashes,
            "data_meta": hashlib.sha256(json.dumps(
                data.meta, sort_keys=True).encode()).hexdigest(),
            "config": hashlib.sha256(json.dumps(
                config, sort_keys=True).encode()).hexdigest(),
            "template": data.meta["chat_template"]["version"],
        }

    if args.resume:
        prev_path = os.path.join(config["out_dir"], "ckpt_prev.pt")
        if not os.path.exists(checkpoint_path) and os.path.exists(prev_path):
            print(f"WARNING: {checkpoint_path} missing (interrupted save?) — "
                  f"falling back to {prev_path}")
            checkpoint_path = prev_path
        if not os.path.exists(checkpoint_path):
            parser.error(f"resume checkpoint not found: {checkpoint_path}")
        source = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
        saved_fp, live_fp = source.get("fingerprint"), fingerprint()
        if saved_fp != live_fp:
            parser.error("resume refused: dataset/config/template changed "
                         f"since checkpoint\n  saved: {saved_fp}\n"
                         f"  live:  {live_fp}")
        start_step = int(source["step"]) + 1
        base_checkpoint = source.get("base_checkpoint")
    else:
        if not args.init_from:
            parser.error("--init-from is required unless --resume is used")
        source = torch.load(args.init_from, map_location=device,
                            weights_only=False)
        start_step = 0
        base_checkpoint = os.path.abspath(args.init_from)

    model_config = ModelConfig(**source["model_config"])
    if data.seq_len > model_config.max_seq_len:
        raise ValueError(f"data seq_len={data.seq_len} exceeds model context "
                         f"{model_config.max_seq_len}")
    if data.meta["vocab_size"] != model_config.vocab_size:
        raise ValueError("SFT tokenizer vocab does not match base checkpoint")
    model = GPT(model_config).to(device)
    model.load_state_dict(source["model"])

    decay = [parameter for parameter in model.parameters()
             if parameter.requires_grad and parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters()
                if parameter.requires_grad and parameter.dim() < 2]
    # 8-bit AdamW (bitsandbytes) is opt-in via config: it cuts optimizer-state
    # memory ~4x (fp32 m+v = 8 B/param -> 2 B/param, ~7.8 GB saved on a 1.31B
    # model) with negligible fine-tuning quality impact. Needed to fit the
    # reasoning-SFT (long <think> traces) on a 40 GB A100; default stays the
    # exact fp32 AdamW so other SFT runs are byte-for-byte unchanged.
    param_groups = [
        {"params": decay, "weight_decay": config.get("weight_decay", 0.1)},
        {"params": no_decay, "weight_decay": 0.0}]
    betas = (config.get("beta1", 0.9), config.get("beta2", 0.95))
    if config.get("optimizer_8bit", False):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            param_groups, lr=config["lr"], betas=betas)
        print("optimizer: bitsandbytes AdamW8bit (memory-saving)")
    else:
        optimizer = torch.optim.AdamW(
            param_groups, lr=config["lr"], betas=betas)
    if args.resume:
        optimizer.load_state_dict(source["opt"])
        # RNG restore must come AFTER model/optimizer construction — those
        # consume RNG draws and would desync a mid-run restore otherwise
        if source.get("rng") is not None:
            torch.set_rng_state(source["rng"]["torch"].cpu())
            if torch.cuda.is_available() and source["rng"].get("cuda"):
                torch.cuda.set_rng_state_all(
                    [t.cpu() for t in source["rng"]["cuda"]])
        print(f"resumed {checkpoint_path} at step {start_step}")
    else:
        print(f"warm-started {args.init_from} at base step {source.get('step')}")
    print(f"device={device} params={model.num_params()/1e6:.1f}M "
          f"chat={data.meta['chat_template']['version']} "
          f"rows={data.n_rows('train')}/{data.n_rows('val')}")

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.autocast("cpu", enabled=False))

    @torch.no_grad()
    def validation_loss():
        model.eval()
        loss_sum = token_sum = 0.0
        for eval_step in range(config["eval_iters"]):
            x, targets, attention, positions = data.batch(
                "val", config["batch_size"], seed, eval_step, 0, 1, device)
            n_sup = int((targets != -100).sum())
            with autocast:
                _, loss = model(x, targets, attn_mask=attention,
                                position_ids=positions)
            if torch.isfinite(loss) and n_sup:
                loss_sum += loss.item() * n_sup     # CE mean -> CE sum
                token_sum += n_sup
        model.train()
        if not token_sum:
            raise RuntimeError("validation produced no finite response loss")
        return loss_sum / token_sum

    model.train()
    tokens_per_step = (config["batch_size"] * config["grad_accum"]
                       * data.seq_len)
    log_started = time.time()
    for step in range(start_step, config["max_steps"]):
        learning_rate = lr_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        # Token-weighted objective: each micro-batch's CE mean is re-weighted
        # by its supervised-token count so short-response micro-batches don't
        # get disproportionate gradient weight. Counts come from numpy before
        # the GPU pass (batches are pure functions of their indices).
        micro_counts = [data.supervised_count(
            "train", config["batch_size"], seed, step, micro,
            config["grad_accum"]) for micro in range(config["grad_accum"])]
        total_sup = sum(micro_counts) or 1
        accumulated_loss = 0.0
        for micro in range(config["grad_accum"]):
            x, targets, attention, positions = data.batch(
                "train", config["batch_size"], seed, step, micro,
                config["grad_accum"], device)
            with autocast:
                _, loss = model(x, targets, attn_mask=attention,
                                position_ids=positions)
            (loss * (micro_counts[micro] / total_sup)).backward()
            accumulated_loss += loss.item() * micro_counts[micro] / total_sup
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       config.get("grad_clip", 1.0))
        optimizer.step()

        if step % config["log_interval"] == 0 or step == config["max_steps"] - 1:
            elapsed = time.time() - log_started
            interval_steps = config["log_interval"] if step > start_step else 1
            throughput = tokens_per_step * interval_steps / max(elapsed, 1e-9)
            print(f"step {step:>6} | response loss {accumulated_loss:.4f} | "
                  f"lr {learning_rate:.2e} | {throughput:,.0f} tok/s")
            log_started = time.time()
        if step and step % config["eval_interval"] == 0:
            print(f"step {step:>6} | VAL response loss {validation_loss():.4f}")
        if (step and step % config["ckpt_interval"] == 0) \
                or step == config["max_steps"] - 1:
            payload = {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "step": step,
                "model_config": model_config.__dict__,
                "train_config": config,
                "data_meta": data.meta,
                "chat_template": data.meta["chat_template"],
                "sft": "navya-sft-v2",
                "base_checkpoint": base_checkpoint,
                "fingerprint": fingerprint(),
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None)},
            }
            save_path = os.path.join(config["out_dir"], "ckpt_last.pt")
            tmp_path = save_path + ".tmp"
            torch.save(payload, tmp_path)
            if os.path.exists(save_path):            # keep one generation back
                os.replace(save_path,
                           os.path.join(config["out_dir"], "ckpt_prev.pt"))
            os.replace(tmp_path, save_path)

    print(f"final VAL response loss {validation_loss():.4f}")
    print(f"done. checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
