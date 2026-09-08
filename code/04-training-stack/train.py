#!/usr/bin/env python3
"""Single-node trainer for research models (300M--1B) and smoke tests.

Phase-0 requirements it implements (execution plan, Workstream 4):
  - deterministic, seekable batch sampling: batch at step k depends only on
    (seed, k), so a resumed run consumes exactly the data an unbroken run would
  - resumable checkpointing to the exact step (save/kill/resume is the Phase-0
    node-failure drill)
  - MFU + tokens/sec logging
  - AdamW, cosine schedule with linear warmup, gradient clipping, bf16 autocast

Multi-GPU: single-node DDP via torchrun. The config's batch_size/grad_accum
are GLOBAL: each of W ranks runs grad_accum/W micro-steps, and the global
micro index (rank*ga_local + m) feeds the same deterministic sampler — so a
run consumes bitwise-identical data at any world size, and a checkpoint
from a single-GPU run resumes correctly on 8 GPUs (and vice versa).
FSDP is the 8B+ increment per the decision log.

Usage:
  python train.py --config configs/smoke.json
  python train.py --config configs/smoke.json --resume    # from latest ckpt
  torchrun --standalone --nproc_per_node=8 train.py --config configs/navya-1b.json

Env: NAVYA_PEAK_FLOPS overrides the config's per-device peak_flops for MFU
(A10 125e12, A100 312e12, H100 989e12, B200 2250e12).
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, ModelConfig


def setup_dist() -> tuple[int, int, int]:
    """Init the process group under torchrun; no-op single-process otherwise.

    Returns (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if os.environ.get("NAVYA_DIST_FILE"):   # socket-free rendezvous (mac/tests)
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{os.environ['NAVYA_DIST_FILE']}",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]))
    else:
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank


@dataclass
class TrainConfig:
    data_dir: str = "data/smoke"
    out_dir: str = "out/smoke"
    batch_size: int = 8            # sequences per micro-batch
    ce_chunk: int = 0              # rows per chunked-CE block (0 = full logits)
    compile_mode: str = ""         # torch.compile mode ("" = eager)
    # curriculum cooldown (Qwen3-style): after cooldown_start_frac of steps,
    # switch to a second stream (high-quality math/finance/Hinglish upweight)
    # while the cosine schedule is already decaying LR
    cooldown_data_dir: str = ""
    cooldown_start_frac: float = 0.85
    grad_accum: int = 1
    max_steps: int = 50
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 50
    eval_iters: int = 20
    ckpt_interval: int = 50
    peak_flops: float = 0.0        # per-device peak FLOP/s for MFU; 0 = skip
    # 4B-class memory strategy (navya-2): both are execution choices, not
    # training semantics — excluded from the resume fingerprint like peak_flops
    zero1: bool = False            # ZeRO-1: shard AdamW state across DDP ranks
    act_ckpt: bool = False         # per-block activation checkpointing
    # LR schedule: "cosine" (warmup -> cosine to min_lr at max_steps) or
    # "wsd" (warmup -> constant -> linear decay over the LAST decay_steps;
    # decay_steps 0 = constant forever). WSD is the continued-pretraining
    # schedule: the main run never decays, and a short DECAY BRANCH
    # (--branch-from <ckpt> with a config whose max_steps = step + decay_steps)
    # produces the evaluable checkpoint whenever wall-clock says so.
    lr_schedule: str = "cosine"
    decay_steps: int = 0


def load_configs(path: str) -> tuple[ModelConfig, TrainConfig]:
    raw = json.load(open(path))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    mnames = {f.name for f in fields(ModelConfig)}
    tnames = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - mnames - tnames
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    mc = ModelConfig(**{k: v for k, v in raw.items() if k in mnames})
    tc = TrainConfig(**{k: v for k, v in raw.items() if k in tnames})
    return mc, tc


def pick_device() -> str:
    if os.environ.get("NAVYA_DEVICE"):      # explicit override (tests, CI)
        return os.environ["NAVYA_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _VirtualTrain:
    """Sliceable view over a --virtual stream manifest (PERF item 9).

    Presents the exact byte sequence a physical train.bin would contain —
    domain pools repeated `full` times plus `rem` head tokens, concatenated
    in manifest order — assembled on read from the domain part memmaps.
    A window costs at most a few contiguous part reads (pool wrap or
    domain boundary), so training throughput is memmap-bound either way;
    what disappears is the full-corpus write+re-read at build time.
    """

    def __init__(self, segments, dtype):
        import bisect
        self._bisect = bisect
        self.segs = []
        maps = {}
        for seg in segments:
            pieces, off = [], 0
            for part in seg["parts"]:
                pp = part["path"]
                if pp not in maps:
                    maps[pp] = np.memmap(pp, dtype=dtype, mode="r")
                pieces.append((off, off + part["tokens"], maps[pp]))
                off += part["tokens"]
            pool = seg["pool_tokens"]          # pool = concat(parts)[:pool]
            length = seg["full"] * pool + seg["rem"]
            self.segs.append({"start": seg["start"],
                              "end": seg["start"] + length,
                              "pool": pool, "pieces": pieces})
        self.starts = [g["start"] for g in self.segs]
        self.total = self.segs[-1]["end"] if self.segs else 0

    def __len__(self):
        return self.total

    def _read(self, g0, n):
        out = np.empty(n, dtype=self.segs[0]["pieces"][0][2].dtype)
        filled = 0
        g = g0
        while filled < n:
            si = self._bisect.bisect_right(self.starts, g) - 1
            seg = self.segs[si]
            pool_idx = (g - seg["start"]) % seg["pool"]
            run = min(n - filled, seg["end"] - g, seg["pool"] - pool_idx)
            # map pool_idx into the concatenated parts
            for p0, p1, mm in seg["pieces"]:
                if pool_idx < p1:
                    take = min(run, p1 - pool_idx)
                    lo = pool_idx - p0
                    out[filled:filled + take] = mm[lo:lo + take]
                    filled += take
                    g += take
                    run -= take
                    pool_idx += take
                    if run == 0:
                        break
        return out

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            start = sl.start or 0
            stop = self.total if sl.stop is None else min(sl.stop, self.total)
            return self._read(start, max(0, stop - start))
        return self._read(sl, 1)[0]


class Data:
    """Memory-mapped token stream with deterministic EPOCH-COVERAGE sampling.

    Pretraining mode: the stream is cut into non-overlapping seq_len windows;
    each epoch is a seeded permutation of ALL windows, consumed sequentially by
    a global micro-batch counter. Guarantees: every token position is seen
    exactly once per epoch (the old random-with-replacement sampling touched
    only ~63% of the stream in one nominal pass), and the batch at any step is
    still a pure function of (seed, step, micro) — exact-step resume needs no
    extra loader state.

    SFT mode (meta sets align_block): docs are eos-padded to that multiple and
    windows start uniformly at random block boundaries, so sequences regularly
    begin at a doc start ("Q: ..." at position 0 — the inference shape).
    Coverage doesn't matter there: the corpus is tiny and trained many epochs.
    """

    def __init__(self, data_dir: str, seq_len: int, grad_accum: int = 1):
        meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.meta = meta
        self.vocab_size = meta["vocab_size"]
        self.align = meta.get("align_block")
        dtype = np.dtype(meta["dtype"])
        if meta.get("virtual"):
            self.train = _VirtualTrain(meta["segments"], dtype)
        else:
            self.train = np.memmap(os.path.join(data_dir, "train.bin"),
                                   dtype=dtype, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"),
                             dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.grad_accum = grad_accum
        self._perm_cache = {}   # (split, epoch) -> permutation of window ids

    def _perm(self, split: str, epoch: int, n_win: int, seed: int):
        key = (split, epoch)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 3:   # keep it bounded
                self._perm_cache.clear()
            rng = np.random.default_rng((seed, split == "val", epoch, 7))
            self._perm_cache[key] = rng.permutation(n_win)
        return self._perm_cache[key]

    def batch(self, split: str, batch_size: int, seed: int, step: int,
              micro: int, device: str):
        arr = self.train if split == "train" else self.val
        L = self.seq_len
        if self.align:   # SFT: random block-aligned starts (see docstring)
            rng = np.random.default_rng((seed, step, micro, split == "val"))
            max_start = len(arr) - L - 1
            starts = rng.integers(0, max_start // self.align + 1,
                                  size=batch_size) * self.align
        else:            # pretraining: full-coverage epoch permutation
            n_win = (len(arr) - 1) // L
            base = (step * self.grad_accum + micro) * batch_size
            counters = base + np.arange(batch_size)
            starts = np.empty(batch_size, dtype=np.int64)
            for j, c in enumerate(counters):
                epoch, pos = divmod(int(c), n_win)
                starts[j] = self._perm(split, epoch, n_win, seed)[pos] * L
        x = np.stack([arr[s:s + L] for s in starts]).astype(np.int64)
        y = np.stack([arr[s + 1:s + L + 1] for s in starts]).astype(np.int64)
        return (torch.from_numpy(x).to(device),
                torch.from_numpy(y).to(device))


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    min_lr = tc.lr * tc.min_lr_frac
    if tc.lr_schedule == "wsd":
        if not tc.decay_steps:
            return tc.lr
        d0 = tc.max_steps - tc.decay_steps
        if step < d0:
            return tc.lr
        t = min(1.0, (step - d0 + 1) / tc.decay_steps)
        return tc.lr - (tc.lr - min_lr) * t
    t = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return min_lr + 0.5 * (tc.lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def eval_loss(model, data: Data, tc: TrainConfig, device: str) -> float:
    model.eval()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    losses = []
    for i in range(tc.eval_iters):
        x, y = data.batch("val", tc.batch_size, tc.seed, i, 0, device)
        with amp:
            _, loss = model(x, y, ce_chunk=tc.ce_chunk)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", help="checkpoint to warm-start WEIGHTS from "
                    "(fresh optimizer and step 0 — for SFT; distinct from --resume)")
    ap.add_argument("--branch-from", help="checkpoint to CONTINUE from (weights, "
                    "optimizer and step) under a NEW schedule/out_dir — the WSD "
                    "decay branch; same data stream required, fingerprint not")
    args = ap.parse_args()

    mc, tc = load_configs(args.config)
    rank, world, local_rank = setup_dist()
    device = pick_device()
    if world > 1 and device == "cuda":
        device = f"cuda:{local_rank}"
    if os.environ.get("NAVYA_PEAK_FLOPS"):
        tc.peak_flops = float(os.environ["NAVYA_PEAK_FLOPS"])
    if tc.grad_accum % world:
        raise SystemExit(f"grad_accum {tc.grad_accum} not divisible by "
                         f"world size {world}")
    ga_local = tc.grad_accum // world
    log = print if rank == 0 else (lambda *a, **k: None)
    # identical initialization across runs (LR/batch sweeps depend on it):
    # model init draws from the global torch RNG, so seed it explicitly
    torch.manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tc.seed)

    data = Data(tc.data_dir, mc.max_seq_len, tc.grad_accum)
    cooldown_data = None
    cooldown_step = tc.max_steps + 1
    if tc.cooldown_data_dir:
        cooldown_data = Data(tc.cooldown_data_dir, mc.max_seq_len, tc.grad_accum)
        assert cooldown_data.vocab_size == data.vocab_size
        cooldown_step = int(tc.max_steps * tc.cooldown_start_frac)
        log(f"curriculum: cooldown stream from step {cooldown_step} "
            f"({tc.cooldown_data_dir})")
    if mc.vocab_size != data.vocab_size:
        log(f"config vocab {mc.vocab_size} -> data vocab {data.vocab_size}")
        mc.vocab_size = data.vocab_size

    model = GPT(mc).to(device)
    n_params = model.num_params()
    log(f"device={device} world={world} params={n_params/1e6:.1f}M "
        f"layers={mc.n_layers} dim={mc.dim} heads={mc.n_heads}/{mc.n_kv_heads} "
        f"micro-steps/rank={ga_local}")

    # weight decay on matrices, none on norms/1-D params
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    opt_kw = dict(lr=tc.lr, betas=(tc.beta1, tc.beta2))
    if device.startswith("cuda"):
        opt_kw["fused"] = True      # fused AdamW; falls back below if unsupported
    use_zero = tc.zero1 and world > 1

    def make_opt(kw):
        if use_zero:
            # ZeRO-1 (navya-2, 4B): each rank keeps 1/world of the AdamW state
            # and master weights (~6GB instead of 48GB at 4B on 8 ranks) and
            # broadcasts its updated shard after step(). Gradients are still
            # the full DDP all-reduce, so clipping and the data stream are
            # unchanged; checkpoints are consolidated to rank 0 so they stay
            # loadable at ANY world size (plain AdamW included).
            from torch.distributed.optim import ZeroRedundancyOptimizer
            return ZeroRedundancyOptimizer(
                groups, optimizer_class=torch.optim.AdamW,
                parameters_as_bucket_view=True, **kw)
        return torch.optim.AdamW(groups, **kw)

    try:
        opt = make_opt(opt_kw)
    except (RuntimeError, TypeError) as e:
        log(f"fused AdamW unavailable ({e}); using default")
        opt_kw.pop("fused", None)
        opt = make_opt(opt_kw)
    if use_zero:
        log("ZeRO-1: AdamW state sharded across ranks")
    if tc.act_ckpt:
        model.act_ckpt = True
        log("activation checkpointing: per block")

    # run fingerprint: a resumed checkpoint must come from the SAME data
    # stream and training config — shape-compatibility alone is not identity
    stream_id = data.meta.get("stream_id", "unknown")
    # peak_flops is MFU display only — excluding it lets a checkpoint resume
    # on a different device class (A10 -> A100 -> H100/B200) without refusal
    cfg_fp = hashlib.sha256(json.dumps(
        {**mc.__dict__, **{k: v for k, v in tc.__dict__.items()
                           if k not in ("peak_flops", "zero1", "act_ckpt")}},
        sort_keys=True).encode()).hexdigest()

    start_step = 0
    ckpt_path = os.path.join(tc.out_dir, "ckpt_last.pt")
    prev_path = os.path.join(tc.out_dir, "ckpt_prev.pt")
    if args.resume and not os.path.exists(ckpt_path) and os.path.exists(prev_path):
        log(f"WARNING: {ckpt_path} missing (interrupted save?) — "
            f"falling back to {prev_path}")
        ckpt_path = prev_path
    if args.init_from and not (args.resume and os.path.exists(ckpt_path)):
        base = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(base["model"])
        log(f"warm-started weights from {args.init_from} "
            f"(step {base.get('step')}); fresh optimizer, step 0")
    if args.branch_from and not (args.resume and os.path.exists(ckpt_path)):
        ckpt = torch.load(args.branch_from, map_location=device, weights_only=False)
        saved_sid = ckpt.get("stream_id")
        if saved_sid is not None and saved_sid != stream_id:
            raise SystemExit(f"branch refused: checkpoint trained on "
                             f"{saved_sid}, data dir has {stream_id}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"] + 1
        if start_step >= tc.max_steps:
            raise SystemExit(f"branch refused: checkpoint step {ckpt['step']} "
                             f">= max_steps {tc.max_steps}")
        log(f"branched from {args.branch_from} at step {start_step} "
            f"(schedule {tc.lr_schedule}, decay {tc.decay_steps} of "
            f"{tc.max_steps} steps)")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_sid = ckpt.get("stream_id")
        saved_cfg = ckpt.get("config_fingerprint")
        if saved_sid is not None and saved_sid != stream_id:
            raise SystemExit(f"resume refused: checkpoint trained on "
                             f"{saved_sid}, data dir has {stream_id}")
        if saved_cfg is not None and saved_cfg != cfg_fp:
            raise SystemExit("resume refused: model/train config changed "
                             "since checkpoint")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"] + 1
        log(f"resumed from {ckpt_path} at step {start_step}")

    os.makedirs(tc.out_dir, exist_ok=True)
    ddp_model = None
    run_model = model
    if world > 1:
        ddp_model = DDP(model, device_ids=[local_rank]
                        if device.startswith("cuda") else None)
        run_model = ddp_model
    if tc.compile_mode:
        log(f"torch.compile mode={tc.compile_mode} …")
        run_model = torch.compile(run_model, mode=tc.compile_mode)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    tokens_per_step = tc.batch_size * tc.grad_accum * mc.max_seq_len
    flops_per_token = 6 * n_params
    t_log = time.time()

    model.train()
    for step in range(start_step, tc.max_steps):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        src = cooldown_data if (cooldown_data and step >= cooldown_step) else data
        loss_acc = None    # stays on-device; .item() only at log time
        for m in range(ga_local):
            # global micro index: rank r owns micros [r*ga_local, (r+1)*ga_local)
            # — together the ranks consume exactly the micros a single-GPU run
            # would, so the data stream is identical at any world size
            micro = rank * ga_local + m
            x, y = src.batch("train", tc.batch_size, tc.seed, step, micro, device)
            # skip the per-micro allreduce; grads sync on the last micro
            sync = (ddp_model.no_sync() if ddp_model and m < ga_local - 1
                    else contextlib.nullcontext())
            with sync, autocast:
                _, loss = run_model(x, y, ce_chunk=tc.ce_chunk)
            (loss / ga_local).backward()
            d = loss.detach()
            loss_acc = d if loss_acc is None else loss_acc + d
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if step % tc.log_interval == 0 or step == tc.max_steps - 1:
            loss_mean = loss_acc / ga_local
            if world > 1:   # average the local means for a global loss reading
                dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
                loss_mean /= world
            dt = time.time() - t_log
            t_log = time.time()
            steps_done = tc.log_interval if step else 1
            tps = tokens_per_step * steps_done / dt
            mfu = (f" mfu={tps * flops_per_token / (tc.peak_flops * world):5.1%}"
                   if tc.peak_flops else "")
            log(f"step {step:>6} | loss {loss_mean.item():7.4f} | "
                f"lr {lr:.2e} | {tps:,.0f} tok/s{mfu}")

        if step and step % tc.eval_interval == 0 and rank == 0:
            print(f"step {step:>6} | VAL loss {eval_loss(model, data, tc, device):.4f}")

        if (step and step % tc.ckpt_interval == 0) or step == tc.max_steps - 1:
            if use_zero:    # collective: every rank ships its shard to rank 0
                opt.consolidate_state_dict(to=0)
            if rank == 0:
                # atomic save: tmp -> keep previous generation -> promote
                save_path = os.path.join(tc.out_dir, "ckpt_last.pt")
                tmp = save_path + ".tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "model_config": mc.__dict__,
                            "train_config": tc.__dict__,
                            "stream_id": stream_id,
                            "config_fingerprint": cfg_fp}, tmp)
                if os.path.exists(save_path):
                    os.replace(save_path,
                               os.path.join(tc.out_dir, "ckpt_prev.pt"))
                os.replace(tmp, save_path)
            if world > 1:   # no rank runs ahead of an unfinished save
                dist.barrier()

    if rank == 0:
        print(f"final VAL loss {eval_loss(model, data, tc, device):.4f}")
        print(f"done. checkpoint: {ckpt_path}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
