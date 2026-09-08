#!/usr/bin/env python3
"""Reproducible training micro-benchmark for optimization decisions.

Runs a configuration matrix over {micro_batch, ce_chunk, compile_mode} while
holding the GLOBAL batch fixed (grad_accum adjusts inversely), timing with
CUDA events after warmup and reporting tokens/sec, MFU and peak memory.
Decisions are made from this table, not from single-step eyeballing.

Usage (on the GPU box):
  python bench_train.py --config configs/navya-1a.json \
      --steps 30 --warmup 5 --json-out bench_results.json
"""

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, ModelConfig  # noqa: E402
from train import Data, load_configs  # noqa: E402


def bench_one(mc, tc_raw, data_dir, micro_b, ce_chunk, compile_mode,
              steps, warmup, device):
    global_tokens = tc_raw["batch_size"] * tc_raw["grad_accum"] * mc.max_seq_len
    grad_accum = global_tokens // (micro_b * mc.max_seq_len)
    # per-config Data so counter math uses THIS config's grad_accum —
    # every candidate consumes the same contiguous global window sequence
    data = Data(data_dir, mc.max_seq_len, grad_accum)
    torch.manual_seed(7)
    model = GPT(mc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    run_model = torch.compile(model, mode=compile_mode) if compile_mode else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device == "cuda"))
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()

    def one_step(step):
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            x, y = data.batch("train", micro_b, 1337, step, micro, device)
            with amp:
                _, loss = run_model(x, y, ce_chunk=ce_chunk)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    try:
        for s in range(warmup):
            one_step(s)
        torch.cuda.synchronize()
        startup_peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()   # steady-state peak, sans warmup
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for s in range(warmup, warmup + steps):
            one_step(s)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000.0
        tps = global_tokens * steps / secs
        mfu = tps * 6 * n_params / 125e12
        peak = torch.cuda.max_memory_allocated() / 1e9
        return {"tok_s": round(tps), "mfu": round(mfu, 4),
                "steady_peak_gb": round(peak, 2),
                "startup_peak_gb": round(startup_peak, 2),
                "grad_accum": grad_accum}
    except torch.cuda.OutOfMemoryError:
        return {"oom": True, "grad_accum": grad_accum}
    finally:
        del model, run_model, opt
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--micro-batches", default="4,8,16")
    ap.add_argument("--ce-chunks", default="0,8192")
    ap.add_argument("--compile-modes", default=",default")
    ap.add_argument("--gqa-modes", default="native,expanded")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark requires the GPU box"

    mc, tc = load_configs(args.config)
    device = "cuda"
    results = []
    for micro_b, ce_chunk, mode, gqa in itertools.product(
            [int(x) for x in args.micro_batches.split(",")],
            [int(x) for x in args.ce_chunks.split(",")],
            args.compile_modes.split(","),
            args.gqa_modes.split(",")):
        os.environ["NAVYA_NATIVE_GQA"] = "1" if gqa == "native" else "0"
        import importlib
        import model as model_mod
        importlib.reload(model_mod)          # re-evaluate the GQA switch
        globals()["GPT"] = model_mod.GPT
        label = (f"micro={micro_b} ce_chunk={ce_chunk} compile='{mode}' "
                 f"gqa={gqa}")
        print(f"=== {label}", flush=True)
        r = bench_one(mc, tc.__dict__, tc.data_dir, micro_b, ce_chunk, mode,
                      args.steps, args.warmup, device)
        r.update({"micro_batch": micro_b, "ce_chunk": ce_chunk,
                  "compile": mode, "gqa": gqa})
        results.append(r)
        print(f"    {r}", flush=True)

    ok = [r for r in results if not r.get("oom")]
    if ok:
        best = max(ok, key=lambda r: r["tok_s"])
        print(f"\nBEST: micro={best['micro_batch']} ce_chunk={best['ce_chunk']} "
              f"compile='{best['compile']}' gqa={best['gqa']} -> "
              f"{best['tok_s']:,} tok/s (mfu {best['mfu']:.1%}, "
              f"steady peak {best['steady_peak_gb']}GB)")
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
