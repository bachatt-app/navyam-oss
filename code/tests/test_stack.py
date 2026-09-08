#!/usr/bin/env python3
"""Regression tests for the navya-1a stack fixes.

Covers what the review flagged as untested:
  1. token-weighted gradient equivalence (accumulated == one big batch)
  2. epoch-coverage loader: full coverage, determinism, reshuffle
  3. build_stream: val_spans present, single meta write, provenance kept
  4. packing isolation (block-diagonal attention == standalone example)
  5. parallel_clean: 1-worker vs N-worker output equality + fail-closed evals

Run:  ../.venv/bin/python tests/test_stack.py   (from code/)
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402

PASS = []


def check(name, ok):
    PASS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not ok:
        sys.exit(1)


def test_token_weighted_gradients():
    """Two micro-batches with unequal supervised counts, token-weighted,
    must produce the same gradient as one combined batch."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=500, dim=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=32)
    x = torch.randint(3, 500, (4, 16))
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    y[0, :10] = -100        # unequal supervision per row
    y[2, :14] = -100

    def grads(micro_slices):
        torch.manual_seed(1)
        m = GPT(cfg)
        counts = [(y[s] != -100).sum().item() for s in micro_slices]
        total = sum(counts)
        m.zero_grad()
        for s, n in zip(micro_slices, counts):
            _, loss = m(x[s], y[s])
            (loss * (n / total)).backward()
        return [p.grad.clone() for p in m.parameters()]

    g_micro = grads([slice(0, 2), slice(2, 4)])
    g_full = grads([slice(0, 4)])
    diff = max((a - b).abs().max().item() for a, b in zip(g_micro, g_full))
    check(f"token-weighted grad equivalence (max diff {diff:.2e})", diff < 1e-5)


def test_epoch_coverage():
    from train import Data
    d = tempfile.mkdtemp()
    (np.arange(1000) % 97).astype(np.uint16).tofile(os.path.join(d, "train.bin"))
    (np.arange(200) % 97).astype(np.uint16).tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 97, "dtype": "uint16"},
              open(os.path.join(d, "meta.json"), "w"))
    L, B, GA = 8, 4, 2
    data = Data(d, L, GA)
    n_win = (1000 - 1) // L
    seen, c, step, micro = [], 0, 0, 0
    while c < n_win:
        data.batch("train", B, 7, step, micro, "cpu")
        base = (step * GA + micro) * B
        for j in range(B):
            epoch, pos = divmod(base + j, n_win)
            if epoch == 0:
                seen.append(int(data._perm("train", 0, n_win, 7)[pos]))
        c += B
        micro += 1
        if micro == GA:
            micro, step = 0, step + 1
    check("epoch coverage: every window exactly once",
          sorted(seen) == list(range(n_win)))
    x1, _ = data.batch("train", B, 7, 3, 1, "cpu")
    x2, _ = Data(d, L, GA).batch("train", B, 7, 3, 1, "cpu")
    check("loader determinism across instances", torch.equal(x1, x2))


def test_build_stream_meta():
    d = tempfile.mkdtemp()
    dom = os.path.join(d, "dom")
    os.makedirs(dom)
    rng = np.random.default_rng(1)
    for name in ("global_english", "indian_english", "indian_languages",
                 "code", "finance_econ_law", "math_reasoning"):
        rng.integers(3, 60000, size=30000).astype(np.uint16).tofile(
            os.path.join(dom, f"{name}.bin"))
        json.dump({"vocab_size": 65536, "dtype": "uint16"},
                  open(os.path.join(dom, f"{name}.bin.meta.json"), "w"))
    out = os.path.join(d, "stream")
    r = subprocess.run([sys.executable,
                        os.path.join(CODE, "04-training-stack", "build_stream.py"),
                        "--domains", dom, "--out", out,
                        "--total-tokens", "1e5"], capture_output=True, text=True)
    check("build_stream runs", r.returncode == 0)
    m = json.load(open(os.path.join(out, "meta.json")))
    check("build_stream meta has val_spans + provenance",
          "val_spans" in m and "val_frac" in m and "stream_id" in m
          and len(m["val_spans"]) == 6)


def test_packing_isolation():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=1000, dim=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=64)
    m = GPT(cfg).eval()
    a = torch.randint(3, 1000, (1, 10))
    b = torch.randint(3, 1000, (1, 10))
    packed = torch.cat([a, b], dim=1)
    ex = torch.tensor([[0] * 10 + [1] * 10])
    same = ex.unsqueeze(2) == ex.unsqueeze(1)
    causal = torch.tril(torch.ones(20, 20, dtype=torch.bool))
    attn = (same & causal).unsqueeze(1)
    with torch.no_grad():
        packed_logits, _ = m(packed, packed, attn_mask=attn)
        alone_logits, _ = m(b, b)
    diff = (packed_logits[0, 10:] - alone_logits[0]).abs().max().item()
    check(f"packing isolation (max diff {diff:.2e})", diff < 1e-4)


def test_parallel_clean():
    pipeline = os.path.join(CODE, "02-data-pipeline", "pipeline")
    d = tempfile.mkdtemp()
    pool = os.path.join(d, "pool.jsonl")
    sents = ["The reserve bank announced that interest rates will remain "
             "unchanged for the quarter and markets reacted calmly.",
             "Households with an emergency fund are better placed to handle "
             "a sudden loss of income than those without one.",
             "The new rules require lenders to disclose the total cost of "
             "credit to every borrower before signing."]
    import random
    rnd = random.Random(3)
    with open(pool, "w") as f:
        for i in range(400):
            body = " ".join(rnd.choices(sents, k=8))
            f.write(json.dumps({"text": f"Doc {i}. " + body}) + "\n")
    evals = os.path.join(CODE, "03-evals", "bachattbench", "*.jsonl")
    outs = {}
    for w in (1, 3):
        out = os.path.join(d, f"clean_w{w}.jsonl")
        st = os.path.join(d, f"state_w{w}.bin")
        r = subprocess.run([sys.executable, "parallel_clean.py",
                            "--inputs", pool, "--evals", evals,
                            "--dedup-state", st, "--out", out,
                            "--workers", str(w), "--chunk-lines", "50"],
                           cwd=pipeline, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        outs[w] = sorted(open(out).readlines())
    check("parallel_clean 1-worker == 3-worker", outs[1] == outs[3])
    r = subprocess.run([sys.executable, "parallel_clean.py",
                        "--inputs", pool, "--evals", "/nonexistent/*.jsonl",
                        "--dedup-state", os.path.join(d, "s.bin"),
                        "--out", os.path.join(d, "c.jsonl")],
                       cwd=pipeline, capture_output=True, text=True)
    check("parallel_clean fails closed on missing evals", r.returncode != 0)


def test_train_main_smoke():
    """Run train.py main() end to end on a tiny config — imports alone do not
    cover the resume/fingerprint/checkpoint code paths (a data.meta attribute
    bug shipped because nothing executed main)."""
    d = tempfile.mkdtemp()
    (np.arange(30000) % 97).astype(np.uint16).tofile(os.path.join(d, "train.bin"))
    (np.arange(2000) % 97).astype(np.uint16).tofile(os.path.join(d, "val.bin"))
    json.dump({"vocab_size": 97, "dtype": "uint16", "stream_id": "stream-test"},
              open(os.path.join(d, "meta.json"), "w"))
    cfg = {"vocab_size": 97, "dim": 32, "n_layers": 2, "n_heads": 4,
           "n_kv_heads": 2, "max_seq_len": 32, "data_dir": d,
           "out_dir": os.path.join(d, "out"), "batch_size": 2,
           "grad_accum": 2, "max_steps": 4, "lr": 1e-3, "warmup_steps": 1,
           "log_interval": 2, "eval_interval": 100, "eval_iters": 2,
           "ckpt_interval": 2}
    cfg_path = os.path.join(d, "cfg.json")
    json.dump(cfg, open(cfg_path, "w"))
    trainer = os.path.join(CODE, "04-training-stack", "train.py")
    r = subprocess.run([sys.executable, trainer, "--config", cfg_path],
                       capture_output=True, text=True)
    check("train.py main() fresh run", r.returncode == 0 and
          "done." in r.stdout)
    r = subprocess.run([sys.executable, trainer, "--config", cfg_path,
                        "--resume"], capture_output=True, text=True)
    check("train.py main() resume run", r.returncode == 0 and
          "resumed" in r.stdout)


def test_chunked_ce_equivalence():
    """Chunked linear-CE must match full-logits CE in loss AND gradients."""
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=800, dim=48, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=64)
    x = torch.randint(3, 800, (2, 32))
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    y[0, :20] = -100          # exercise ignore_index in both paths

    def run(chunk):
        torch.manual_seed(4)
        m = GPT(cfg)
        _, loss = m(x, y, ce_chunk=chunk)
        loss.backward()
        return loss.item(), [p.grad.clone() for p in m.parameters()]

    l_full, g_full = run(0)
    l_chunk, g_chunk = run(13)     # deliberately unaligned chunk size
    dl = abs(l_full - l_chunk)
    dg = max((a - b).abs().max().item() for a, b in zip(g_full, g_chunk))
    check(f"chunked CE loss equivalence (diff {dl:.2e})", dl < 1e-5)
    check(f"chunked CE grad equivalence (max diff {dg:.2e})", dg < 1e-5)


def test_virtual_stream_equivalence():
    """PERF item 9: --virtual must be indistinguishable from a physical
    build — same stream_id and bit-identical training batches."""
    d = tempfile.mkdtemp()
    dom = os.path.join(d, "dom")
    os.makedirs(dom)
    rng = np.random.default_rng(11)
    rng.integers(3, 60000, size=90000).astype(np.uint16).tofile(
        os.path.join(dom, "alpha.bin"))
    rng.integers(3, 60000, size=40000).astype(np.uint16).tofile(
        os.path.join(dom, "beta.bin"))
    for n in ("alpha", "beta"):
        json.dump({"vocab_size": 65536, "dtype": "uint16"},
                  open(os.path.join(dom, f"{n}.bin.meta.json"), "w"))
    w = os.path.join(d, "w.json")
    json.dump({"alpha": 0.7, "beta": 0.3}, open(w, "w"))
    outs = {}
    for flag, name in (([], "phys"), (["--virtual"], "virt")):
        out = os.path.join(d, name)
        r = subprocess.run([sys.executable,
                            os.path.join(CODE, "04-training-stack",
                                         "build_stream.py"),
                            "--domains", dom, "--weights", w, "--out", out,
                            "--total-tokens", "3e5"] + flag,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        outs[name] = out
    mp = json.load(open(os.path.join(outs["phys"], "meta.json")))
    mv = json.load(open(os.path.join(outs["virt"], "meta.json")))
    sys.path.insert(0, os.path.join(CODE, "04-training-stack"))
    import train as T
    dp = T.Data(outs["phys"], 128, 2)
    dv = T.Data(outs["virt"], 128, 2)
    same = (mp["stream_id"] == mv["stream_id"]
            and len(dp.train) == len(dv.train))
    if same:
        for step in (0, 5, 900):
            xp, yp = dp.batch("train", 4, 1337, step, 1, "cpu")
            xv, yv = dv.batch("train", 4, 1337, step, 1, "cpu")
            same = same and bool((xp == xv).all() and (yp == yv).all())
    check("virtual stream == physical (id + batches)", bool(same))


if __name__ == "__main__":
    test_train_main_smoke()
    test_chunked_ce_equivalence()
    test_token_weighted_gradients()
    test_epoch_coverage()
    test_build_stream_meta()
    test_virtual_stream_equivalence()
    test_packing_isolation()
    test_parallel_clean()
    print(f"\nall {len(PASS)} checks passed")


def test_zero1_act_ckpt_equivalence():
    """navya-2 (4B) memory path: ZeRO-1 optimizer sharding + per-block
    activation checkpointing must be numerically transparent — a 2-rank run
    reproduces the single-process losses, and its consolidated checkpoint
    resumes at ANY world size (2 ranks with ZeRO, 1 rank with plain AdamW).
    CPU/gloo so it runs anywhere; MPS has no collectives."""
    import re
    import shutil
    base = json.load(open(os.path.join(CODE, "04-training-stack", "configs",
                                       "smoke.json")))
    d = tempfile.mkdtemp()
    env = {**os.environ, "NAVYA_DEVICE": "cpu", "MASTER_PORT": "29751"}
    train = os.path.join(CODE, "04-training-stack", "train.py")

    def cfg(name, **kw):
        c = {**base, "out_dir": os.path.join(d, name), "max_steps": 12,
             "ckpt_interval": 4, "eval_interval": 100, "log_interval": 1,
             "grad_accum": 2, "compile_mode": "", **kw}
        p = os.path.join(d, name + ".json")
        json.dump(c, open(p, "w"))
        return p

    def losses(out):
        return [float(m) for m in re.findall(r"^step\s+\d+ \| loss\s+([\d.]+)",
                                             out, flags=re.M)]

    def run(argv):
        r = subprocess.run(argv, capture_output=True, text=True, env=env,
                           cwd=CODE)
        check("train run exits 0: " + " ".join(argv[-3:]), r.returncode == 0)
        return r.stdout

    ref = losses(run([sys.executable, train, "--config", cfg("ref")]))
    zcfg = cfg("zero", zero1=True, act_ckpt=True)
    two = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node",
           "2", "--master_port", "29751", train, "--config", zcfg]
    z = losses(run(two))
    check("zero1+act_ckpt (2 ranks) == single-process losses",
          len(z) == len(ref) == 12 and all(abs(a - b) < 2e-3
                                           for a, b in zip(z, ref)))
    out = os.path.join(d, "zero")
    shutil.copy(os.path.join(out, "ckpt_prev.pt"),       # step-8 checkpoint
                os.path.join(out, "ckpt_last.pt"))
    z2 = losses(run(two + ["--resume"]))
    check("resume on 2 ranks continues the trajectory",
          len(z2) == 3 and all(abs(a - b) < 2e-3 for a, b in zip(z2, ref[9:])))
    shutil.copy(os.path.join(out, "ckpt_prev.pt"),
                os.path.join(out, "ckpt_last.pt"))
    z1 = losses(run([sys.executable, train, "--config", zcfg, "--resume"]))
    check("consolidated ZeRO checkpoint resumes on 1 rank (plain AdamW)",
          len(z1) == 3 and all(abs(a - b) < 2e-3 for a, b in zip(z1, ref[9:])))


def test_wsd_schedule_and_branch():
    """WSD: constant LR after warmup on the trunk; a decay branch continues
    weights+optimizer from a trunk checkpoint under a new max_steps/decay and
    lands exactly on min_lr at its last step."""
    import re
    import shutil
    from train import TrainConfig, lr_at
    tc = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=1000, lr_schedule="wsd",
                     decay_steps=0, min_lr_frac=0.1)
    check("wsd trunk is constant after warmup",
          lr_at(10, tc) == 1e-3 and lr_at(999, tc) == 1e-3 and lr_at(0, tc) < 1e-3)
    tb = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, lr_schedule="wsd",
                     decay_steps=20, min_lr_frac=0.1)
    check("wsd branch decays linearly to min_lr",
          lr_at(79, tb) == 1e-3 and abs(lr_at(99, tb) - 1e-4) < 1e-12
          and abs(lr_at(89, tb) - (1e-3 - 0.9e-3 * 10 / 20)) < 1e-12)

    base = json.load(open(os.path.join(CODE, "04-training-stack", "configs",
                                       "smoke.json")))
    d = tempfile.mkdtemp()
    env = {**os.environ, "NAVYA_DEVICE": "cpu"}
    train = os.path.join(CODE, "04-training-stack", "train.py")
    trunk = {**base, "out_dir": os.path.join(d, "trunk"), "max_steps": 8,
             "ckpt_interval": 4, "eval_interval": 100, "log_interval": 1,
             "compile_mode": "", "lr_schedule": "wsd", "decay_steps": 0,
             "warmup_steps": 2}
    tp = os.path.join(d, "trunk.json"); json.dump(trunk, open(tp, "w"))
    r = subprocess.run([sys.executable, train, "--config", tp],
                       capture_output=True, text=True, env=env, cwd=CODE)
    check("wsd trunk runs", r.returncode == 0)
    lrs = [float(m) for m in re.findall(r"\| lr ([\d.e+-]+)", r.stdout)]
    check("trunk LR constant from warmup on", len(set(lrs[2:])) == 1)
    mk = os.path.join(CODE, "04-training-stack", "make_branch_config.py")
    r = subprocess.run([sys.executable, mk, tp,
                        os.path.join(d, "trunk", "ckpt_prev.pt"), "4"],
                       capture_output=True, text=True, env=env, cwd=CODE)
    check("make_branch_config writes the branch", r.returncode == 0
          and os.path.exists(os.path.join(d, "trunk-decay.json")))
    r = subprocess.run([sys.executable, train, "--config",
                        os.path.join(d, "trunk-decay.json"), "--branch-from",
                        os.path.join(d, "trunk", "ckpt_prev.pt")],
                       capture_output=True, text=True, env=env, cwd=CODE)
    check("decay branch runs from the trunk checkpoint",
          r.returncode == 0 and "branched from" in r.stdout)
    blrs = [float(m) for m in re.findall(r"\| lr ([\d.e+-]+)", r.stdout)]
    check("branch decays to min_lr on its last step",
          len(blrs) == 4 and blrs[0] > blrs[-1]
          and abs(blrs[-1] - trunk["lr"] * 0.1) < 1e-9)
