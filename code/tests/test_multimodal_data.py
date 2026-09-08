#!/usr/bin/env python3
"""End-to-end tests for the navya-1b-av data tooling (doc 14 build steps).

Covers:
  1. build_asr_shards: synthetic Opus (24 kHz/32 kbps archival format) +
     timestamped transcript -> shard with correct token layout and refs;
     coarse 300 s segments (ytdl today) are skipped and counted
  2. ASRShardDataset + collate_av: ffmpeg window decode matches n_aud contract
  3. forward_av: finite loss through a tiny MultimodalGPT on the real vocab
  4. render_ocr_pairs: PNG + shard with 196 image-pad layout round-trips

Needs ffmpeg on PATH (present on the dev box and A10).
Run:  ../.venv/bin/python tests/test_multimodal_data.py   (from code/)
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
from model import ModelConfig  # noqa: E402
from multimodal import AUD_END, AUD_START, AUDIO_PAD, MultimodalGPT  # noqa: E402
import multimodal_data as md  # noqa: E402

TOKENIZER = os.path.join(CODE, "01-tokenizer", "tokenizer-v0.3-64k.json")
PIPELINE = os.path.join(CODE, "02-data-pipeline")
PY = sys.executable

PASS = []


def check(name, ok):
    PASS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not ok:
        sys.exit(1)


def make_opus(path, seconds=8.0, sr=16_000):
    """Two-tone synthetic speech-band signal -> archival Opus 24k mono 32kbps."""
    t = np.arange(int(seconds * sr)) / sr
    wave = (0.4 * np.sin(2 * np.pi * 300 * t)
            + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
         "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k", "-ar", "24000",
         "-ac", "1", path],
        input=wave.tobytes(), check=True)


def test_asr_build_and_load(tmp):
    opus = os.path.join(tmp, "call.opus")
    make_opus(opus)
    rows = [
        {"id": "call_a", "lang": "hi", "audio_path": opus,
         "segments": [
             {"start": 0.0, "end": 3.0, "text": "namaste SIP cancel karna hai"},
             {"start": 3.2, "end": 6.5, "text": "mutual fund KYC ho gaya kya"},
         ]},
        # ytdl-style coarse block: must be skipped + counted, not ingested
        {"id": "yt_coarse", "lang": "hi", "audio_path": opus,
         "segments": [{"start": 0.0, "end": 300.0, "text": "long block " * 40}]},
    ]
    tr = os.path.join(tmp, "transcripts.jsonl")
    with open(tr, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = os.path.join(tmp, "shards")
    res = subprocess.run(
        [PY, os.path.join(PIPELINE, "build_asr_shards.py"),
         "--transcripts", tr, "--tokenizer", TOKENIZER, "--out", out,
         "--name", "t0", "--audio-field", "audio_path"],
        capture_output=True, text=True)
    check("build_asr_shards runs", res.returncode == 0)
    check("coarse ytdl block skipped + counted",
          "'too_coarse': 1" in res.stdout and "examples 1" in res.stdout)

    refs = [json.loads(l) for l in
            open(os.path.join(out, "asr_hi", "asr", "t0.refs.jsonl"))]
    n_aud = refs[0]["n_aud"]
    check("n_aud = ceil(6.5s * 25) = 163", n_aud == 163)

    arr = np.fromfile(os.path.join(out, "asr_hi", "asr", "t0.bin"),
                      dtype=np.uint16)
    ok_layout = (arr[0] == 0 and arr[1] == AUD_START
                 and (arr[2:2 + n_aud] == AUDIO_PAD).all()
                 and arr[2 + n_aud] == AUD_END and arr[-1] == 1
                 and arr.size == refs[0]["tok_len"])
    check("token layout <bos><aud_start><pad*n><aud_end>...<eos>", ok_layout)

    ds = md.ASRShardDataset(out)
    item = ds[0]
    check("dataset wave matches n_aud contract (163 frames * 640)",
          item["wave"].numel() == 163 * 640
          and item["wave"].abs().max() <= 1.0 + 1e-6)

    batch = md.collate_av([ds[0]])
    check("collate shapes consistent",
          batch["x"].shape[1] == batch["y"].shape[1] == arr.size - 1)

    torch.manual_seed(0)
    mm = MultimodalGPT(ModelConfig(vocab_size=65536, dim=64, n_layers=2,
                                   n_heads=4, n_kv_heads=2,
                                   max_seq_len=batch["x"].shape[1]))
    _, loss = md.forward_av(mm, batch)
    check("forward_av finite loss on real-vocab tiny model",
          torch.isfinite(loss).item())
    loss.backward()
    check("audio projector gets gradient from real shard batch",
          mm.audio_proj.proj.weight.grad.abs().sum() > 0)


def test_ocr_render(tmp):
    txt = os.path.join(tmp, "passages.txt")
    with open(txt, "w") as f:
        for i in range(4):
            f.write("Section 80C of the Income Tax Act allows deduction "
                    f"up to one point five lakh rupees case {i}.\n")
    out = os.path.join(tmp, "ocr_out")
    res = subprocess.run(
        [PY, os.path.join(PIPELINE, "render_ocr_pairs.py"),
         "--text", txt, "--tokenizer", TOKENIZER, "--out", out,
         "--name", "t0", "--n", "3", "--seed", "0"],
        capture_output=True, text=True)
    check("render_ocr_pairs runs", res.returncode == 0)

    refs = [json.loads(l) for l in
            open(os.path.join(out, "ocr", "shards", "t0.refs.jsonl"))]
    check("3 OCR pairs rendered", len(refs) == 3
          and all(os.path.exists(r["image"]) for r in refs))

    arr = np.fromfile(os.path.join(out, "ocr", "shards", "t0.bin"),
                      dtype=np.uint16)
    r0 = refs[0]
    ex = arr[r0["tok_off"]:r0["tok_off"] + r0["tok_len"]]
    ok = (ex[0] == 0 and ex[1] == 8 and (ex[2:198] == 10).all()
          and ex[198] == 9 and ex[-1] == 1)
    check("OCR layout <bos><img_start><pad*196><img_end>...<eos>", ok)

    from PIL import Image
    check("images are 384x384 RGB",
          Image.open(refs[0]["image"]).size == (384, 384))


def test_train_av_smoke(tmp):
    """2-step CPT on the fixture shards + CER eval on the saved ckpt."""
    replay_dir = os.path.join(tmp, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(14, 60000, 10_000, dtype=np.uint16).tofile(
        os.path.join(replay_dir, "replay.bin"))

    stack = os.path.join(CODE, "04-training-stack")
    cfg = {"vocab_size": 65536, "dim": 32, "n_layers": 1, "n_heads": 4,
           "n_kv_heads": 2, "max_seq_len": 256,
           "asr_shards": os.path.join(tmp, "shards"),
           "text_replay_dir": replay_dir,
           "out_dir": os.path.join(tmp, "out_av"),
           "p_audio": 0.5, "audio_batch_size": 1, "text_batch_size": 1,
           "replay_seq_len": 64, "grad_accum": 1, "max_steps": 2,
           "lr": 1e-4, "warmup_steps": 1, "ce_chunk": 64,
           "loader_workers": 0, "log_interval": 1, "ckpt_interval": 2,
           "seed": 0}
    cfg_path = os.path.join(tmp, "av_smoke.json")
    json.dump(cfg, open(cfg_path, "w"))

    res = subprocess.run(
        [PY, os.path.join(stack, "train_av.py"), "--config", cfg_path],
        capture_output=True, text=True, cwd=stack)
    ckpt = os.path.join(cfg["out_dir"], "ckpt_av.pt")
    check("train_av 2-step smoke runs and checkpoints",
          res.returncode == 0 and os.path.exists(ckpt)
          and "step 2" in res.stdout)

    res = subprocess.run(
        [PY, os.path.join(stack, "eval_asr_cer.py"), "--ckpt", ckpt,
         "--shards", cfg["asr_shards"], "--tokenizer", TOKENIZER,
         "--limit", "1", "--max-new", "4"],
        capture_output=True, text=True, cwd=stack)
    check("eval_asr_cer runs on the smoke ckpt and reports CER",
          res.returncode == 0 and "CER[all]" in res.stdout)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_asr_build_and_load(tmp)
        test_ocr_render(tmp)
        test_train_av_smoke(tmp)
    print(f"\n{len(PASS)}/{len(PASS)} checks passed")
