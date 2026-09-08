"""Loader for the navya-1b-av multimodal shards (docs/14_navya_1b_av_plan.md).

Reads the shard format written by 02-data-pipeline/build_asr_shards.py:
a uint16 token .bin plus a .refs.jsonl sidecar whose rows carry each
example's token span (tok_off/tok_len) and its audio window (archival file
path + start_ms/end_ms). Audio is decoded per batch by ffmpeg to the model
contract — 16 kHz mono float32, peak-normalized — so shards stay tiny and
the archival Opus/m4a files remain the single source of truth.

Smoke:  ../.venv/bin/python multimodal_data.py --shards <dir> \
            (loads one batch, runs a tiny MultimodalGPT forward)
"""
import glob
import json
import os
import subprocess

import numpy as np
import torch

from model import ModelConfig
from multimodal import (FRAME, SAMPLE_RATE, MultimodalGPT,
                        mask_modality_targets, splice_multimodal)

PAD_ID = 2


def decode_audio(path, start_ms=None, end_ms=None):
    """ffmpeg decode -> np.float32 [S] @ 16 kHz mono, peak-normalized."""
    cmd = ["ffmpeg", "-v", "error"]
    if start_ms is not None:
        cmd += ["-ss", f"{start_ms / 1000:.3f}"]
    if end_ms is not None:
        cmd += ["-to", f"{end_ms / 1000:.3f}"]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    wave = np.frombuffer(out, dtype=np.float32).copy()
    peak = np.abs(wave).max()
    if peak > 0:
        wave /= peak
    return wave


class ASRShardDataset(torch.utils.data.Dataset):
    """All (bin, refs) pairs under a directory tree; one item per example."""

    def __init__(self, shard_dir):
        self.examples = []          # (memmap, ref)
        for bin_path in sorted(glob.glob(
                os.path.join(shard_dir, "**", "*.bin"), recursive=True)):
            refs_path = bin_path[:-4] + ".refs.jsonl"
            if not os.path.exists(refs_path):
                continue
            arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
            for line in open(refs_path, encoding="utf-8"):
                self.examples.append((arr, json.loads(line)))
        if not self.examples:
            raise FileNotFoundError(f"no (bin, refs.jsonl) pairs in {shard_dir}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        arr, ref = self.examples[i]
        tokens = torch.from_numpy(
            arr[ref["tok_off"]:ref["tok_off"] + ref["tok_len"]]
            .astype(np.int64))
        wave = decode_audio(ref["audio"], ref["start_ms"], ref["end_ms"])
        # the ref's n_aud is the contract; pad/trim decode jitter to match
        need = ref["n_aud"] * FRAME
        if wave.size < need:
            wave = np.pad(wave, (0, need - wave.size))
        return {"tokens": tokens, "wave": torch.from_numpy(wave[:need]),
                "n_aud": ref["n_aud"]}


def collate_av(batch):
    """Pad tokens (PAD_ID) and waves (silence) to batch max; build x/y."""
    T = max(b["tokens"].numel() for b in batch)
    S = max(b["wave"].numel() for b in batch)
    toks = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    wave = torch.zeros(len(batch), S)
    for i, b in enumerate(batch):
        toks[i, :b["tokens"].numel()] = b["tokens"]
        wave[i, :b["wave"].numel()] = b["wave"]
    x, y = toks[:, :-1], toks[:, 1:].clone()
    y[y == PAD_ID] = -100
    return {"x": x, "y": y, "wave": wave,
            "n_aud": [b["n_aud"] for b in batch]}


def forward_av(mm: MultimodalGPT, batch, ce_chunk=0):
    """One AV training step's forward: per-row audio embeds spliced into x."""
    emb = mm.audio_proj(batch["wave"])                    # [B, F_max, dim]
    audio_embeds = [emb[b, :n] for b, n in enumerate(batch["n_aud"])]
    x = splice_multimodal(mm.gpt.tok_emb, batch["x"], audio_embeds=audio_embeds)
    y = mask_modality_targets(batch["x"], batch["y"])
    return mm.gpt(batch["x"], y, ce_chunk=ce_chunk, inputs_embeds=x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    ds = ASRShardDataset(args.shards)
    print(f"{len(ds)} examples")
    batch = collate_av([ds[i] for i in range(min(args.batch, len(ds)))])
    print("x", tuple(batch["x"].shape), "wave", tuple(batch["wave"].shape),
          "n_aud", batch["n_aud"])
    mm = MultimodalGPT(ModelConfig(vocab_size=65536, dim=64, n_layers=2,
                                   n_heads=4, n_kv_heads=2,
                                   max_seq_len=batch["x"].shape[1]))
    _, loss = forward_av(mm, batch)
    print("smoke loss", float(loss))
