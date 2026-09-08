#!/usr/bin/env python3
"""M0 tests for the multimodal front-ends (docs/14_navya_1b_av_plan.md).

Covers:
  1. inputs_embeds path is a no-op refactor (== tok_emb path, logits + grads)
  2. text-only GPT checkpoints load into MultimodalGPT.gpt strict=True
  3. audio framing: 1 s @ 16 kHz -> 25 tokens, tail zero-padded
  4. vision projector: 729 SigLIP patches -> 196 tokens
  5. splicing places modality embeddings exactly at pad positions
  6. modality positions are excluded from the loss (plain and chunked CE agree)
  7. end-to-end audio forward: finite loss, gradient reaches the projector

Run:  ../.venv/bin/python tests/test_multimodal.py   (from code/)
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "04-training-stack"))
from model import GPT, ModelConfig  # noqa: E402
from multimodal import (AUDIO_PAD, FRAME, AudioFrameProjector,  # noqa: E402
                        MultimodalGPT, VisionProjector, frame_audio,
                        mask_modality_targets, splice_multimodal)

PASS = []


def check(name, ok):
    PASS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not ok:
        sys.exit(1)


CFG = ModelConfig(vocab_size=500, dim=64, n_layers=2, n_heads=4,
                  n_kv_heads=2, max_seq_len=64)


def test_inputs_embeds_noop():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(14, 500, (2, 16))
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    logits_a, loss_a = m(x, y)
    logits_b, loss_b = m(None, y, inputs_embeds=m.tok_emb(x))
    check("inputs_embeds == tok_emb path (logits)",
          torch.equal(logits_a, logits_b) and torch.equal(loss_a, loss_b))

    ga = torch.autograd.grad(loss_a, m.tok_emb.weight, retain_graph=False)[0]
    _, loss_b2 = m(None, y, inputs_embeds=m.tok_emb(x))
    gb = torch.autograd.grad(loss_b2, m.tok_emb.weight)[0]
    check("inputs_embeds == tok_emb path (grads)", torch.allclose(ga, gb))


def test_text_checkpoint_compat():
    torch.manual_seed(1)
    plain = GPT(CFG)
    mm = MultimodalGPT(CFG, audio=True, vision=True)
    missing, unexpected = mm.gpt.load_state_dict(plain.state_dict(),
                                                 strict=True), None
    x = torch.randint(14, 500, (1, 8))
    same = torch.equal(mm.gpt(x, x)[0], plain(x, x)[0])
    check("text-only checkpoint loads strict=True into MultimodalGPT.gpt",
          same)


def test_audio_framing():
    wave = torch.rand(2, 16_000) * 2 - 1          # 1 s @ 16 kHz
    frames = frame_audio(wave)
    check("1 s @ 16 kHz -> 25 frames of 640", frames.shape == (2, 25, FRAME))

    ragged = torch.ones(1, FRAME + 7)             # tail must zero-pad
    fr = frame_audio(ragged)
    check("tail zero-padded to whole frame",
          fr.shape == (1, 2, FRAME) and fr[0, 1, 7:].abs().sum() == 0
          and fr[0, 1, :7].sum() == 7)

    proj = AudioFrameProjector(CFG)
    check("audio projector shape [B, 25, dim]",
          proj(wave).shape == (2, 25, CFG.dim))
    check("audio token count helper", AudioFrameProjector.n_tokens(16_000) == 25)


def test_vision_projector():
    proj = VisionProjector(CFG)
    feats = torch.randn(2, 729, 1152)
    out = proj(feats)
    check("729 SigLIP patches -> 196 tokens",
          out.shape == (2, 196, CFG.dim) and proj.n_tokens() == 196)


def test_splice_positions():
    torch.manual_seed(2)
    m = GPT(CFG)
    B, T, n_aud = 2, 20, 5
    x = torch.randint(14, 500, (B, T))
    x[:, 3] = 11                                   # <|aud_start|>
    x[:, 4:4 + n_aud] = AUDIO_PAD
    x[:, 4 + n_aud] = 12                           # <|aud_end|>
    aud = torch.randn(B, n_aud, CFG.dim)
    spliced = splice_multimodal(m.tok_emb, x, audio_embeds=aud)
    base = m.tok_emb(x)
    at_pads = torch.equal(spliced[:, 4:4 + n_aud], aud)
    elsewhere = (torch.equal(spliced[:, :4], base[:, :4])
                 and torch.equal(spliced[:, 4 + n_aud:], base[:, 4 + n_aud:]))
    check("splice hits exactly the audio_pad positions", at_pads and elsewhere)

    bad = torch.randn(B, n_aud + 1, CFG.dim)       # count mismatch must raise
    try:
        splice_multimodal(m.tok_emb, x, audio_embeds=bad)
        check("splice count mismatch raises", False)
    except AssertionError:
        check("splice count mismatch raises", True)


def test_loss_masking():
    torch.manual_seed(3)
    m = GPT(CFG)
    x = torch.randint(14, 500, (2, 24))
    x[:, 5:13] = AUDIO_PAD
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    masked = mask_modality_targets(x, y)
    check("all special-id targets masked to -100",
          ((masked >= 8) & (masked <= 13)).sum() == 0
          and (masked[:, 20] == y[:, 20]).all())

    _, loss_plain = m(x, masked)
    _, loss_chunk = m(x, masked, ce_chunk=7)
    check("plain and chunked CE agree under modality mask",
          torch.allclose(loss_plain, loss_chunk, atol=1e-5))


def test_end_to_end_audio():
    torch.manual_seed(4)
    mm = MultimodalGPT(CFG, audio=True)
    B, n_aud = 2, 25
    wave = torch.rand(B, 16_000) * 2 - 1
    x = torch.randint(14, 500, (B, 40))
    x[:, 2] = 11
    x[:, 3:3 + n_aud] = AUDIO_PAD
    x[:, 3 + n_aud] = 12
    y = torch.full_like(x, -100)
    y[:, :-1] = x[:, 1:]
    _, loss = mm(x, wave=wave, targets=y)
    check("end-to-end audio forward gives finite loss",
          torch.isfinite(loss).item())
    loss.backward()
    g = mm.audio_proj.proj.weight.grad
    check("gradient reaches the audio projector",
          g is not None and g.abs().sum() > 0)
    check("base GPT params also receive gradient",
          mm.gpt.blocks[0].attn.wq.weight.grad is not None)


if __name__ == "__main__":
    test_inputs_embeds_noop()
    test_text_checkpoint_compat()
    test_audio_framing()
    test_vision_projector()
    test_splice_positions()
    test_loss_masking()
    test_end_to_end_audio()
    print(f"\n{len(PASS)}/{len(PASS)} checks passed")
