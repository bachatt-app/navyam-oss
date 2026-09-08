"""Multimodal front-ends for the Navya stack (plan: docs/14_navya_1b_av_plan.md).

M0 mechanics, all additive: the base GPT is untouched parameter-wise (text
checkpoints load strict=True); modality projectors live here and their
embeddings enter the transformer via GPT.forward(inputs_embeds=...).

Audio contract (doc 14 §1): 16 kHz mono float32 in [-1, 1], non-overlapping
640-sample (40 ms) frames, one frame = one token, 25 tokens/s. Archival is
Opus 24 kHz mono 32 kbps; decode+resample to 16 kHz happens in the data
pipeline, not here — this module only ever sees 16 kHz waveforms.

Vision contract (doc 13): SigLIP-so400m-384 (frozen, external) emits
[B, 729, 1152] patch features; the projector maps them to d_model and merges
2x2 neighbours (27x27 grid padded to 28x28 -> 196 image tokens).

Splicing: the tokenized text stream carries runs of <|audio_pad|>/<|image_pad|>
exactly as long as the projected clip/image; splice_multimodal() replaces those
positions' embeddings. Targets at modality positions must be -100 (both CE
paths honor ignore_index).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, ModelConfig, RMSNorm

# tokenizer v0.3 reserved-block ids (doc 14 §1/§2)
IMG_START, IMG_END, IMAGE_PAD = 8, 9, 10
AUD_START, AUD_END, AUDIO_PAD = 11, 12, 13

SAMPLE_RATE = 16_000          # model-boundary rate, Hz (archival is Opus 24k)
FRAME = 640                   # samples per audio token = 40 ms @ 16 kHz
SIGLIP_DIM = 1152             # so400m patch feature width
SIGLIP_GRID = 27              # 384px / patch14 -> 27x27 = 729 patches
VISION_MERGE = 2              # 2x2 neighbour merge -> 196 tokens per image


def frame_audio(wave: torch.Tensor) -> torch.Tensor:
    """[B, S] or [S] float32 waveform -> [B, ceil(S/FRAME), FRAME].

    Pads the tail with zeros (silence) to a whole frame. Peak normalization
    is the data pipeline's job; asserted loosely here to catch int16 leaks.
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    assert wave.dtype.is_floating_point, "waveform must be float in [-1, 1]"
    B, S = wave.shape
    pad = (-S) % FRAME
    if pad:
        wave = F.pad(wave, (0, pad))
    return wave.view(B, -1, FRAME)


class AudioFrameProjector(nn.Module):
    """Encoder-free audio: one linear per 640-sample frame, Gemma-4-12B style."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(FRAME, cfg.dim, bias=False)
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """[B, S] waveform -> [B, n_frames, dim] embeddings."""
        frames = frame_audio(wave).to(self.proj.weight.dtype)
        return self.norm(self.proj(frames))

    @staticmethod
    def n_tokens(n_samples: int) -> int:
        return (n_samples + FRAME - 1) // FRAME


class VisionProjector(nn.Module):
    """SigLIP patch features -> LM space; 2x2 merge to cut token count 4x."""

    def __init__(self, cfg: ModelConfig, in_dim: int = SIGLIP_DIM,
                 grid: int = SIGLIP_GRID, merge: int = VISION_MERGE):
        super().__init__()
        self.grid, self.merge = grid, merge
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, grid*grid, in_dim] -> [B, ceil(grid/merge)^2, dim]."""
        B, N, C = feats.shape
        assert N == self.grid ** 2, f"expected {self.grid ** 2} patches, got {N}"
        g, m = self.grid, self.merge
        x = feats.view(B, g, g, C)
        pad = (-g) % m
        if pad:  # 27 -> 28 for 2x2 merge
            x = F.pad(x, (0, 0, 0, pad, 0, pad))
        gm = x.shape[1] // m
        # group each m x m neighbourhood into one token (channel concat is the
        # usual pixel-shuffle merge; average instead to keep in_dim fixed)
        x = x.view(B, gm, m, gm, m, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, gm * gm, m * m, C).mean(dim=2)
        return self.norm(self.mlp(x))

    def n_tokens(self) -> int:
        return ((self.grid + self.merge - 1) // self.merge) ** 2


def splice_multimodal(tok_emb: nn.Embedding, idx: torch.Tensor,
                      audio_embeds=None, image_embeds=None) -> torch.Tensor:
    """Build inputs_embeds: text embeddings with modality embeddings dropped
    into the <|audio_pad|>/<|image_pad|> positions.

    audio_embeds/image_embeds: [B, n, dim] or list of [n_i, dim] per row; the
    total pad count per row must equal the supplied embedding count.
    """
    x = tok_emb(idx).clone()
    for pad_id, embeds in ((AUDIO_PAD, audio_embeds), (IMAGE_PAD, image_embeds)):
        if embeds is None:
            continue
        for b in range(idx.shape[0]):
            pos = (idx[b] == pad_id).nonzero(as_tuple=True)[0]
            e = embeds[b]
            assert pos.numel() == e.shape[0], (
                f"row {b}: {pos.numel()} pad id={pad_id} slots, "
                f"{e.shape[0]} embeddings")
            x[b, pos] = e.to(x.dtype)
    return x


def mask_modality_targets(idx: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
    """Loss on text only: -100 wherever the *predicted* token is a modality
    filler or delimiter (positions whose target is one of the special ids)."""
    special = ((targets >= IMG_START) & (targets <= AUDIO_PAD))
    return targets.masked_fill(special, -100)


class MultimodalGPT(nn.Module):
    """GPT + projectors. Text-only checkpoints load into .gpt unchanged;
    projectors are the only new parameters (~1.3M audio, ~8.5M vision @ 2048)."""

    def __init__(self, cfg: ModelConfig, audio: bool = True,
                 vision: bool = False):
        super().__init__()
        self.gpt = GPT(cfg)
        self.audio_proj = AudioFrameProjector(cfg) if audio else None
        self.vision_proj = VisionProjector(cfg) if vision else None

    def forward(self, idx, wave=None, image_feats=None, targets=None,
                attn_mask=None, position_ids=None, ce_chunk=0):
        audio_embeds = self.audio_proj(wave) if wave is not None else None
        image_embeds = (self.vision_proj(image_feats)
                        if image_feats is not None else None)
        if audio_embeds is None and image_embeds is None:
            return self.gpt(idx, targets, attn_mask, position_ids, ce_chunk)
        x = splice_multimodal(self.gpt.tok_emb, idx, audio_embeds, image_embeds)
        if targets is not None:
            targets = mask_modality_targets(idx, targets)
        return self.gpt(idx, targets, attn_mask, position_ids, ce_chunk,
                        inputs_embeds=x)
