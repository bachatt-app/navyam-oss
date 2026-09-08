"""Decoder-only transformer, research-scale reference implementation.

Implements the frozen starting recipe from the execution plan (Workstream 3):
RoPE positions, grouped-query attention, RMSNorm (pre-norm), SwiGLU FFN,
no biases, tied embeddings, next-token objective.

This maps 1:1 onto the derivation in ../../main.tex, with three modern
substitutions:
    learned position embeddings  ->  RoPE (applied to Q and K)
    LayerNorm                    ->  RMSNorm
    GELU 4x FFN                  ->  SwiGLU (~8/3 x, rounded)

Role: research models (300M--1B) and ablations on single nodes. Alpha-scale
(8B+) training adopts a distributed framework per the decision log; this file
is the executable spec that framework must reproduce.
"""

from dataclasses import dataclass

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2        # GQA: n_heads % n_kv_heads == 0
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize in fp32 for stability, cast back to input dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).type_as(x)


def rope_cache(cfg: ModelConfig, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables, each [max_seq_len, head_dim/2]."""
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_base **
                      (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               position_ids=None) -> torch.Tensor:
    """Rotate Q/K pairs; packed SFT may reset positions per conversation."""
    T = x.shape[2]
    if position_ids is None:
        cos = cos[:T].view(1, 1, T, -1).to(x.dtype)
        sin = sin[:T].view(1, 1, T, -1).to(x.dtype)
    else:
        cos = cos[position_ids].unsqueeze(1).to(x.dtype)
        sin = sin[position_ids].unsqueeze(1).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _probe_sdpa_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 4, 8)
        kv = torch.zeros(1, 1, 4, 8)
        F.scaled_dot_product_attention(q, kv, kv, is_causal=True,
                                       enable_gqa=True)
        return True
    except TypeError:
        return False


# native GQA is used when the API supports it AND it is not disabled via env
# (NAVYA_NATIVE_GQA=0 forces the expanded-KV path — used by benchmarks to
# compare both, and as an escape hatch if a backend misbehaves)
_SDPA_GQA = _probe_sdpa_gqa() and os.environ.get("NAVYA_NATIVE_GQA", "1") == "1"


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        d, hd = cfg.dim, cfg.head_dim
        self.wq = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, d, bias=False)

    def forward(self, x, cos, sin, attn_mask=None, position_ids=None):
        B, T, _ = x.shape
        cfg, hd = self.cfg, self.cfg.head_dim
        q = self.wq(x).view(B, T, cfg.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, cfg.n_kv_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, cfg.n_kv_heads, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin, position_ids)
        k = apply_rope(k, cos, sin, position_ids)
        # GQA: native SDPA broadcast when available; else expand KV heads
        rep = cfg.n_heads // cfg.n_kv_heads
        if rep > 1 and not _SDPA_GQA:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        gqa = {"enable_gqa": True} if (rep > 1 and _SDPA_GQA) else {}
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **gqa)
        else:
            # SFT packing: [B,1,T,T] bool mask (True = may attend), already
            # causal AND block-diagonal per packed example
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 **gqa)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.dim / 3)
        hidden = 256 * ((hidden + 255) // 256)  # round up to multiple of 256
        self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None, position_ids=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask,
                          position_ids)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        cos, sin = rope_cache(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # set by the trainer (act_ckpt): recompute each block's activations in
        # backward instead of storing them — ~1/3 more compute, memory for a
        # 4B model drops from ~1GB/layer/sequence to the block input only
        self.act_ckpt = False

        self.apply(self._init)
        # scaled init for residual-path output projections
        std = 0.02 / (2 * cfg.n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=std)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def embed(self, idx):
        """Mean-pooled final-layer hidden state — a semantic sentence embedding
        from the model ITSELF. The 1.3B model's generation is too weak to follow
        a 'reply YES/NO' classification instruction, but its internal
        representation still captures meaning, so we match intent (e.g. identity
        questions) by cosine similarity to exemplars instead of brittle regex."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, None, None)
        x = self.final_norm(x)
        return x.mean(dim=1)   # [B, dim]

    def forward(self, idx, targets=None, attn_mask=None, position_ids=None,
                ce_chunk=0):
        x = self.tok_emb(idx)
        ckpt = self.act_ckpt and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, self.rope_cos, self.rope_sin, attn_mask,
                    position_ids, use_reentrant=False)
            else:
                x = block(x, self.rope_cos, self.rope_sin, attn_mask,
                          position_ids)
        x = self.final_norm(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None  # generation: last position
        if ce_chunk:
            # chunked linear+CE: the full [B*T, vocab] logits tensor (~1GiB at
            # 4x2048x64K bf16) is never materialized; each chunk's logits are
            # recomputed during backward (checkpoint). Same objective, ~2x CE
            # compute, peak memory = one chunk.
            flat_h = x.reshape(-1, x.size(-1))
            flat_t = targets.reshape(-1)
            n_valid = (flat_t != -100).sum().clamp(min=1)

            def _chunk_ce(h, t):
                lg = F.linear(h, self.lm_head.weight).float()
                return F.cross_entropy(lg, t, reduction="sum",
                                       ignore_index=-100)

            loss_sum = flat_h.new_zeros((), dtype=torch.float32)
            for i in range(0, flat_h.size(0), ce_chunk):
                loss_sum = loss_sum + torch.utils.checkpoint.checkpoint(
                    _chunk_ce, flat_h[i:i + ce_chunk], flat_t[i:i + ce_chunk],
                    use_reentrant=False)
            return None, loss_sum / n_valid
        logits = self.lm_head(x)
        # ignore_index: SFT masks prompt/pad positions with -100
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self) -> int:
        # tied lm_head counted once
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
