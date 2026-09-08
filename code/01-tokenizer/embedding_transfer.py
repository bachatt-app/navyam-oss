#!/usr/bin/env python3
"""FVT (Fast Vocabulary Transfer) embedding transfer between Navya tokenizers.

De-risks a tokenizer upgrade (e.g. v0.3-64k -> v0.5-B96k) by re-using a
checkpoint's LEARNED token embeddings instead of re-initializing them from
scratch, so the model only needs a short continued-pretrain to adapt rather
than a from-scratch run.

Method (Gee et al. 2022, "Fast Vocabulary Transfer for Language Model
Compression" -- mean-pooling variant): for every token in the NEW vocab,
decompose its surface form into a sequence of OLD-vocab tokens and set the
new embedding row to the MEAN of the corresponding old embedding rows. A
token shared verbatim between both vocabs decomposes to exactly one old
token, so it is (trivially) copied directly -- no special-casing needed.

Both Navya tokenizers are byte-level BPE over the SAME byte->unicode
alphabet (see tokenizer-v0.*.json pre_tokenizer/decoder: ByteLevel with
matching config), so a new-vocab token string can be re-tokenized directly
in that shared alphabet space using the OLD tokenizer's raw BPE merges --
`Tokenizer.model.tokenize(s)` -- WITHOUT re-running the old tokenizer's
normalizer/pre-tokenizer pipeline (which would double-apply the byte-level
mapping and corrupt multi-byte characters, e.g. Indic scripts). This is the
key correctness detail: do not call `old_tok.encode(surface_text)` here.

Usage:
  python embedding_transfer.py \
      --ckpt /tmp/navya_proxy/ckpt_last.pt \
      --old-tokenizer tokenizer-v0.3-64k.json \
      --new-tokenizer tokenizer-v0.5-B96k.json \
      --out-transfer /tmp/navya_proxy/proxy_v05_transfer_init.pt \
      --out-random   /tmp/navya_proxy/proxy_v05_random_init.pt \
      --report embedding_transfer_report_navya0_v05B96k.json

Outputs two checkpoint files in the exact `train.py --init-from` shape
(top-level "model" state_dict + "model_config" with the NEW vocab_size):
  --out-transfer: all non-embedding weights copied verbatim from --ckpt;
                  tok_emb/lm_head (tied) = the FVT-transferred table.
  --out-random:   identical non-embedding weights; tok_emb/lm_head =
                  freshly nn.init.normal_(std=0.02) (the baseline for
                  "does transfer actually help" comparison).
"""

import argparse
import json
import sys

import torch
from tokenizers import Tokenizer


def build_transferred_embedding(old_emb: torch.Tensor, old_tok: Tokenizer,
                                 new_tok: Tokenizer) -> tuple[torch.Tensor, dict]:
    """Return (new_emb [V_new, D], stats)."""
    new_vocab = new_tok.get_vocab()  # token string -> id
    v_new = new_tok.get_vocab_size()
    dim = old_emb.shape[1]
    new_emb = torch.empty(v_new, dim, dtype=old_emb.dtype)

    # id -> surface string, in id order
    id_to_str = [None] * v_new
    for s, i in new_vocab.items():
        id_to_str[i] = s

    n_verbatim = 0        # decomposed to exactly 1 old token (incl. exact match)
    n_pooled = 0           # decomposed to >1 old tokens (mean-pooled)
    pool_sizes = []
    missing = []            # decomposition produced zero ids (should not happen
                             # for byte-level BPE; old vocab covers all 256 bytes)

    for new_id, s in enumerate(id_to_str):
        toks = old_tok.model.tokenize(s)
        old_ids = [t.id for t in toks]
        if not old_ids:
            missing.append((new_id, s))
            new_emb[new_id] = torch.empty(dim).normal_(mean=0.0, std=0.02)
            continue
        if len(old_ids) == 1:
            n_verbatim += 1
        else:
            n_pooled += 1
        pool_sizes.append(len(old_ids))
        new_emb[new_id] = old_emb[old_ids].mean(dim=0)

    stats = {
        "v_old": old_emb.shape[0],
        "v_new": v_new,
        "n_verbatim": n_verbatim,
        "n_pooled": n_pooled,
        "n_missing_fallback_random": len(missing),
        "mean_old_tokens_per_new_token": sum(pool_sizes) / len(pool_sizes),
        "max_old_tokens_per_new_token": max(pool_sizes),
        "pct_verbatim": round(100 * n_verbatim / v_new, 2),
    }
    return new_emb, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="proxy checkpoint (train.py format)")
    ap.add_argument("--old-tokenizer", required=True)
    ap.add_argument("--new-tokenizer", required=True)
    ap.add_argument("--out-transfer", required=True)
    ap.add_argument("--out-random", required=True)
    ap.add_argument("--report", default=None, help="optional stats JSON path")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"loading proxy checkpoint: {args.ckpt}", file=sys.stderr)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    old_mc = dict(ckpt["model_config"])
    sd = ckpt["model"]
    old_emb = sd["tok_emb.weight"].clone()
    assert torch.equal(old_emb, sd["lm_head.weight"]), \
        "expected tied tok_emb/lm_head in the proxy checkpoint"
    dim = old_emb.shape[1]
    assert old_emb.shape[0] == old_mc["vocab_size"], \
        f"embedding rows {old_emb.shape[0]} != model_config.vocab_size {old_mc['vocab_size']}"

    old_tok = Tokenizer.from_file(args.old_tokenizer)
    new_tok = Tokenizer.from_file(args.new_tokenizer)
    assert old_tok.get_vocab_size() == old_emb.shape[0], (
        f"--old-tokenizer vocab {old_tok.get_vocab_size()} != checkpoint "
        f"embedding rows {old_emb.shape[0]} -- wrong tokenizer for this ckpt")
    v_new = new_tok.get_vocab_size()

    print(f"proxy: vocab={old_mc['vocab_size']} dim={dim} "
          f"layers={old_mc['n_layers']} -> target vocab={v_new}", file=sys.stderr)

    print("building FVT-transferred embedding ...", file=sys.stderr)
    transfer_emb, stats = build_transferred_embedding(old_emb, old_tok, new_tok)
    print(json.dumps(stats, indent=2), file=sys.stderr)

    print("building random-init embedding (baseline) ...", file=sys.stderr)
    random_emb = torch.empty(v_new, dim, dtype=old_emb.dtype).normal_(mean=0.0, std=0.02)

    new_mc = dict(old_mc)
    new_mc["vocab_size"] = v_new

    non_emb_keys = [k for k in sd.keys() if k not in ("tok_emb.weight", "lm_head.weight")]

    def make_ckpt(emb: torch.Tensor) -> dict:
        new_sd = {k: sd[k].clone() for k in non_emb_keys}
        new_sd["tok_emb.weight"] = emb.clone()
        new_sd["lm_head.weight"] = new_sd["tok_emb.weight"]  # re-tie
        return {"model": new_sd, "step": 0, "model_config": new_mc,
                "train_config": ckpt.get("train_config", {})}

    torch.save(make_ckpt(transfer_emb), args.out_transfer)
    print(f"wrote {args.out_transfer}", file=sys.stderr)
    torch.save(make_ckpt(random_emb), args.out_random)
    print(f"wrote {args.out_random}", file=sys.stderr)

    if args.report:
        full_report = {
            "source_ckpt": args.ckpt,
            "old_tokenizer": args.old_tokenizer,
            "new_tokenizer": args.new_tokenizer,
            "old_model_config": old_mc,
            "new_model_config": new_mc,
            **stats,
        }
        json.dump(full_report, open(args.report, "w"), indent=2)
        print(f"wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
