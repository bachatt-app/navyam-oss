"""M1 gate metric: transcription CER of a navya-1b-av checkpoint (doc 14 §3).

For each held-out ASR example, the model is prompted with
<|bos|><|aud_start|><audio embeds><|aud_end|> and greedy-decodes text until
<|eos|> or --max-new. CER = levenshtein(hyp, ref) / len(ref) over characters,
reported per language and overall. The M1 gate is not an absolute WER target —
it is "clearly improving with data and far better than chance"; the
Whisper-small comparison comes at scale-up time.

Generation re-runs the full prefix each step (no KV cache) — fine for eval on
short clips, not a serving path.

  ../.venv/bin/python eval_asr_cer.py --ckpt out/navya-1b-av/ckpt_av.pt \
      --shards data/asr_v0_val --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json
"""
import argparse
import collections
import json

import torch

from model import ModelConfig
from multimodal import (AUD_END, AUD_START, AUDIO_PAD, MultimodalGPT,
                        splice_multimodal)
from multimodal_data import ASRShardDataset

BOS, EOS = 0, 1
MODEL_KEYS = {f.name for f in ModelConfig.__dataclass_fields__.values()}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def transcribe(mm, tok, wave, n_aud, device, max_new=200):
    emb = mm.audio_proj(wave.unsqueeze(0).to(device))[:, :n_aud]
    ids = [BOS, AUD_START] + [AUDIO_PAD] * n_aud + [AUD_END]
    out = []
    for _ in range(max_new):
        idx = torch.tensor([ids + out], dtype=torch.long, device=device)
        x = splice_multimodal(mm.gpt.tok_emb, idx, audio_embeds=emb)
        logits, _ = mm.gpt(idx, inputs_embeds=x)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        out.append(nxt)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ModelConfig(**{k: v for k, v in ck["config"].items()
                          if k in MODEL_KEYS})
    mm = MultimodalGPT(mcfg, audio=True).to(device).eval()
    mm.load_state_dict(ck["model"])

    ds = ASRShardDataset(args.shards)
    dist, chars = collections.Counter(), collections.Counter()
    for i in range(min(args.limit, len(ds))):
        item = ds[i]
        arr, ref_row = ds.examples[i]
        toks = item["tokens"].tolist()
        ref = tok.decode(
            [t for t in toks[toks.index(AUD_END) + 1:] if t != EOS])
        hyp = transcribe(mm, tok, item["wave"], item["n_aud"], device,
                         args.max_new)
        lang = ref_row.get("lang", "und")
        dist[lang] += levenshtein(hyp, ref)
        chars[lang] += max(len(ref), 1)
        if i < 3:
            print(f"[{lang}] ref: {ref[:80]}\n      hyp: {hyp[:80]}")
    total_d, total_c = sum(dist.values()), sum(chars.values())
    for lang in sorted(chars):
        print(f"CER[{lang}] = {dist[lang] / chars[lang]:.3f} "
              f"({chars[lang]} ref chars)")
    print(f"CER[all] = {total_d / max(total_c, 1):.3f}")
    print(json.dumps({"cer": total_d / max(total_c, 1),
                      "per_lang": {k: dist[k] / chars[k] for k in chars}}))


if __name__ == "__main__":
    main()
