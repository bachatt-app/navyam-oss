#!/usr/bin/env python3
"""Render text into images -> self-labeled OCR pairs for navya-1b-av
(docs/14_navya_1b_av_plan.md §4: free perfect labels, all 13 languages).

Takes text we already own (jsonl with a text field, or plain .txt lines),
renders short passages onto document-like images with mild real-world
degradations (off-white background, sensor noise, slight rotation, blur),
and writes the multimodal shard format next to a directory of PNGs:

    <|bos|> <|img_start|> <|image_pad|> x 196 <|img_end|> text <|eos|>

196 = SigLIP-so400m 384px -> 729 patches -> 2x2 merge (multimodal.py
VisionProjector.n_tokens()). The refs sidecar points each example at its
PNG; SigLIP feature extraction happens later/offline (the A10 loop trains
the projector on cached features, never running the encoder per step).

Indic scripts need real fonts: pass --fonts with .ttf/.ttc paths (macOS
ships e.g. /System/Library/Fonts/Supplemental/Kohinoor*.ttc, ITF Devanagari;
Linux: lohit/noto). Without --fonts a PIL default is used (Latin only).

    python3 render_ocr_pairs.py --text passages.jsonl --out ~/ocr_pairs \
        --name hi_batch0 --n 500 --seed 0 \
        --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json \
        --fonts /System/Library/Fonts/Supplemental/Kohinoor.ttc
"""
import argparse
import json
import os
import random
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tokenizers import Tokenizer

BOS, EOS = 0, 1
IMG_START, IMG_END, IMAGE_PAD = 8, 9, 10
N_IMG_TOKENS = 196            # SigLIP 729 patches -> 2x2 merge
CANVAS = 384                  # SigLIP-so400m input size


def iter_texts(path):
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            t = json.loads(line).get("text", "").strip()
            if t:
                yield t
    else:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                yield line.strip()


def load_fonts(paths, rng):
    fonts = []
    for p in paths:
        for size in (18, 22, 26, 30):
            try:
                fonts.append(ImageFont.truetype(p, size))
            except OSError:
                print(f"warn: cannot load font {p} @ {size}")
    return fonts or [ImageFont.load_default()]


def render(text, font, rng):
    """One passage -> degraded 384x384 document-style image + exact text."""
    # wrap to fit; keep only the lines that fit the canvas, return what's shown
    wrap_w = rng.randint(26, 40)
    lines = textwrap.wrap(text, width=wrap_w)
    bg = rng.randint(232, 255)
    img = Image.new("L", (CANVAS, CANVAS), color=bg)
    draw = ImageDraw.Draw(img)
    x, y = rng.randint(8, 24), rng.randint(8, 24)
    line_h = int(getattr(font, "size", 12) * 1.45)
    shown = []
    for ln in lines:
        if y + line_h > CANVAS - 8:
            break
        draw.text((x, y), ln, fill=rng.randint(0, 60), font=font)
        shown.append(ln)
        y += line_h
    if not shown:
        return None, None
    if rng.random() < 0.5:                          # slight scan skew
        img = img.rotate(rng.uniform(-2.0, 2.0), expand=False,
                         fillcolor=bg, resample=Image.BILINEAR)
    if rng.random() < 0.4:                          # focus softness
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    arr = np.asarray(img, dtype=np.int16)           # sensor/print noise
    noise = rng.randint(2, 10)
    arr = arr + np.random.default_rng(rng.getrandbits(32)).integers(
        -noise, noise + 1, arr.shape)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGB")
    return img, " ".join(shown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="jsonl (text field) or txt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--fonts", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=600,
                    help="clip passages before wrapping")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    fonts = load_fonts(args.fonts, rng)

    img_dir = os.path.join(args.out, "ocr", "imgs", args.name)
    shard_dir = os.path.join(args.out, "ocr", "shards")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    stream, refs, tok_off = [], [], 0
    for i, text in enumerate(iter_texts(args.text)):
        if len(refs) >= args.n:
            break
        img, shown = render(text[:args.max_chars], rng.choice(fonts), rng)
        if img is None:
            continue
        fname = f"{args.name}_{len(refs):06d}.png"
        img.save(os.path.join(img_dir, fname))
        ids = ([BOS, IMG_START] + [IMAGE_PAD] * N_IMG_TOKENS + [IMG_END]
               + tok.encode(shown).ids + [EOS])
        stream.append(np.asarray(ids, dtype=np.uint16))
        refs.append({"image": os.path.join(img_dir, fname),
                     "n_img": N_IMG_TOKENS, "tok_off": tok_off,
                     "tok_len": len(ids), "text_chars": len(shown)})
        tok_off += len(ids)

    if not refs:
        raise SystemExit("no examples rendered")
    arr = np.concatenate(stream)
    arr.tofile(os.path.join(shard_dir, f"{args.name}.bin"))
    with open(os.path.join(shard_dir, f"{args.name}.refs.jsonl"), "w",
              encoding="utf-8") as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(refs)} OCR pairs, {arr.size} tokens -> "
          f"{shard_dir}/{args.name}.bin (+ imgs in {img_dir})")


if __name__ == "__main__":
    main()
