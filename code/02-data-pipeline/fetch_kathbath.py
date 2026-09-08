#!/usr/bin/env python3
"""Fetch a Kathbath subset -> navya-1b-av transcript+audio layout (doc 14 M1).

Kathbath (AI4Bharat): 1,684 h human-labelled read speech across 12 Indian
languages — the human-labelled anchor of the M1 audio mix (>=20% floor
against Whisper pseudo-label noise). The HF dataset `ai4bharat/Kathbath`
is GATED: accept the terms at
https://huggingface.co/datasets/ai4bharat/Kathbath once (account is already
logged in via huggingface_hub), or this script exits with the gate error.

Each utterance is one clip; we archive it per the fleet contract (Opus
24 kHz mono 32 kbps) and emit one transcript row with a single full-clip
segment, ready for build_asr_shards.py:

    {id, lang, audio_path, dur_s, segments: [{start: 0, end: dur, text}]}

    python3 fetch_kathbath.py --langs hindi tamil --split valid \
        --per-lang 2000 --out ~/kathbath
    # then:
    python3 build_asr_shards.py --transcripts ~/kathbath/transcripts.jsonl \
        --audio-field audio_path --out ~/shards_local --name kathbath_v0 \
        --tokenizer ../01-tokenizer/tokenizer-v0.3-64k.json
"""
import argparse
import json
import os
import subprocess

import numpy as np

LANG_ISO = {"hindi": "hi", "tamil": "ta", "telugu": "te", "marathi": "mr",
            "bengali": "bn", "gujarati": "gu", "kannada": "kn",
            "malayalam": "ml", "odia": "or", "punjabi": "pa",
            "sanskrit": "sa", "urdu": "ur"}


def to_opus(arr, sr, path):
    """float waveform -> Opus 24 kHz mono 32 kbps (the archival contract)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
         "-ac", "1", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k",
         "-ar", "24000", "-ac", "1", path],
        input=np.asarray(arr, dtype=np.float32).tobytes(), check=True)


def load_split(repo, lang, split):
    from datasets import load_dataset
    for kwargs in ({"name": lang}, {"data_dir": lang}, {}):
        try:
            return load_dataset(repo, split=split, streaming=True, **kwargs)
        except Exception as e:  # noqa: BLE001 — config layout probing
            last = e
    raise SystemExit(f"cannot load {repo} [{lang}/{split}]: {last}\n"
                     "If this is a gated-access error, accept the terms at "
                     f"https://huggingface.co/datasets/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ai4bharat/Kathbath")
    ap.add_argument("--langs", nargs="+", default=["hindi", "tamil"])
    ap.add_argument("--split", default="valid",
                    help="valid is small — right for the M1 pilot; train for scale")
    ap.add_argument("--per-lang", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr_path = os.path.join(args.out, "transcripts.jsonl")
    os.makedirs(args.out, exist_ok=True)
    n_total = 0
    with open(tr_path, "a", encoding="utf-8") as fout:
        for lang in args.langs:
            iso = LANG_ISO.get(lang, lang)
            audio_dir = os.path.join(args.out, "audio", iso)
            os.makedirs(audio_dir, exist_ok=True)
            ds = load_split(args.repo, lang, args.split)
            n = 0
            for row in ds:
                if n >= args.per_lang:
                    break
                text = (row.get("text") or "").strip()
                audio = row.get("audio") or row.get("audio_filepath")
                if not text or audio is None:
                    continue
                arr, sr = audio["array"], audio["sampling_rate"]
                dur = round(len(arr) / sr, 2)
                if dur < 0.5:
                    continue
                uid = f"kb_{iso}_{n:06d}"
                opus = os.path.join(audio_dir, f"{uid}.opus")
                to_opus(arr, sr, opus)
                fout.write(json.dumps(
                    {"id": uid, "lang": iso, "audio_path": opus,
                     "dur_s": dur, "source": "kathbath",
                     "segments": [{"start": 0.0, "end": dur, "text": text}]},
                    ensure_ascii=False) + "\n")
                n += 1
                if n % 200 == 0:
                    print(f"  {lang}: {n}")
            print(f"{lang}: {n} utterances -> {audio_dir}")
            n_total += n
    print(f"done: {n_total} utterances, transcripts -> {tr_path}")


if __name__ == "__main__":
    main()
