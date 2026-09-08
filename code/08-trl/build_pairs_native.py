#!/usr/bin/env python3
"""Build native DPO preference pairs for Navya — no trl/datasets.

Emits jsonl of {"prompt": [messages], "chosen": str, "rejected": str} that
dpo_native.py consumes directly.

Two negative sources, combined:

  mined  (default, FREE, no GPU) — the rejected answer is another canonical
         item's response. It is fluent, correct-looking prose that is simply
         the WRONG answer for this prompt: exactly the "topic-blending" failure
         the SFT 1.31B still shows, and rewards.reward_canonical_overlap scores
         it far below the true answer. We keep a pair only when the reward
         margin (chosen - rejected) clears --margin, so every negative is a
         genuine downgrade under the same reward the model is judged by.
         A slice of these are LANGUAGE-mismatch negatives (Hinglish/Hindi prompt
         paired with an English answer to the same topic, or vice versa) — the
         '₹25k SIP karoon ya loan prepay?' answered-in-English failure.

  gen    (--gen K, uses the GPU) — for K sampled prompts, generate candidates
         from the live SFT model and take the lowest-reward sample as the
         rejected. This captures real degeneration (repetition, truncation)
         that mining can't synthesise. Bounded by K because KV-cache-free
         generation of the 1.31B is the expensive part; default 0 keeps the
         A100 doing only DPO training.

  python build_pairs_native.py --out dpo_pairs.jsonl --n 2000 \
      [--gen 200 --ckpt <sft ckpt> --tokenizer <tok.json>]
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rewards  # noqa: E402

CORPUS_DIRS = [
    os.path.join(HERE, "..", "07-sft", "training_data", "corpus_v2"),
    os.path.join(HERE, "..", "07-sft", "training_data"),
]


def load_corpus():
    """All canonical {instruction, response, lang, topic} rows, deduped by
    (instruction, response)."""
    rows, seen = [], set()
    files = []
    for d in CORPUS_DIRS:
        files += sorted(glob.glob(os.path.join(d, "*.jsonl")))
    for path in files:
        try:
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ins, resp = r.get("instruction"), r.get("response")
                if not isinstance(ins, str) or not isinstance(resp, str):
                    continue
                ins, resp = ins.strip(), resp.strip()
                if len(ins) < 4 or len(resp) < 20:
                    continue
                key = (ins, resp[:80])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"instruction": ins, "response": resp,
                             "lang": r.get("lang", ""),
                             "topic": r.get("topic", os.path.basename(path))})
        except Exception as e:  # noqa: BLE001
            print(f"skip {path}: {e}")
    return rows


def rand_seq(seed, n):
    """Deterministic LCG permutation-ish index stream (Math.random is banned in
    the workflow runtime; here we just want reproducibility without numpy)."""
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        yield x % n


def lang_of(row):
    return rewards._lang(row["instruction"])


def mine_pairs(corpus, n, margin, seed=1337):
    """Free hard-negative mining. For each chosen item, try a mismatched
    canonical response (prefer different topic; a fraction language-mismatched)
    and keep it only if the reward margin clears `margin`."""
    pairs = []
    idx = rand_seq(seed, len(corpus))
    order = list(range(len(corpus)))
    # simple deterministic shuffle
    for i in range(len(order) - 1, 0, -1):
        j = next(idx) % (i + 1)
        order[i], order[j] = order[j], order[i]
    lang_target = int(n * 0.30)   # aim ~30% language-mismatch negatives
    lang_got = 0
    for oi in order:
        if len(pairs) >= n:
            break
        chosen = corpus[oi]
        msgs = [{"role": "user", "content": chosen["instruction"]}]
        want_lang = len(pairs) - lang_got < lang_target and lang_of(chosen) in (
            "hinglish", "hi", "en")
        # find a rejected candidate
        rej = None
        for _ in range(24):
            cand = corpus[next(idx)]
            if cand is chosen or cand["response"] == chosen["response"]:
                continue
            same_topic = cand["topic"] == chosen["topic"]
            cand_lang = rewards._lang(cand["response"])
            if want_lang:
                # cross-language, ideally same topic so only language differs
                if cand_lang == lang_of(chosen):
                    continue
                if same_topic:
                    rej = cand
                    break
                rej = rej or cand
            else:
                if same_topic:      # different item, same topic = subtle blend
                    rej = cand
                    break
                rej = rej or cand
        if rej is None:
            continue
        r_ch = rewards.total_reward(msgs, chosen["response"], chosen["response"])
        r_rj = rewards.total_reward(msgs, rej["response"], chosen["response"])
        if r_ch - r_rj < margin:
            continue
        if want_lang and rewards._lang(rej["response"]) != lang_of(chosen):
            lang_got += 1
        pairs.append({"prompt": msgs, "chosen": chosen["response"],
                      "rejected": rej["response"],
                      "_src": "mine", "_margin": round(r_ch - r_rj, 3)})
    return pairs


def gen_pairs(corpus, k, ckpt, tokenizer, samples, margin, seed=1337):
    """Model-sampled negatives (GPU). Lowest-reward sample below the canonical
    by `margin` becomes the rejected."""
    sys.path.insert(0, os.path.join(HERE, "..", "06-inference"))
    import serve  # noqa: E402
    import chat_format  # noqa: E402
    device = serve.pick_device()
    model, cfg, tok, eos, _ = serve.load(ckpt, tokenizer, device)
    special = chat_format.token_ids(tok)
    stop_ids = (special["end_turn"], special["eos"])
    idx = rand_seq(seed + 7, len(corpus))
    pairs = []
    picked = set()
    tries = 0
    while len(pairs) < k and tries < k * 4:
        tries += 1
        oi = next(idx)
        if oi in picked:
            continue
        picked.add(oi)
        chosen = corpus[oi]
        msgs = [{"role": "user", "content": chosen["instruction"]}]
        prompt_ids = chat_format.encode_prompt(tok, msgs)
        worst, worst_r = None, 1e9
        for s in range(samples):
            text, *_ = serve.generate(
                model, cfg, tok, eos, device, prompt_ids, max_new_tokens=200,
                temperature=0.85, top_k=50, top_p=0.95,
                repetition_penalty=1.1, stop_ids=stop_ids)
            text = text.strip()
            if len(text) < 8:
                continue
            r = rewards.total_reward(msgs, text, chosen["response"])
            if r < worst_r:
                worst, worst_r = text, r
        if worst is None:
            continue
        r_ch = rewards.total_reward(msgs, chosen["response"], chosen["response"])
        if r_ch - worst_r < margin:
            continue
        pairs.append({"prompt": msgs, "chosen": chosen["response"],
                      "rejected": worst, "_src": "gen",
                      "_margin": round(r_ch - worst_r, 3)})
        if len(pairs) % 20 == 0:
            print(f"gen pairs: {len(pairs)}/{k}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000, help="mined pairs target")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--gen", type=int, default=0, help="model-sampled pairs (GPU)")
    ap.add_argument("--samples", type=int, default=4, help="candidates per gen prompt")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    corpus = load_corpus()
    print(f"corpus: {len(corpus)} canonical items")
    if len(corpus) < 100:
        sys.exit("corpus too small — check CORPUS_DIRS")

    pairs = mine_pairs(corpus, args.n, args.margin, args.seed)
    print(f"mined pairs: {len(pairs)} (margin>={args.margin})")

    if args.gen > 0:
        if not args.ckpt or not args.tokenizer:
            sys.exit("--gen requires --ckpt and --tokenizer")
        gp = gen_pairs(corpus, args.gen, args.ckpt, args.tokenizer,
                       args.samples, args.margin, args.seed)
        print(f"generated pairs: {len(gp)}")
        pairs += gp

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    langs = {}
    for p in pairs:
        langs[p["_src"]] = langs.get(p["_src"], 0) + 1
    print(f"wrote {len(pairs)} pairs -> {args.out}  {langs}")


if __name__ == "__main__":
    main()
