#!/usr/bin/env python3
"""Generate DeepSeek-R1 reasoning traces on India-finance prompts (teacher step
of reasoning distillation). CPU/API only — no GPU.

For each India-finance instruction we ask R1 to think step-by-step and answer as
Navya (India personal-finance assistant): reply in the question's language, never
name a specific fund/AMC, keep the final answer concise. We keep R1's chain of
thought AND its final answer; a later SFT pass distills {prompt -> <think>reason
</think> answer} into navya-1c-sft-dpo-grpo.

Teacher access is provider-agnostic (whichever key is present):
  * DeepSeek official  — DEEPSEEK_API_KEY,  https://api.deepseek.com,  model
    "deepseek-reasoner" (returns message.reasoning_content + message.content).
  * OpenRouter         — OPENROUTER_API_KEY, https://openrouter.ai/api/v1, model
    "deepseek/deepseek-r1" (reasoning in message.reasoning or <think> in content).
Key is read from env or the vault file passed via --key-file (KEY=VALUE lines).

Prompts come from the SFT corpus (corpus_v2), NEVER from BachattBench — training
on eval prompts would contaminate the benchmark.

  # validate on a handful first (cheap):
  python build_reasoning_traces.py --limit 5 --out traces.smoke.jsonl
  # then the full set:
  python build_reasoning_traces.py --n 2000 --out traces.jsonl --workers 6
"""
import argparse, glob, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "08-trl"))
import rewards  # noqa: E402  (shared reward checks for filtering)

CORPUS_DIRS = [
    os.path.join(HERE, "..", "07-sft", "corpus_v2"),
    os.path.join(HERE, "..", "07-sft", "training_data", "corpus_v2"),
    os.path.join(HERE, "..", "07-sft", "training_data"),
]
UNREVIEWED_GENERATED_SUFFIXES = ("_paraphrase.jsonl",)
UNREVIEWED_GENERATED_FILES = {"thin_intent_aug.jsonl"}

SYSTEM = (
    "You are Navya, an India-first personal-finance assistant. Think step by step "
    "about the user's question, then give a clear, correct answer grounded in the "
    "Indian context (INR, Indian tax/regulations, SIP/PPF/EPF/NPS, CIBIL, etc.). "
    "Rules for your FINAL answer: reply in the same language as the question "
    "(English, Hindi, or Hinglish); never name a specific mutual fund, AMC, stock, "
    "or product to buy; be concise (roughly 40-120 words); this is education, not "
    "individual investment advice."
)

PROVIDERS = {
    "deepseek": {"env": "DEEPSEEK_API_KEY",
                 "url": "https://api.deepseek.com/chat/completions",
                 "model": "deepseek-reasoner"},
    "openrouter": {"env": "OPENROUTER_API_KEY",
                   "url": "https://openrouter.ai/api/v1/chat/completions",
                   "model": "deepseek/deepseek-r1"},
    # Local Ollama (OpenAI-compatible). No key, no cost; teacher is an R1-Distill
    # (Qwen/Llama fine-tuned on R1) — not the 671B R1, but a real reasoner.
    "local": {"env": None,
              "url": "http://localhost:11434/v1/chat/completions",
              "model": "deepseek-r1:14b"},
}


def resolve_provider(key_file, forced, model_override):
    """Return (provider, api_key, model). Forced wins; else a cloud key if present
    (env or the KEY=VALUE vault file); else local Ollama (no key)."""
    env = {}
    if key_file and os.path.exists(key_file):
        for line in open(key_file, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()

    def finish(prov, key):
        model = model_override or PROVIDERS[prov]["model"]
        return prov, key, model

    if forced and forced != "auto":
        if forced == "local":
            return finish("local", "ollama-local")
        val = os.environ.get(PROVIDERS[forced]["env"]) or env.get(PROVIDERS[forced]["env"])
        return finish(forced, val)          # key may be None -> caller aborts
    for prov in ("deepseek", "openrouter"):
        val = os.environ.get(PROVIDERS[prov]["env"]) or env.get(PROVIDERS[prov]["env"])
        if val:
            return finish(prov, val)
    return finish("local", "ollama-local")  # default: free local teacher


def load_prompts(n, seed=1337):
    files, rows, seen = [], [], set()
    for d in CORPUS_DIRS:
        if os.path.isdir(d):
            for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
                name = os.path.basename(path)
                if (name in UNREVIEWED_GENERATED_FILES
                        or name.endswith(UNREVIEWED_GENERATED_SUFFIXES)):
                    continue
                files.append(path)
    for path in files:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            instr = (r.get("instruction") or "").strip()
            resp = (r.get("response") or "").strip()
            if not instr or instr in seen or len(instr) < 8:
                continue
            seen.add(instr)
            rows.append({"instruction": instr, "canonical": resp})
    # deterministic shuffle so a --limit smoke set is a representative slice
    import random
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def call_r1(url, model, key, instr, timeout, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": instr}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    # OpenRouter sometimes inlines <think>...</think> in content
    if not reasoning and "<think>" in content and "</think>" in content:
        pre, _, rest = content.partition("<think>")
        reasoning, _, post = rest.partition("</think>")
        content = (pre + post).strip()
        reasoning = reasoning.strip()
    return reasoning, content


def acceptable(instr, answer):
    """Filter with the same reward rules used across post-training."""
    msgs = [{"role": "user", "content": instr}]
    if len(answer.split()) < 5:
        return False, "too short"
    # The prompt asks for the same language/script. Previously a partial 0.5
    # score let Hinglish prompts paired with garbled Devanagari answers through.
    if rewards.reward_language_match(msgs, answer) < 1.0:
        return False, "language mismatch"
    if rewards.reward_no_fund_naming(msgs, answer) < 0:
        return False, "named a fund"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000, help="target prompts")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap prompts")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "deepseek", "openrouter", "local"])
    ap.add_argument("--model", default="", help="override the provider's model")
    ap.add_argument("--key-file", default=os.path.join(
        os.path.expanduser("~"),
        "Bachatt/devops/azure/navyam-gpt-secrets/deepseek.txt"))
    args = ap.parse_args()

    prov, key, model = resolve_provider(args.key_file, args.provider,
                                        args.model or "")
    url = PROVIDERS[prov]["url"]
    if prov != "local" and not key:
        sys.exit(f"NO API KEY for provider '{prov}': set its key in env or "
                 f"{args.key_file} (KEY=VALUE). Aborting before any spend.")
    if prov == "local":
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        except Exception:
            sys.exit("LOCAL provider chosen but Ollama isn't answering on "
                     "localhost:11434 — start Ollama / pull the model first.")
    print(f"provider={prov} model={model} url={url}")
    call = lambda instr: call_r1(url, model, key, instr, args.timeout, args.max_tokens)  # noqa: E731

    n = args.limit or args.n
    prompts = load_prompts(n)
    print(f"prompts={len(prompts)} (from corpus_v2; BachattBench excluded)")

    def work(row):
        instr = row["instruction"]
        for attempt in range(args.retries + 1):
            try:
                reasoning, answer = call(instr)
                if not reasoning or not answer:
                    return None, "empty reasoning/answer"
                ok, why = acceptable(instr, answer)
                if not ok:
                    return None, why
                return {"prompt": [{"role": "user", "content": instr}],
                        "reasoning": reasoning, "answer": answer,
                        "canonical": row["canonical"]}, "ok"
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < args.retries:
                    time.sleep(2 * (attempt + 1)); continue
                return None, f"http {e.code}"
            except Exception as e:  # noqa: BLE001
                if attempt < args.retries:
                    time.sleep(2 * (attempt + 1)); continue
                return None, f"err {type(e).__name__}"
        return None, "exhausted"

    kept = 0
    reasons = {}
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in prompts}
        for i, fut in enumerate(as_completed(futs), 1):
            rec, why = fut.result()
            reasons[why] = reasons.get(why, 0) + 1
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                kept += 1
            if i % 25 == 0 or args.limit:
                print(f"  {i}/{len(prompts)} done | kept {kept} | "
                      f"{i/max(time.time()-t0,1e-9):.2f} req/s | last: {why}",
                      flush=True)
    print(f"DONE kept={kept}/{len(prompts)} -> {args.out}")
    print("reasons:", json.dumps(reasons, ensure_ascii=False))
    if args.limit and kept:
        ex_rec = json.loads(open(args.out, encoding="utf-8").readline())
        print("\n--- sample trace ---")
        print("Q:", ex_rec["prompt"][0]["content"][:120])
        print("REASONING (first 300):", ex_rec["reasoning"][:300])
        print("ANSWER:", ex_rec["answer"][:300])


if __name__ == "__main__":
    main()
