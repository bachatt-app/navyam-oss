#!/usr/bin/env python3
"""traces.jsonl (R1-14b reasoning + answer pairs) -> conversation jsonl for SFT
packing, distilling reasoning into navya-1c-sft-dpo-grpo.

Assistant target = "<think>{reasoning}</think>\n\n{canonical}" so the trained
model learns to emit a rationale before the source-corpus canonical answer,
matching the same <think>...</think> convention the R1-family teacher uses.

Steps:
  1. Load + dedup traces (by user-question text) across one or more --traces
     files — a re-run of build_reasoning_traces.py with the same seed/--n
     regenerates (a prefix of) the same prompt set, and traces are also
     merged in from the recovery copy, so duplicates are expected.
  2. Build one conversation per trace: user question -> assistant
     "<think>reasoning</think>\n\nanswer".
  3. Decontaminate against BachattBench (all sections + tier1 eval texts)
     using the SAME hash n-gram method build_chat_data.py applies to every
     other SFT source, via code/02-data-pipeline/pipeline/decontaminate.py.
     (build_chat_data.py decontaminates again at pack time — this pass is a
     belt-and-braces check specific to reasoning traces, and reports the
     drop count before any packing happens.)
  4. Write conversations/*.jsonl in the shape build_chat_data.py expects:
     {id, messages, lang, topic, source, reviewer, safety, licence}.

Usage:
  python format_reasoning_sft.py \
      --traces traces.jsonl /Users/.../navyam-recover/traces.jsonl \
      --out reasoning_conversations/reasoning_v1.jsonl --min-n 300
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "02-data-pipeline", "pipeline"))
sys.path.insert(0, os.path.join(CODE, "08-trl"))
import decontaminate as qdecon  # noqa: E402
import rewards  # noqa: E402  (language heuristic, reused for the `lang` field)

SHORT_NGRAM = 7  # matches build_chat_data.py's short-ngram eval-overlap check
TOPIC = "reasoning-distill"
SOURCE = "S066-r1-14b-reasoning-distillation"
REVIEWER = "unreviewed-teacher-distillation"
SAFETY = "teacher-distilled-unreviewed"
LICENCE = "first-party-project-content"


def load_traces(paths):
    """Load one or more traces.jsonl files, de-duplicating by user-question
    text (later files win — pass the freshest/most-complete file last)."""
    by_prompt = {}
    total = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"  (skip, not found: {path})")
            continue
        n_here = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = rec.get("prompt")
            reasoning = (rec.get("reasoning") or "").strip()
            answer = (rec.get("answer") or "").strip()
            if not prompt or not reasoning or not answer:
                continue
            instr = (prompt[0].get("content") or "").strip() \
                if isinstance(prompt, list) and prompt else ""
            if not instr:
                continue
            by_prompt[instr] = rec  # last file wins on duplicate prompts
            n_here += 1
        print(f"  {path}: {n_here} usable records")
    return list(by_prompt.values()), total


def make_conversation(rec):
    instr = rec["prompt"][0]["content"].strip()
    reasoning = rec["reasoning"].strip()
    # The local 14B teacher produced a number of fluent but incorrect or
    # garbled final answers. The source corpus already carries a canonical
    # response; use that as the supervised final answer. Both the pairing and
    # teacher reasoning remain explicitly unreviewed until a human approves it.
    answer = (rec.get("canonical") or "").strip()
    if not answer:
        raise ValueError("reasoning trace is missing its canonical answer")
    assistant = f"<think>{reasoning}</think>\n\n{answer}"
    cid = "reasoning-" + hashlib.sha1(instr.encode("utf-8")).hexdigest()[:12]
    lang = rewards._lang(instr)
    return {
        "id": cid,
        "topic": TOPIC,
        "lang": lang,
        "messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": assistant},
        ],
        "source": SOURCE,
        "reviewer": REVIEWER,
        "safety": SAFETY,
        "licence": LICENCE,
    }


def decontaminate(conversations):
    """Drop any conversation overlapping BachattBench (all sections + tier1),
    identical method to build_chat_data.py: hash 13-gram + short 7-gram."""
    eval_paths = sorted(glob.glob(os.path.join(
        CODE, "03-evals", "bachattbench", "*.jsonl")))
    tier1 = os.path.join(CODE, "03-evals", "decontam", "tier1_eval_texts.jsonl")
    if os.path.exists(tier1):
        eval_paths.append(tier1)
    banned13, banned7 = set(), set()
    for path in eval_paths:
        for text in qdecon.eval_texts(path):
            banned13.update(qdecon.ngrams(text))
            banned7.update(qdecon.ngrams(text, SHORT_NGRAM))
    kept, dropped = [], []
    for c in conversations:
        text = " ".join(m["content"] for m in c["messages"])
        if any(g in banned13 for g in qdecon.ngrams(text)) \
                or any(g in banned7 for g in qdecon.ngrams(text, SHORT_NGRAM)):
            dropped.append(c["id"])
        else:
            kept.append(c)
    return kept, dropped, eval_paths, len(banned13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True,
                    help="one or more traces.jsonl files to merge + dedup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=0,
                    help="exit non-zero if fewer than this many conversations "
                         "survive dedup + decontamination")
    args = ap.parse_args()

    print(f"loading {len(args.traces)} trace file(s)...")
    records, total_lines = load_traces(args.traces)
    print(f"total lines read={total_lines}; unique prompts={len(records)}")

    conversations = [make_conversation(r) for r in records]

    # build_chat_data.py enforces a link-free authored corpus; the teacher
    # occasionally hallucinates a URL (e.g. a bogus govt-portal link) in its
    # final answer, which would otherwise abort packing downstream. Drop
    # those here so the failure is visible and attributed to this stage.
    no_url, url_dropped = [], []
    for c in conversations:
        text = " ".join(m["content"] for m in c["messages"])
        if "http" in text or "www." in text:
            url_dropped.append(c["id"])
        else:
            no_url.append(c)
    if url_dropped:
        print(f"dropped {len(url_dropped)} conversation(s) with a hallucinated "
              f"URL: {url_dropped}")
    conversations = no_url

    kept, dropped, eval_paths, banned = decontaminate(conversations)
    print(f"decontamination vs {len(eval_paths)} eval file(s) "
          f"({banned:,} banned 13-grams): kept={len(kept)} dropped={len(dropped)}")
    if dropped:
        preview = dropped[:20]
        print("  dropped ids:", preview, "..." if len(dropped) > 20 else "")

    lang_counts = {}
    for c in kept:
        lang_counts[c["lang"]] = lang_counts.get(c["lang"], 0) + 1
    print("lang distribution:", lang_counts)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} conversations -> {args.out}")

    if kept:
        s = kept[0]
        print("\n--- sample conversation ---")
        print("id:", s["id"], "| lang:", s["lang"], "| topic:", s["topic"])
        print("USER:", s["messages"][0]["content"][:200])
        print("ASSISTANT:", s["messages"][1]["content"][:400])

    if args.min_n and len(kept) < args.min_n:
        sys.exit(f"FATAL: only {len(kept)} conversations survived; "
                 f"need >= {args.min_n}")


if __name__ == "__main__":
    main()
