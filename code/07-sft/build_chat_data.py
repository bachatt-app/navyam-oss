#!/usr/bin/env python3
"""Build SFT-v2 chat rows with response masks and example boundaries.

Native input is conversation JSONL with ``messages`` plus per-example source,
language, topic, reviewer, safety, and licence metadata. The existing authored
single-turn corpus is adapted automatically. Whole conversations are greedily
packed into fixed-length rows; ``*_ex.bin`` preserves their boundaries so the
trainer can build block-diagonal causal attention and reset positions.

Outputs:
  train_x.bin / val_x.bin   token ids, uint16, shape [rows, seq_len]
  train_m.bin / val_m.bin   assistant loss mask, uint8, same shape
  train_ex.bin / val_ex.bin packed-example ids, uint16, same shape
  manifest.jsonl            split, placement, and provenance per conversation
  meta.json                 format and count contract

Usage:
  python build_chat_data.py --out ../04-training-stack/data/navya-1a-sft-v2 \
      --seq-len 2048
"""

import argparse
import hashlib
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "02-data-pipeline", "pipeline"))
sys.path.insert(0, HERE)
import decontaminate as qdecon  # noqa: E402
from chat_format import (encode_conversation, template_metadata,
                         validate_messages)  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

PAD_EX = 65535
SEED = 20260818
REQUIRED_METADATA = ("source", "lang", "topic", "reviewer", "safety",
                     "licence")
UNREVIEWED_GENERATED_SUFFIXES = ("_paraphrase.jsonl",)
UNREVIEWED_GENERATED_FILES = {"thin_intent_aug.jsonl"}


def is_unreviewed_generated(path):
    """Whether ``path`` was produced by regex intent-to-answer mapping.

    These files contain authored questions and answers, but the *pairing* was
    automatic (first matching regex wins) and has produced known semantic
    mismatches. They are useful for experiments, not safe default SFT input.
    """
    name = os.path.basename(path)
    return (name in UNREVIEWED_GENERATED_FILES
            or name.endswith(UNREVIEWED_GENERATED_SUFFIXES))


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_conversations(include_legacy=True, conversations_dir=None,
                       include_unreviewed_generated=False):
    conversations = []
    conversations_dir = conversations_dir or os.path.join(HERE, "conversations")
    for path in sorted(glob.glob(os.path.join(conversations_dir, "*.jsonl"))):
        for item in _read_jsonl(path):
            missing = [key for key in ("id", "messages", "lang", "topic")
                       if key not in item]
            if missing:
                raise ValueError(f"{path}: {item.get('id', '?')} missing {missing}")
            # Early seed files predate the provenance schema. Preserve that
            # fact explicitly rather than inventing a reviewer or safety signoff.
            item.setdefault("source", f"first-party:{os.path.basename(path)}")
            item.setdefault("reviewer", "review-not-recorded")
            item.setdefault("safety", "review-not-recorded")
            item.setdefault("licence", "first-party-project-content")
            conversations.append(item)

    if include_legacy:
        paths = sorted(glob.glob(os.path.join(HERE, "corpus_v2", "*.jsonl")))
        paths.append(os.path.join(HERE, "seed_finance_instructions.jsonl"))
        for path in paths:
            generated = is_unreviewed_generated(path)
            if generated and not include_unreviewed_generated:
                print(f"  skipped unreviewed auto-mapped corpus: "
                      f"{os.path.basename(path)}")
                continue
            for item in _read_jsonl(path):
                conversations.append({
                    "id": item["id"],
                    "source": item.get("source", "S046-first-party-authored"),
                    "lang": item.get("lang", "mixed"),
                    "topic": item.get("topic", "misc"),
                    "reviewer": item.get(
                        "reviewer", "unreviewed-auto-intent-mapping"
                        if generated else "S046-review-record"),
                    "safety": item.get(
                        "safety", "unreviewed-auto-intent-mapping"
                        if generated else "general-finance-reviewed"),
                    "licence": item.get("licence", "first-party-project-content"),
                    "messages": [
                        {"role": "user", "content": item["instruction"]},
                        {"role": "assistant", "content": item["response"]},
                    ],
                })

    # NOTE: greeting-context augmentation used to happen HERE, before the
    # decontamination + train/holdout split — which let a conversation and its
    # "-ctx" twin (same final user question) land on opposite sides, leaking 46%
    # of holdout questions into training. Augmentation now runs in main() AFTER
    # the split, on the TRAIN split only (see augment_greetings).

    seen = set()
    for conversation in conversations:
        cid = conversation.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("every conversation needs a non-empty string id")
        if cid in seen:
            raise ValueError(f"duplicate conversation id {cid}")
        seen.add(cid)
        conversation["messages"] = validate_messages(
            conversation["messages"], conversation.get("system"),
            require_final_assistant=True)
        for field in REQUIRED_METADATA:
            if not isinstance(conversation.get(field), str) \
                    or not conversation[field].strip():
                raise ValueError(f"{cid}: metadata {field!r} must be non-empty")
        for message in conversation["messages"]:
            if "http" in message["content"] or "www." in message["content"]:
                raise ValueError(f"URL in {cid} — authored corpus is link-free")
    return conversations


GREETINGS = [
    ("hi", "Hi! I'm Navya, an AI built in India to help with personal "
           "finance — savings, SIPs, loans, credit scores, insurance, "
           "tax. What would you like to know?"),
    ("hello", "Hello! I'm Navya. Ask me anything about money — savings, "
              "investments, loans, credit cards, insurance or tax."),
    ("namaste", "Namaste! Main Navya hoon — India ke liye banaya gaya "
                "finance assistant. Savings, SIP, loan, credit score, "
                "insurance ya tax — kuch bhi poochhiye."),
]


def augment_greetings(conversations):
    """Greeting-prefixed context twins. Run AFTER the split, on the TRAIN split
    ONLY, so a twin can never carry its origin's final user question into the
    other split (the v1 leak: 46% of holdout questions also sat in training
    because twins were made before the split). Returns conversations + twins,
    each twin re-validated for role ordering."""
    augmented = []
    for i, conversation in enumerate(conversations):
        if len(conversation["messages"]) == 2 and i % 3 == 0:
            g_q, g_a = GREETINGS[i % len(GREETINGS)]
            messages = validate_messages(
                [{"role": "user", "content": g_q},
                 {"role": "assistant", "content": g_a},
                 *conversation["messages"]],
                conversation.get("system"), require_final_assistant=True)
            augmented.append({**conversation, "id": conversation["id"] + "-ctx",
                              "messages": messages})
    return conversations + augmented


# Short n-gram used ONLY for eval overlap. BachattBench prompts/options are short
# and rarely form a 13-gram, so a shared 13-gram misses them; a shared 7-gram
# catches any MCQ stem/option of >=7 words. Both checks are hash-based (O(tokens))
# so this stays fast on the whole corpus — no O(docs x probes) substring scan.
SHORT_NGRAM = 7


def decontaminate(conversations):
    """Drop any training conversation overlapping BachattBench (ALL sections:
    seed_v0 + bench_v1 + date_aware_v0) or the tier-1 eval texts. Two hash-based
    checks:
      (1) shares >=1 13-gram with any eval text (long rationales / passages);
      (2) shares >=1 7-gram with any eval text — catches the short MCQ stems and
          options that never form a 13-gram (the gap that let bench items leak).
    Eval texts now include MCQ options/rationale/rubric (see decontaminate.eval_texts)."""
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
    for conversation in conversations:
        text = " ".join(message["content"]
                        for message in conversation["messages"])
        if any(g in banned13 for g in qdecon.ngrams(text)) \
                or any(g in banned7 for g in qdecon.ngrams(text, SHORT_NGRAM)):
            dropped.append(conversation["id"])
        else:
            kept.append(conversation)
    return kept, dropped, eval_paths, len(banned13)


def split_by_topic(conversations, seed=SEED):
    """Deterministic topic-stratified holdout; no random token-tail validation."""
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for conversation in conversations:
        by_topic[conversation["topic"]].append(conversation)
    train, holdout = [], []
    for _, group in sorted(by_topic.items()):
        rng.shuffle(group)
        n_holdout = max(1, len(group) // 10) if len(group) >= 5 else 0
        holdout.extend(group[:n_holdout])
        train.extend(group[n_holdout:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def pack_conversations(encoded, seq_len, pad_id, on_overflow="error"):
    """Pack whole encoded conversations and return arrays plus placements."""
    rows_x, rows_m, rows_e, placements = [], [], [], []
    cur_x, cur_m, cur_e = [], [], []
    row_index = 0
    local_example = 0
    dropped = []

    def flush():
        nonlocal cur_x, cur_m, cur_e, row_index, local_example
        if not cur_x:
            return
        pad_n = seq_len - len(cur_x)
        rows_x.append(cur_x + [pad_id] * pad_n)
        rows_m.append(cur_m + [0] * pad_n)
        rows_e.append(cur_e + [PAD_EX] * pad_n)
        cur_x, cur_m, cur_e = [], [], []
        row_index += 1
        local_example = 0

    for conversation, ids, loss_mask in encoded:
        if len(ids) > seq_len:
            if on_overflow == "drop":
                dropped.append(conversation["id"])
                continue
            raise ValueError(f"{conversation['id']} is {len(ids)} tokens; "
                             f"exceeds seq_len={seq_len}")
        if len(cur_x) + len(ids) > seq_len:
            flush()
        offset = len(cur_x)
        placements.append({"id": conversation["id"], "row": row_index,
                           "example": local_example, "offset": offset,
                           "tokens": len(ids), "loss_tokens": sum(loss_mask)})
        cur_x.extend(ids)
        cur_m.extend(loss_mask)
        cur_e.extend([local_example] * len(ids))
        local_example += 1
    flush()

    shape = (0, seq_len) if not rows_x else None
    x = np.empty(shape, dtype=np.uint16) if shape else np.asarray(rows_x,
                                                                  dtype=np.uint16)
    m = np.empty(shape, dtype=np.uint8) if shape else np.asarray(rows_m,
                                                                 dtype=np.uint8)
    e = np.empty(shape, dtype=np.uint16) if shape else np.asarray(rows_e,
                                                                  dtype=np.uint16)
    return x, m, e, placements, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tokenizer", default=os.path.join(
        CODE, "01-tokenizer", "tokenizer-v0.3-64k.json"))
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--on-overflow", choices=("error", "drop"),
                        default="error")
    parser.add_argument("--conversations-dir", default=None,
                        help="override the conversations/*.jsonl source dir "
                             "(default: the conversations/ dir next to this "
                             "script). Use this to pack a standalone corpus "
                             "(e.g. reasoning traces) without touching the "
                             "default first-party conversations/ files.")
    parser.add_argument(
        "--include-unreviewed-generated", action="store_true",
        help="include regex auto-mapped *_paraphrase/thin_intent_aug rows; "
             "off by default because known prompt/answer mismatches require "
             "review before production training")
    args = parser.parse_args()
    if args.seq_len < 8:
        parser.error("--seq-len must be at least 8")

    conversations = load_conversations(include_legacy=not args.no_legacy,
                                       conversations_dir=args.conversations_dir,
                                       include_unreviewed_generated=
                                       args.include_unreviewed_generated)
    print(f"loaded {len(conversations)} conversations / "
          f"{sum(len(c['messages']) for c in conversations)} messages")
    kept, contaminated, eval_paths, banned_count = decontaminate(conversations)
    print(f"decontamination: kept={len(kept)} dropped={len(contaminated)} "
          f"banned_ngrams={banned_count:,}")
    train, holdout = split_by_topic(kept)
    # Augment AFTER the split, TRAIN only — holdout stays real + twin-free, and
    # no twin can leak its origin's question across the split.
    train = augment_greetings(train)
    if not train or not holdout:
        raise ValueError("both train and holdout splits must be non-empty")
    print(f"split: train={len(train)} holdout={len(holdout)}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    template = template_metadata(tokenizer)
    if tokenizer.get_vocab_size() > 2**16:
        raise ValueError("SFT-v2 binary format requires vocab <= 65,536")
    pad_id = template["pad_id"]
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    metadata = {
        "format": "navya-sft-v2",
        "seq_len": args.seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "token_dtype": "uint16",
        "mask_dtype": "uint8",
        "example_dtype": "uint16",
        "pad_example_id": PAD_EX,
        "chat_template": template,
        "decontamination": {
            "eval_files": [os.path.relpath(path, CODE) for path in eval_paths],
            "banned_ngrams": banned_count,
            "dropped_ids": contaminated,
        },
        "topics": dict(sorted(Counter(c["topic"] for c in kept).items())),
        "included_unreviewed_generated": args.include_unreviewed_generated,
    }
    for split_name, split in (("train", train), ("val", holdout)):
        encoded = [(c, *encode_conversation(tokenizer, c["messages"]))
                   for c in split]
        x, mask, example, placements, overflow = pack_conversations(
            encoded, args.seq_len, pad_id, args.on_overflow)
        x.tofile(os.path.join(args.out, f"{split_name}_x.bin"))
        mask.tofile(os.path.join(args.out, f"{split_name}_m.bin"))
        example.tofile(os.path.join(args.out, f"{split_name}_ex.bin"))
        placement_by_id = {item["id"]: item for item in placements}
        for conversation in split:
            if conversation["id"] not in placement_by_id:
                continue
            entry = {key: conversation[key]
                     for key in ("id", *REQUIRED_METADATA)}
            entry.update({"split": split_name,
                          "messages": len(conversation["messages"]),
                          **placement_by_id[conversation["id"]]})
            manifest.append(entry)
        non_pad = int((example != PAD_EX).sum())
        metadata[split_name] = {
            "rows": len(x),
            "conversations": len(placements),
            "loss_tokens": int(mask.sum()),
            "content_tokens": non_pad,
            "packing_efficiency": non_pad / max(1, x.size),
            "overflow_dropped_ids": overflow,
        }
        print(f"  {split_name}: {len(placements)} conversations -> {len(x)} rows; "
              f"{int(mask.sum()):,} supervised tokens; "
              f"{metadata[split_name]['packing_efficiency']:.1%} packed")

    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"sft v2 chat data -> {args.out}")


if __name__ == "__main__":
    main()
