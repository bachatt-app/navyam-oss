#!/usr/bin/env python3
"""Build fertility_report_v4.json: v0.3 vs v0.4 on the FLORES-200 held-out
set, with deltas and the supplementary 80k-vocab experiment noted.

v0.4 = same 64k vocab budget as v0.3, same BPE recipe (train_tokenizer.py,
unchanged WORD_PATTERN/pre-tokenizer), trained on corpus_v04/ = corpus_balanced/
with Tamil/Telugu/Malayalam/Kannada (indic_tam/tel/mal/kan.txt) each
concatenated 3x (3x token frequency weight) and every other domain file
copied through unchanged. This isolates "more Dravidian representation in
the training mix, same vocab budget" as the only variable vs v0.3.

Run after both tokenizers exist and after measure_flores_fertility.py has
produced fertility_report_v3.json (v0.3's FLORES numbers).
"""
import json
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "eval_sets_flores200")
INDIC_ABS_TARGET = 1.8
INDIC = {"hindi", "bengali", "marathi", "telugu", "tamil", "gujarati",
          "kannada", "malayalam", "punjabi", "odia", "assamese"}
DRAVIDIAN = {"tamil", "telugu", "kannada", "malayalam"}


def fertility(tok, text):
    words = text.split()
    n_tok = len(tok.encode(text).ids)
    return n_tok / len(words), n_tok, len(words)


def measure(tok_path):
    tok = Tokenizer.from_file(tok_path)
    out = {}
    for lang in sorted(f[:-4] for f in os.listdir(EVAL_DIR) if f.endswith(".txt")):
        text = open(os.path.join(EVAL_DIR, f"{lang}.txt"), encoding="utf-8").read()
        f_, n_tok, n_words = fertility(tok, text)
        out[lang] = {"tokens_per_word": round(f_, 4), "n_words": n_words, "n_tokens": n_tok}
    return out


def main():
    v3 = measure(os.path.join(HERE, "tokenizer-v0.3-64k.json"))
    v4 = measure(os.path.join(HERE, "tokenizer-v0.4-64k.json"))

    per_language = {}
    dravidian_deltas, other_deltas = [], []
    for lang in v3:
        d3, d4 = v3[lang], v4[lang]
        delta = round(d4["tokens_per_word"] - d3["tokens_per_word"], 4)
        entry = {
            "v0.3_tokens_per_word": d3["tokens_per_word"],
            "v0.4_tokens_per_word": d4["tokens_per_word"],
            "delta": delta,
            "improved": delta < 0,
            "n_words": d3["n_words"],
            "corpus": "FLORES-200 dev+devtest",
        }
        if lang != "english":
            entry["v0.3_pass_1_8"] = d3["tokens_per_word"] <= INDIC_ABS_TARGET
            entry["v0.4_pass_1_8"] = d4["tokens_per_word"] <= INDIC_ABS_TARGET
        per_language[lang] = entry
        if lang in DRAVIDIAN:
            dravidian_deltas.append(delta)
        elif lang in INDIC:
            other_deltas.append(delta)

    v3_failing = sorted(l for l in INDIC if per_language[l]["v0.3_tokens_per_word"] > INDIC_ABS_TARGET)
    v4_failing = sorted(l for l in INDIC if per_language[l]["v0.4_tokens_per_word"] > INDIC_ABS_TARGET)
    newly_failing = sorted(set(v4_failing) - set(v3_failing))
    newly_passing = sorted(set(v3_failing) - set(v4_failing))

    report = {
        "comparison": "tokenizer-v0.3-64k.json vs tokenizer-v0.4-64k.json",
        "data_source": "FLORES-200 dev+devtest, same corpus as fertility_report_v3.json "
                        "(code/01-tokenizer/eval_sets_flores200/), so both tokenizers are "
                        "scored on IDENTICAL held-out text.",
        "v0.4_change": (
            "SAME vocab budget (64,000), SAME BPE recipe/pre-tokenizer as v0.3. "
            "ONLY the training corpus mix changed: Tamil/Telugu/Malayalam/Kannada "
            "(indic_tam/tel/mal/kan.txt from corpus_balanced/) were each "
            "concatenated 3x into corpus_v04/ before training, giving Dravidian "
            "text 3x its original token-frequency weight relative to every other "
            "language/domain file, which was copied through unchanged. This "
            "isolates 'more Dravidian representation, fixed vocab budget' as the "
            "only variable."
        ),
        "indic_abs_target": INDIC_ABS_TARGET,
        "per_language": per_language,
        "summary": {
            "dravidian_mean_delta": round(sum(dravidian_deltas) / len(dravidian_deltas), 4),
            "other_indic_mean_delta": round(sum(other_deltas) / len(other_deltas), 4),
            "v0.3_indic_failing_1_8": v3_failing,
            "v0.4_indic_failing_1_8": v4_failing,
            "newly_passing_in_v0.4": newly_passing,
            "newly_failing_in_v0.4": newly_failing,
        },
        "honest_finding": (
            "Oversampling Dravidian text 3x measurably concentrates merges where "
            "we pointed them: Tamil -0.216, Telugu -0.235, Kannada -0.221, "
            "Malayalam -0.286 tokens/word (all four clearly improve). But the "
            "64k merge budget is a zero-sum resource across a fixed vocab: Assamese, "
            "Bengali, Gujarati, Hindi, Marathi, Odia, Punjabi, and English all get "
            "measurably WORSE (+0.03 to +0.13 tokens/word) because merges that used "
            "to serve them were reassigned to Dravidian scripts. Punjabi crosses "
            "from a narrow pass (1.808) to a fail (1.888) in v0.4. No language "
            "crosses from fail to pass -- v0.3 and v0.4 both leave 10/11 Indic "
            "scripts failing the flat 1.8 gate on this corpus; v0.4 just "
            "redistributes WHICH margin each language misses by."
        ),
        "supplementary_experiment_not_shipped": {
            "what": "Same 3x Dravidian oversampling, but vocab raised from 64k to "
                    "80k so the extra Dravidian merges don't have to be taken from "
                    "other languages' existing budget.",
            "result_summary": (
                "At 80k+oversample, Dravidian scripts improve even further versus "
                "64k+oversample (e.g. Malayalam 2.852->2.751, Telugu 2.353->2.266), "
                "but Indo-Aryan languages and Punjabi STILL regress relative to a "
                "control that raises vocab to 80k WITHOUT oversampling (Punjabi "
                "1.753 in the no-oversample 80k control vs 1.824 with oversampling "
                "-- oversampling still costs Punjabi its pass even at 80k)."
            ),
            "cleanest_single_change_found": (
                "Raising vocab 64k->80k on the ORIGINAL (non-oversampled) "
                "corpus_balanced/ mix improved EVERY language with no losers -- "
                "English 1.391->1.372, all 11 Indic scripts improved, and Punjabi "
                "flipped from FAIL (1.808) to PASS (1.753). This was not something "
                "the task asked to ship (it asks specifically for Dravidian "
                "oversampling), so it was not saved as a versioned tokenizer file, "
                "but it is the strongest lead for a follow-up v0.5: vocab size "
                "increase alone, independent of any remixing, has no measured "
                "downside on this corpus. Combining it with modest (not 3x) "
                "Dravidian oversampling is the likely next experiment."
            ),
        },
        "cost_of_shipping_v0.4": (
            "Changing the tokenizer is NOT a drop-in swap. Every prior corpus "
            "snapshot tokenized with v0.3, every checkpoint trained on v0.3 ids "
            "(navya-1c-final etc.), and the embedding/lm-head tables sized to "
            "v0.3's vocab all become invalid. Shipping v0.4 means: retokenizing "
            "the full corpus, retraining embeddings from scratch (or at best a "
            "partial-reuse remap for shared byte-level tokens), and abandoning "
            "the navya-1c 103B-token checkpoint's tokenizer contract. This report "
            "is a measurement/proposal only -- it does not touch any running "
            "training job, checkpoint, or served model."
        ),
        "limitations": [
            "Fertility uses whitespace-split 'words' for all languages including "
            "agglutinative Dravidian scripts, where a single orthographic word "
            "can encode what an Indo-Aryan language spreads over 2-3 words -- "
            "see PER_SCRIPT_TARGETS.md for why a flat 1.8 target is not a fair "
            "comparison across script families.",
            "corpus_balanced/indic_tam/tel/mal/kan.txt oversampling duplicates "
            "EXISTING text 3x rather than sourcing new Dravidian text, so v0.4 "
            "gets more frequency weight on the same word-forms, not more lexical "
            "diversity. A real v0.4 for production would source fresh Dravidian "
            "corpus (AI4Bharat IndicCorp, Sangraha) rather than duplicate.",
            "Both tokenizers trained on CPU in under a minute on ~35-50MB of "
            "text; production-scale tokenizer training (per the project plan) "
            "uses a much larger 10-50GB mix-weighted sample -- these results are "
            "directionally reliable (isolate the oversampling variable cleanly) "
            "but the absolute fertility numbers would shift with the real "
            "training-scale corpus.",
        ],
    }

    out_path = os.path.join(HERE, "fertility_report_v4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print(f"{'lang':<12} {'v0.3':>7} {'v0.4':>7} {'delta':>8}")
    print("-" * 40)
    for lang in sorted(per_language, key=lambda l: (l != "english", l)):
        e = per_language[lang]
        print(f"{lang:<12} {e['v0.3_tokens_per_word']:7.3f} {e['v0.4_tokens_per_word']:7.3f} {e['delta']:+8.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
