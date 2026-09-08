#!/usr/bin/env python3
"""Shard-flywheel ingestion — the unit of the 2T-token corpus build.

The download-everything-then-clean design cannot reach SmolLM2 scale (2T
tokens ~ 8-10TB text) on bounded disks. This worker processes ONE shard
end-to-end and leaves nothing behind:

    pull ~N GB from a HF parquet dataset (from a shard cursor)
    -> clean in-process (langid/filters/PII/decontam via docfeat, exact +
       near dedup with PER-SHARD state — see honesty note)
    -> tokenize to uint16 bin
    -> upload clean.jsonl.zst (cool tier: future re-tokenization) + bin +
       manifest (sha256, counts, pipeline version) to blob
    -> delete local, advance cursor

Restartable: the cursor file records the next parquet index per (dataset,
config); an interrupted shard leaves no partial blob (upload-then-commit).
Run many workers in parallel on different datasets — each shard is
independent.

HONESTY NOTE (dedup at scale): near-dedup state across 2T tokens would need
hundreds of GB; at bulk scale state is per-shard, and we rely on upstream
dedup of the curated sources (FineWeb-Edu/DCLM/FineMath are deduplicated at
origin). Cross-shard exact dedup and distributed MinHash are the documented
1T-stage upgrades (PERF_TODO item 3's Rust kernel is the prerequisite).

Usage:
  python shard_ingest.py --dataset HuggingFaceFW/fineweb-edu --config sample-100BT \
      --split train --source-id S010 --domain global_english \
      --shard-bytes 3e9 --max-shards 4 --evals "../../03-evals/bachattbench/*.jsonl ..."
"""

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
import decontaminate  # noqa: E402
import parallel_clean  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

UA = "navyam-gpt-research/0.1 (corpus ingest; contact: bachattapp@gmail.com)"
# gated datasets (e.g. ai4bharat/Shrutilipi): export HF_TOKEN after accepting
# the dataset's terms on huggingface.co — never bypass a gate without it
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _headers():
    h = {"User-Agent": UA}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def fetch_urls(dataset, config, split):
    url = (f"https://huggingface.co/api/datasets/{dataset}/parquet/"
           f"{config}/{split}")
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


AUTH_MODE = os.environ.get("NAVYA_BLOB_AUTH", "login")   # login on VMs (MSI)
# Artifact store. Default: Azure blob `corpora` via MSI (Azure VMs). Off-Azure
# workers (the GCP fleet) set ONE of:
#   BLOB_SAS=<container SAS>          -> Azure blob via SAS (cross-cloud egress)
#   NAVYA_BLOB_TARGET=gs://bucket     -> GCS (no credential on GCE; zero egress;
#                                        mirror to Azure at sprint end)
BLOB_SAS = os.environ.get("BLOB_SAS", "")
BLOB_TARGET = os.environ.get("NAVYA_BLOB_TARGET", "az")


def _az_base(container):
    cmd = ["az", "storage", "blob"]
    auth = (["--sas-token", BLOB_SAS] if BLOB_SAS
            else ["--auth-mode", AUTH_MODE])
    return cmd, ["--account-name", "your-storage-account", "--container-name",
                 container, "-o", "none"] + auth


def upload(local, blob_name, container="corpora", tier=None):
    if BLOB_TARGET.startswith("gs://"):
        cmd = ["gcloud", "storage", "cp", "-q", local,
               f"{BLOB_TARGET.rstrip('/')}/{blob_name}"]
    else:
        head, tail = _az_base(container)
        cmd = head + ["upload", "--name", blob_name, "--file", local,
                      "--overwrite"] + tail
        if tier:
            cmd += ["--tier", tier]
    for _ in range(5):
        if subprocess.run(cmd).returncode == 0:
            return True
    return False


def download(blob_name, local, container="corpora"):
    """Best-effort fetch (cursor mirror). Returns True iff the blob existed."""
    if BLOB_TARGET.startswith("gs://"):
        cmd = ["gcloud", "storage", "cp", "-q",
               f"{BLOB_TARGET.rstrip('/')}/{blob_name}", local]
    else:
        head, tail = _az_base(container)
        cmd = head + ["download", "--name", blob_name, "--file", local] + tail
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--filter-field",
                    help="keep only rows whose field is in --filter-values "
                         "(e.g. license allow-list for code)")
    ap.add_argument("--filter-values", default="")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--shard-bytes", type=float, default=3e9,
                    help="raw text per shard before cleaning")
    ap.add_argument("--max-shards", type=int, default=1,
                    help="shards to process this invocation (loop outside)")
    ap.add_argument("--evals", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "01-tokenizer", "tokenizer-v0.3-64k.json"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--work-dir", default="shard_work")
    ap.add_argument("--skip-filters", action="store_true")
    ap.add_argument("--skip-near-dedup", action="store_true",
                    help="fast path for corpora deduplicated at origin "
                         "(FineWeb-Edu/DCLM/FineMath): keep exact-dedup, "
                         "PII, decontam, tokenize; drop per-shard MinHash")
    ap.add_argument("--worker-index", type=int, default=0,
                    help="cursor-sharding: this worker takes parquet files "
                         "i where i %% num-workers == worker-index")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--text-template", default=None,
                    help="render each row into text from several columns, "
                         "python str.format over column names, e.g. "
                         "'Question: {question}\\nAnswer: {exp}' (Q/A sets "
                         "such as MedMCQA); --text-field is then ignored")
    ap.add_argument("--keep-fields", default="",
                    help="comma list of parquet columns carried into each "
                         "doc (e.g. int_score,dump for FineWeb-Edu)")
    ap.add_argument("--partition-field", default=None,
                    help="write one bin per distinct value of this doc field "
                         "(NNNNN.s<val>.bin) so streams can SELECT from the "
                         "pool by quality bucket without re-tokenizing")
    args = ap.parse_args()
    keep_fields = [f for f in args.keep_fields.split(",") if f.strip()]
    if args.partition_field and args.partition_field not in keep_fields:
        keep_fields.append(args.partition_field)
    if not 0 <= args.worker_index < args.num_workers:
        sys.exit("--worker-index must be in [0, --num-workers)")

    tag = f"{args.dataset.replace('/', '_')}_{args.config}"
    args.work_dir = os.path.abspath(args.work_dir)   # subprocess cwd differs
    os.makedirs(args.work_dir, exist_ok=True)
    # Cursor-sharding: N workers stripe one dataset's parquet list without
    # duplicating; each (worker, N) pair owns its own cursor and blob prefix.
    # The cursor is mirrored to the artifact store after every commit so a
    # recreated spot VM (fresh disk) resumes instead of restarting at 0.
    wtag = (f"w{args.worker_index:03d}of{args.num_workers:03d}"
            if args.num_workers > 1 else "")
    ltag = f"{tag}{'.' + wtag if wtag else ''}"     # local scratch prefix
    cursor_name = f"{ltag}.cursor.json"
    cursor_path = os.path.join(args.work_dir, cursor_name)
    cursor_blob = f"cursors/{args.domain}/{cursor_name}"
    if not os.path.exists(cursor_path):
        if download(cursor_blob, cursor_path):
            print(f"cursor restored from store: {cursor_blob}", flush=True)
    cursor = json.load(open(cursor_path)) if os.path.exists(cursor_path) \
        else {"parquet_idx": 0, "shard": 0}
    all_urls = fetch_urls(args.dataset, args.config, args.split)
    urls = all_urls[args.worker_index::args.num_workers]
    print(f"{tag}: {len(all_urls)} parquet files, {len(urls)} for "
          f"{wtag or 'single worker'}; cursor {cursor}", flush=True)

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|eos|>")

    for _ in range(args.max_shards):
        if cursor["parquet_idx"] >= len(urls):
            print("dataset exhausted", flush=True)
            break
        shard_id = cursor["shard"]
        first_idx = cursor["parquet_idx"]
        raw_path = os.path.join(args.work_dir, f"{ltag}_{shard_id}.raw.jsonl")
        # ---- 1. pull raw text up to shard-bytes
        written = 0
        docs = 0
        with open(raw_path, "w", encoding="utf-8") as out:
            while written < args.shard_bytes and cursor["parquet_idx"] < len(urls):
                u = urls[cursor["parquet_idx"]]
                print(f"  shard {shard_id}: parquet {cursor['parquet_idx']}",
                      flush=True)
                req = urllib.request.Request(u, headers=_headers())
                with urllib.request.urlopen(req, timeout=1200) as r:
                    buf = io.BytesIO(r.read())
                allowed = {v.strip() for v in args.filter_values.split(",")
                           if v.strip()}
                for batch in pq.ParquetFile(buf).iter_batches(batch_size=4096):
                    cols = batch.to_pydict()
                    if args.text_template:
                        # materialize a synthetic text column from the template
                        nrow = len(next(iter(cols.values())))
                        cols["__text__"] = [
                            args.text_template.replace("\\n", "\n").format(
                                **{k: ("" if v[j] is None else v[j])
                                   for k, v in cols.items()}) for j in range(nrow)]
                        args.text_field = "__text__"
                    if args.text_field not in cols:
                        # ASR/gated corpora vary: fall back to common names
                        for cand in ("transcription", "transcript",
                                     "sentence", "text", "raw_text"):
                            if cand in cols:
                                print(f"  text field '{args.text_field}' -> "
                                      f"'{cand}'", flush=True)
                                args.text_field = cand
                                break
                        else:
                            sys.exit(f"no text field among {list(cols)[:8]}")
                    texts = cols[args.text_field]
                    fvals = cols.get(args.filter_field) \
                        if args.filter_field else None
                    kept_cols = [(f, cols[f]) for f in keep_fields if f in cols]
                    for j, text in enumerate(texts):
                        if fvals is not None and str(fvals[j]) not in allowed:
                            continue
                        if not text or not text.strip():
                            continue
                        rec = {"text": text, "source": args.source_id}
                        for f, col in kept_cols:
                            rec[f] = col[j]
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += len(text)
                        docs += 1
                cursor["parquet_idx"] += 1
        print(f"  shard {shard_id}: raw {written/1e9:.2f}GB / {docs} docs",
              flush=True)

        # ---- 2. clean (per-shard dedup state; see honesty note)
        clean_path = os.path.join(args.work_dir, f"{ltag}_{shard_id}.clean.jsonl")
        state_path = os.path.join(args.work_dir, f"{ltag}_{shard_id}.state.bin")
        rc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "..", "pipeline",
                "parallel_clean.py"),
             "--inputs", raw_path, "--evals", args.evals,
             "--dedup-state", state_path, "--out", clean_path,
             "--workers", str(args.workers)]
            + (["--skip-filters"] if args.skip_filters else [])
            + (["--skip-near-dedup"] if args.skip_near_dedup else []),
            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "pipeline")).returncode
        if rc != 0:
            sys.exit(f"cleaning failed for shard {shard_id} (rc={rc})")

        # ---- 3. tokenize — optionally partitioned by a per-doc field
        # (FineWeb-Edu int_score): NNNNN.s3.bin / .s4.bin / .s5.bin, so a
        # stream can select "int_score >= 4" from the pool by picking bins.
        # PERF item 4: per-doc uint16 arrays, one concatenate per flush.
        eos_arr = np.array([eos], dtype=np.uint16)
        parts = {}          # partition key -> state
        total = 0

        def part_for(val):
            key = "all" if args.partition_field is None else \
                ("na" if val is None else str(val))
            p = parts.get(key)
            if p is None:
                suffix = "" if key == "all" else f".s{key}"
                path = os.path.join(args.work_dir,
                                    f"{ltag}_{shard_id}{suffix}.bin")
                p = parts[key] = {"f": open(path + ".part", "wb"),
                                  "chunks": [], "pending": 0, "total": 0,
                                  "path": path, "suffix": suffix}
            return p

        def flush(p):
            nonlocal total
            if p["chunks"]:
                np.concatenate(p["chunks"]).tofile(p["f"])
                p["total"] += p["pending"]
                total += p["pending"]
                p["chunks"], p["pending"] = [], 0

        texts, vals = [], []

        def encode_flush():
            nonlocal texts, vals
            for enc, val in zip(tok.encode_batch(texts), vals):
                p = part_for(val)
                ids = np.fromiter(enc.ids, dtype=np.uint16, count=len(enc.ids))
                p["chunks"].append(ids)
                p["chunks"].append(eos_arr)
                p["pending"] += len(ids) + 1
                if p["pending"] >= 2_000_000:
                    flush(p)
            texts, vals = [], []

        for line in open(clean_path, encoding="utf-8"):
            d = json.loads(line)
            texts.append(d["text"])
            vals.append(d.get(args.partition_field)
                        if args.partition_field else None)
            if len(texts) >= 512:
                encode_flush()
        if texts:
            encode_flush()
        if not parts:                       # fully-filtered shard: empty bin
            part_for(None if args.partition_field else None)
        bins = []                           # (suffix, path, tokens)
        for key, p in sorted(parts.items()):
            flush(p)
            p["f"].close()
            os.replace(p["path"] + ".part", p["path"])
            bins.append((p["suffix"], p["path"], p["total"]))

        # ---- 4. compress clean text (kept for future re-tokenization)
        subprocess.run(["zstd", "-q", "-f", "--rm", clean_path])

        # ---- 5. upload shard + manifest, then advance cursor (commit point)
        # flat under <domain>/<dataset>/ — build_stream discovers bins with
        # shards/*/*/*.bin, so worker striping lives in the file name
        prefix = (f"shards/{args.domain}/{tag}/"
                  + (f"{wtag}_" if wtag else "") + f"{shard_id:05d}")
        meta = json.load(open(clean_path + ".meta.json"))
        manifest = {"dataset": args.dataset, "config": args.config,
                    "source_id": args.source_id, "domain": args.domain,
                    "shard": shard_id, "raw_bytes": written,
                    "tokens": total, "cleaning": meta,
                    "tokenizer": os.path.basename(args.tokenizer),
                    "worker_index": args.worker_index,
                    "num_workers": args.num_workers,
                    "parquet_files": urls[first_idx:cursor["parquet_idx"]],
                    "partition_field": args.partition_field,
                    "bins": {os.path.basename(prefix) + sfx + ".bin": t
                             for sfx, _, t in bins},
                    "store": BLOB_TARGET}
        man_path = os.path.join(args.work_dir, f"{ltag}_{shard_id}.manifest.json")
        json.dump(manifest, open(man_path, "w"), indent=1)
        ok = (all(upload(path, prefix + sfx + ".bin") for sfx, path, _ in bins)
              and upload(clean_path + ".zst", prefix + ".clean.jsonl.zst",
                         tier="Cool")
              and upload(man_path, prefix + ".manifest.json"))
        if not ok:
            sys.exit(f"upload failed for shard {shard_id} — cursor NOT advanced")
        for p in [raw_path, clean_path + ".zst", clean_path + ".meta.json",
                  state_path, man_path] + [b[1] for b in bins]:
            if os.path.exists(p):
                os.remove(p)
        cursor["shard"] += 1
        json.dump(cursor, open(cursor_path + ".tmp", "w"))
        os.replace(cursor_path + ".tmp", cursor_path)
        upload(cursor_path, cursor_blob)     # mirror; local copy is the truth
        print(f"  shard {shard_id} COMMITTED: {total/1e6:.0f}M tokens "
              f"-> {prefix}.*", flush=True)


if __name__ == "__main__":
    main()
