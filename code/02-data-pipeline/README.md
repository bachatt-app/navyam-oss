# 02 — Data pipeline v1

The highest-leverage workstream. Model quality is downstream of corpus quality
more than of any architecture choice.

## Stage order

```
RAW SOURCES (web crawl, open corpora, licensed, regulator/public-domain)
  -> EXTRACTION        html/pdf -> clean text, tables preserved
  -> LANGID            language + script + code-mix detection    (pipeline/langid.py)
  -> QUALITY FILTER    heuristics now, trained classifier later  (pipeline/filters.py)
  -> DEDUP             exact + MinHash near-dup, cross-source    (pipeline/dedup_minhash.py)
  -> DECONTAMINATE     strip eval-set overlap. NON-NEGOTIABLE    (pipeline/decontaminate.py)
  -> PII/SAFETY SCRUB
  -> MIX + SNAPSHOT    weighted sampling manifest                (pipeline/mix.py)
  -> TOKENIZE + SHARD  versioned, reproducible
```

## Rules

1. **Snapshots are immutable.** `mix.py` emits a manifest whose hash is the
   snapshot ID. A training run must be reproducible from that ID alone.
2. **Every source has a recorded legal basis** in the data-source legal register
   (`legal_register.md`) *before* it enters the pipeline. This is a compliance
   artifact, not paperwork.
3. **Decontamination runs against every eval set we use** — Tier 1, Tier 2, and
   BachattBench — before every snapshot.
4. Documents flow through as JSONL: one object per doc with `text`, `source`,
   `lang`, `url`, `ts`, plus stage annotations. Never mutate in place; each stage
   reads one JSONL and writes another.

## Status of this code

These are working *reference implementations* to develop the logic and process
the first tens of GB on a single machine. At the 1T-token scale they get ported
onto a distributed runner (Spark/Ray/slurm array — decision by month 3) with the
same logic and the same tests.

Known upgrades already planned:
- `langid.py`: script heuristics now → fastText LID + trained romanized-Hindi
  classifier
- `filters.py`: Gopher-style heuristics now → trained quality classifier
  (exit criterion: classifier beats heuristics on downstream 1B evals)
- `dedup_minhash.py`: in-memory LSH now → distributed MinHash at scale

## First 90 days of data work

1. Stand up the India-focused crawl list (news, finance portals, regulator
   sites, forums with permissive terms) + Common Crawl processing.
2. Ingest open corpora: FineWeb-class English, AI4Bharat Sangraha, Samanantar.
3. Start the finance library: RBI/SEBI/IRDAI/PFRDA archives, budgets, acts —
   these are mostly PDF; extraction quality is the work.
4. Legal register live from day one.
5. Target: 100B clean tokens by month 3 (enough for all research-model runs).

## Durable storage: Azure Blob

Everything downloaded from the internet (and everything derived from it) is
mirrored to Azure Blob Storage — local disk is a working cache, the blob is
the durable copy.

| | |
| --- | --- |
| Resource group | `your-webapp-rg` (Central India) |
| Storage account | `your-storage-account` (StorageV2, Standard_LRS, Hot, no public access) |
| `corpora` container | `pools/<domain>/*.jsonl`, `snapshots/*.json`, `meta/` (mix configs, legal register) |
| `tokenizers` container | trained `tokenizer-v*.json` artifacts |
| `checkpoints` container | model checkpoints worth keeping |

Run `./sync_blob.sh` after every ingestion session (needs `az login`).
