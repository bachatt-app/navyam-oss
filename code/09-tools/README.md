# 09-tools — live-data tool layer (three-layer architecture, layer 2)

Weights hold concepts. Live numbers come from tools. Trained reasoning over
product picks comes later. This directory is layer 2's v1.

## Design decision that matters

The router answers live-data queries **deterministically from templates** —
the model never sees the fetched numbers. At 151-338M params, a model WILL
corrupt injected digits; template formatting makes the number path
provably correct. When models are big enough to be trusted with tool
outputs, the chat template's reserved `tool` role (token id 6) is already
in the tokenizer + SFT format, so the migration path is data, not plumbing.

## Components

- `tools.py` — fetchers with on-disk cache:
  - `mf_nav(query)` — AMFI `NAVAll.txt` (official, all schemes, daily; 6h TTL)
  - `fx_rate(base, quote)` — ECB reference via frankfurter.app (no key; 1h TTL)
  - `web_search(query)` — pluggable: `BRAVE_API_KEY` > `SERPER_API_KEY` >
    DuckDuckGo Instant Answers (keyless, best-effort)
- `router.py` — regex intent detection (NAV/FX + "live-ness" words like
  aaj/abhi/current) → formatted Hinglish reply with **source + as-of date**.
  No match → returns None → serve.py falls through to the model.
- Integration: `06-inference/serve.py` calls `router.route(messages)` before
  generation on `/v1/chat/completions`; tool replies return
  `model: "<id>+tools"`. Disable with `NAVYA_TOOLS=0`.

## Adding a tool

1. Fetcher in `tools.py` returning `{"ok", "data", "asof", "source"}` with
   `_cached()` and a real TTL.
2. Detection regex + template in `router.py`. Template must state source and
   date — a number without a date is misinformation waiting to happen.
3. Probe it: `python tools.py <name> <query>` then a router test.

Candidates next: RBI repo rate + small-savings rates (official pages),
gold/silver (IBJA), index levels (NSE), stock quotes — each needs a terms
check before ingesting/serving (register the source like corpus sources).

## web_search status

Implemented in `tools.py`, NOT yet routed — general web search needs answer
synthesis, which the current model is too small to do reliably. It becomes
useful at navya-1b/2 when a summarize-with-citations SFT task lands. Keys:
set `BRAVE_API_KEY` or `SERPER_API_KEY` on the serving box to upgrade the
backend from DuckDuckGo instant answers.
