#!/usr/bin/env python3
"""Live-data tools for Navya — layer 2 of the three-layer architecture.

Weights hold CONCEPTS; live numbers (NAV, FX, market data) must come from
tools, never from the model. Each tool returns
``{"ok", "data", "asof", "source"}`` or ``{"ok": False}`` — the router
formats answers deterministically, because a 151M model must never be
trusted to copy injected numbers.

Sources (all free / official):
  - AMFI NAVAll.txt — every mutual fund scheme NAV, updated daily (official)
  - frankfurter.app — ECB reference FX rates incl. INR (no key)
  - web search — pluggable: BRAVE_API_KEY > SERPER_API_KEY > DuckDuckGo
    Instant Answers (keyless, best-effort)

Every fetch is cached on disk (CACHE_DIR) with a per-tool TTL so a burst of
queries costs one upstream hit.
"""

import difflib
import json
import os
import time
import urllib.parse
import urllib.request

CACHE_DIR = os.environ.get(
    "NAVYA_TOOL_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".cache"))
UA = {"User-Agent": "navyam-tools/0.1 (contact: bachattapp@gmail.com)"}


def _cached(name, ttl, fetch):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
        with open(path, encoding="utf-8") as f:
            return f.read()
    data = fetch()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)
    return data


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- mf_nav ----

def mf_nav(scheme_query, top=3):
    """Fuzzy-match a scheme name against AMFI's official daily NAV dump."""
    def _fetch():
        # portal.amfiindia.com is AMFI's canonical NAV endpoint; www mirror
        # is the fallback.
        try:
            return _get("https://portal.amfiindia.com/spages/NAVAll.txt",
                        timeout=30)
        except Exception:  # noqa: BLE001
            return _get("https://www.amfiindia.com/spages/NAVAll.txt",
                        timeout=30)
    try:
        raw = _cached("amfi_navall.txt", 6 * 3600, _fetch)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"AMFI fetch failed: {e}"}

    rows = []
    for line in raw.splitlines():
        parts = line.split(";")
        # code;isin1;isin2;scheme name;plan;option;nav;date
        if len(parts) >= 8 and parts[0].strip().isdigit():
            name = " - ".join(p.strip() for p in parts[3:6] if p.strip())
            rows.append({"name": name, "nav": parts[6].strip(),
                         "date": parts[7].strip()})
    if not rows:
        return {"ok": False, "error": "AMFI dump parse failed"}

    q = scheme_query.lower()
    scored = []
    for r in rows:
        n = r["name"].lower()
        hit = sum(1 for w in q.split() if w in n)
        if hit:
            scored.append((hit + difflib.SequenceMatcher(None, q, n).ratio(),
                           r))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return {"ok": False, "error": "no scheme matched"}
    best = [r for _, r in scored[:top]]
    return {"ok": True, "data": best, "asof": best[0]["date"],
            "source": "AMFI (amfiindia.com)"}


# --------------------------------------------------------------- fx_rate ----

def fx_rate(base="USD", quote="INR"):
    try:
        raw = _cached(f"fx_{base}_{quote}.json", 3600,
                      lambda: _get("https://api.frankfurter.app/latest?"
                                   f"from={base}&to={quote}"))
        d = json.loads(raw)
        return {"ok": True, "data": {"rate": d["rates"][quote],
                                     "base": base, "quote": quote},
                "asof": d["date"], "source": "ECB reference via frankfurter.app"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"fx fetch failed: {e}"}


# ------------------------------------------------------------ web_search ----

def web_search(query, count=3):
    """Pluggable search: Brave > Serper (if keys set) > DDG instant answer."""
    brave = os.environ.get("BRAVE_API_KEY")
    serper = os.environ.get("SERPER_API_KEY")
    try:
        if brave:
            req = urllib.request.Request(
                "https://api.search.brave.com/res/v1/web/search?q="
                + urllib.parse.quote(query) + f"&count={count}",
                headers={**UA, "X-Subscription-Token": brave})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            hits = [{"title": w["title"], "url": w["url"],
                     "snippet": w.get("description", "")}
                    for w in d.get("web", {}).get("results", [])[:count]]
            return {"ok": bool(hits), "data": hits, "source": "brave"}
        if serper:
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=json.dumps({"q": query, "num": count}).encode(),
                headers={**UA, "X-API-KEY": serper,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            hits = [{"title": w["title"], "url": w["link"],
                     "snippet": w.get("snippet", "")}
                    for w in d.get("organic", [])[:count]]
            return {"ok": bool(hits), "data": hits, "source": "serper"}
        raw = _get("https://api.duckduckgo.com/?format=json&no_html=1&q="
                   + urllib.parse.quote(query))
        d = json.loads(raw)
        hits = []
        if d.get("AbstractText"):
            hits.append({"title": d.get("Heading", query),
                         "url": d.get("AbstractURL", ""),
                         "snippet": d["AbstractText"]})
        for t in d.get("RelatedTopics", [])[:count]:
            if isinstance(t, dict) and t.get("Text"):
                hits.append({"title": t["Text"][:60],
                             "url": t.get("FirstURL", ""),
                             "snippet": t["Text"]})
        return {"ok": bool(hits), "data": hits[:count],
                "source": "duckduckgo-instant"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"search failed: {e}"}


if __name__ == "__main__":
    import sys
    tool = sys.argv[1] if len(sys.argv) > 1 else "fx"
    arg = " ".join(sys.argv[2:])
    out = {"nav": lambda: mf_nav(arg or "uti nifty 50 index direct growth"),
           "fx": lambda: fx_rate(arg or "USD"),
           "search": lambda: web_search(arg or "rbi repo rate")}[tool]()
    print(json.dumps(out, ensure_ascii=False, indent=1)[:2000])
