#!/usr/bin/env python3
"""Ingest Indian financial-regulator and legislation text (S047).

Polite same-host BFS crawler over an allowlist of Indian government sites:
RBI, SEBI (investor education), IRDAI, PFRDA, India Code. HTML only (PDF
extraction is a later stage); respects robots.txt; bounded by pages/site,
total bytes and a fixed delay. Extracted text goes through the normal
cleaning pipeline afterwards, so extraction here can be simple.

These are Indian government works reproduced for research/training with
attribution — see legal_register.md S047.

Usage:
  python ingest_gov_in.py --out pools/finance_econ_law/raw_bulk_govin.jsonl \
      --max-pages-per-site 3000 --max-bytes 300e6
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from html.parser import HTMLParser

UA = "navyam-gpt-research/0.1 (corpus ingest; contact: bachattapp@gmail.com)"

# Seed families — each maps to a legal-register row and a target pool.
# Select with --family (default: regulators, the original S047 crawl).
SEED_FAMILIES = {
    # S047: regulator consumer/education text -> finance_econ_law
    "regulators": [
        "https://www.rbi.org.in/Scripts/FAQDisplay.aspx",
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
        "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
        "https://www.rbi.org.in/FinancialEducation/Home.aspx",
        "https://investor.sebi.gov.in/",
        "https://www.sebi.gov.in/sebiweb/investment/publication.jsp",
        "https://irdai.gov.in/consumer-affairs",
        "https://irdai.gov.in/rules",
        "https://www.pfrda.org.in/index1.cshtml?lsid=237",
    ],
    # S050+S052: acts, rules, circulars, FAQs -> india_tax_authoritative.
    # NOTE: incometaxindia.gov.in (S051/CBDT) robots-blocks crawlers (403) —
    # respected; the Income Tax Act text itself comes via India Code.
    "tax": [
        # 2026-08-20: indiacode.nic.in is BACK (robots.txt 200, User-agent:*
        # allows all but /discover and /simple-search — urllib robotparser
        # enforces this). The .gov.in mirror serves pages but its robots.txt
        # 502s, so the crawler fail-closes there; use the old domain.
        "https://www.indiacode.nic.in/",
        "https://www.indiacode.nic.in/browse?type=title",
        "https://cbic-gst.gov.in/gst-goods-services-rates.html",
        "https://cbic-gst.gov.in/hindi/cgst-act.html",
        "https://cbic-gst.gov.in/gst-acts.html",
        "https://gstcouncil.gov.in/gst-knowledge",
    ],
    # S053-adjacent: Companies/LLP act text via India Code.
    # NOTE: mca.gov.in robots-blocks crawlers (403) — respected.
    "corporate": [
        "https://indiacode.gov.in/browse-by-central-acts",
    ],
}
SEEDS = SEED_FAMILIES["regulators"]   # backward-compatible default

SKIP_EXT = re.compile(r"\.(pdf|zip|doc|docx|xls|xlsx|ppt|jpg|jpeg|png|gif|svg"
                      r"|mp3|mp4|css|js|ico|rss|xml)$", re.I)
DROP_TAGS = {"script", "style", "noscript", "nav", "header", "footer",
             "form", "aside", "iframe"}


class TextAndLinks(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.chunks, self.links = [], []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in DROP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urllib.parse.urljoin(self.base, href))

    def handle_endtag(self, tag):
        if tag in DROP_TAGS and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()[:200]
        elif not self._skip:
            t = data.strip()
            if t:
                self.chunks.append(t)


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if "text/html" not in r.headers.get("Content-Type", "text/html"):
            return None
        return r.read(3_000_000).decode("utf-8", "replace")


def norm(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc.lower(),
                                    p.path, p.query, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-id", default="S047")
    ap.add_argument("--max-pages-per-site", type=int, default=3000)
    ap.add_argument("--max-bytes", type=float, default=300e6)
    ap.add_argument("--delay", type=float, default=0.7)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--family", default="regulators",
                    choices=sorted(SEED_FAMILIES),
                    help="seed family (maps to a legal-register row + pool)")
    args = ap.parse_args()
    global SEEDS
    SEEDS = SEED_FAMILIES[args.family]

    robots = {}

    def allowed(url):
        host = urllib.parse.urlsplit(url).netloc
        if host not in robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(f"https://{host}/robots.txt")
                rp.read()
            except Exception:
                rp = None
            robots[host] = rp
        rp = robots[host]
        return rp is None or rp.can_fetch(UA, url)

    # per-host queues: hosts are crawled round-robin, each honouring its own
    # politeness delay — requests to DIFFERENT hosts overlap in wall-clock
    # instead of serializing behind one global sleep
    seen, per_site = set(), {}
    queues: dict[str, deque] = {}
    next_ok: dict[str, float] = {}
    for u in SEEDS:
        queues.setdefault(urllib.parse.urlsplit(u).netloc, deque()).append(u)
    total = kept = 0
    out = open(args.out, "w", encoding="utf-8")
    while any(queues.values()) and total < args.max_bytes:
        progressed = False
        for site in list(queues):
            q = queues[site]
            if not q or per_site.get(site, 0) >= args.max_pages_per_site:
                continue
            if time.monotonic() < next_ok.get(site, 0):
                continue
            url = q.popleft()
            key = norm(url)
            if key in seen:
                continue
            seen.add(key)
            progressed = True
            if SKIP_EXT.search(urllib.parse.urlsplit(url).path) \
                    or not allowed(url):
                continue
            next_ok[site] = time.monotonic() + args.delay
            try:
                html = fetch(url)
            except Exception:
                continue
            if html is None:
                continue
            per_site[site] = per_site.get(site, 0) + 1
            p = TextAndLinks(url)
            try:
                p.feed(html)
            except Exception:
                continue
            text = "\n".join(p.chunks)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text) >= args.min_chars:
                doc = {"text": (p.title + "\n\n" if p.title else "") + text,
                       "source_id": args.source_id, "url": url}
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
                total += len(text)
                if total >= args.max_bytes:
                    break
            for link in p.links:
                if urllib.parse.urlsplit(link).netloc == site and \
                        norm(link) not in seen:
                    q.append(link)
            if kept and kept % 200 == 0:
                print(f"  {kept} pages, {total/1e6:.0f}MB, "
                      f"queued={sum(len(q) for q in queues.values())}",
                      file=sys.stderr)
        if not progressed:
            # every live host is inside its delay window — sleep the shortest
            pending = [next_ok.get(s, 0) for s in queues
                       if queues[s] and per_site.get(s, 0) < args.max_pages_per_site]
            if not pending:
                break
            time.sleep(max(0.05, min(pending) - time.monotonic()))
    out.close()
    print(f"gov_in: {kept} pages, {total/1e6:.1f}MB text → {args.out}",
          file=sys.stderr)
    print(f"per-site: { {k: v for k, v in sorted(per_site.items())} }",
          file=sys.stderr)


if __name__ == "__main__":
    main()
