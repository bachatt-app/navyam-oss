#!/usr/bin/env python3
"""Ingest Python source from an allowlist of permissively licensed repos.

Laptop-scale code pool: a few real, high-quality codebases with unambiguous
licenses, one document per file. At scale this is replaced by a
Stack-v2-class pipeline (license detection, per-file dedup) — same document
format.

Only repos in ALLOWLIST are fetched; adding one means recording its license
here AND a row in the legal register.
"""

import argparse
import io
import json
import sys
import tarfile
import urllib.request

UA = "bachatt-phase0-research/0.1 (data pipeline; contact: bachattapp@gmail.com)"

# (owner/repo, ref, license) — permissive only
ALLOWLIST = [
    ("psf/requests", "main", "Apache-2.0"),
    ("pallets/flask", "main", "BSD-3-Clause"),
    ("pallets/click", "main", "BSD-3-Clause"),
    ("more-itertools/more-itertools", "master", "MIT"),
]
EXTS = (".py",)
MAX_FILE_BYTES = 200_000     # skip generated monsters


def fetch_tar(repo: str, ref: str) -> bytes:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for repo, ref, license_ in ALLOWLIST:
            blob = fetch_tar(repo, ref)
            n_repo = 0
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or member.size > MAX_FILE_BYTES:
                        continue
                    if not member.name.endswith(EXTS):
                        continue
                    text = tar.extractfile(member).read().decode(
                        "utf-8", errors="replace")
                    if not text.strip():
                        continue
                    path = member.name.split("/", 1)[-1]
                    f.write(json.dumps({
                        "id": f"github/{repo}/{path}",
                        "source": args.source_id,
                        "url": f"https://github.com/{repo}",
                        "license": license_,
                        "text": text,
                    }, ensure_ascii=False) + "\n")
                    n_repo += 1
            written += n_repo
            print(f"{repo} ({license_}): {n_repo} files", file=sys.stderr)
    print(f"code pool -> {written} files -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
