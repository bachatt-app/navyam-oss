#!/usr/bin/env python3
"""Data-reality launch mix: start from a target mix, drop domains that have NO
bins in the local shard/domain dirs, and give their weight to the nearest
sibling (indic_* -> indic_hin, india_* -> finance_econ_law, anything else ->
global_english) so the launch stream keeps the target's block shares.
Writes <mix>-launch.json and prints what moved.  No silent drops.

  python make_launch_mix.py --mix configs/mix-navya1b-v2-main.json \
      --shards ~/shards_local --domains data/domains1b [--bin-include REGEX]
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--mix", required=True)
ap.add_argument("--shards", default="")
ap.add_argument("--domains", default="")
ap.add_argument("--bin-include", default=None)
ap.add_argument("--total-tokens", type=float, default=0,
                help="stream size; a domain whose bins hold fewer than "
                     "weight*total/4 tokens (the 4-epoch cap) is folded too")
args = ap.parse_args()
mix = json.load(open(args.mix))
weights = {k: v for k, v in mix.items() if not k.startswith("_")}
inc = re.compile(args.bin_include) if args.bin_include else None

tokens = {}          # domain -> uint16 tokens available
if args.domains:
    for p in glob.glob(os.path.join(args.domains, "*.bin")):
        tokens[os.path.basename(p)[:-4]] = tokens.get(os.path.basename(p)[:-4], 0) + os.path.getsize(p) // 2
if args.shards:
    for b in glob.glob(os.path.join(args.shards, "*", "*", "*.bin")):
        if inc and not inc.search(b):
            continue
        sdom = b.split(os.sep)[-3]
        dom = sdom
        while dom not in weights and "_" in dom:
            dom = dom.rsplit("_", 1)[0]
        if dom in weights:
            tokens[dom] = tokens.get(dom, 0) + os.path.getsize(b) // 2
have = {d for d, t in tokens.items()
        if t > 0 and (not args.total_tokens or t >= weights[d] * args.total_tokens / 4)}

def sibling(d):
    if d.startswith("indic_"):
        return "indic_hin"
    if d.startswith("india_"):
        return "finance_econ_law"
    return "global_english"

out = dict(weights)
moved = []
for d, w in weights.items():
    if d not in have:
        s = sibling(d)
        if s == d or s not in have:
            s = "global_english"
        out[s] = out.get(s, 0.0) + w
        del out[d]
        moved.append(f"{d} ({w:.3f}, {tokens.get(d, 0)/1e6:.0f}M tokens) -> {s}")
tot = sum(out.values())
assert abs(tot - 1.0) < 1e-9, tot
dst = args.mix.replace(".json", "-launch.json")
json.dump({"_comment": f"LAUNCH variant of {os.path.basename(args.mix)} built from data reality "
                       f"({len(have)} domains with bins). Moved: {'; '.join(moved) or 'nothing'}.",
           **out}, open(dst, "w"), indent=2)
print(f"{dst}: {len(out)} domains; moved {len(moved)}: {'; '.join(moved) or 'nothing'}")
