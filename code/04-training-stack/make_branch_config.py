#!/usr/bin/env python3
"""Write the WSD decay-branch config for a running trunk.

  python make_branch_config.py <trunk_config.json> <trunk_ckpt.pt> <decay_steps> [cooldown_data_dir]

The branch continues weights+optimizer from the checkpoint (train.py
--branch-from) and decays the LR linearly to min_lr over decay_steps; the
cooldown curriculum stream (if given) switches in at the branch start.
"""
import json
import os
import sys

import torch

cfg_path, ckpt_path, decay = sys.argv[1], sys.argv[2], int(sys.argv[3])
cooldown = sys.argv[4] if len(sys.argv) > 4 else ""
c = json.load(open(cfg_path))
step = torch.load(ckpt_path, map_location="cpu", weights_only=False)["step"]
start = step + 1
c["_comment"] = (f"DECAY BRANCH of {cfg_path} from {ckpt_path} (step {step}): "
                 f"linear LR decay over {decay} steps; "
                 + (f"cooldown stream {cooldown} from the branch start." if cooldown
                    else "no cooldown stream."))
c["lr_schedule"] = "wsd"
c["decay_steps"] = decay
c["max_steps"] = start + decay
c["out_dir"] = c["out_dir"].rstrip("/") + f"-decay{step}"
if cooldown:
    c["cooldown_data_dir"] = cooldown
    c["cooldown_start_frac"] = start / c["max_steps"]
out = cfg_path.replace(".json", "-decay.json")
json.dump(c, open(out, "w"), indent=2)
print(f"{out}: steps {start}..{c['max_steps']-1}, out_dir {c['out_dir']}")
print(f"launch: python train.py --config {out} --branch-from {ckpt_path}")
