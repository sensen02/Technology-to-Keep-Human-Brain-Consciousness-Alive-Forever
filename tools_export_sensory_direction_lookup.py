"""NEURAL INTERFACE REBUILD, step 25: DERIVE the antagonistic partition from the data, instead of
splitting by index parity.

WHY
---
Step 21 split each signed sensory channel into a "+" and a "-" group by index PARITY, and declared
that choice as arbitrary because the dataset carries no per-cell sign.  Step 24 then measured the
consequence at the motor neurons: a positive and a negative deviation both INCREASED the motor
population's spike count (1611 vs 1804), so the sign did not survive to the motor command.

But the dataset is not sign-free after all.  `data/banc888/banc_888_meta.feather` labels motor
neurons with a PER-JOINT ACTION:

    flex_femur_tibia_joint 101   /  extend_femur_tibia_joint 12
    flex_coxa_trochanter_joint 62 /  extend_coxa_trochanter_joint 48
    flex_tibia_tarsus_joint 6    /  extend_tibia_tarsus_joint 24

So a sensory cell's direction can be READ OFF ITS OWN CONNECTOME: the ratio of its synapses into a
joint's flexor pool versus that joint's extensor pool.  Cells that project to the flexor side are
one half of the antagonistic pair; cells that project to the extensor side are the other.  That is
the data-derived partition that step 21 lacked, and step 7's failure was that it formed ONE axis per
LEG instead of one per JOINT.

WHAT THIS SCRIPT DOES
---------------------
Exports, for every leg sensory cell, its signed direction = (synapses into the same leg's flexor
pools - into the extensor pools) / total, per joint, summed over the joints it scores on.  The
result is a lookup that `AntagonisticEncoder` can split on: `+` group = cells with a positive
derived direction, `-` group = cells with a negative one, and cells with a ZERO direction (balanced
projection) go to neither and are reported.

Run with an interpreter that has pandas: venv/bin/python tools_export_sensory_direction_lookup.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FEATHER = HERE / "data/banc888/banc_888_meta.feather"
NPZ = HERE / "data/flywire/banc_connectome.npz"
OUT = HERE / "outputs/neural_rebuild/sensory_direction_lookup.json"

LEG_BODY = {"lf": "front_leg", "lm": "middle_leg", "lh": "hind_leg",
            "rf": "front_leg", "rm": "middle_leg", "rh": "hind_leg"}
SIDE_PREFIX = {"left": "L", "right": "R"}
LEG_CODE = {"front_leg": "F", "middle_leg": "M", "hind_leg": "H"}
FLEX = re.compile(r"flex", re.I)
EXTEND = re.compile(r"extend", re.I)
JOINTS = ("femur_tibia", "coxa_trochanter", "tibia_tarsus")


def main() -> None:
    import pandas as pd

    if not FEATHER.exists():
        raise SystemExit(f"missing {FEATHER}")
    fh = hashlib.sha256(FEATHER.read_bytes()).hexdigest()

    d = pd.read_feather(FEATHER)
    root = d["root_id"].to_numpy()
    fun = np.array([str(x) for x in d["cell_function_detailed"].to_numpy()])
    sub = np.array([str(x) for x in d["cell_sub_class"].to_numpy()])
    side = np.array([str(x) for x in d["side"].to_numpy()])

    # ---- per-joint flexor / extensor motor pools, per leg and side -------------
    pools = defaultdict(lambda: defaultdict(list))
    for i, f in enumerate(fun):
        for joint in JOINTS:
            is_flex = any(FLEX.search(f) and joint in f and "extend" not in f for _ in [0])
            is_ext = EXTEND.search(f) is not None and joint in f
            if FLEX.search(f) and joint in f and "extend" not in f:
                is_flex = True
            if not (is_flex or is_ext):
                continue
            ident = None
            for body_name, code in LEG_CODE.items():
                if sub[i].startswith(body_name):
                    pre = SIDE_PREFIX.get(side[i])
                    if pre:
                        ident = f"{pre}{code}"
                    break
            if ident is None:
                continue
            pools[joint][(ident, "flex" if is_flex else "extend")].append(int(root[i]))

    # ---- connectivity: synapses from sensory cells into those pools ------------
    z = np.load(NPZ, allow_pickle=True)
    pre, post, syn = z["pre"], z["post"], z["syn"]
    n = int(z["root_ids"].size)
    row_of_root = {int(r): i for i, r in enumerate(np.asarray(z["root_ids"]))}
    W = sp.csr_matrix((syn.astype(float), (post, pre)), shape=(n, n))

    # sensory cells = the leg sensory sub classes we already use as channels
    CH_SUB = {"claw_chordotonal_organ_neuron": "claw", "hook_chordotonal_organ_neuron": "hook",
              "hair_plate_neuron": "hair_plate"}
    sens_rows, sens_meta = [], {}
    for i, s in enumerate(sub):
        ident = None
        for body_name, code in LEG_CODE.items():
            if s.startswith(body_name):
                pre_ = SIDE_PREFIX.get(side[i])
                if pre_:
                    ident = f"{pre_}{code}"
                break
        if ident is None:
            continue
        ch = next((v for k, v in CH_SUB.items() if k in s), None)
        if ch is None:
            continue
        sens_rows.append(i)
        sens_meta[int(root[i])] = {"leg": ident, "channel": ch}

    direction = {}
    zero_direction = 0
    per_leg_channel = Counter()
    for i in sens_rows:
        r = int(root[i])
        meta = sens_meta[r]
        leg = meta["leg"]
        row = row_of_root.get(r)
        if row is None:
            continue
        total_f = total_e = 0.0
        for joint in JOINTS:
            fids = pools[joint].get((leg, "flex"), [])
            eids = pools[joint].get((leg, "extend"), [])
            fr = [row_of_root[x] for x in fids if x in row_of_root]
            er = [row_of_root[x] for x in eids if x in row_of_root]
            if fr:
                total_f += float(W[fr, row].sum())
            if er:
                total_e += float(W[er, row].sum())
        tot = total_f + total_e
        sgn = ((total_f - total_e) / tot) if tot > 0 else 0.0
        if abs(sgn) < 1e-9:
            zero_direction += 1
        direction[str(r)] = {"leg": leg, "channel": meta["channel"],
                             "signed_direction": sgn,
                             "to_flexor": total_f, "to_extensor": total_e}
        per_leg_channel[f"{leg}/{meta['channel']}"] += 1

    pos = sum(1 for v in direction.values() if v["signed_direction"] > 0)
    neg = sum(1 for v in direction.values() if v["signed_direction"] < 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "question": "what is each leg sensory cell's derived directional sign?",
        "status": "PASS",
        "rule": ("signed_direction = (synapses into the same leg's PER-JOINT flexor pools - into "
                 "the extensor pools) / total, summed over the three joint action pairs the "
                 "dataset labels: flex/extend femur_tibia, coxa_trochanter, tibia_tarsus"),
        "source": {"feather": str(FEATHER), "feather_sha256": fh, "npz": str(NPZ)},
        "joint_pools": {j: {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(pools[j].items())}
                        for j in pools},
        "n_cells": len(direction),
        "n_positive_direction": pos, "n_negative_direction": neg,
        "n_zero_direction": zero_direction,
        "counts_by_leg_channel": dict(per_leg_channel),
        "direction_by_root": direction,
        "what_this_does_not_support": [
            "The sign is a NAME-based reading: the dataset labels motor neuron ACTIONS "
            "(flex_femur_tibia_joint) and this script turns that into a direction; it does not "
            "prove the animal's convention.",
            "A synapse count carries no transmission sign in BANC, so projecting to the flexor "
            "pool is a functional assertion, not an excitatory one.",
        ],
    }, indent=0))
    print(f"cells with a derived direction: {len(direction)}  "
          f"(positive {pos}, negative {neg}, zero {zero_direction})")
    for j in JOINTS:
        print(f"  {j}: " + ", ".join(f"{k[0]}/{k[1]}={len(v)}"
                                     for k, v in sorted(pools[j].items())))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
