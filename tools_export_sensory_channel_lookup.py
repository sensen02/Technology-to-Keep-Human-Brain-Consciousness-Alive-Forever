"""Export the root_id -> (leg, channel) sensory lookup, once.

WHY A CACHE
-----------
The channel partition comes from the dataset's `cell_sub_class` column in
`data/banc888/banc_888_meta.feather`.  Reading that file needs pandas/pyarrow, and in this
project the interpreter that can run MuJoCo does NOT have pandas (measured: `venv_body` has
mujoco but no pandas; `venv` has pandas but no mujoco).  Rather than make the runtime depend on
a second environment, the published mapping is exported here ONCE and read as plain JSON by
`engine/embodied/sensory_channels.py`.

The export is a declared step, not a hidden one: the JSON records the source file, its sha256,
and the rule used, so a reader can tell exactly which dataset version the channels came from.

Run with an interpreter that has pandas:  venv/bin/python tools_export_sensory_channel_lookup.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FEATHER = HERE / "data/banc888/banc_888_meta.feather"
OUT = HERE / "outputs/neural_rebuild/sensory_channel_lookup.json"

CHANNEL_SPEC = {
    "claw_position": "claw_chordotonal_organ_neuron",
    "hook_direction": "hook_chordotonal_organ_neuron",
    "hairplate_angle": "hair_plate_neuron",
    "campaniform_strain": "campaniform_sensillum_neuron",
    "club_vibration": "club_chordotonal_organ_neuron",
    "bristle_touch": "bristle_neuron",
    "multidendritic_noxious": "multidendritic_neuron",
}
LEG_CODE = {"front_leg": "F", "middle_leg": "M", "hind_leg": "H"}
SIDE_PREFIX = {"left": "L", "right": "R"}


def main() -> None:
    import pandas as pd

    if not FEATHER.exists():
        raise SystemExit(f"missing {FEATHER}")
    h = hashlib.sha256(FEATHER.read_bytes()).hexdigest()
    d = pd.read_feather(FEATHER)
    sub = [str(x) for x in d["cell_sub_class"].to_numpy()]
    side = [str(x) for x in d["side"].to_numpy()]
    root = d["root_id"].to_numpy()

    by_root = {}
    counts = Counter()
    for i, s in enumerate(sub):
        leg = None
        for body_name, code in LEG_CODE.items():
            if s.startswith(body_name):
                pre = SIDE_PREFIX.get(side[i])
                if pre:
                    leg = f"{pre}{code}".lower()  # lowercase: matches adapters.LEGS
                break
        if leg is None:
            continue
        ch = next((c for c, substr in CHANNEL_SPEC.items() if substr in s), None)
        if ch is None:
            continue
        by_root[str(int(root[i]))] = [leg, ch]
        counts[f"{leg}/{ch}"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "question": "which (leg, channel) does each leg sensory cell belong to?",
        "status": "PASS",
        "rule": ("cell_sub_class must start with front_leg / middle_leg / hind_leg, and contain "
                 "one of the seven subtype strings; the leg is formed from the body-part prefix "
                 "AND the side column, because the prefix alone carries no side (a first version "
                 "used the prefix only and filed every right-leg cell under the left)"),
        "channels": CHANNEL_SPEC,
        "source": {"file": str(FEATHER), "sha256": h, "rows": int(len(d))},
        "n_cells_with_a_channel": len(by_root),
        "counts": dict(counts),
        "leg_channel_by_root": by_root,
        "what_this_does_not_support": [
            "A channel assignment comes from an ANNOTATION column, not from a measurement of "
            "what the cell senses; within a subtype the dataset gives no per-cell difference.",
            "No sign is attached: the sign convention is a declared human decision made "
            "downstream.",
        ],
    }, indent=0))
    print(f"cells with a channel: {len(by_root)}")
    for k in sorted(counts):
        print(f"  {k:34s} {counts[k]}")
    print("source sha256", h)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
