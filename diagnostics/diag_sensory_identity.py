#!/usr/bin/env python3
"""WHAT ARE THE CELLS THE LOAD CHANNEL DRIVES?  Check the annotations, do not assume.

The reflex matrix shows the per-leg sensory groups are annotated SNta and SNch.  Driving them with a
load signal is only defensible if those are load-sensing cells, so this prints every annotation column
for the groups and for their top types.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
from engine.embodied.adapters import build_neural_tier, TierConfig, LEGS, LEG_NERVE  # noqa: E402

COLS = ("ann_super_class", "ann_class", "ann_primary_type", "ann_nerve", "ann_body_part",
        "ann_function", "ann_flow")


def main():
    built = build_neural_tier(TierConfig())
    z = np.load(HERE / "data" / "flywire" / "banc_connectome.npz", allow_pickle=True)
    roots = np.asarray(z["root_ids"])
    order = {int(r): i for i, r in enumerate(roots)}
    tier_roots = np.asarray(built["tier"]["root_ids"])
    ann = {c: np.asarray(z[c]) for c in COLS if c in z}
    out = {}
    for leg in LEGS:
        loc = np.asarray(built["sensory"][leg])
        idx = np.array([order.get(int(r), -1) for r in tier_roots[loc]])
        idx = idx[idx >= 0]
        print(f"\n=== {leg} ({LEG_NERVE[leg]}): {loc.size} mechanosensory neurons in the tier ===")
        rec = {"n_in_tier": int(loc.size)}
        for c in COLS:
            if c not in ann:
                continue
            vals, cnt = np.unique(ann[c][idx], return_counts=True)
            top = sorted(zip(cnt, vals), reverse=True)[:5]
            rec[c] = {str(v): int(n) for n, v in top}
            print(f"  {c:<18}" + ", ".join(f"{v}×{n}" for n, v in top))
        out[leg] = rec
    # the decisive question: are the dominant classes mechanosensory and are the types tarsal or
    # chordotonal (load/position sensors of the leg), or something else entirely?
    print("\n=== the dominant primary types across all six legs, with their class ===")
    allidx = []
    for leg in LEGS:
        loc = np.asarray(built["sensory"][leg])
        for r in tier_roots[loc]:
            i = order.get(int(r), -1)
            if i >= 0:
                allidx.append(i)
    allidx = np.array(allidx)
    vals, cnt = np.unique(ann["ann_primary_type"][allidx], return_counts=True)
    for n, v in sorted(zip(cnt, vals), reverse=True)[:12]:
        m = ann["ann_primary_type"][allidx] == v
        cls, _c2 = np.unique(ann["ann_class"][allidx][m], return_counts=True)
        fn, _c3 = np.unique(ann["ann_function"][allidx][m], return_counts=True)
        print(f"  {v:<10} ×{n:<5} class={list(cls)[:2]} function={list(fn)[:2]}")
    (HERE / "outputs" / "sensory_identity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'sensory_identity.json'}")


if __name__ == "__main__":
    main()
