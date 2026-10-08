#!/usr/bin/env python3
"""PER-LEG motor-neuron -> muscle map: 6 leg nerves x muscle-named MN types x neuron ids.

WHY THIS IS THE ARTEFACT THAT UNBLOCKS THE MOTOR PATH.  The biomechanical muscle model ships for
the FORELAG ONLY (Ozdi et al., ICLR 2026, arXiv 2509.06426: "We modeled 15 muscle-tendon units
(MTUs) per foreleg"), and no other project models the remaining legs.  But the CONNECTOME has
muscle-specific motor neurons for ALL SIX legs already, partitioned by leg nerve -- measured here:
29 / 29 / 45 / 46 / 48 / 46 muscle-named motor neurons for the six leg nerves, 243 in total.  So
the neural side of the motor path is complete; only the muscle actuators are missing, and the same
paper supplies the warrant for mirroring them: "the muscle structure across legs is nearly
identical, with a few exceptions including tergal depressor of the trochanter (TDT) muscles in the
middle legs".

PROVENANCE: the join is by NOMENCLATURE (connectome MN type names vs muscle names), so it is
DERIVED, not a cited table.  The authoritative table is Azevedo et al., 'Tools for comprehensive
reconstruction and analysis of Drosophila motor circuits' (bioRxiv 2022.12.15.520299).

Run:
    OPENBLAS_NUM_THREADS=1 venv/bin/python tools_mn_muscle_map_per_leg.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools_mn_muscle_map import muscle_actuators, tokens, canon  # noqa: E402

BANC = HERE / "data/flywire/banc_connectome.npz"

#: leg nerve -> (side letter used by the muscle model, leg letter F/M/H)
NERVE_TO_LEG = {
    "left_prothoracic_leg_nerve": ("L", "F"),
    "right_prothoracic_leg_nerve": ("R", "F"),
    "left_mesothoracic_leg_nerve": ("L", "M"),
    "right_mesothoracic_leg_nerve": ("R", "M"),
    "left_metathoracic_leg_nerve": ("L", "H"),
    "right_metathoracic_leg_nerve": ("R", "H"),
}
KW = ("promotor", "remotor", "rotator", "adductor", "abductor", "flex", "extend",
      "depressor", "levator", "trochanter", "tibia", "tarsus", "sternal", "tergal",
      "tergopleural", "pleural", "tergotrochanter", "sternotrochanter")


def main() -> int:
    z = np.load(BANC, allow_pickle=False)
    nerve = np.asarray(z["ann_nerve"]); prim = np.asarray(z["ann_primary_type"])
    root = np.asarray(z["root_ids"]); cls = np.asarray(z["ann_class"])
    muscles = muscle_actuators()

    def is_muscle_named(s) -> bool:
        s = str(s)
        return bool(s) and not s.startswith("MN") and not s.endswith("_neuron") \
            and any(k in s for k in KW)

    table = {}
    print(f"{'leg':<8}{'nerve':<34}{'muscle-named MNs':>17}")
    for n, (side, leg) in NERVE_TO_LEG.items():
        m = (nerve == n) & np.array([is_muscle_named(s) for s in prim])
        per_type = {}
        for t in np.unique(prim[m]):
            sel = m & (prim == t)
            per_type[str(t)] = {"n": int(sel.sum()),
                                "ids": [int(x) for x in root[sel][:64]]}
        table[f"{side}{leg}"] = {"nerve": n, "n_muscle_named_mn": int(m.sum()),
                                 "by_type": per_type,
                                 "all_mn_like": int(((nerve == n) &
                                                     np.array([is_muscle_named(s) or
                                                               str(s).startswith("MN")
                                                               for s in prim])).sum())}
        print(f"{side + leg:<8}{n:<34}{int(m.sum()):>17}")

    # for each model muscle, which per-leg MN groups can drive it
    print(f"\n{'muscle actuator':<40}" + "".join(f"{k:>8}" for k in table))
    link = []
    for mus in muscles:
        mt = {canon(x) for x in tokens(mus)}
        row = {"muscle": mus, "per_leg": {}}
        cells = []
        for legkey, info in table.items():
            best, bestj = None, 0.0
            for t, d in info["by_type"].items():
                tt = {canon(x) for x in tokens(t)}
                if not tt:
                    continue
                j = len(mt & tt) / len(mt | tt)
                if j > bestj:
                    best, bestj = t, j
            if bestj >= 0.40:
                row["per_leg"][legkey] = {"mn_type": best, "jaccard": round(bestj, 3),
                                          "ids": info["by_type"][best]["ids"]}
                cells.append(f"{info['by_type'][best]['n']:>8}")
            else:
                row["per_leg"][legkey] = None
                cells.append(f"{'-':>8}")
        link.append(row)
        print(f"{mus:<40}" + "".join(cells))

    dest = HERE / "outputs" / "mn_muscle_map_per_leg.json"
    dest.write_text(json.dumps({
        "provenance": ("DERIVED by naming; the muscle model is FORELAG-ONLY (Ozdi et al., ICLR "
                       "2026, arXiv 2509.06426) while the connectome has per-leg muscle-specific "
                       "motor neurons for all six legs.  Authoritative MN->muscle table to check "
                       "against: bioRxiv 2022.12.15.520299."),
        "n_legs_with_motor_neurons": len(table),
        "total_muscle_named_mn": sum(v["n_muscle_named_mn"] for v in table.values()),
        "legs": table, "muscle_to_legs": link,
        "mirroring_warrant": ("Ozdi et al. state: 'the muscle structure across legs is nearly "
                              "identical, with a few exceptions including tergal depressor of "
                              "the trochanter (TDT) muscles in the middle legs that facilitate "
                              "jump escape' -- this is the licence for mirroring the 15 foreleg "
                              "MTUs to the other five legs, with those exceptions recorded."),
        "model_exclusions_to_carry_over": [
            "no muscles housed in the tibia (the tibia was only partially captured in the X-ray data)",
            "no trochanter muscles (function unclear)",
            "no tergal depressor of the trochanter (TDT), which exists in the middle legs in vivo",
            "15 MTUs per leg modelled where the real leg has approximately 19 muscles and 69 MNs",
        ],
    }, indent=2, sort_keys=True))
    print(f"\ntotal muscle-named motor neurons across six legs: "
          f"{sum(v['n_muscle_named_mn'] for v in table.values())}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
