"""NEURAL INTERFACE REBUILD, step 11: resolve motor-neuron pools from the RIGHT column.

THE FINDING THIS IMPLEMENTS
---------------------------
Rounds 2-36 resolved motor-neuron pools from `banc_connectome.npz`'s coarse `ann_primary_type`
plus an `ann_nerve` suffix filter.  Measured consequence: 22/90 actuators resolved in round 2,
44/90 after the anatomy pass.

The same dataset ships `data/banc888/banc_888_meta.feather` (81 columns, 188,508 rows) with a
PER-NEURON MUSCLE TARGET field, `peripheral_target_type` (e.g. `trochanter_flexor_muscle`), plus
`body_part_effector` (front_leg / middle_leg / hind_leg / ...) and `side` (left/right).  None of
rounds 2-36 read those three columns.

WHAT THIS SCRIPT DOES (read-only on every input)
------------------------------------------------
  1. reads the 90 actuator names from the canonical asset;
  2. normalises each name to a muscle target with a DECLARED alias table;
  3. looks up the pool as the set of motor neurons whose (target, leg, side) matches;
  4. reports coverage, and reports separately every actuator that does NOT get a pool of its own
     leg - including the eight middle/hind tergopleural actuators that the current runtime wires
     to FRONT-leg pools (`tools_build_design_runtime.py:151` takes a pool by side only, with no
     leg term), and the six `pleural_promotor` actuators whose target does not exist anywhere in
     BANC because the annotation merges Miller 28 and 30 into one label.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not silently substitute one leg's pool for another's.  An actuator whose own leg has no
annotated pool is reported as UNRESOLVED, with the reason.  Round 39's lesson was that a label
which does not describe its content is worse than an admitted gap.

Output: outputs/neural_rebuild/neural_rebuild_11_mn_pools.json, plus a per-leg wiring report.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
FEATHER = HERE / "data/banc888/banc_888_meta.feather"

# DECLARED alias table: model actuator name -> BANC peripheral_target_type.
# Two of these were already written down by round 2's own manifest under
# `declared_aliases_not_accepted`; they are accepted here, and that choice is declared.
ALIASES = {
    "accesory_trochanter_flexor": "accessory_trochanter_flexor_muscle",
    "trochanter_flexor": "trochanter_flexor_muscle",
    "trochanter_flexor_a": "trochanter_flexor_muscle",
    "trochanter_flexor_b": "trochanter_flexor_muscle",
    "trochanter_extensor": "trochanter_extensor_muscle",
    "accessory_trochanter_flexor": "accessory_trochanter_flexor_muscle",
    "tibia_flex": "tibia_flexor_muscle",
    "tibia_extensor": "tibia_extensor_muscle",
    "tergopleural_promotor_a": "tergopleural_promotor_muscle",
    "tergopleural_promotor_b": "tergopleural_promotor_muscle",
    "pleural_remotor_and_abductor": "pleural_remotor_and_abductor_muscle",
    "pleural_promotor": "pleural_promotor_muscle",
    "sternal_anterior_rotator": "sternal_anterior_rotator_muscle",
    "sternal_posterior_rotator": "sternal_posterior_rotator_muscle",
    "sternal_adductor": "sternal_adductor_muscle",
    # the model merges two muscles into one name; both halves exist in BANC
    "sterno-tergo-trochanter_extensor_a": "sternotrochanter_extensor_muscle",
    "sterno-tergo-trochanter_extensor_b": "tergotrochanter_extensor_muscle",
}
LEG_OF = {"LF": "front_leg", "LM": "middle_leg", "LH": "hind_leg",
          "RF": "front_leg", "RM": "middle_leg", "RH": "hind_leg"}
SIDE_OF = {c: ("left" if c == "L" else "right") for c in "LR"}
JOINT_NAMES = ("coxa", "trochanter", "femur", "tibia", "tarsus")


def parse_actuator(name: str) -> dict:
    """Two naming patterns exist in the canonical asset (measured: 78 + 12):

        LFC_tergopleural_promotor_a   -> leg LF, side L, part C, muscle tergopleural_promotor_a
        LFTibia_flex_93434            -> leg LF, side L, part (tibia), muscle tibia_flex
        LFTibia_extensor_93932        -> leg LF, side L, part (tibia), muscle tibia_extensor

    A first version matched only the first pattern and reported the 12 tibia actuators as "name
    does not parse", which understated the coverage by 12.  That is recorded because it is the
    third time in this rebuild that a label/naming mismatch was mistaken for missing data.
    """
    m = re.match(r"^([LR])([FMH])([CFT])_(.+)$", name)
    if m:
        side_c, leg_c, part, muscle = m.groups()
        return {"ok": True, "name": name, "leg": f"{side_c}{leg_c}", "side": SIDE_OF[side_c],
                "body_part": LEG_OF[f"{side_c}{leg_c}"], "part": part, "muscle": muscle}
    m2 = re.match(r"^([LR])([FMH])Tibia_([a-z]+)_(\d+)$", name)
    if m2:
        side_c, leg_c, action, _id = m2.groups()
        return {"ok": True, "name": name, "leg": f"{side_c}{leg_c}", "side": SIDE_OF[side_c],
                "body_part": LEG_OF[f"{side_c}{leg_c}"], "part": "T",
                "muscle": f"tibia_{action}"}
    return {"ok": False, "name": name}


def main() -> None:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    if digest != cp.CANONICAL_XML_SHA256:
        raise SystemExit(f"canonical asset changed: {digest}")
    stored = json.loads((OUT / "neural_rebuild_11_mn_pools.json").read_text()) \
        if (OUT / "neural_rebuild_11_mn_pools.json").exists() else None

    xml = cp.CANONICAL_XML.read_text()
    act_names = re.findall(r'<general name="([^"]+)"', xml)
    print(f"actuators in the canonical asset: {len(act_names)}", flush=True)

    d = pd.read_feather(FEATHER)
    tgt = np.array([str(x) for x in d["peripheral_target_type"].to_numpy()])
    eff = np.array([str(x) for x in d["body_part_effector"].to_numpy()])
    sd = np.array([str(x) for x in d["side"].to_numpy()])
    flow = np.array([str(x) for x in d["flow"].to_numpy()])
    root = d["root_id"].to_numpy()
    ctype = np.array([str(x) for x in d["cell_type"].to_numpy()])

    efferent = flow == "efferent"
    # (target, body_part, side) -> pool
    pool = defaultdict(list)
    for i in np.flatnonzero(efferent):
        pool[(tgt[i], eff[i], sd[i])].append(i)

    resolved, unresolved = [], []
    per_leg = defaultdict(lambda: {"resolved": 0, "unresolved": 0})
    for name in act_names:
        p = parse_actuator(name)
        if not p["ok"]:
            unresolved.append({"actuator": name, "reason": "name does not parse"})
            continue
        target = ALIASES.get(p["muscle"])
        if target is None:
            unresolved.append({"actuator": name, "reason": "no declared alias for muscle",
                               "muscle": p["muscle"]})
            per_leg[p["leg"]]["unresolved"] += 1
            continue
        key = (target, p["body_part"], p["side"])
        rows = pool.get(key, [])
        if rows:
            resolved.append({
                "actuator": name, "leg": p["leg"], "side": p["side"],
                "muscle_target": target,
                "n_pool": len(rows),
                "pool_root_ids": [int(root[i]) for i in rows],
                "pool_cell_types": dict(Counter(ctype[i] for i in rows).most_common(4)),
            })
            per_leg[p["leg"]]["resolved"] += 1
        else:
            # does the target exist at all, and on which legs?  Report rather than substitute.
            where = Counter(eff[i] for i in np.flatnonzero(tgt == target))
            unresolved.append({
                "actuator": name, "leg": p["leg"], "side": p["side"],
                "muscle_target": target,
                "reason": ("target absent from BANC entirely" if not where
                           else "target exists but not on this leg/side"),
                "target_exists_on": dict(where),
            })
            per_leg[p["leg"]]["unresolved"] += 1

    n_res = len(resolved)
    n_unres = len(unresolved)
    by_reason = Counter(
        ("target absent from BANC entirely" if u.get("target_exists_on") == {}
         else u.get("reason", "?")) for u in unresolved)

    # cross-leg substitution the CURRENT runtime performs, for the record
    cross_leg = [{"actuator": u["actuator"], "would_be_wired_to": u["target_exists_on"]}
                 for u in unresolved if u.get("target_exists_on")]

    cp.write_result(
        OUT / "neural_rebuild_11_mn_pools.json",
        question=("How many of the 90 actuators get a motor-neuron pool on their OWN leg and "
                  "side when the resolver reads `peripheral_target_type` + `body_part_effector` "
                  "+ `side` instead of `ann_primary_type` + a nerve-suffix filter?"),
        status="PASS",
        measured={
            "source": {"feather": str(FEATHER), "rows": int(len(d))},
            "columns_used": ["peripheral_target_type", "body_part_effector", "side", "flow",
                             "root_id", "cell_type"],
            "n_actuators": len(act_names),
            "n_resolved_own_leg": n_res,
            "n_unresolved": n_unres,
            "coverage_fraction": n_res / len(act_names),
            "comparison_to_history": {
                "round2_lexical": "22/90 = 0.244",
                "anatomy_resolution": "44/90 = 0.489",
                "this_step": f"{n_res}/{len(act_names)} = {n_res/len(act_names):.3f}",
            },
            "per_leg": {k: dict(v) for k, v in sorted(per_leg.items())},
            "unresolved_by_reason": dict(by_reason),
            "unresolved": unresolved,
            "cross_leg_substitution_the_old_runtime_performs": cross_leg,
            "resolved": resolved,
            "alias_table": ALIASES,
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_11_mn_pools.py"),
            "read_only": True,
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
            "old_runtime_defect": "tools_build_design_runtime.py:151 takes the tergopleural pool "
                                  "by side only, with no leg term, so the 8 middle/hind "
                                  "tergopleural actuators are driven by FRONT-leg neurons",
        },
        what_this_does_not_support=[
            "A resolved pool is a SET OF NEURONS whose annotation names that muscle; it is not a "
            "validated 1:1 innervation, and no transmission sign is claimed.",
            "The alias table is a DECLARED human decision (17 entries). Two of them were already "
            "written down by round 2's manifest; accepting them is a choice, and it is declared "
            "here rather than hidden.",
            "Actuators reported unresolved are NOT substituted: this step refuses to wire one "
            "leg's pool to another leg's muscle, which is what the old runtime did.",
            "No dynamics, no body and no closed loop are involved: this is a wiring table.",
        ],
    )

    print(f"resolved on own leg/side: {n_res}/{len(act_names)} "
          f"({n_res/len(act_names):.3f})   unresolved: {n_unres}", flush=True)
    print("unresolved by reason:", dict(by_reason), flush=True)
    for u in unresolved[:12]:
        print(f"   {u['actuator']:44s} {u.get('reason')} "
              f"{u.get('target_exists_on','')}", flush=True)
    print("wrote", OUT / "neural_rebuild_11_mn_pools.json")


if __name__ == "__main__":
    main()
