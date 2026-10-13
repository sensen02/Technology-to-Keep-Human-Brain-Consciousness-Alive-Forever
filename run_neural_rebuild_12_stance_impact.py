"""NEURAL INTERFACE REBUILD, step 12: what the corrected MN mapping does to a real stance.

WHY
---
Step 11 resolved 74 of 90 actuators to a motor-neuron pool on their OWN leg and side (against
22/90 for round 2's lexical pass and 44/90 after the anatomy pass), and refused to substitute
one leg's pool for another's.  Sixteen actuators stay unresolved: 6 whose muscle target does not
exist anywhere in BANC (`pleural_promotor`, whose annotation merges Miller 28 and 30 into the
tergopleural label) and 10 whose target exists but not on that leg (the middle/hind tergopleural
pairs, one middle sternal adductor, one right-middle pleural remotor).

The engineering question is: if the runtime used the corrected mapping and honestly zeroed what
it cannot source, HOW MUCH of a feasible stance's activation would it lose?  That number decides
whether the corrected wiring is usable or whether the geometry must be re-searched.

WHAT IS MEASURED
----------------
For each of the project's own feasible zero-adhesion stances (the round-37 margin pose and the
round-36 search pose), partition the activation vector by:
    A  actuator has a pool on its own leg and side            -> sourced
    B  actuator currently driven by ANOTHER leg's pool        -> cross-leg substitution (defect)
    C  actuator has no pool anywhere                          -> unsourced
and report the activation share of each.  B is the group the old runtime wires wrongly; C is the
group it holds at zero.

Read-only.  The canonical asset is not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
POOLS = OUT / "neural_rebuild_11_mn_pools.json"
STANCES = {
    "round37_margin_pose": HERE / "outputs/stance_margin_round37/stance_margin.json",
    "round36_search_pose": HERE / "outputs/muscle_only_stance_round36/design_motor_pools.json",
}


def activation_of(path) -> dict:
    d = json.loads(path.read_text())
    if "best" in d and isinstance(d["best"], dict) and "activation" in d["best"]:
        return {k: float(v) for k, v in d["best"]["activation"].items()}
    # the round-36 artifact nests the search pose differently
    try:
        a = d["physics"]["design_search_over_poses"]["best"]["activation"]
        return {k: float(v) for k, v in a.items()}
    except Exception:
        return {}


def main() -> None:
    pools = json.loads(POOLS.read_text())["measured"]
    resolved = {r["actuator"] for r in pools["resolved"]}
    unresolved = {u["actuator"] for u in pools["unresolved"]}
    cross = {c["actuator"] for c in pools["cross_leg_substitution_the_old_runtime_performs"]}

    rows = {}
    for label, path in STANCES.items():
        if not path.exists():
            rows[label] = {"note": "artifact missing"}
            continue
        act = activation_of(path)
        if not act:
            rows[label] = {"note": "no activation vector found in this artifact"}
            continue
        groups = {"A_own_leg_pool": [], "B_cross_leg_substitution": [], "C_unsourced": []}
        for k, v in act.items():
            if k in resolved:
                groups["A_own_leg_pool"].append((k, v))
            elif k in cross:
                groups["B_cross_leg_substitution"].append((k, v))
            elif k in unresolved:
                groups["C_unsourced"].append((k, v))
        total = sum(v for _, v in act.items())
        rows[label] = {
            "n_actuators_with_activation": len(act),
            "total_activation": total,
            "groups": {
                g: {"n": len(items), "n_nonzero": sum(1 for _, v in items if v > 0),
                    "activation_sum": sum(v for _, v in items),
                    "share": (sum(v for _, v in items) / total) if total else None,
                    "nonzero_members": [{"actuator": k, "activation": v}
                                        for k, v in items if v > 0]}
                for g, items in groups.items()
            },
        }
        r = rows[label]
        print(f"{label}: total activation {total:.4f} over "
              f"{r['n_actuators_with_activation']} actuators", flush=True)
        for g in ("A_own_leg_pool", "B_cross_leg_substitution", "C_unsourced"):
            d = r["groups"][g]
            print(f"   {g:28s} n={d['n']:2d} nonzero={d['n_nonzero']:2d} "
                  f"sum={d['activation_sum']:.4f} share={d['share']:.4f}", flush=True)

    a_shares = [v["groups"]["A_own_leg_pool"]["share"] for v in rows.values()
                if isinstance(v, dict) and "groups" in v]
    b_shares = [v["groups"]["B_cross_leg_substitution"]["share"] for v in rows.values()
                if isinstance(v, dict) and "groups" in v]
    c_shares = [v["groups"]["C_unsourced"]["share"] for v in rows.values()
                if isinstance(v, dict) and "groups" in v]

    cp.write_result(
        OUT / "neural_rebuild_12_stance_impact.json",
        question=("If the runtime used the corrected MN mapping and zeroed what it cannot "
                  "source, how much of a feasible stance's activation would it lose, and how "
                  "much was previously wired from the wrong leg?"),
        status="PASS" if any(isinstance(v, dict) and "groups" in v for v in rows.values())
               else "NOT_RUN",
        measured={
            "pool_source": str(POOLS),
            "stanced": rows,
            "summary": {
                "mean_share_own_leg_pool": float(np.mean(a_shares)) if a_shares else None,
                "mean_share_cross_leg_substitution": float(np.mean(b_shares)) if b_shares else None,
                "mean_share_unsourced": float(np.mean(c_shares)) if c_shares else None,
                "reading": ("correcting the mapping and zeroing the unsourced actuators removes "
                            "the cross-leg substitution entirely; the activation share that has "
                            "no source at all is the price of honesty, and it is reported here "
                            "rather than hidden"),
            },
            "correction_of_the_59_percent_claim": (
                "The often-quoted 'about 59% of the activation has no motor-neuron source' "
                "conflates two different things: actuators with no pool ANYWHERE, and actuators "
                "the runtime drives from ANOTHER leg's pool.  They are separated here."),
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_12_stance_impact.py"),
            "read_only": True,
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Activation share is a property of the STANCE, not of the fly: a different pose "
            "redistributes it, which is exactly why the geometry must be re-searched after the "
            "mapping is corrected.",
            "A pool that exists on the correct leg is still only an annotated set of neurons; no "
            "innervation strength or sign is claimed.",
            "No dynamics and no closed loop: this is an accounting of a wiring table against a "
            "stored activation vector.",
        ],
    )
    print("\nwrote", OUT / "neural_rebuild_12_stance_impact.json")


if __name__ == "__main__":
    main()
