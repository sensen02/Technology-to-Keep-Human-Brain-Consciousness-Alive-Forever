#!/usr/bin/env python3
"""WIRE THE MUSCLES TO THE CONNECTOME: every muscle's activation IS its motor neurons' firing.

THE ONE RULE THIS FILE OBEYS, AND WHY IT MATTERS GIVEN THE PROHIBITION ON HAND-WRITTEN ACTIONS:
there is no function from state to action here.  A muscle's activation is the measured firing rate
of THAT MUSCLE'S OWN motor neurons in the BANC tier, normalised by a reference rate.  The only
hand-authored ingredients are
  * the ANATOMICAL map from motor neuron to muscle, which comes from the connectome's own
    annotation (ann_primary_type names motor neurons after the muscle they innervate), and
  * the normalisation constant, which is a calibration exactly like the one the existing decoder
    already uses (a fixed reference interval), NOT a controller.
Nothing reads the body state and computes an action.  The body state reaches the muscles only by
driving the tier's edges.

MEASURED FEASIBILITY, which is why this is possible at all: of the 308 muscle-named motor neurons
in BANC, 281 are inside the 12000-neuron tier; of the 243 in the per-leg map, 218 resolve to a
tier-local index.  The rest are reported as missing rather than silently dropped.

Run:
    OPENBLAS_NUM_THREADS=1 venv/bin/python muscle_wiring.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TIER_CFG_MUSCLES = 41  # not used; kept out of the way


def build_mn_muscle_wiring(verbose: bool = True):
    """{leg: {muscle_name: {'local': [...], 'root_ids': [...], 'missing': [...]}}} plus stats."""
    from engine.embodied.adapters import build_neural_tier, TierConfig
    built = build_neural_tier(TierConfig())
    n_tier = int(built["n"])
    tier = built["tier"]
    # ROOT ID -> TIER-LOCAL INDEX, built from the tier itself.  MEASURED: ``local_of_global`` is
    # indexed by the neuron's ROW in the BANC table, not by its root id, and a root id like
    # 720575941406488494 is far outside a 153962-entry array -- indexing it directly raised
    # IndexError.  The tier already publishes its own root ids in local order, so the lookup is
    # built from that and needs no assumption about the row ordering.
    local_of_root = {int(rid): li for li, rid in enumerate(np.asarray(tier["root_ids"]))}

    mp = json.loads((HERE / "outputs" / "mn_muscle_map_per_leg.json").read_text())
    out, stats = {}, {"total": 0, "wired": 0, "missing": 0, "per_leg": {}}
    for leg, info in mp["legs"].items():
        out[leg] = {}
        leg_wired = leg_missing = 0
        for mn_type, v in info["by_type"].items():
            loc, roots, miss = [], [], []
            for rid in v["ids"]:
                li = local_of_root.get(int(rid), -1)
                if li >= 0:
                    loc.append(li)
                    roots.append(int(rid))
                else:
                    miss.append(int(rid))
            out[leg][mn_type] = {"local": loc, "root_ids": roots, "missing": miss}
            leg_wired += len(loc)
            leg_missing += len(miss)
        stats["per_leg"][leg] = {"wired": leg_wired, "missing": leg_missing}
        stats["total"] += leg_wired + leg_missing
        stats["wired"] += leg_wired
        stats["missing"] += leg_missing
    if verbose:
        print(f"tier: {n_tier} neurons")
        print(f"{'leg':<5}{'MN->muscle groups':>18}{'wired':>8}{'missing':>9}")
        for leg in sorted(out):
            print(f"{leg:<5}{len(out[leg]):>18}{stats['per_leg'][leg]['wired']:>8}"
                  f"{stats['per_leg'][leg]['missing']:>9}")
        print(f"total motor neurons wired to muscles: {stats['wired']} of {stats['total']} "
              f"({stats['missing']} missing from the tier)")
    return out, stats, built


def actuator_to_mn_type(xml_path=None, verbose=True):
    """{actuator name: (leg, motor-neuron type)} for ALL muscles in a model, not just the foreleg.

    MEASURED BUG THIS FIXES: the loop built this map from ``outputs/mn_muscle_map.json``, which is
    derived from the SHIPPED model and therefore contains ONLY the 15 foreleg muscles.  Every lookup
    for an LM/RM/LH/RH actuator missed, so those 75 muscles were never driven at all -- and, worse,
    ablating their motor neurons then changed NOTHING, which reads exactly like "the neurons do not
    drive these legs" when the truth was "these muscles were never connected".  The map is now built
    from the model actually being simulated, by the same name matching used everywhere else.
    """
    import xml.etree.ElementTree as ET
    from tools_mn_muscle_map import tokens, canon
    path = Path(xml_path) if xml_path else (
        HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml")
    root = ET.parse(path).getroot()
    names = [a.get("name") for a in root.find("actuator").findall("general")
             if a.get("class") == "muscle" and a.get("name")]
    mp = json.loads((HERE / "outputs" / "mn_muscle_map_per_leg.json").read_text())
    types_by_leg = {leg: list(info["by_type"]) for leg, info in mp["legs"].items()}
    out, unmatched = {}, []
    for nm in names:
        leg = nm[:2]
        if leg not in types_by_leg:
            unmatched.append(nm)
            continue
        mt = {canon(x) for x in tokens(nm)}
        best, bestj = None, 0.0
        for t in types_by_leg[leg]:
            tt = {canon(x) for x in tokens(t)}
            if not tt:
                continue
            j = len(mt & tt) / len(mt | tt)
            if j > bestj:
                best, bestj = t, j
        if best is None or bestj < 0.40:
            unmatched.append(nm)
        else:
            out[nm] = (leg, best)
    if verbose:
        import collections
        c = collections.Counter(v[0] for v in out.values())
        print(f"actuators mapped to a motor-neuron type: {len(out)} of {len(names)}")
        print("  per leg:", dict(sorted(c.items())))
        print(f"  unmatched: {len(unmatched)}" + (f" e.g. {unmatched[:4]}" if unmatched else ""))
    return out


def muscle_activation(spikes, wiring, leg, mn_type, reference_hz, dt_s, max_rate_factor=2.0):
    """Mean firing rate of that muscle's own motor neurons, normalised to [0, 1].

    ``spikes`` is a boolean (n_substeps, n_neurons) array for one command interval.  The
    activation is the rate of THIS muscle's neurons only, so ablating them silences exactly this
    muscle and nothing else -- which is what makes the ablation test meaningful.
    """
    grp = wiring.get(leg, {}).get(mn_type)
    if not grp or not grp["local"]:
        return 0.0
    sel = spikes[:, grp["local"]]
    rate = float(sel.sum()) / (sel.shape[0] * dt_s) if sel.size else 0.0
    return float(min(1.0, rate / (reference_hz * max_rate_factor)))


if __name__ == "__main__":
    wiring, stats, built = build_mn_muscle_wiring()
    # also report the reverse direction: which tier neurons are muscle motor neurons at all
    allw = {li for leg in wiring for t in wiring[leg] for li in wiring[leg][t]["local"]}
    print(f"\ndistinct tier-local indices that are muscle motor neurons: {len(allw)}")
    dest = HERE / "outputs" / "muscle_wiring.json"
    dest.write_text(json.dumps({
        "provenance": ("motor-neuron-to-muscle map DERIVED by naming from BANC "
                       "ann_primary_type; tier membership measured"),
        "stats": stats,
        "wiring": {leg: {t: {"n_local": len(v["local"]), "n_missing": len(v["missing"])}
                         for t, v in g.items()} for leg, g in wiring.items()},
        "distinct_tier_neurons_used": len(allw),
    }, indent=2, sort_keys=True))
    print(f"wrote {dest}")
