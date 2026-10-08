#!/usr/bin/env python3
"""DOES DRIVING A LEG'S MECHANOSENSORY NEURONS RECRUIT ITS EXTENSORS OR ITS FLEXORS?

WHY THIS IS THE NEXT QUESTION, MEASURED: the fly cannot stand passively and no steady hand-set muscle
pattern holds it up, so the loop has to close through load.  The biological route is the insect
RESISTANCE REFLEX: a loaded leg recruits its extensor motor neurons.  This project's loop currently
drives each leg's sensory neurons from JOINT ANGLE only, which carries no load information, and the
measured consequence is that the intact loop is WORSE than silencing the motor neurons (0.801 body
weights on the body against 0.629).

Before wiring a load signal in, measure what that pathway actually does.  This is a purely NEURAL
measurement -- no physics, no body -- so it is cheap and unambiguous:

    baseline: run the tier with no external current, record the firing rate of every muscle's own
              motor-neuron group
    then for one leg at a time: add a constant depolarising current to THAT leg's mechanosensory
              neurons and record the change in every muscle's motor-neuron firing rate

A positive change on the extensor types means the connectome's own wiring implements the resistance
reflex for these cells.  A negative change means it implements the opposite, and the honest response
is to report that rather than to flip the sign by hand.

Also reports what the sensory groups actually ARE, by annotation, so driving them with load is not a
claim about cells whose identity was never checked.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")

from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,
                                      build_neural_tier, LEGS, LEG_NERVE)  # noqa: E402
from muscle_wiring import build_mn_muscle_wiring  # noqa: E402

DRIVE_NA = float(os.environ.get("REFLEX_DRIVE_NA", "2.0"))
DUR_MS = float(os.environ.get("REFLEX_MS", "200"))


#: the MUSCLE model names its legs in upper case and the connectome adapter in lower case
MUSCLE_LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")


def main():
    built = build_neural_tier(TierConfig())
    tier_cfg = TierConfig()
    tier = NeuralTier(built, tier_cfg, OwnershipLedger())
    n = int(built["n"])
    print(f"tier: {n} neurons, sensory group sizes "
          + ", ".join(f"{k}={built['sensory'][k].size}" for k in LEGS))

    # ---- what ARE these sensory neurons?  Report the annotations, do not assume.
    z = np.load(HERE / "data" / "flywire" / "banc_connectome.npz", allow_pickle=True)
    root_ids = np.asarray(z["root_ids"])
    order = {int(r): i for i, r in enumerate(root_ids)}
    tier_roots = np.asarray(built["tier"]["root_ids"])
    ptype = np.asarray(z["ann_primary_type"])
    ntype = np.asarray(z["ann_nerve"])
    print("\n=== what the per-leg sensory groups contain (ann_primary_type) ===")
    sensory_ann = {}
    for leg in LEGS:
        loc = np.asarray(built["sensory"][leg])
        gl = tier_roots[loc]
        idx = np.array([order.get(int(r), -1) for r in gl])
        idx = idx[idx >= 0]
        vals, cnt = np.unique(ptype[idx], return_counts=True)
        top = sorted(zip(cnt, vals), reverse=True)[:6]
        sensory_ann[leg] = {str(v): int(c) for c, v in top}
        print(f"  {leg} ({LEG_NERVE[leg]}, {loc.size} in tier): "
              + ", ".join(f"{v}×{c}" for c, v in top))

    # ---- the muscle groups: (leg, muscle type) -> tier-local motor neuron indices
    wiring, _stats, _ = build_mn_muscle_wiring(verbose=False)
    groups = {}
    for leg in ("LF", "RF", "LM", "RM", "LH", "RH"):
        lk = leg.lower()
        for mtype, g in wiring.get(leg, {}).items():
            if g.get("local"):
                groups[(leg, mtype)] = np.asarray(sorted(set(g["local"])), dtype=np.int64)
    print(f"\nmuscle groups with motor neurons inside the tier: {len(groups)}")
    # USE THE WIRING'S OWN TYPE VOCABULARY.  MEASURED: this summary first hard-coded type names
    # ("trochanter_extensor", "tibia_flexor", ...) and every lookup came back nan, because the
    # derived map's vocabulary is different -- it contains tibia_extensor, tibia_flexor,
    # sternotrochanter, tarsus_levator, tarsus_depressor and more.  Classify from the names that are
    # actually present instead of from names I assumed.
    types_present = sorted({t for _leg, t in groups})
    print(f"motor-neuron type vocabulary ({len(types_present)}): {types_present}")

    def rates(extra):
        tier.reset()
        n_sub = int(round(DUR_MS / tier_cfg.dt_ms))
        cur = np.zeros(n, dtype=float)
        for leg, loc in extra.items():
            cur[np.asarray(built["sensory"][leg])] += loc
        sp = tier.advance(n_sub, cur)
        win = n_sub * tier_cfg.dt_ms / 1000.0
        return {k: float(sp[:, v].sum()) / (v.size * win) for k, v in groups.items()}

    base = rates({})
    print(f"baseline (no external current): mean muscle-MN rate "
          f"{np.mean(list(base.values())):.2f} Hz over {DUR_MS:.0f} ms")

    # ---- the reflex matrix
    reflex = {}
    print(f"\n=== reflex: +{DRIVE_NA} nA into ONE leg's mechanosensory neurons ===")
    for leg in LEGS:
        r = rates({leg: DRIVE_NA})
        d = {k: r[k] - base[k] for k in base}
        reflex[leg] = d
        exc = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        inh = sorted(d.items(), key=lambda kv: kv[1])[:4]
        print(f"\n  {leg}: mean Δ {np.mean(list(d.values())):+.2f} Hz")
        print("    most EXCITED: " + ", ".join(f"{k[0]}/{k[1]} {v:+.1f}" for k, v in exc))
        print("    most INHIBITED: " + ", ".join(f"{k[0]}/{k[1]} {v:+.1f}" for k, v in inh))

    # ---- the decisive summary: do the extensors of the driven leg get recruited?
    print("\n=== does a leg's own load pathway recruit ITS extensors? (Δ Hz) ===")
    print(f"{'leg':<5}{'own tib_ext':>13}{'all extensors':>14}{'all flexors':>12}"
          f"{'ext-flex':>10}")
    verdict = {}
    def is_ext(t):
        return "extensor" in t
    def is_flx(t):
        return "flexor" in t
    for leg in MUSCLE_LEGS:
        d = reflex[leg.lower()]
        ext = float(sum(v for (l, t), v in d.items() if l == leg and is_ext(t)))
        flx = float(sum(v for (l, t), v in d.items() if l == leg and is_flx(t)))
        tib = float(d.get((leg, "tibia_extensor"), float("nan")))
        verdict[leg] = {"extensor_sum": ext, "flexor_sum": flx, "net": ext - flx,
                        "own_tibia_extensor": tib}
        print(f"{leg:<5}{tib:>11.2f}{ext:>11.2f}{flx:>12.2f}{ext-flx:>12.2f}")
    n_ext_pos = sum(1 for v in verdict.values() if v["net"] > 0)
    print(f"\nlegs whose mechanosensory drive recruits extensors more than flexors: "
          f"{n_ext_pos} / 6")
    if n_ext_pos >= 4:
        print("  -> the connectome's own wiring gives a resistance reflex SIGN for these cells; "
              "load feedback can be wired in without choosing a sign by hand")
    else:
        print("  -> the connectome's own wiring does NOT give a resistance reflex for these cells; "
              "wiring load feedback with a positive sign would drive the leg DOWN, and that must be "
              "reported rather than fixed by hand")
    (HERE / "outputs" / "reflex_matrix.json").write_text(json.dumps(
        {"drive_nA": DRIVE_NA, "duration_ms": DUR_MS,
         "baseline_hz": {f"{k[0]}|{k[1]}": v for k, v in base.items()},
         "reflex_delta_hz": {lw: {f"{k[0]}|{k[1]}": v for k, v in d.items()}
                             for lw, d in reflex.items()},
         "sensory_annotations": sensory_ann, "verdict": verdict,
         "legs_with_extensor_recruitment": int(n_ext_pos)}, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'reflex_matrix.json'}")


if __name__ == "__main__":
    main()
