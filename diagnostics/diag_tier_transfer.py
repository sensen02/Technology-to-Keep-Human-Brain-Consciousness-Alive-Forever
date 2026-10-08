#!/usr/bin/env python3
"""WHY IS THE LOOP'S OUTPUT A SWITCH?  Measure the tier's input-output curve.

MEASURED MOTIVATION: with every receptor gain scaled over a 150x range the motor neurons stayed at
450-507 Hz and the mean muscle activation at 0.48-0.52, while the same network with zero external
current fired at 0.2 Hz.  A 150x change in input producing no change in output is not a gain problem,
it is a threshold: below some input the population is silent, above it the population saturates, and
there is no graded regime in between for a posture to be represented in.

This maps that curve directly and cheaply -- neural only, no physics.  For a range of uniform currents
injected into EVERY tier neuron, it reports the population firing rate, the fraction of neurons
active, and the rate of the muscle motor-neuron groups.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
from engine.embodied.adapters import (NeuralTier, OwnershipLedger, TierConfig,  # noqa: E402
                                      build_neural_tier)
from muscle_wiring import build_mn_muscle_wiring  # noqa: E402

DUR_MS = float(os.environ.get("TRANSFER_MS", "500"))


def sweep_one(scale, currents, dur_ms, mn_local):
    """the transfer curve of the tier at one synaptic weight scale."""
    built = build_neural_tier(TierConfig(weight_scale_uS_per_synapse=scale))
    tier = NeuralTier(built, TierConfig(weight_scale_uS_per_synapse=scale), OwnershipLedger())
    n = int(built["n"])
    n_sub = int(round(dur_ms / tier.cfg.dt_ms))
    win = n_sub * tier.cfg.dt_ms / 1000.0
    rows = []
    for cur in currents:
        tier.reset()
        sp = tier.advance(n_sub, np.full(n, cur))
        rows.append({
            "current_nA": cur,
            "population_hz": float(sp.sum()) / (n * win),
            "frac_active": float(sp.any(axis=0).mean()),
            "mn_hz": float(sp[:, mn_local].sum()) / (mn_local.size * win),
            "mn_frac_active": float(sp[:, mn_local].any(axis=0).mean())})
    return rows


def gradedness(rows):
    """how many decades of input the output TRACKS, and where the cliff is.

    A graded sensorimotor loop must change its output progressively as the input rises.  The measure
    used here is the input span, in decades, over which the motor-neuron rate climbs from 10% to 90%
    of its own maximum: a wide span means graded, a narrow one means a threshold switch.
    """
    cur = np.array([r["current_nA"] for r in rows], float)
    mn = np.array([r["mn_hz"] for r in rows], float)
    nz = cur > 0
    cur, mn = cur[nz], mn[nz]
    if mn.size == 0 or mn.max() <= 0:
        return {"decades_10_to_90": None, "note": "no response"}
    hi, lo = 0.9 * mn.max(), 0.1 * mn.max()
    def first_above(t):
        idx = np.flatnonzero(mn >= t)
        return float(cur[idx[0]]) if idx.size else None
    a, b = first_above(lo), first_above(hi)
    dec = (np.log10(b / a) if (a and b and a > 0 and b > a) else None)
    return {"decades_10_to_90": dec,
            "input_at_10pct_hz": a, "input_at_90pct_hz": b,
            "max_mn_hz": float(mn.max())}


def main():
    built = build_neural_tier(TierConfig())
    cfg = TierConfig()
    tier = NeuralTier(built, cfg, OwnershipLedger())
    n = int(built["n"])
    wiring, _s, _ = build_mn_muscle_wiring(verbose=False)
    mn = np.asarray(sorted({li for leg in ("LF", "RF", "LM", "RM", "LH", "RH")
                            for g in wiring.get(leg, {}).values() for li in g["local"]}),
                    dtype=np.int64)
    n_sub = int(round(DUR_MS / cfg.dt_ms))
    win = n_sub * cfg.dt_ms / 1000.0
    print(f"tier {n} neurons, {mn.size} motor-neuron cells; {DUR_MS:.0f} ms per point")
    print(f"{'current nA':>11}{'population Hz':>15}{'frac active':>13}{'MN group Hz':>13}"
          f"{'MN frac active':>16}")
    CURRENTS = (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0)
    scales_env = os.environ.get("TIER_SCALES")
    if scales_env:
        # ---- SWEEP THE SYNAPTIC WEIGHT SCALE.  MEASURED MOTIVATION: at the shipped 5e-4 the
        # network is a threshold switch (a 10x input change takes it from 0.02 Hz to 90% of the
        # motor neurons firing), so no posture can be represented.  The weight scale sets the
        # recurrent gain, and this finds the regime where the transfer curve is graded instead.
        out = {"scales": [], "currents": list(CURRENTS), "duration_ms": DUR_MS}
        for sc in [float(v) for v in scales_env.split(",")]:
            print(f"\n=== weight_scale = {sc:g} uS/synapse ===")
            rows = sweep_one(sc, CURRENTS, DUR_MS, mn)
            g = gradedness(rows)
            for r in rows:
                print(f"  {r['current_nA']:>9.2e} nA  pop {r['population_hz']:>8.2f} Hz  "
                      f"active {r['frac_active']:>6.3f}  MN {r['mn_hz']:>8.2f} Hz  "
                      f"MN active {r['mn_frac_active']:>6.3f}")
            print(f"  -> decades of input from 10% to 90% of max MN rate: "
                  f"{g['decades_10_to_90']}; threshold {g['input_at_10pct_hz']}; "
                  f"max {g['max_mn_hz']:.1f} Hz")
            out["scales"].append({"weight_scale_uS_per_synapse": sc, "rows": rows,
                                  "gradedness": g})
        (HERE / "outputs" / "tier_transfer_scales.json").write_text(json.dumps(out, indent=2))
        print(f"\nwrote {HERE / 'outputs' / 'tier_transfer_scales.json'}")
        print("\n=== which scale is GRADED? (more decades of tracking = more graded) ===")
        for e in out["scales"]:
            d = e["gradedness"]["decades_10_to_90"]
            print(f"  {e['weight_scale_uS_per_synapse']:>9.1e}  decades={d if d is None else round(d,3)}"
                  f"  max MN {e['gradedness']['max_mn_hz']:.1f} Hz")
        return
    rows = []
    for cur_nA in CURRENTS:
        tier.reset()
        sp = tier.advance(n_sub, np.full(n, cur_nA))
        pop = float(sp.sum()) / (n * win)
        frac = float(sp.any(axis=0).mean())
        mnr = float(sp[:, mn].sum()) / (mn.size * win)
        mnf = float(sp[:, mn].any(axis=0).mean())
        rows.append({"current_nA": cur_nA, "population_hz": pop, "frac_active": frac,
                     "mn_hz": mnr, "mn_frac_active": mnf})
        print(f"{cur_nA:>11.5f}{pop:>15.2f}{frac:>13.4f}{mnr:>13.2f}{mnf:>16.4f}")
    # where is the transition?
    pops = np.array([r["population_hz"] for r in rows])
    curs = np.array([r["current_nA"] for r in rows])
    lo = np.argmax(pops > 1.0)
    print(f"\nfirst current at which the population exceeds 1 Hz: {curs[lo]:.5f} nA "
          f"-> population {pops[lo]:.2f} Hz")
    print(f"at {curs[-1]:.3f} nA the population is {pops[-1]:.1f} Hz; ratio to the 1 Hz point: "
          f"{pops[-1]/max(pops[lo],1e-9):.1f}x over an input ratio of {curs[-1]/max(curs[lo],1e-9):.0f}x")
    print("a graded sensorimotor loop needs the output to TRACK the input over decades of input; "
          "if the output jumps from silence to saturation within a decade, there is no regime in "
          "which a posture can be represented")
    (HERE / "outputs" / "tier_transfer.json").write_text(json.dumps(
        {"rows": rows, "duration_ms": DUR_MS, "n_mn_cells": int(mn.size)}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'tier_transfer.json'}")


if __name__ == "__main__":
    main()
