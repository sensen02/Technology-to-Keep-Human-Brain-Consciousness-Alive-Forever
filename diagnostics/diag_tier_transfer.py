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
    rows = []
    for cur_nA in (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0):
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
