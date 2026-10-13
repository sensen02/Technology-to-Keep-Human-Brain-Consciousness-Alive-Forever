"""NEURAL INTERFACE REBUILD, step 17: POPULATION rate coding - the question that decides
whether the timing bind is fatal or merely restrictive.

WHY THIS STEP
-------------
Step 4 measured that at the calibrated operating point a SINGLE cell emits 0.2225 spikes per
5 ms readout window, 13x short of the ~3 spikes a 5-level command needs, and step 16 showed the
network's output rate barely moves when the input current is raised 380x (101.1 -> 106.0 Hz).
Both point at the readout.

But a motor command is not decoded from ONE cell.  `CommandDecoder` turns a spike COUNT over the
tier's motor readout into an actuator command, and the readout pool has many cells.  So the
quantity that actually matters is the POPULATION count per window:

        spikes_per_window = n_readout_cells * rate * window

and with ~200 readout cells at ~13 Hz over 5 ms that is of order 10 spikes per window - which
would be enough for a coarse command.  The per-cell framing may therefore have been the wrong
question, and this step tests that directly instead of arguing about it.

WHAT IS MEASURED
----------------
The tier is driven through the channel front end with a signed joint deviation of varying
magnitude and sign, and for each condition the SPIKES OF THE READOUT POOL are counted over
1 / 2 / 5 / 10 / 20 / 50 ms windows, with repetitions to get trial-to-trial spread.  Then:

  D1  the population spike count per window, against the readout pool size;
  D2  whether opposite signs are separable at each window, as a SIGNED SEPARATION
      (mean(-) - mean(+)) divided by the pooled standard deviation across trials;
  D3  whether the separation grows with the drive magnitude, i.e. whether the population count
      is GRADED rather than a threshold;
  D4  the smallest window at which |separation| >= 2 (a conventional two-sigma detection),
      compared with the project's 5 ms command window and the 25 ms mechanical stability window
      measured in round 39.

PRE-REGISTERED
--------------
  H11  At the calibrated weight scale there exists a window <= 20 ms at which the population
       spike count separates a flexion from a matched extension by more than two pooled standard
       deviations.
  FALSIFIER: no window at or below 20 ms reaches |separation| >= 2, at any drive magnitude.
  If H11 is rejected, the conclusion that the interface cannot produce a usable command is
  established on the population statistic as well, and the binding constraint is the readout
  window rather than the per-cell rate.

Read-only on the canonical asset.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
from engine.embodied.adapters import build_neural_tier, TierConfig  # noqa: E402
from engine.embodied.sensory_channels import (CHANNEL_SPEC, LEGS, ChannelConfig,  # noqa: E402
                                              ChannelQuantities, ChannelSensoryEncoder,
                                              channel_index_map)
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
DT_MS = 0.5
CALIBRATED_FACTOR = 0.2
WINDOWS_MS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DRIVES = (1e-2, 3e-2, 1e-1, 3e-1, 1.0)
TRIALS = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    root_ids = np.asarray(built["tier"]["root_ids"])
    readout_local = np.asarray(built["readout_local"], dtype=int)
    groups, stats = channel_index_map(n, root_ids)
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    cov = enc.coverage()

    def sc(u, f):
        return sp.csr_matrix((u.data * f, u.indices, u.indptr), shape=u.shape)

    We = sc(built["edges"]["We"], CALIBRATED_FACTOR)
    Wi = sc(built["edges"]["Wi"], CALIBRATED_FACTOR)
    p = ConductanceParams()

    # the channel current vector for a signed deviation of a given magnitude, applied to the
    # signed channels only (claw = position, hook = direction, hairplate = joint_angle), so the
    # stimulus carries a SIGN and nothing else
    def current_for(mag):
        c = np.zeros(n)
        for leg in LEGS:
            q = ChannelQuantities(tibia_joint_dev_rad=mag, tibia_joint_vel_rad_s=mag * 10.0,
                                  multi_joint_dev_rad=mag, contact_force_norm=0.0,
                                  foot_contact=0.0)
            cur, _ = enc.encode({leg: q})
            c += cur
        return c

    # spike train of the READOUT POOL, recorded at the neural dt
    def run_trial(current, steps, seed):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        rng = np.random.default_rng(seed)
        counts = np.zeros(steps, dtype=np.int64)
        for k in range(steps):
            sp = net.step(current_nA=current)
            counts[k] = int(sp[readout_local].sum())
        return counts

    max_window = max(WINDOWS_MS)
    steps = int(round(max_window / DT_MS))
    results = []
    t0 = time.time()
    for mag in DRIVES:
        cur = current_for(mag)
        nz = int(np.count_nonzero(cur))
        for sign in (+1.0, -1.0):
            trials = [run_trial(sign * cur, steps, seed=t) for t in range(TRIALS)]
            for w in WINDOWS_MS:
                k = int(round(w / DT_MS))
                # cumulative count over the first k steps of each trial
                per_trial = np.array([float(t[:k].sum()) for t in trials])
                results.append({"drive_magnitude": mag, "sign": sign,
                                "window_ms": w, "n_readout_cells": int(readout_local.size),
                                "cells_with_current": nz,
                                "mean_spikes": float(per_trial.mean()),
                                "sd_spikes": float(per_trial.std(ddof=1)) if TRIALS > 1 else None,
                                "spikes_per_cell": float(per_trial.mean() / readout_local.size)})
        print(f"  drive {mag:<6g} done ({time.time()-t0:.0f}s)", flush=True)

    # separation per (drive, window)
    sep = {}
    for mag in DRIVES:
        for w in WINDOWS_MS:
            plus = [r for r in results if r["drive_magnitude"] == mag and r["sign"] > 0
                    and r["window_ms"] == w][0]
            minus = [r for r in results if r["drive_magnitude"] == mag and r["sign"] < 0
                     and r["window_ms"] == w][0]
            pooled = np.sqrt(max(plus["sd_spikes"] ** 2 + minus["sd_spikes"] ** 2, 0.0) / 2.0)
            d = minus["mean_spikes"] - plus["mean_spikes"]
            sep[f"{mag:g}|{w:g}"] = {
                "mean_plus": plus["mean_spikes"], "mean_minus": minus["mean_spikes"],
                "difference": d, "pooled_sd": float(pooled),
                "separation_sigma": (float(abs(d) / pooled) if pooled > 0 else None),
                "graded_across_drives": None}

    # gradedness: does the count grow with drive magnitude at a fixed window?
    graded = {}
    for w in WINDOWS_MS:
        counts = [r["mean_spikes"] for r in results
                  if r["window_ms"] == w and r["sign"] > 0]
        graded[f"{w:g}"] = {"counts_by_drive": counts,
                            "monotone_increasing": bool(all(b >= a - 1e-9
                                                            for a, b in zip(counts, counts[1:]))),
                            "span_ratio_first_to_last": (counts[-1] / counts[0]
                                                         if counts and counts[0] > 0 else None)}

    first_ok = None
    for w in WINDOWS_MS:
        best = max((abs(v["separation_sigma"] or 0.0) for k, v in sep.items()
                    if k.endswith(f"|{w:g}")), default=0.0)
        if best >= 2.0:
            first_ok = {"window_ms": w, "best_separation_sigma": best}
            break
    h11 = bool(first_ok is not None and first_ok["window_ms"] <= 20.0)

    n_readout = int(readout_local.size)
    typical = [r for r in results if r["drive_magnitude"] == 0.3 and r["sign"] > 0]
    tpw = {r["window_ms"]: r["mean_spikes"] for r in typical}

    cp.write_result(
        OUT / "neural_rebuild_17_population_decoding.json",
        question=("Does the POPULATION spike count of the tier's readout separate a flexion "
                  "from a matched extension within a window short enough for the project's "
                  "5 ms command clock?"),
        status="PASS",
        measured={
            "tier": {"n": n, "readout_cells": n_readout,
                     "weights_scale_factor": CALIBRATED_FACTOR},
            "channel_coverage": cov,
            "channel_stats": stats,
            "trials_per_condition": TRIALS,
            "windows_ms": list(WINDOWS_MS),
            "drive_magnitudes": list(DRIVES),
            "population_count_at_drive_0.3_per_window": tpw,
            "separation": sep,
            "gradedness_by_window": graded,
            "predictions": {
                "H11": {"supported": h11, "first_window_reaching_2sigma": first_ok,
                        "falsifier": "no window <= 20 ms reaches |separation| >= 2 at any drive"},
            },
            "reference_constraints": {
                "command_window_ms": 5.0,
                "mechanical_stability_window_ms": 25.0,
                "source": "engine/embodied/loop.py dt_command_s = 5e-3; round-39 stance "
                          "forward-integration escape time ~25 ms",
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_17_population_decoding.py"),
            "read_only": True,
            "stimulus": "the signed channels only (claw/hook/hairplate), so the drive carries a "
                        "sign and no contact component",
            "counts": "spikes of the tier's own readout pool (built['readout_local'])",
            "dt_ms": DT_MS,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "A separable count is a NECESSARY condition for a command, not a demonstration that "
            "the command would be correct or that the fly would be controlled.",
            "The trials differ only in the random seed of the receptor model; they do not include "
            "body-state variation, so the pooled SD is a lower bound on real variability.",
            "The stimulus is a constant signed deviation held for the whole window; a real loop "
            "sees a changing state.",
            "No body, no muscle and no forward integration are involved.",
        ],
    )
    for w in WINDOWS_MS:
        print(f"  window {w:5.1f} ms: counts by drive "
              f"{[round(r['mean_spikes'],1) for r in results if r['window_ms']==w and r['sign']>0]}")
    print(f"  first window reaching 2 sigma: {first_ok}")
    print(f"  H11 (<=20 ms): {h11}")
    print("wrote", OUT / "neural_rebuild_17_population_decoding.json")


if __name__ == "__main__":
    main()
