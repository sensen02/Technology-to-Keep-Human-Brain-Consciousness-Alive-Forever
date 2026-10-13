"""NEURAL INTERFACE REBUILD, step 4: the timing bind, measured, and what a usable
interface would require.

WHAT STEPS 1-3 ESTABLISHED
--------------------------
  1. Recurrence gain 978x at the declared weight scale, crossing below 1 near scale factor
     0.25 (measured 977.9 / 444.6 / 2.26 / 0.00 at factors 1 / 0.5 / 0.333 / 0.2), with a
     self-sustained fraction of 0.372 at the declared scale.
  2. Below that crossing the population response to a 1% subset drive spans only ~0.5 decades
     of drive current, identically at factors 0.333 / 0.1 / 0.05 / 0.02 / 0.01.  So H2 was
     rejected: the weight scale alone cannot make the interface graded.
  3. Adaptation can widen the nominal span to 1.0 decade, but only at population rates of
     0.1-0.3 Hz, and the widest drive range tested plateaus at 13.7 Hz.

Those numbers become decisive only when combined with THE READOUT, which is what this step
does.  The project's readout is a sliding spike-count rate over a 5 ms window
(`engine/embodied/loop.py` dt_command_s = 5e-3, decoded from a 5 ms spike count).  Two hard
facts follow from the model's own declared parameters:

  * a rate computed over a 5 ms window is QUANTISED: one spike per window = 200 Hz, so the
    command can only take multiples of 200 Hz;
  * with a 2 ms refractory period a neuron can emit at most 2 spikes per 5 ms window, so the
    top of that quantisation is 400 Hz.

So the number of spikes available per neuron per readout window is the ONLY quantity the
motor decoder can use.  Step 4 measures it, across the operating regimes step 1-2 found:

    spikes per neuron per window = rate_hz * window_s

and compares it with what a decoder needs.  A decoder that must distinguish, say, 5 command
levels needs at least ~5 spikes per window per neuron (Poisson relative error ~1/sqrt(n)), and
more realistically >= 3 usable spikes for the lowest level.

This is a negative-space measurement: it asks what the interface would have to deliver, and
checks whether ANY declared setting can deliver it.  It does not build the interface.

PRE-REGISTERED, BEFORE THE RUN
------------------------------
  H4  Under the declared dynamics at the calibrated weight scale, at least one reachable
      drive gives >= 3 spikes per neuron per 5 ms window AND a self-sustained fraction < 0.5.
  FALSIFIER: no drive satisfies both at any tested scale.
  A rejected H4 is the decisive statement that the interface cannot be rebuilt by tuning:
  the readout window, the refractory period and the chosen operating point are mutually
  incompatible, and one of them must be changed by design.

A declared network, not a fly.  No body, no controller, no weight written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402
from engine.embodied.adapters import build_neural_tier, TierConfig  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
DT_MS = 0.5
DECLARED_SCALE = 5e-4
READOUT_MS = 5.0                      # the project's declared command window
CALIBRATED_FACTOR = 0.2


def scaled(unit, factor):
    return sparse.csr_matrix((unit.data * factor, unit.indices, unit.indptr),
                             shape=unit.shape)


def measure(We, Wi, p, n, *, drive_nA, frac, steps, seed=1):
    """Run with a subset drive held constant; report spikes per cell per readout window and
    the self-sustained fraction (same run, drive removed for the last third)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
    net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
    net.set_connectivity(We, Wi)
    drive = np.zeros(n)
    drive[idx] = float(drive_nA)
    n_drive = int(round(0.8 * steps))
    total, seed_spikes = 0, 0
    for _ in range(n_drive):
        sp = net.step(current_nA=drive)
        total += int(sp.sum())
        seed_spikes += int(sp[idx].sum())
    quiet = 0
    for _ in range(steps - n_drive):
        quiet += int(net.step(current_nA=0.0).sum())
    dur_s = n_drive * DT_MS / 1000.0
    quiet_s = (steps - n_drive) * DT_MS / 1000.0
    rate = total / (n * dur_s)
    quiet_rate = quiet / (n * quiet_s)
    return {
        "drive_nA": float(drive_nA),
        "population_rate_hz": rate,
        "spikes_per_cell_per_readout_window": rate * READOUT_MS / 1000.0,
        "seeded_cell_rate_hz": seed_spikes / (idx.size * dur_s),
        "spikes_per_seeded_cell_per_window": (seed_spikes / (idx.size * dur_s))
                                             * READOUT_MS / 1000.0,
        "quiet_rate_hz": quiet_rate,
        "self_sustained_ratio": (quiet_rate / rate) if rate > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    rows = []
    steps = 600 if not args.quick else 300

    for factor in (1.0, 0.5, CALIBRATED_FACTOR, 0.05):
        We = scaled(built["edges"]["We"], factor)
        Wi = scaled(built["edges"]["Wi"], factor)
        p = ConductanceParams()
        for drive in (1e-2, 3e-2, 1e-1, 3e-1, 1.0):
            r = measure(We, Wi, p, n, drive_nA=drive, frac=0.01, steps=steps)
            r["scale_factor"] = factor
            rows.append(r)
            print(f"  f={factor:<5.2f} drive={drive:<6g} rate {r['population_rate_hz']:8.2f} Hz  "
                  f"spk/cell/window {r['spikes_per_cell_per_readout_window']:.4f}  "
                  f"quiet/active {r['self_sustained_ratio'] if r['self_sustained_ratio'] is None else round(r['self_sustained_ratio'],3)}",
                  flush=True)

    # also the adaptation arm, at the calibrated scale, since step 3's best span came from there
    We = scaled(built["edges"]["We"], CALIBRATED_FACTOR)
    Wi = scaled(built["edges"]["Wi"], CALIBRATED_FACTOR)
    for inc in (0.002, 0.05, 0.2):
        p = ConductanceParams(adaptation_increment_nA=inc)
        for drive in (3e-1, 1.0, 3.0):
            r = measure(We, Wi, p, n, drive_nA=drive, frac=0.01, steps=steps)
            r["scale_factor"] = CALIBRATED_FACTOR
            r["adaptation_increment_nA"] = inc
            rows.append(r)
            print(f"  f={CALIBRATED_FACTOR} adapt={inc:<6g} drive={drive:<5g} rate "
                  f"{r['population_rate_hz']:8.2f} Hz  spk/cell/window "
                  f"{r['spikes_per_cell_per_readout_window']:.4f}", flush=True)

    ok = [r for r in rows
          if r["spikes_per_cell_per_readout_window"] >= 3.0
          and (r["self_sustained_ratio"] is not None and r["self_sustained_ratio"] < 0.5)]
    best_spk = max(rows, key=lambda r: r["spikes_per_cell_per_readout_window"])
    h4 = bool(ok)

    cp.write_result(
        OUT / "neural_rebuild_04_timing_bind.json",
        question=("Is there any declared operating point that delivers both enough spikes per "
                  "readout window for a motor command and no self-sustaining activity?"),
        status="PASS",
        measured={
            "tier": {"n": n, "fingerprint": built.get("fingerprint")},
            "readout_window_ms": READOUT_MS,
            "quantisation_hz": 1000.0 / READOUT_MS,
            "refractory_ms": ConductanceParams().refractory_ms,
            "max_spikes_per_window_from_refractory": float(
                int(READOUT_MS // ConductanceParams().refractory_ms)),
            "rows": rows,
            "requirement_used": {"spikes_per_cell_per_window_for_5_levels": 3.0,
                                 "justification": "a 5-level command needs ~3 usable spikes "
                                                 "for the lowest level; Poisson relative "
                                                 "error is 1/sqrt(n)"},
            "predictions": {
                "H4": {"supported": h4,
                       "max_spikes_per_cell_per_window_observed":
                           best_spk["spikes_per_cell_per_readout_window"],
                       "window_ms_that_would_give_3_spikes_at_that_rate":
                           (3000.0 / best_spk["population_rate_hz"]
                            if best_spk["population_rate_hz"] > 0 else None),
                       "at_scale_factor": best_spk["scale_factor"],
                       "at_drive_nA": best_spk["drive_nA"],
                       "falsifier": "no drive satisfies >= 3 spikes/window with self-sustained "
                                    "ratio < 0.5 at any tested scale"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_04_timing_bind.py"),
            "steps_1_3": ["outputs/neural_rebuild/neural_rebuild_01_operating_point.json",
                          "outputs/neural_rebuild/neural_rebuild_02_critical_scale.json",
                          "outputs/neural_rebuild/neural_rebuild_03_gradedness.json"],
            "readout": "engine/embodied/loop.py dt_command_s = 5e-3",
            "dt_ms": DT_MS,
            "subset_fraction": 0.01,
            "wall_s": time.time() - t0,
            "interpreter": sys.executable,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
        },
        what_this_does_not_support=[
            "Spikes per window is a NECESSARY condition for a rate decoder, not a sufficient "
            "one: it says nothing about whether the pattern carries posture information.",
            "A larger drive current is not a fix if the current itself is unrealistic; the "
            "nA unit here is a declared parameter.",
            "No receptor, no body, no muscle, no controller are involved.",
        ],
    )
    print(f"\nH4 (a usable window exists): {h4}")
    print(f"  best observed: {best_spk['spikes_per_cell_per_readout_window']:.4f} spikes/cell/"
          f"window at scale factor {best_spk['scale_factor']}, drive {best_spk['drive_nA']} nA")
    print("wrote", OUT / "neural_rebuild_04_timing_bind.json")


if __name__ == "__main__":
    main()
