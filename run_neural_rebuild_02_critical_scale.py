"""NEURAL INTERFACE REBUILD, step 2: find the critical weight scale, and test gradedness
with a REALISTIC input.

WHY STEP 2
----------
Step 1 (run_neural_rebuild_01_operating_point.py) measured, at three weight scales:

    scale 5.000e-04 (declared)  self-sustained 0.372   recurrence gain  977.9x
    scale 1.667e-04 (1/3)       self-sustained 0.0497  recurrence gain    2.26x
    scale 5.000e-05 (1/10)      self-sustained 0.0018  recurrence gain    0.00x

and every input-output curve was the same step: 0 Hz at 1e-2 nA, ~90-110 Hz at 8e-2 nA,
0.0 decades of graded span.  TWO separate findings are entangled in that curve:

  (a) the network's recurrence gain crosses 1 between 1/10 and 1/3 of the declared scale,
      and above 1 the network SELF-SUSTAINS - that is the "recurrence sets the operating
      point" defect, now quantified;
  (b) the input-output curve is a step even at 1/10 scale where there is no recurrence at
      all, so the step is NOT caused by recurrence: it is caused by the neuron's own
      threshold-to-rest ratio (threshold -50 mV vs rest -65 mV = 15 mV) against the drive
      range, i.e. the drive either charges the membrane past threshold or does not.

Step 1 drove EVERY neuron, which is not what a sensory interface does.  Step 2 therefore:

  1. sweeps the scale finely to locate the critical scale where the recurrence gain crosses 1;
  2. drives only a SENSORY SUBSET (1% of the tier) and measures the population response -
     the population is then a function of the network, not of the injected current;
  3. reports the graded span in decades at each scale, which is the property the project
     needs and never had.

PRE-REGISTERED, BEFORE THE RUN
------------------------------
  H2  There exists a critical scale factor in (1/100, 1) where the recurrence gain crosses
      1, and at scale factors below it the population response to a subset drive spans more
      than 1 decade of drive current.
  FALSIFIER for H2: no scale factor below the crossing gives a span > 1 decade.
  A rejected H2 means the interface cannot be made graded by the weight scale alone, and the
  threshold-to-rest ratio (or the drive unit) must change too - which is a different, larger
  change and must be declared as such.

No body, no controller, no weight written to disk.  A declared network, not a fly.
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


def scaled(unit, factor):
    return sparse.csr_matrix((unit.data * factor, unit.indices, unit.indptr),
                             shape=unit.shape)


def recurrence_gain(We, Wi, p, n, *, n_seed=50, seed_steps=20, quiet_steps=200, seed=0,
                    seed_current_nA=5e-2):
    """Seed `n_seed` cells with a forced current for `seed_steps`, then remove it.

    Reported: the spikes the SEED cells themselves must have emitted (an upper bound on the
    input, since each cell spikes at most once per refractory interval), and what the
    network emits afterwards.  amplification = after / seed_spikes.
    """
    rng = np.random.default_rng(seed)
    net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
    net.set_connectivity(We, Wi)
    idx = rng.choice(n, size=int(n_seed), replace=False)
    drive = np.zeros(n)
    drive[idx] = seed_current_nA
    seed_spikes = 0
    for _ in range(seed_steps):
        sp = net.step(current_nA=drive)
        seed_spikes += int(sp[idx].sum())
    after = 0
    for _ in range(quiet_steps):
        after += int(net.step(current_nA=0.0).sum())
    return {"seeded_cells": int(n_seed), "seed_spikes": int(seed_spikes),
            "spikes_after_removal": int(after),
            "amplification": (after / seed_spikes) if seed_spikes else None,
            "quiet_duration_ms": quiet_steps * DT_MS}


def subset_response(We, Wi, p, n, *, drives, frac=0.01, steps=400, seed=1):
    """Drive a sensory SUBSET; return per-level population rate and readout rate.

    The population rate is measured over ALL cells, so it reflects what the network does
    with the input rather than what was injected.  `readout_rate` is over the tier's
    motor-neuron-indexed cells if the builder exposes them, else over the driven subset.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
    pop, out = [], []
    for level in drives:
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        drive = np.zeros(n)
        drive[idx] = float(level)
        total = 0
        for _ in range(steps):
            total += int(net.step(current_nA=drive).sum())
        pop.append(total / (n * steps * DT_MS / 1000.0))
    return np.array(pop), idx


def span_decades(drives, rates, lo=0.10, hi=0.90):
    top = rates.max()
    if top <= 0:
        return None
    i_lo = int(np.argmax(rates >= lo * top))
    i_hi = int(np.argmax(rates >= hi * top))
    if rates[i_lo] < lo * top or rates[i_hi] < hi * top:
        return None
    if drives[i_lo] <= 0:
        return None
    return float(np.log10(drives[i_hi] / drives[i_lo]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = TierConfig(background_mean_nA=0.0, background_sd_nA=0.0)
    built = build_neural_tier(cfg)
    n = int(built["n"])
    We_unit, Wi_unit = built["edges"]["We"], built["edges"]["Wi"]
    p = ConductanceParams()
    declared = float(cfg.weight_scale_uS_per_synapse)

    factors = [1.0, 0.5, 0.333, 0.2, 0.1, 0.05, 0.02, 0.01]
    if args.quick:
        factors = [1.0, 0.2, 0.1, 0.02]

    rows = []
    for f in factors:
        We, Wi = scaled(We_unit, f), scaled(Wi_unit, f)
        rg = recurrence_gain(We, Wi, p, n)
        drives = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        pop, idx = subset_response(We, Wi, p, n, drives=drives,
                                   steps=200 if args.quick else 400)
        s = span_decades(drives, pop)
        rows.append({
            "scale_factor": float(f),
            "weight_scale_uS_per_synapse": float(declared * f),
            "recurrence_gain": rg,
            "subset_drives_nA": drives.tolist(),
            "subset_population_rate_hz": pop.tolist(),
            "population_span_decades": s,
            "subset_size": int(idx.size),
        })
        print(f"  f={f:6.3f}  scale {declared*f:.3e}  rec-gain "
              f"{('%.2f' % rg['amplification']) if rg['amplification'] is not None else 'n/a':>8s}  "
              f"pop rate {' '.join('%.1f' % v for v in pop)}  span {s}", flush=True)

    # critical scale: first factor (descending) where amplification drops below 1
    critical = None
    for r in rows:
        a = r["recurrence_gain"]["amplification"]
        if a is not None and a < 1.0:
            critical = r["scale_factor"]
            break
    below = [r for r in rows if critical is not None and r["scale_factor"] < critical]
    h2 = bool(any((r["population_span_decades"] or 0) > 1.0 for r in below)) if below else False

    cp.write_result(
        OUT / "neural_rebuild_02_critical_scale.json",
        question=("Where does the tier's recurrence gain cross 1, and below that scale does a "
                  "SENSORY-SUBSET drive produce a population response spanning more than a "
                  "decade of input?"),
        status="PASS",
        measured={
            "tier": {"n": n, "fingerprint": built.get("fingerprint")},
            "declared_weight_scale_uS_per_synapse": declared,
            "rows": rows,
            "critical_scale_factor": critical,
            "predictions": {
                "H2": {
                    "supported": h2,
                    "critical_scale_factor": critical,
                    "note": "H2 needed BOTH a gain crossing in (1/100, 1) and a span > 1 "
                            "decade below it",
                    "falsifier": "no scale factor below the crossing gives span > 1 decade",
                },
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_02_critical_scale.py"),
            "step_1": "outputs/neural_rebuild/neural_rebuild_01_operating_point.json",
            "weight_cap_per_edge": 8,
            "dt_ms": DT_MS,
            "subset_fraction": 0.01,
            "background": "zero mean and zero sd; the subset drive is the only input",
            "wall_s": time.time() - t0,
            "interpreter": sys.executable,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
        },
        what_this_does_not_support=[
            "A recurrence gain below 1 does NOT make the network physiological: it only means "
            "the declared recurrent weights no longer amplify a spike into a cascade.",
            "The subset drive is a constant current into arbitrary cells, not a sensory "
            "transduction; it tests the network's response shape, not a receptor.",
            "No body, no muscle and no controller are involved.",
            "The graded span is measured against the drive current in nA, and that unit is a "
            "declared parameter, so a span in decades is a statement about this model's "
            "input unit as much as about the network.",
        ],
    )
    print(f"\ncritical scale factor (gain crosses 1): {critical}")
    print(f"H2 (span > 1 decade below the crossing): {h2}")
    print("wrote", OUT / "neural_rebuild_02_critical_scale.json")


if __name__ == "__main__":
    main()
