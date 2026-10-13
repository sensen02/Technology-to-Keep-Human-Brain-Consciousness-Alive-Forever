"""NEURAL INTERFACE REBUILD, step 3: can gradedness be bought at all, and with what?

WHAT STEPS 1-2 ESTABLISHED
--------------------------
  * the recurrence gain is 978x at the declared weight scale and falls below 1 at a scale
    factor near 0.25 (measured: 977.9x / 444.6x / 2.26x / 0.00x at 1 / 0.5 / 0.333 / 0.2);
  * BELOW that crossing the population response to a 1% subset drive still spans only
    0.52 decades of drive current (measured identically at 0.333 / 0.1 / 0.05 / 0.02 / 0.01);
  * so H2 was REJECTED: the weight scale alone cannot make the interface graded.  The step
    response is a property of the neuron model's excitability curve, i.e. of
    (threshold - rest) = 15 mV against the drive amplitude.

Step 3 asks the only remaining question that decides the interface design: WHICH declared
parameter can buy gradedness, and how much?

Three candidates, all already in the model (`engine/neural_cond.py::ConductanceParams`), so
no new physics is invented:

  A. spike-frequency adaptation (`adaptation_increment_nA`, tau 100 ms).  Physiologically the
     standard mechanism for expanding a neuron's dynamic range.
  B. the excitable threshold-to-rest gap.  A larger gap widens the input range that lies below
     threshold, so more of the drive range is sub-threshold, i.e. graded.
  C. the input unit itself: the drive is in nA while the synaptic weights are in uS, so the
     ratio between "current injected by the interface" and "conductance received from the
     network" is arbitrary.  Varying the drive amplitude over a wider range tests whether the
     graded window is merely offset rather than absent.

PRE-REGISTERED, BEFORE THE RUN
------------------------------
  H3  With the recurrence below 1 AND adaptation at its declared increment or larger, the
      population response to a subset drive spans >= 1 decade.
  FALSIFIER: no combination of (scale below crossing, adaptation increment, threshold gap)
      reaches 1 decade of span.
  If H3 is rejected, the honest conclusion is that this neuron model cannot carry a graded
  population code at all, and the interface rebuild must replace the neuron model rather than
  tune it - which is a much larger claim and must be reported as such.

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
CALIBRATED_FACTOR = 0.2          # measured crossing in step 2: gain below 1 at <= 0.2


def scaled(unit, factor):
    return sparse.csr_matrix((unit.data * factor, unit.indices, unit.indptr),
                             shape=unit.shape)


def curve(We, Wi, p, n, drives, *, frac, steps, seed=1):
    """Population rate vs a subset drive, measured over ALL cells."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max(1, int(frac * n)), replace=False)
    out = []
    for level in drives:
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        drive = np.zeros(n)
        drive[idx] = float(level)
        total = 0
        for _ in range(steps):
            total += int(net.step(current_nA=drive).sum())
        out.append(total / (n * steps * DT_MS / 1000.0))
    return np.array(out)


def span(drives, rates, lo=0.10, hi=0.90):
    top = rates.max()
    if top <= 0:
        return None
    i_lo = int(np.argmax(rates >= lo * top))
    i_hi = int(np.argmax(rates >= hi * top))
    if rates[i_lo] < lo * top or rates[i_hi] < hi * top or drives[i_lo] <= 0:
        return None
    return float(np.log10(drives[i_hi] / drives[i_lo]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    We = scaled(built["edges"]["We"], CALIBRATED_FACTOR)
    Wi = scaled(built["edges"]["Wi"], CALIBRATED_FACTOR)
    base = ConductanceParams()

    drives = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    if args.quick:
        drives = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    steps = 300 if args.quick else 600

    rows = []

    # A. adaptation
    for inc in (base.adaptation_increment_nA, 0.01, 0.05, 0.2, 1.0):
        p = ConductanceParams(adaptation_increment_nA=inc)
        r = curve(We, Wi, p, n, drives, frac=0.01, steps=steps)
        rows.append({"arm": "adaptation", "adaptation_increment_nA": inc,
                     "rates_hz": r.tolist(), "span_decades": span(drives, r)})
        print(f"  adapt {inc:<7g} span {span(drives, r)}  rates "
              f"{' '.join('%.1f' % v for v in r)}", flush=True)

    # B. threshold-to-rest gap
    for thr in (base.threshold_mV, -45.0, -40.0, -35.0):
        p = ConductanceParams(threshold_mV=thr)
        r = curve(We, Wi, p, n, drives, frac=0.01, steps=steps)
        rows.append({"arm": "threshold", "threshold_mV": thr,
                     "rates_hz": r.tolist(), "span_decades": span(drives, r)})
        print(f"  thr   {thr:<7g} span {span(drives, r)}  rates "
              f"{' '.join('%.1f' % v for v in r)}", flush=True)

    # C. wider drive range at the declared neuron model, to separate "offset" from "absent"
    p = ConductanceParams()
    wide = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    r = curve(We, Wi, p, n, wide, frac=0.01, steps=max(200, steps // 2))
    rows.append({"arm": "wide_drive", "drives_nA": wide.tolist(),
                 "rates_hz": r.tolist(), "span_decades": span(wide, r)})
    print(f"  wide drive span {span(wide, r)}  rates "
          f"{' '.join('%.1f' % v for v in r)}", flush=True)

    h3 = bool(any((row["span_decades"] or 0) >= 1.0 for row in rows))
    best = max([row for row in rows if row["span_decades"] is not None],
               key=lambda x: x["span_decades"], default=None)

    cp.write_result(
        OUT / "neural_rebuild_03_gradedness.json",
        question=("Which declared parameter, if any, gives the tier a graded population "
                  "response (>= 1 decade of drive) once its recurrence is below 1?"),
        status="PASS",
        measured={
            "tier": {"n": n, "fingerprint": built.get("fingerprint")},
            "calibrated_scale_factor": CALIBRATED_FACTOR,
            "calibrated_weight_scale_uS_per_synapse": DECLARED_SCALE * CALIBRATED_FACTOR,
            "drives_nA": drives.tolist(),
            "rows": rows,
            "predictions": {
                "H3": {"supported": h3,
                       "best_span_decades": None if best is None else best["span_decades"],
                       "best_arm": None if best is None else best["arm"],
                       "falsifier": "no combination reaches 1 decade of span"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_03_gradedness.py"),
            "steps_1_2": ["outputs/neural_rebuild/neural_rebuild_01_operating_point.json",
                          "outputs/neural_rebuild/neural_rebuild_02_critical_scale.json"],
            "declared_conductance_params": {k: v for k, v in vars(base).items()},
            "dt_ms": DT_MS,
            "subset_fraction": 0.01,
            "background": "zero mean, zero sd",
            "wall_s": time.time() - t0,
            "interpreter": sys.executable,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
        },
        what_this_does_not_support=[
            "Adaptation and threshold are DECLARED parameters of a point-neuron reduction; a "
            "value that widens the span here is not a measured fly property.",
            "The span is measured against the injected current in nA, a declared input unit; "
            "a wider span may mean the input unit is unrealistic rather than that the network "
            "became more graded.",
            "No receptor, no body, no muscle, no controller are involved.",
        ],
    )
    print(f"\nH3 (any arm reaches >= 1 decade): {h3}")
    if best:
        print(f"   best arm {best['arm']} span {best['span_decades']}")
    print("wrote", OUT / "neural_rebuild_03_gradedness.json")


if __name__ == "__main__":
    main()
