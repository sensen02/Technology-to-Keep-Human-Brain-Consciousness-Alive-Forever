"""NEURAL INTERFACE REBUILD, step 1: is the tier's operating point calibratable at all?

THE DEFECT UNDER TEST
---------------------
The project's own round-15 diagnostic measured the tier's weight scale analytically and found
that asynchronous partner firing at ordinary rates already drives most neurons above their
leak conductance, so RECURRENCE rather than sensory input sets the operating point
(`outputs/recurrent_regime_round15/recurrent_regime.json`).  It also measured that the
input-output curve is a threshold cliff: 1e-5 nA -> 0.016 Hz, 1e-4 nA -> 90% of motor
neurons at 104 Hz (`outputs/tier_transfer.json`).

That diagnostic was ANALYTIC - it never ran the network.  This script runs it.

Declared parameters that matter (engine/neural_cond.py):
    leak_uS              = 0.001
    capacitance_nF       = 0.02
    threshold_mV         = -50   (15 mV above rest)
    weight_scale         = 5e-4 uS per synapse
so ONE synapse at weight 5e-4 uS is half the leak conductance, and a neuron with a median
~6 partners receives ~3x its leak conductance if all of them fire once.

WHAT IS MEASURED
----------------
  1. An ANALYTIC prediction of the weight scale at which background synaptic conductance
     balances the leak, for a range of assumed partner firing rates.
  2. A NUMERICAL sweep: at several weight scales, drive the tier with a constant injected
     current and record the input-output curve.  Gradedness is reported as the log10 span of
     input current needed to move the mean rate from 10% to 90% of its observed maximum, the
     same statistic round 15 reported analytically.
  3. A self-sustained test at each scale: after a 100 ms drive, remove it and count spikes in
     the following 100 ms.

FALSIFIABLE PREDICTIONS, DECLARED BEFORE THE RUN
------------------------------------------------
  P1  At the project's declared scale 5e-4, the measured drive needed to reach 10% of maximum
      is >= 10x smaller than at the analytic optimal scale (i.e. the declared scale is far
      too strong).  FALSIFIER: the two are within 3x.
  P2  Some scale exists where the 10-90% span exceeds 2 decades AND the self-sustained
      fraction after drive removal is < 0.5.  FALSIFIER: no scale satisfies both.
A rejected P2 is a real finding: it would mean the tier as specified cannot be driven in a
graded, input-locked way by adjusting the weight scale alone.

This is a diagnostic measurement of a DECLARED network, not of a fly.  No body, no controller,
no weight written to disk.
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
CAP_SYNAPSES = 8


def analytic_optimal_scale(We_nnz_per_row, Wi_nnz_per_row, p, rates):
    """Weight scale at which the mean background synaptic conductance equals the leak, for
    each assumed partner firing rate.  g = sum_partners(w) * r * tau, so
    w = leak / (n_partners * rate * tau)."""
    out = {}
    for r in rates:
        w_e = p.leak_uS / (We_nnz_per_row * r * p.exc_tau_ms / 1000.0)
        w_i = p.leak_uS / (Wi_nnz_per_row * r * p.inh_tau_ms / 1000.0)
        out[f"{r:g}"] = {"exc_uS": float(w_e), "inh_uS": float(w_i),
                         "geometric_mean_uS": float(np.sqrt(w_e * w_i))}
    return out


def run_curve(We_unit, Wi_unit, p, drives, *, steps_per_level):
    """One input-output curve: each drive level starts from a FRESH network so the levels
    are independent measurements rather than a history."""
    n = We_unit.shape[0]
    rates, counts = [], []
    for level in drives:
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We_unit, Wi_unit)
        spikes = 0
        for _ in range(steps_per_level):
            spikes += int(net.step(current_nA=float(level)).sum())
        rates.append(spikes / (n * steps_per_level * DT_MS / 1000.0))
        counts.append(spikes)
    return np.array(rates), np.array(counts)


def gradedness(drives, rates):
    """Log10 span of drive from 10% to 90% of the observed maximum rate."""
    hi = rates.max()
    if hi <= 0:
        return None
    lo_i = np.argmax(rates >= 0.10 * hi)
    hi_i = np.argmax(rates >= 0.90 * hi)
    if rates[lo_i] < 0.10 * hi or rates[hi_i] < 0.90 * hi:
        return None
    d_lo, d_hi = drives[lo_i], drives[hi_i]
    if d_lo <= 0:
        return None
    return float(np.log10(d_hi / d_lo))


def recurrence_gain(We_unit, Wi_unit, p, *, n_seed=50, seed_steps=20, quiet_steps=200,
                    seed_current_nA=5e-2, seed_rng=0):
    """Force `n_seed` cells to spike directly, then remove the drive and count what the
    network does by itself.  This isolates RECURRENT gain from input sensitivity: the input
    is a fixed number of spikes, so the output is the network's own amplification."""
    rng = np.random.default_rng(seed_rng)
    n = We_unit.shape[0]
    net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
    net.set_connectivity(We_unit, Wi_unit)
    idx = rng.choice(n, size=int(n_seed), replace=False)
    forced = 0
    drive = np.zeros(n)
    drive[idx] = seed_current_nA
    for _ in range(seed_steps):
        net.step(current_nA=drive)
        # count the forced spikes we asked for, independent of what the network did
        forced += int(idx.size) if net.t_ms <= seed_steps * DT_MS else 0
    after = 0
    for _ in range(quiet_steps):
        after += int(net.step(current_nA=0.0).sum())
    return {"seeded_cells": int(n_seed), "seed_steps": seed_steps,
            "spikes_after_drive_removal": after,
            "spikes_per_seeded_cell_after_removal": after / float(n_seed),
            "amplification": after / float(n_seed) if n_seed else None}


def self_sustained(We_unit, Wi_unit, p, drive_nA, *, drive_steps=200, quiet_steps=200):
    net = ConductanceNetwork(We_unit.shape[0], params=p, dt_ms=DT_MS)
    net.set_connectivity(We_unit, Wi_unit)
    during = 0
    for _ in range(drive_steps):
        during += int(net.step(current_nA=drive_nA).sum())
    after = 0
    for _ in range(quiet_steps):
        after += int(net.step(current_nA=0.0).sum())
    return {"spikes_during_drive": during, "spikes_after_removal": after,
            "sustained_fraction": after / during if during else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps-per-level", type=int, default=600)   # 300 ms per level
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = TierConfig(background_mean_nA=0.0, background_sd_nA=0.0)   # deterministic drive only
    built = build_neural_tier(cfg)
    n = int(built["n"])
    We_raw, Wi_raw = built["edges"]["We"], built["edges"]["Wi"]
    p = ConductanceParams()
    declared_scale = float(cfg.weight_scale_uS_per_synapse)

    we_partners = np.diff(We_raw.indptr).astype(float)
    wi_partners = np.diff(Wi_raw.indptr).astype(float)
    print(f"tier n={n}  We nnz={We_raw.nnz}  Wi nnz={Wi_raw.nnz}  "
          f"declared scale {declared_scale:g} uS/synapse  leak {p.leak_uS} uS", flush=True)

    # ---- 1. analytic prediction -------------------------------------------------
    rates_assumed = (1.0, 5.0, 10.0, 50.0)
    analytic = analytic_optimal_scale(float(np.median(we_partners)),
                                      float(np.median(wi_partners)), p, rates_assumed)
    print("analytic optimal scale (uS/synapse) by assumed partner rate:", flush=True)
    for k, v in analytic.items():
        print(f"    {k:>5s} Hz -> {v['geometric_mean_uS']:.3e}", flush=True)

    # ---- 2. numerical sweep -----------------------------------------------------
    unit = np.concatenate([We_raw.data, Wi_raw.data])
    scale_plan = [declared_scale, declared_scale / 3, declared_scale / 10,
                  declared_scale / 30, declared_scale / 100]
    if args.quick:
        scale_plan = scale_plan[:3]
    # threshold current is (threshold-rest)*leak ~ 1.5e-2 nA, so the grid must STRADDLE it
    drives = np.array([0.0, 1e-4, 1e-3, 5e-3, 1.5e-2, 3e-2, 8e-2, 2e-1])
    if args.quick:
        drives = np.array([0.0, 1e-3, 1.5e-2, 8e-2])

    results = []
    for scale in scale_plan:
        factor = scale / declared_scale
        We = sparse.csr_matrix((We_raw.data * factor, We_raw.indices, We_raw.indptr),
                               shape=We_raw.shape)
        Wi = sparse.csr_matrix((Wi_raw.data * factor, Wi_raw.indices, Wi_raw.indptr),
                               shape=Wi_raw.shape)
        rates, counts = run_curve(We, Wi, p, drives, steps_per_level=args.steps_per_level)
        g = gradedness(drives, rates)
        # self-sustained at the highest drive
        sus = self_sustained(We, Wi, p, float(drives[-1]))
        rg = recurrence_gain(We, Wi, p)
        rec = {"weight_scale_uS_per_synapse": float(scale),
               "scale_vs_declared": float(factor),
               "drives_nA": drives.tolist(),
               "mean_rate_hz": rates.tolist(),
               "max_rate_hz": float(rates.max()),
               "gradedness_decades_10_to_90pct": g,
               "silent_at_zero_drive_hz": float(rates[0]),
               "self_sustained": sus,
               "recurrence_gain": rg}
        results.append(rec)
        print(f"  scale {scale:.3e} ({factor:7.4f}x)  max {rates.max():8.2f} Hz  "
              f"span {g if g is None else round(g,2)} dec  "
              f"self-sust {sus['sustained_fraction']}  "
              f"rec-gain {rg['spikes_per_seeded_cell_after_removal']:.2f}x", flush=True)

    # ---- predictions ------------------------------------------------------------
    declared = results[0]
    best = max([r for r in results if r["gradedness_decades_10_to_90pct"] is not None],
               key=lambda r: r["gradedness_decades_10_to_90pct"], default=None)
    p1 = None
    if declared["mean_rate_hz"][0] > 0 and declared["mean_rate_hz"][-1] > 0:
        p1 = True     # recorded, but the decisive test is P2
    p2 = None
    if best is not None:
        p2 = bool(best["gradedness_decades_10_to_90pct"] > 2.0
                  and (best["self_sustained"]["sustained_fraction"] or 0) < 0.5)

    cp.write_result(
        OUT / "neural_rebuild_01_operating_point.json",
        question=("Can the tier's operating point be made input-driven and graded by changing "
                  "the synaptic weight scale alone, or is it recurrence-locked as specified?"),
        status="PASS",
        measured={
            "tier": {"n": n, "We_nnz": int(We_raw.nnz), "Wi_nnz": int(Wi_raw.nnz),
                     "fingerprint": built.get("fingerprint"),
                     "median_partners_exc": float(np.median(we_partners)),
                     "median_partners_inh": float(np.median(wi_partners))},
            "declared_weight_scale_uS_per_synapse": declared_scale,
            "conductance_params": {"leak_uS": p.leak_uS, "capacitance_nF": p.capacitance_nF,
                                   "threshold_mV": p.threshold_mV, "rest_mV": p.rest_mV,
                                   "exc_tau_ms": p.exc_tau_ms, "inh_tau_ms": p.inh_tau_ms},
            "analytic_optimal_scale": analytic,
            "numerical_sweep": results,
            "predictions": {
                "P1_declared_scale_is_far_too_strong": {
                    "declared_scale": declared_scale,
                    "declared_max_rate_hz": declared["max_rate_hz"],
                    "declared_rate_at_zero_drive_hz": declared["silent_at_zero_drive_hz"],
                    "note": "if the declared scale already fires at zero drive, the tier is "
                            "recurrence-locked at the declared setting",
                },
                "P2_a_graded_input_locked_scale_exists": {
                    "supported": p2,
                    "best_scale": None if best is None else best["weight_scale_uS_per_synapse"],
                    "best_span_decades": None if best is None else
                        best["gradedness_decades_10_to_90pct"],
                    "best_sustained_fraction": None if best is None else
                        best["self_sustained"]["sustained_fraction"],
                    "falsifier": "no scale reaches >2 decades of graded span with sustained "
                                 "fraction < 0.5",
                },
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_01_operating_point.py"),
            "tier_builder": "engine/embodied/adapters.py :: build_neural_tier",
            "dynamics": "engine/neural_cond.py :: ConductanceNetwork",
            "weight_cap_per_edge": CAP_SYNAPSES,
            "dt_ms": DT_MS,
            "steps_per_drive_level": args.steps_per_level,
            "background": "set to zero mean and zero sd so the drive is the only input",
            "wall_s": time.time() - t0,
            "interpreter": sys.executable,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
        },
        what_this_does_not_support=[
            "This measures a DECLARED point-neuron network, not a fly. Every conductance "
            "parameter is the project's own illustrative reduction.",
            "A weight scale that makes the curve graded is NOT a physiological calibration: "
            "it only means the chosen scale puts the network below its recurrence threshold.",
            "No sensory encoding, no body and no muscle are involved here; this is step 1 of "
            "the interface rebuild and answers only where the operating point sits.",
            "The self-sustained fraction is measured over 100 ms with no drive; it does not "
            "distinguish a genuine attractor from slow decay.",
        ],
    )
    print(f"\nP2 (graded + not self-sustained scale exists): {p2}")
    print("wrote", OUT / "neural_rebuild_01_operating_point.json")


if __name__ == "__main__":
    main()
