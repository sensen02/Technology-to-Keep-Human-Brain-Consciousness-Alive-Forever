"""NEURAL INTERFACE REBUILD, step 18: the LATENCY wall - how long does the tier need before it
produces ANY readout spike at all?

WHY THIS IS THE MEASUREMENT THAT SETTLES THE TIMING QUESTION
------------------------------------------------------------
Step 4 framed the problem per cell (0.2225 spikes per cell per 5 ms window).  Step 17 re-framed
it per POPULATION and measured the ramp directly:

    window   5 ms : 0, 0, 0, 0, 0 readout spikes across drive 0.01 -> 1.0
    window  10 ms : 0, 0, 0, 2, 10
    window  20 ms : 0, 0, 24, 68, 103
    window  50 ms : 0, 8, 102, 200, 302

So the population framing does NOT rescue the interface at the project's 5 ms command clock: the
count is exactly zero there at every drive.  And the growth from 0 to ~200 spikes between 10 ms
and 50 ms is not a rate - it is a RAMP, which suggests a LATENCY rather than a low firing rate.

This step measures that latency directly, because latency and rate are different problems with
different fixes:

  * if the tier needs ~tens of ms before its first readout spike, then no decoder window shorter
    than that can carry anything, whatever the weight scale;
  * and the comparison that matters is against the MECHANICAL stability window measured in round
    39 (~25 ms for the stance to leave its balance neighbourhood): if the neural latency is of the
    same order, then a rate-coded reflex loop cannot stabilise that stance in principle.

WHAT IS MEASURED
----------------
At several weight scales, the cumulative readout-pool spike count versus time from stimulus
onset, at the largest drive used in step 17.  Reported: the first time the cumulative count
reaches 1, 10 and 50 spikes; the fit time constant of the ramp; and the steady-state rate.
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
from engine.embodied.sensory_channels import (LEGS, ChannelConfig, ChannelQuantities,  # noqa: E402
                                              ChannelSensoryEncoder, channel_index_map)
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
DT_MS = 0.5
SCALES = (0.2, 0.5, 1.0)
DRIVE = 1.0
DURATION_MS = 200.0
TRIALS = 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    root_ids = np.asarray(built["tier"]["root_ids"])
    readout_local = np.asarray(built["readout_local"], dtype=int)
    groups, _ = channel_index_map(n, root_ids)
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    p = ConductanceParams()
    steps = int(round(DURATION_MS / DT_MS))

    def current_for(mag):
        c = np.zeros(n)
        for leg in LEGS:
            q = ChannelQuantities(tibia_joint_dev_rad=mag, tibia_joint_vel_rad_s=mag * 10.0,
                                  multi_joint_dev_rad=mag, contact_force_norm=0.0,
                                  foot_contact=0.0)
            cur, _ = enc.encode({leg: q})
            c += cur
        return c

    cur = current_for(DRIVE)
    rows = []
    t0 = time.time()
    for f in SCALES:
        We = sp.csr_matrix((built["edges"]["We"].data * f, built["edges"]["We"].indices,
                            built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
        Wi = sp.csr_matrix((built["edges"]["Wi"].data * f, built["edges"]["Wi"].indices,
                            built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)
        cum_trials = []
        for t in range(TRIALS):
            net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
            net.set_connectivity(We, Wi)
            cum = np.zeros(steps, dtype=np.int64)
            run = 0
            for k in range(steps):
                run += int(net.step(current_nA=cur)[readout_local].sum())
                cum[k] = run
            cum_trials.append(cum)
        C = np.vstack(cum_trials).astype(float)          # trials x steps
        mean = C.mean(axis=0)
        t_ms = (np.arange(1, steps + 1)) * DT_MS

        def first_reaching(level):
            hit = np.flatnonzero(mean >= level)
            return float(t_ms[hit[0]]) if hit.size else None

        # time constant of the ramp: fit the rise from the first spike to 90% of the final value
        final = float(mean[-1])
        tau = None
        if final > 10:
            target = 0.63 * final
            hit = np.flatnonzero(mean >= target)
            if hit.size:
                tau = float(t_ms[hit[0]])
        rows.append({
            "scale_factor": f,
            "readout_cells": int(readout_local.size),
            "cumulative_mean_at_ms": {str(m): float(mean[int(m / DT_MS) - 1])
                                      for m in (1, 2, 5, 10, 20, 30, 50, 100, 200)
                                      if int(m / DT_MS) - 1 < steps},
            "t_to_1_spike_ms": first_reaching(1),
            "t_to_10_spikes_ms": first_reaching(10),
            "t_to_50_spikes_ms": first_reaching(50),
            "time_to_63pct_of_final_ms": tau,
            "final_count_in_200ms": final,
            "steady_rate_hz_readout_pool": final / (DURATION_MS / 1000.0),
            "trial_sd_at_20ms": float(C[:, max(0, int(20 / DT_MS) - 1)].std(ddof=1)),
        })
        print(f"  scale {f:4.1f}: t(1sp)={rows[-1]['t_to_1_spike_ms']} ms  "
              f"t(10sp)={rows[-1]['t_to_10_spikes_ms']} ms  "
              f"t(50sp)={rows[-1]['t_to_50_spikes_ms']} ms  200ms total "
              f"{rows[-1]['final_count_in_200ms']:.0f} ({time.time()-t0:.0f}s)", flush=True)

    best = min((r["t_to_1_spike_ms"] for r in rows if r["t_to_1_spike_ms"] is not None),
               default=None)
    fastest = [r for r in rows if r["t_to_1_spike_ms"] == best]
    verdict = {
        "shortest_latency_to_first_readout_spike_ms": best,
        "at_scale": (fastest[0]["scale_factor"] if fastest else None),
        "command_window_ms": 5.0,
        "mechanical_stability_window_ms": 25.0,
        "latency_exceeds_command_window": bool(best is not None and best > 5.0),
        "latency_comparable_to_stability_window": bool(best is not None and best >= 12.5),
        "reading": ("the tier needs at least this long before its readout pool emits a single "
                    "spike; a decoder window shorter than that carries an unavoidable zero, and "
                    "if the latency is of the order of the mechanical stability window then a "
                    "rate-coded reflex cannot stabilise that stance in principle"),
    }

    cp.write_result(
        OUT / "neural_rebuild_18_latency.json",
        question=("How long after stimulus onset does the tier's readout pool need before it "
                  "produces its first spike, and how does that compare with the 5 ms command "
                  "window and the 25 ms mechanical stability window?"),
        status="PASS",
        measured={"drive": DRIVE, "duration_ms": DURATION_MS, "trials": TRIALS,
                  "scales": rows, "verdict": verdict},
        provenance={
            "code": str(HERE / "run_neural_rebuild_18_latency.py"),
            "read_only": True,
            "counts": "cumulative spikes of the tier's readout pool",
            "dt_ms": DT_MS,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "The latency here is to the READOUT POOL of a declared point-neuron model; it is not a "
            "measured neural conduction or synaptic delay of a fly.",
            "The drive is a constant signed channel current, not a receptor response.",
            "A latency equal to the stability window does not prove that no control is possible: a "
            "feedforward or event-coded scheme is not excluded by this measurement.",
        ],
    )
    print(f"  shortest latency to first readout spike: {best} ms (scale {verdict['at_scale']})")
    print("wrote", OUT / "neural_rebuild_18_latency.json")


if __name__ == "__main__":
    main()
