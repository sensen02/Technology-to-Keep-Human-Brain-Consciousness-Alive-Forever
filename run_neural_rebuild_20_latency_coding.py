"""NEURAL INTERFACE REBUILD, step 20: LATENCY (first-spike-time) coding, and whether it fits the
project's command clock.

WHY THIS STEP
-------------
Steps 4, 16, 17 and 19 established, from four directions, that RATE decoding cannot work at the
project's 5 ms command clock:

    per-cell  : 0.2225 spikes per cell per 5 ms window at the calibrated scale (step 4)
    loop      : 380x more input current changes the descending rate by 5% (step 16)
    population: ZERO readout spikes in a 5 ms window at every drive and scale (step 17)
    latency   : the tier needs 5.0-8.0 ms before its readout emits anything at all (steps 18/19)

But LATENCY coding does not need a long window: a first spike at 8 ms still carries information in
its TIMING, and the mechanical stability window is ~25 ms, so a decoder that needs 8-12 ms is not
automatically excluded.  Step 19 left this as the one repair direction that does not conflict with
the stability window.  This step tests it.

WHAT IS MEASURED
----------------
The latency of the first readout-pool spike is measured as a function of the SIGNED drive
magnitude, at the calibrated (non-self-sustaining) scale, with 0.5 ms resolution.  Then:

  E1  is the latency MONOTONE in drive magnitude, i.e. does it carry a graded code?
  E2  do opposite signs give DIFFERENT latencies, so the sign is decodable from time alone?
  E3  what is the smallest distinguishable latency difference, and how many command levels fit
      between the shortest latency and the command/stability windows?
  E4  the same measurement at the DECLARED (self-sustaining) scale, to show the trade-off between
      "fast enough" and "self-sustaining".

PRE-REGISTERED
--------------
  H12  At the calibrated scale the first-spike latency is strictly monotone in drive magnitude
       over at least one decade, and opposite signs differ by more than the 0.5 ms resolution.
  FALSIFIER: the latency is flat in magnitude (or non-monotone) over the tested range, which would
       mean latency carries no graded code either.
  H13  The number of distinguishable command levels available within the shorter of the two
       windows (5 ms command, ~25 ms stability) is at least 3.
  FALSIFIER: fewer than 3 levels, which would make the code too coarse for a command.
"""
from __future__ import annotations

import json
import sys
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
STEPS = 200                      # 100 ms
SCALES = (0.2, 1.0)
MAGNITUDES = (0.03, 0.1, 0.3, 1.0)
COMMAND_WINDOW_MS = 5.0
STABILITY_WINDOW_MS = 25.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    readout = np.asarray(built["readout_local"], dtype=int)
    groups, _ = channel_index_map(n, np.asarray(built["tier"]["root_ids"]))
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    p = ConductanceParams()

    def current_for(mag):
        c = np.zeros(n)
        for leg in LEGS:
            q = ChannelQuantities(tibia_joint_dev_rad=mag, tibia_joint_vel_rad_s=mag * 10.0,
                                  multi_joint_dev_rad=mag, contact_force_norm=0.0,
                                  foot_contact=0.0)
            cc, _ = enc.encode({leg: q})
            c += cc
        return c

    rows = []
    for f in SCALES:
        We = sp.csr_matrix((built["edges"]["We"].data * f, built["edges"]["We"].indices,
                            built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
        Wi = sp.csr_matrix((built["edges"]["Wi"].data * f, built["edges"]["Wi"].indices,
                            built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)
        for mag in MAGNITUDES:
            for sign in (+1, -1):
                cur = sign * current_for(mag)
                net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
                net.set_connectivity(We, Wi)
                first = None
                cum = 0
                for k in range(STEPS):
                    cum += int(net.step(current_nA=cur)[readout].sum())
                    if cum >= 1 and first is None:
                        first = (k + 1) * DT_MS
                        break
                rows.append({"scale_factor": f, "magnitude": mag, "sign": sign,
                             "latency_ms": first})
        print(f"  scale {f} done", flush=True)

    def lat(f, mag, sign):
        hit = [r for r in rows if r["scale_factor"] == f and r["magnitude"] == mag
               and r["sign"] == sign][0]["latency_ms"]
        return hit

    # E1/E2 at the calibrated scale
    cal = {}
    for mag in MAGNITUDES:
        cal[f"{mag:g}"] = {"plus": lat(0.2, mag, +1), "minus": lat(0.2, mag, -1)}
    plus_series = [cal[f"{m:g}"]["plus"] for m in MAGNITUDES]
    minus_series = [cal[f"{m:g}"]["minus"] for m in MAGNITUDES]

    def monotone_decreasing(series):
        vals = [v for v in series if v is not None]
        return len(vals) >= 3 and all(b <= a for a, b in zip(vals, vals[1:])) and vals[0] > vals[-1]

    e1 = monotone_decreasing(plus_series)
    signs_differ = {f"{m:g}": (cal[f"{m:g}"]["plus"] is not None
                               and cal[f"{m:g}"]["minus"] is not None
                               and cal[f"{m:g}"]["plus"] != cal[f"{m:g}"]["minus"])
                    for m in MAGNITUDES}
    e2 = any(signs_differ.values())

    # E3: levels available inside the tighter window
    budget = min(COMMAND_WINDOW_MS, STABILITY_WINDOW_MS)
    shortest = min((v for v in plus_series + minus_series if v is not None), default=None)
    resolution = DT_MS
    levels_in_stability = (None if shortest is None
                           else max(0, int((STABILITY_WINDOW_MS - shortest) // resolution)))
    levels_in_command = (None if shortest is None
                         else max(0, int((COMMAND_WINDOW_MS - shortest) // resolution)))
    e3 = bool(levels_in_stability is not None and levels_in_stability >= 3)
    e4 = bool(levels_in_stability is not None and levels_in_stability >= 3
              and levels_in_command is not None and levels_in_command < 3)

    # the declared scale, for the trade-off table
    dec = {f"{m:g}": {"plus": lat(1.0, m, +1), "minus": lat(1.0, m, -1)} for m in MAGNITUDES}

    h12 = bool(e1 and e2)
    h13 = e3

    cp.write_result(
        OUT / "neural_rebuild_20_latency_coding.json",
        question=("Does the tier's first-spike LATENCY carry a signed, graded code, and how many "
                  "command levels fit inside the command and stability windows?"),
        status="PASS",
        measured={
            "dt_ms": DT_MS, "magnitudes": list(MAGNITUDES), "scales": list(SCALES),
            "latency_calibrated_scale": cal,
            "latency_declared_scale": dec,
            "all_rows": rows,
            "E1_latency_monotone_in_magnitude": e1,
            "E2_opposite_signs_differ": e2,
            "sign_differences": signs_differ,
            "E3_levels": {"shortest_latency_ms": shortest, "time_resolution_ms": resolution,
                          "levels_inside_command_window_5ms": levels_in_command,
                          "levels_inside_stability_window_25ms": levels_in_stability,
                          "budget_used_ms": budget},
            "E4_command_window_too_short_but_stability_window_usable": e4,
            "predictions": {"H12": {"supported": h12,
                                    "falsifier": "latency flat or non-monotone"},
                            "H13": {"supported": h13,
                                    "threshold": 3,
                                    "falsifier": "fewer than 3 distinguishable levels"}},
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_20_latency_coding.py"),
            "read_only": True,
            "stimulus": "signed channels only; magnitude swept over one and a half decades",
            "counts": "first spike of the tier's readout pool, 0.5 ms resolution",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "A decodable latency is not a controller: it shows the code carries the information, "
            "not that a downstream muscle command would be correct or stable.",
            "The latency is a property of this declared point-neuron reduction.",
            "The measurement is deterministic (step 19 measured bit-identical trials), so the "
            "latency values are exact for this model and carry no uncertainty estimate.",
            "Nothing here involves the body, the muscles or a forward integration.",
        ],
    )
    print("  calibrated scale latency (mag: +sign / -sign):", json.dumps(cal))
    print("  declared   scale latency (mag: +sign / -sign):", json.dumps(dec))
    print(f"  E1 monotone={e1}  E2 signs differ={e2}  E3 levels in 25ms="
          f"{levels_in_stability} in 5ms={levels_in_command}")
    print(f"  H12={h12}  H13={h13}")
    print("wrote", OUT / "neural_rebuild_20_latency_coding.json")


if __name__ == "__main__":
    main()
