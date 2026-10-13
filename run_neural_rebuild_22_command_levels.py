"""NEURAL INTERFACE REBUILD, step 22: the decidable version - how many COMMAND LEVELS can the
front end actually deliver inside its budget?

WHY
---
Step 20/21 replaced "rate" with "latency" coding and measured a 13.5 ms dynamic range
(0.1 rad -> 24 ms, 1.0 rad -> 10.5 ms, and 0.03 rad -> nothing at all).  But a 0.5 ms time
resolution inside a 25 ms window is NOT a count of distinguishable command levels: two drives
that differ by less than the latency jitter cannot be told apart, and this model has a threshold
below which nothing fires.  Reporting "30 levels" from time resolution alone would be the same
class of error as the round-39 unit labels - a number that does not mean what its name says.

So this step measures the thing the objective actually needs, WITH A DECODER:

    a drive magnitude is reported as DECODABLE inside a budget W if a simple nearest-latency
    decoder, calibrated on a set of magnitudes, classifies independent repeats of that magnitude
    correctly, using only spikes that arrive within W of stimulus onset.

Repeats here vary the one thing that CAN vary in this deterministic model: the antagonistic
PARTITION (which half of the cells is '+' and which is '-').  That is a real source of variability
for the code, it is declared, and it is not noise invented to make a decoder look better.

MEASURED
--------
For budgets W = 5, 10, 15, 20, 25 ms and magnitude grids of 3 and 5 levels:
  * which levels ever produce a spike inside the budget (a level that never fires is not a level);
  * the leave-one-partition-out classification accuracy of a nearest-latency decoder;
  * the resulting number of levels that are BOTH reachable and separable inside the budget;
  * the same at the DECLARED (self-sustaining) scale as a comparison, since it is faster.
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
from engine.embodied.sensory_channels import (CHANNEL_SPEC, LEGS,  # noqa: E402
                                              ChannelSensoryEncoder, ChannelConfig,
                                              channel_index_map)
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
DT_MS = 0.5
STEPS = 60                      # 30 ms is enough for every latency measured so far
SCALES = (0.2, 1.0)
BUDGETS_MS = (5.0, 10.0, 15.0, 20.0, 25.0)
GRIDS = {"3_levels": (0.1, 0.3, 1.0), "5_levels": (0.1, 0.2, 0.3, 0.5, 1.0)}
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)
PARTITIONS = ("even_odd", "odd_even", "first_half", "second_half")


def build_partition(groups, kind):
    plus, minus = {}, {}
    for leg in LEGS:
        for ch in SIGNED_CHANNELS:
            idx = np.asarray(groups[leg][ch], dtype=int)
            if kind == "even_odd":
                plus[(leg, ch)], minus[(leg, ch)] = idx[0::2], idx[1::2]
            elif kind == "odd_even":
                plus[(leg, ch)], minus[(leg, ch)] = idx[1::2], idx[0::2]
            else:
                h = idx.size // 2
                plus[(leg, ch)], minus[(leg, ch)] = (idx[:h], idx[h:]) if kind == "first_half" \
                    else (idx[h:], idx[:h])
    return plus, minus


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    readout = np.asarray(built["readout_local"], dtype=int)
    groups, _ = channel_index_map(n, np.asarray(built["tier"]["root_ids"]))
    p = ConductanceParams()

    results = {}
    for f in SCALES:
        We = sp.csr_matrix((built["edges"]["We"].data * f, built["edges"]["We"].indices,
                            built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
        Wi = sp.csr_matrix((built["edges"]["Wi"].data * f, built["edges"]["Wi"].indices,
                            built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)
        for pname in PARTITIONS:
            plus, minus = build_partition(groups, pname)

            def cur_for(dev):
                c = np.zeros(n)
                amp = 3e-1 * min(abs(float(dev)), 1.0)
                tgt = plus if dev >= 0 else minus
                for idx in tgt.values():
                    if idx.size:
                        c[idx] += amp
                return c

            def latency(dev):
                net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
                net.set_connectivity(We, Wi)
                cur = cur_for(dev)
                cum = 0
                for k in range(STEPS):
                    cum += int(net.step(current_nA=cur)[readout].sum())
                    if cum >= 1:
                        return (k + 1) * DT_MS
                return None

            for gname, mags in GRIDS.items():
                for sign, slab in ((+1, "positive"), (-1, "negative")):
                    table = {m: latency(sign * m) for m in mags}
                    key = f"{f}|{pname}|{gname}|{slab}"
                    results[key] = {"scale_factor": f, "partition": pname, "grid": gname,
                                    "sign": slab, "latency_by_magnitude": table}
        print(f"  scale {f} done", flush=True)

    # ---- decoding: nearest latency, grouped by partition -----------------------
    analysis = {}
    for gname, mags in GRIDS.items():
        for slab in ("positive", "negative"):
            for budget in BUDGETS_MS:
                key = f"{gname}|{slab}|{budget:g}"
                per_scale = {}
                for f in SCALES:
                    rows = [v for k, v in results.items()
                            if v["scale_factor"] == f and v["grid"] == gname and v["sign"] == slab]
                    # reference = the mean latency per magnitude over the OTHER partitions
                    correct = total = 0
                    levels_reachable = set()
                    for i, r in enumerate(rows):
                        others = [o for j, o in enumerate(rows) if j != i]
                        ref = {}
                        for m in mags:
                            vals = [o["latency_by_magnitude"][m] for o in others
                                    if o["latency_by_magnitude"][m] is not None
                                    and o["latency_by_magnitude"][m] <= budget]
                            if vals:
                                ref[m] = float(np.mean(vals))
                        if not ref:
                            continue
                        for m in mags:
                            obs = r["latency_by_magnitude"][m]
                            total += 1
                            if obs is None or obs > budget:
                                continue
                            levels_reachable.add(m)
                            guess = min(ref, key=lambda k: abs(ref[k] - obs))
                            if guess == m:
                                correct += 1
                    per_scale[f"{f:g}"] = {
                        "accuracy": (correct / total) if total else None,
                        "n_trials": total,
                        "levels_reachable_within_budget": sorted(levels_reachable),
                        "n_levels_reachable": len(levels_reachable),
                    }
                analysis[key] = per_scale

    # the decidable summary: levels that are BOTH reachable and separable at the calibrated scale
    summary = {}
    for budget in BUDGETS_MS:
        entry = analysis[f"3_levels|positive|{budget:g}"]["0.2"]
        entry5 = analysis[f"5_levels|positive|{budget:g}"]["0.2"]
        summary[f"{budget:g}"] = {
            "budget_ms": budget,
            "n_levels_reachable_3level_grid": entry["n_levels_reachable"],
            "accuracy_3level_grid": entry["accuracy"],
            "n_levels_reachable_5level_grid": entry5["n_levels_reachable"],
            "accuracy_5level_grid": entry5["accuracy"],
            "decodable_levels": (entry["n_levels_reachable"]
                                 if (entry["accuracy"] or 0) >= 0.8 else 0),
        }

    usable = [b for b, v in summary.items() if v["decodable_levels"] >= 3]
    cp.write_result(
        OUT / "neural_rebuild_22_command_levels.json",
        question=("How many command levels can the rebuilt front end actually deliver within a "
                  "given time budget, measured with a decoder rather than from time resolution?"),
        status="PASS",
        measured={
            "scales": list(SCALES), "grids": {k: list(v) for k, v in GRIDS.items()},
            "budgets_ms": list(BUDGETS_MS),
            "partitions_used_as_repeats": list(PARTITIONS),
            "latency_tables": results,
            "decoder_analysis": analysis,
            "summary_by_budget_at_calibrated_scale": summary,
            "budgets_with_at_least_3_decodable_levels": usable,
            "command_window_ms": 5.0,
            "mechanical_stability_window_ms": 25.0,
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_22_command_levels.py"),
            "read_only": True,
            "decoder": "nearest-latency, leave-one-partition-out; the reference latency per "
                       "magnitude is the mean over the other three partitions",
            "repeats": "the four antagonistic PARTITIONS are the source of variability; this is "
                       "declared, not noise added to flatter the decoder",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "The decoder is a diagnostic, not the project's command decoder: it shows how much "
            "information is available in the latency, not how a muscle command would be formed.",
            "A level count inside a budget is not a controller: nothing here closes a loop, and the "
            "body is excluded.",
            "The antagonistic partition is declared (the dataset carries no per-cell sign), so the "
            "partition-to-partition variability is a property of that choice.",
            "No claim is made about fly behaviour or about real proprioceptive coding.",
        ],
    )
    print(json.dumps(summary, indent=1))
    print("budgets with >=3 decodable levels:", usable)
    print("wrote", OUT / "neural_rebuild_22_command_levels.json")


if __name__ == "__main__":
    main()
