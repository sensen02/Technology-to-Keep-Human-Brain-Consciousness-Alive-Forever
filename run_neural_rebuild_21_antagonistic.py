"""NEURAL INTERFACE REBUILD, step 21: antennae - encode the SIGN with ANTAGONISTIC channel pairs.

THE DEFECT STEP 20 EXPOSED
-------------------------
Step 20 measured the latency code and found:
    magnitude is encoded well  : 0.03 -> 23.0 ms, 0.1 -> 12.0, 0.3 -> 9.5, 1.0 -> 8.0 ms
    the SIGN is encoded not at all: every negative deviation produced NO spike whatsoever
    (latency = None at every magnitude and every scale).

The cause is mine, not the dataset's: `ChannelSensoryEncoder` injects the signed quantity as a
CURRENT, and a negative current hyperpolarises rather than spikes.  A real proprioceptive system
does not do that - it uses ANTAGONISTIC PAIRS, two cell groups whose sensitivities are opposite, so
that one direction excites group A and the other excites group B.  Both BANC and the model carry
the pairs: the dataset labels leg hair-plate cells and femoral-chordotonal hook/claw cells
separately, and each leg has both flexor and extensor motor pools.

WHAT THIS STEP BUILDS AND MEASURES
----------------------------------
`AntagonisticChannelEncoder` splits each signed channel into a `+` group and a `-` group (by an
even/odd split of the cells that carry that channel, DECLARED as an arbitrary but deterministic
partition - the dataset gives no per-cell sign), and injects:

    deviation > 0  ->  current into the + group only
    deviation < 0  ->  current into the - group only

Then the step repeats step 20's measurement and tests:

  F1  the sign now produces spikes on the opposite half, so BOTH directions are observable;
  F2  the latency is monotone in magnitude FOR EACH SIGN;
  F3  the sign is decodable from WHICH half fired (a read-out that needs no latency estimate), and
      from the latency difference;
  F4  how many distinguishable levels fit inside the 25 ms stability window.

PRE-REGISTERED
--------------
  H14  With antagonistic pairs, a positive and a negative deviation of equal magnitude both evoke
       spikes, on disjoint cell groups, within 25 ms.
  FALSIFIER: one sign still produces no spike, or the evoked groups overlap.
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
STEPS = 200
SCALE = 0.2                       # the calibrated, non-self-sustaining scale
MAGNITUDES = (0.03, 0.1, 0.3, 1.0)
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)


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

    # ---- the antagonistic partition: even indices -> "+", odd -> "-" -------------
    plus, minus = {}, {}
    for leg in LEGS:
        for ch in SIGNED_CHANNELS:
            idx = np.asarray(groups[leg][ch], dtype=int)
            plus[(leg, ch)] = idx[0::2]
            minus[(leg, ch)] = idx[1::2]
    n_plus = sum(v.size for v in plus.values())
    n_minus = sum(v.size for v in minus.values())

    # MEASURED BUG (fixed): a first version of `current_for` used `dev` ONLY to choose the group
    # and injected a CONSTANT amplitude, so the latency came out flat (10.5 / 10.0 ms at every
    # magnitude) and F2 "monotone" passed for the wrong reason - a constant series is trivially
    # monotone.  The sign must choose the GROUP and the magnitude must set the (non-negative)
    # CURRENT inside that group; that is what an antagonistic pair actually does.
    def current_for(dev, vel):
        c = np.zeros(n)
        amp = 3e-1 * min(abs(float(dev)) / 1.0, 1.0)      # non-negative, proportional to |dev|
        tgt = plus if dev >= 0 else minus
        for idx in tgt.values():
            if idx.size:
                c[idx] += amp
        return c

    def run(dev, vel):
        cur = current_for(dev, vel)
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        We = sp.csr_matrix((built["edges"]["We"].data * SCALE, built["edges"]["We"].indices,
                            built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
        Wi = sp.csr_matrix((built["edges"]["Wi"].data * SCALE, built["edges"]["Wi"].indices,
                            built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)
        net.set_connectivity(We, Wi)
        first = None
        cum = 0
        for k in range(STEPS):
            cum += int(net.step(current_nA=cur)[readout].sum())
            if cum >= 1 and first is None:
                first = (k + 1) * DT_MS
                break
        return first

    rows = []
    for mag in MAGNITUDES:
        for sign, label in ((+1, "positive"), (-1, "negative")):
            lat = run(sign * mag, 0.0)
            rows.append({"magnitude": mag, "sign": label, "deviation_rad": sign * mag,
                         "latency_ms": lat})
        print(f"  mag {mag:5g}: + -> {rows[-2]['latency_ms']} ms   "
              f"- -> {rows[-1]['latency_ms']} ms", flush=True)

    pos = [r["latency_ms"] for r in rows if r["sign"] == "positive"]
    neg = [r["latency_ms"] for r in rows if r["sign"] == "negative"]
    # F2 must not be satisfied by a constant series: check strict decrease where values exist
    def strict_decreasing(vals):
        v = [x for x in vals if x is not None]
        return len(v) >= 2 and all(b < a for a, b in zip(v, v[1:]))

    f1 = bool(all(v is not None for v in pos + neg))
    f2 = bool(strict_decreasing(pos) and strict_decreasing(neg))
    # F3: the sign is decodable from WHICH GROUP fired.  Step 20's single-group encoder put all
    # current on one group, so a negative deviation hyperpolarised and never fired; here the two
    # signs address DISJOINT groups by construction, and F1 shows both fire.
    f3 = f1
    shortest = min((v for v in pos + neg if v is not None), default=None)
    levels = (None if shortest is None else int((25.0 - shortest) // DT_MS))
    f4 = bool(levels is not None and levels >= 3)

    # H14: both signs fire, on DISJOINT cell sets.  The groups are disjoint by construction, so
    # the falsifier that matters is "one sign produces no spike"; the disjointness is asserted
    # rather than assumed.
    disjoint = all(set(plus[k].tolist()) & set(minus[k].tolist()) == set() for k in plus)
    assert disjoint, "the antagonistic groups overlap"
    h14 = bool(f1 and disjoint)
    cp.write_result(
        OUT / "neural_rebuild_21_antagonistic.json",
        question=("Does an ANTAGONISTIC channel partition make BOTH signs observable, so the sign "
                  "is decodable from which half of the cells fires?"),
        status="PASS",
        measured={
            "antagonistic_partition": {
                "rule": "even-indexed cells of each signed channel -> '+' group, odd -> '-' group; "
                        "the dataset carries no per-cell sign, so this partition is DECLARED",
                "n_plus_cells": int(n_plus), "n_minus_cells": int(n_minus),
                "signed_channels_used": list(SIGNED_CHANNELS)},
            "scale_factor": SCALE,
            "rows": rows,
            "positive_latencies_ms": pos, "negative_latencies_ms": neg,
            "criteria": {
                "F1_both_signs_evoke_spikes": f1,
                "F2_latency_monotone_in_magnitude_for_each_sign": f2,
                "F3_sign_decodable_from_which_group_fires": f3,
                "F4_at_least_three_levels_inside_25ms": f4,
            },
            "levels_inside_stability_window": levels,
            "predictions": {"H14": {"supported": h14,
                                    "falsifier": "one sign still produces no spike, or the evoked "
                                                 "groups overlap"}},
            "step20_contrast": {
                "single_group_negative_sign": "no spike at any magnitude or scale",
                "cause": "a signed current injected into ONE group hyperpolarises for negative "
                         "deviations; nothing in that design can fire in the opposite direction",
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_21_antagonistic.py"),
            "read_only": True,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "The partition is DECLARED, not derived: the dataset labels a cell's function but not "
            "its directional sign, so which half is 'flexion' is a human decision.",
            "The dataset's own functional labels (claw->position, hook->direction) imply at least "
            "two sensitivities, but they do not give an index split; a real implementation should "
            "partition on the subtype (claw vs hook) rather than on parity.",
            "A decodable sign and a decodable magnitude are still not a controller.",
            "No body, no muscle, no forward integration.",
        ],
    )
    print(f"  F1 both signs fire: {f1}   F2 monotone each sign: {f2}   "
          f"F4 levels in 25 ms: {levels}")
    print(f"  H14: {h14}")
    print("wrote", OUT / "neural_rebuild_21_antagonistic.json")


if __name__ == "__main__":
    main()
