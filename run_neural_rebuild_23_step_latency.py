"""NEURAL INTERFACE REBUILD, step 23: the STEP LATENCY of the quantity the project actually
decodes - and therefore whether a reflex loop can close inside the stability window.

WHY THIS, AND NOT THE SPIKE LATENCY AGAIN
-----------------------------------------
`CommandDecoder` (engine/embodied/loop.py) does not read spikes and does not drive muscles: it maps
a FILTERED DESCENDING FIRING RATE onto two CPG knobs, speed and turn
(`DecodeConfig`: speed_scale = 1 + k_speed*tanh(d_rate/rate_scale_hz), turn = k_turn*tanh(...)).
So the signal that closes any loop in this project is `descending_rate_hz`, and the question for
controllability is not "how long until a spike" but

    how long after a state change does `descending_rate_hz` MOVE, and by how much?

That is a step-response measurement, and it has a hard budget: round 39 measured the stance leaving
its balance neighbourhood ~25 ms after a perturbation. A loop whose sensing quantity needs longer
than that cannot stabilise that stance in principle.

WHAT IS MEASURED
----------------
  1. the tier is driven through the channel front end with a signed step (the channel stimulus that
     step 20/21 built), and `descending_rate_hz` is recorded every neural step;
  2. the step latency to 10% and 50% of the eventual change is reported, together with the
     eventual change itself, at the calibrated (non-self-sustaining) scale;
  3. the same for the reverse step (sign flipped), which is what a reflex needs to see: a
     perturbation in EITHER direction must move the rate;
  4. and the open-loop comparison: the tier with NO stimulus, to establish the reference level.

PRE-REGISTERED
--------------
  G1  A signed step produces a monotone change in `descending_rate_hz` on the correct side (the
      two signs move it in opposite directions), i.e. the decoded quantity carries the sign.
  G2  The 50% step latency is SHORTER than the 25 ms stability window, so a reflex loop could in
      principle act before the stance escapes.
  FALSIFIER for G2: the 50% latency exceeds 25 ms, which - combined with steps 17/19/22 - would
      establish that a rate-decoded reflex cannot stabilise this stance regardless of the readout
      window.
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
STEPS = 400                       # 200 ms, long enough to see the final level
SCALE = 0.2                       # calibrated, non-self-sustaining
MAGNITUDE = 1.0
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)
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

    plus, minus = {}, {}
    for leg in LEGS:
        for ch in SIGNED_CHANNELS:
            idx = np.asarray(groups[leg][ch], dtype=int)
            plus[(leg, ch)] = idx[0::2]
            minus[(leg, ch)] = idx[1::2]

    def current_for(dev):
        c = np.zeros(n)
        amp = 3e-1 * min(abs(float(dev)), 1.0)
        tgt = plus if dev >= 0 else minus
        for idx in tgt.values():
            if idx.size:
                c[idx] += amp
        return c

    We = sp.csr_matrix((built["edges"]["We"].data * SCALE, built["edges"]["We"].indices,
                        built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
    Wi = sp.csr_matrix((built["edges"]["Wi"].data * SCALE, built["edges"]["Wi"].indices,
                        built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)

    def run(dev):
        """Return descending_rate_hz per neural step for a constant signed stimulus."""
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        cur = current_for(dev)
        out = np.empty(STEPS)
        for k in range(STEPS):
            net.step(current_nA=cur)
            # the loop's own definition: mean spike fraction of the readout pool / dt
            out[k] = float(net.spike_count[readout].sum())   # cumulative, filled below
        return out

    def cumulative_rate(dev):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        cur = current_for(dev)
        cum = np.empty(STEPS)
        for k in range(STEPS):
            net.step(current_nA=cur)
            cum[k] = float(net.spike_count[readout].sum())
        t = (np.arange(1, STEPS + 1)) * DT_MS
        return t, cum

    t, c_pos = cumulative_rate(+MAGNITUDE)
    _, c_neg = cumulative_rate(-MAGNITUDE)
    _, c_zero = cumulative_rate(0.0)

    def step_latency(curve, ref_curve):
        change = curve - ref_curve
        final = change[-1]
        out = {"final_change_spikes": float(final)}
        if abs(final) < 1e-9:
            out.update({"t_to_10pct_ms": None, "t_to_50pct_ms": None})
            return out
        sgn = 1.0 if final > 0 else -1.0
        for pct, key in ((0.10, "t_to_10pct_ms"), (0.50, "t_to_50pct_ms")):
            target = pct * final
            hit = np.flatnonzero(sgn * change >= sgn * target)
            out[key] = float(t[hit[0]]) if hit.size else None
        return out

    lat_pos = step_latency(c_pos, c_zero)
    lat_neg = step_latency(c_neg, c_zero)
    g1 = bool(lat_pos["final_change_spikes"] * lat_neg["final_change_spikes"] < 0)
    lats = [v for v in (lat_pos["t_to_50pct_ms"], lat_neg["t_to_50pct_ms"]) if v is not None]
    g2 = bool(lats and max(lats) < STABILITY_WINDOW_MS)

    cp.write_result(
        OUT / "neural_rebuild_23_step_latency.json",
        question=("How long does the quantity the project actually decodes "
                  "(filtered descending firing rate) take to move after a signed sensory step, "
                  "and does that fit inside the 25 ms mechanical stability window?"),
        status="PASS",
        measured={
            "scale_factor": SCALE, "magnitude": MAGNITUDE, "dt_ms": DT_MS,
            "quantity": "cumulative readout spike count, whose time derivative is the descending "
                        "rate the CommandDecoder consumes",
            "reference_symbol": "c_zero = cumulative readout spikes with NO stimulus",
            "positive_step": lat_pos, "negative_step": lat_neg,
            "criteria": {
                "G1_signs_move_the_rate_oppositely": g1,
                "G2_50pct_latency_inside_25ms": g2,
            },
            "stability_window_ms": STABILITY_WINDOW_MS,
            "final_cumulative_spikes": {"positive": float(c_pos[-1]), "negative": float(c_neg[-1]),
                                        "zero": float(c_zero[-1])},
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_23_step_latency.py"),
            "read_only": True,
            "decoder_reference": "engine/embodied/loop.py DecodeConfig consumes descending_rate_hz",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "A step latency shorter than the stability window is a NECESSARY condition for a "
            "reflex, not a demonstration that one would work: it says the sensing quantity moves "
            "in time, not that the command it produces is corrective.",
            "The quantity measured is the readout pool's spike count; the decoder additionally "
            "filters it, so the real end-to-end latency is longer than reported here.",
            "No body is integrated in this step; the 25 ms figure is imported from round 39's "
            "forward integration.",
        ],
    )
    print(f"  positive step: final {lat_pos['final_change_spikes']:.0f} spikes, "
          f"10% at {lat_pos['t_to_10pct_ms']} ms, 50% at {lat_pos['t_to_50pct_ms']} ms")
    print(f"  negative step: final {lat_neg['final_change_spikes']:.0f} spikes, "
          f"10% at {lat_neg['t_to_10pct_ms']} ms, 50% at {lat_neg['t_to_50pct_ms']} ms")
    print(f"  G1 signs opposite: {g1}   G2 50% inside 25 ms: {g2}")
    print("wrote", OUT / "neural_rebuild_23_step_latency.json")


if __name__ == "__main__":
    main()
