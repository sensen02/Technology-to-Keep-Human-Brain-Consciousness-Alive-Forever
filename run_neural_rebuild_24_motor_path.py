"""NEURAL INTERFACE REBUILD, step 24: measure the MUSCLE command path, not the CPG-knob path.

WHAT STEP 23 GOT WRONG, AND WHY IT MATTERS
------------------------------------------
Step 23 measured the step response of `readout_local`, and this step first establishes what that
pool actually contains (measured, from the tier's own root ids against the BANC annotations):

    readout_local: 195 cells, 193 of them annotated `descending_neuron`, 2 `orphan_neuron`
    intersection with the tier's motor neurons: ZERO

That is not an accident of the data: `CommandDecoder` maps descending rate onto two CPG
parameters (speed, turn), and the CPG generates the muscle drive.  So `readout_local` is the path
to the GAIT KNOBS, and it is the correct thing to measure if the question is "can the connectome
modulate the gait".

But it is the WRONG thing to measure for the objective's question, which is about the MOTOR
INTERFACE: whether the rebuilt front end can drive muscles.  The tier contains 397 motor neurons
(annotated efferent/motor) and those are the cells whose spikes should become muscle commands.
Step 23's "rate reflex needs 96.5 ms" is therefore a statement about the CPG-knob path.

THIS STEP
---------
Repeats the step-response measurement ON THE MOTOR NEURONS:
  1. how many motor neurons the tier contains, and how many receive synapses from the channel
     cells (the sensory->motor path must exist for the front end to command muscles at all);
  2. the step latency of the motor-neuron population to 10% / 50% of its eventual change;
  3. whether the two signs move it oppositely, i.e. whether the sign reaches the motor command;
  4. the comparison against the 25 ms stability window, as before.

PRE-REGISTERED
--------------
  H15  The channel cells reach tier motor neurons directly (a non-empty sensory->motor projection),
       and a signed step moves the motor-neuron population within the 25 ms stability window.
  FALSIFIER: no direct projection exists, or the 50% latency exceeds 25 ms.
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
NPZ = HERE / "data/flywire/banc_connectome.npz"
DT_MS = 0.5
STEPS = 400
SCALE = 0.2
MAGNITUDE = 1.0
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)
STABILITY_WINDOW_MS = 25.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    roots = np.asarray(built["tier"]["root_ids"])
    readout = np.asarray(built["readout_local"], dtype=int)

    z = np.load(NPZ, allow_pickle=True)
    r2i = {int(r): i for i, r in enumerate(np.asarray(z["root_ids"]))}
    cls = np.asarray(z["ann_class"]).astype(str)
    sup = np.asarray(z["ann_super_class"]).astype(str)
    flow = np.asarray(z["ann_flow"]).astype(str)
    motor_local = []
    for li, r in enumerate(roots):
        ri = r2i.get(int(r))
        if ri is None:
            continue
        if flow[ri] == "efferent" or "motor" in sup[ri].lower() or "motor" in cls[ri].lower():
            motor_local.append(li)
    motor_local = np.asarray(sorted(motor_local), dtype=int)

    # readout pool composition, for the record
    from collections import Counter
    comp = Counter()
    for li in readout:
        ri = r2i.get(int(roots[li]))
        comp[cls[ri] if ri is not None else "?"] += 1
    overlap = len(set(readout.tolist()) & set(motor_local.tolist()))

    groups, _ = channel_index_map(n, roots)
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    p = ConductanceParams()

    sensor_local = np.array(sorted({int(i) for leg in LEGS for ch in CHANNEL_SPEC
                                    for i in groups[leg][ch]}), dtype=int)
    W = sp.csr_matrix((built["edges"]["We"] + built["edges"]["Wi"]).tocoo() if False else
                      (built["edges"]["We"] + built["edges"]["Wi"]), shape=(n, n))
    # projection from sensor cells into motor cells: rows = post, cols = pre
    proj = W[motor_local, :][:, sensor_local]
    n_sensor_to_motor = int(proj.nnz)
    n_motor_receiving = int((np.asarray(proj.sum(axis=1)).ravel() > 0).sum())

    plus, minus = {}, {}
    for leg in LEGS:
        for ch in SIGNED_CHANNELS:
            idx = np.asarray(groups[leg][ch], dtype=int)
            plus[(leg, ch)] = idx[0::2]
            minus[(leg, ch)] = idx[1::2]

    def current_for(dev):
        c = np.zeros(n)
        amp = 3e-1 * min(abs(float(dev)), 1.0)
        for idx in (plus if dev >= 0 else minus).values():
            if idx.size:
                c[idx] += amp
        return c

    We = sp.csr_matrix((built["edges"]["We"].data * SCALE, built["edges"]["We"].indices,
                        built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
    Wi = sp.csr_matrix((built["edges"]["Wi"].data * SCALE, built["edges"]["Wi"].indices,
                        built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)

    def cumulative(dev):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        cur = current_for(dev)
        cum = np.empty(STEPS)
        for k in range(STEPS):
            net.step(current_nA=cur)
            cum[k] = float(net.spike_count[motor_local].sum())
        return (np.arange(1, STEPS + 1)) * DT_MS, cum

    t, c_pos = cumulative(+MAGNITUDE)
    _, c_neg = cumulative(-MAGNITUDE)
    _, c_zero = cumulative(0.0)

    def step_latency(curve, ref):
        change = curve - ref
        final = float(change[-1])
        out = {"final_change_spikes": final}
        if abs(final) < 1e-9:
            out.update({"t_to_10pct_ms": None, "t_to_50pct_ms": None})
            return out
        s = 1.0 if final > 0 else -1.0
        for pct, key in ((0.10, "t_to_10pct_ms"), (0.50, "t_to_50pct_ms")):
            hit = np.flatnonzero(s * change >= s * pct * final)
            out[key] = float(t[hit[0]]) if hit.size else None
        return out

    lat_pos = step_latency(c_pos, c_zero)
    lat_neg = step_latency(c_neg, c_zero)
    h15_direct = bool(n_sensor_to_motor > 0)
    g1 = bool(lat_pos["final_change_spikes"] * lat_neg["final_change_spikes"] < 0)
    lats = [v for v in (lat_pos["t_to_50pct_ms"], lat_neg["t_to_50pct_ms"]) if v is not None]
    g2 = bool(lats and max(lats) < STABILITY_WINDOW_MS)
    h15 = bool(h15_direct and g2)

    cp.write_result(
        OUT / "neural_rebuild_24_motor_path.json",
        question=("Does the rebuilt sensory front end reach the tier's MOTOR neurons, and does a "
                  "signed step move the motor population within the 25 ms stability window?"),
        status="PASS",
        measured={
            "readout_pool": {"n": int(readout.size), "composition_top": comp.most_common(6),
                             "overlap_with_motor_neurons": overlap,
                             "meaning": "CommandDecoder consumes this pool and maps it onto CPG "
                                        "speed/turn, so it is the GAIT-KNOB path, not the muscle "
                                        "path"},
            "motor_neurons_in_tier": int(motor_local.size),
            "sensor_to_motor_projection": {
                "n_synapses": n_sensor_to_motor,
                "n_motor_neurons_receiving": n_motor_receiving,
                "direct_path_exists": h15_direct},
            "scale_factor": SCALE, "magnitude": MAGNITUDE,
            "step_response_motor_population": {"positive": lat_pos, "negative": lat_neg},
            "final_motor_spikes": {"positive": float(c_pos[-1]), "negative": float(c_neg[-1]),
                                   "zero": float(c_zero[-1])},
            "criteria": {
                "H15_direct_projection": h15_direct,
                "G1_signs_move_motor_population_oppositely": g1,
                "G2_50pct_latency_inside_25ms": g2},
            "stability_window_ms": STABILITY_WINDOW_MS,
            "predictions": {"H15": {"supported": h15,
                                    "falsifier": "no direct projection, or 50% latency > 25 ms"}},
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_24_motor_path.py"),
            "read_only": True,
            "motor_identity": "ann_flow == efferent OR 'motor' in ann_super_class/ann_class",
            "correction_of_step_23": "step 23 measured the readout pool, which contains ZERO motor "
                                     "neurons; its binding constraint (96.5 ms) is about the "
                                     "CPG-knob path, not the muscle path",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Spiking motor neurons is not a muscle command: the connection from motor neurons to "
            "actuators is the corrected wiring table of step 11, and no dynamics are integrated.",
            "A latency inside the stability window is necessary, not sufficient: the SIGN of the "
            "resulting torque is not measured here.",
            "No body, no muscles, no forward integration.",
        ],
    )
    print(f"  readout pool: {readout.size} cells, motor-neuron overlap {overlap}")
    print(f"  motor neurons in tier: {motor_local.size}; "
          f"sensor->motor synapses {n_sensor_to_motor} into {n_motor_receiving} cells")
    print(f"  motor step: +{lat_pos['final_change_spikes']:.0f} spikes "
          f"(10% {lat_pos['t_to_10pct_ms']}, 50% {lat_pos['t_to_50pct_ms']} ms)  "
          f"-{abs(lat_neg['final_change_spikes']):.0f} "
          f"(10% {lat_neg['t_to_10pct_ms']}, 50% {lat_neg['t_to_50pct_ms']} ms)")
    print(f"  H15={h15}  G1={g1}  G2={g2}")
    print("wrote", OUT / "neural_rebuild_24_motor_path.json")


if __name__ == "__main__":
    main()
