"""NEURAL INTERFACE REBUILD, step 26: does the SIGN change WHICH motor neurons respond?

WHY THIS BEFORE ASKING FOR EXTERNAL DATA
----------------------------------------
Steps 24 and 25 measured the motor population's TOTAL spike count for a positive and a negative
deviation and found them nearly equal (1611 vs 1804 with an arbitrary partition, 2771 vs 2844 with
a data-derived one).  I concluded that the sign does not survive to the motor output, and that the
blocker was the missing transmission polarity (BANC's `nt_pair` and `nt_neuron` are all -1).

But that conclusion may have been too quick: a near-equal TOTAL says nothing about WHICH cells
fired.  If a positive deviation drives one set of motor neurons and a negative deviation drives a
DISJOINT set, then the sign IS present as a population pattern, and a decoder could read it without
any polarity data - the polarity would only be needed to know which pattern means "corrective",
not to tell the two patterns apart.

So before declaring an external-data gap, this step measures the PATTERN:
  * the per-motor-neuron spike counts under a positive and a negative step, and under no stimulus;
  * the Jaccard overlap of the activated sets;
  * the correlation of the two count vectors;
  * and, as a positive control, the overlap between two runs that SHOULD be identical (the same
    stimulus twice) and between the stimulus and the zero-stimulus run.

PRE-REGISTERED
--------------
  H16  The activated motor-neuron sets differ substantially between the two signs (Jaccard overlap
       below 0.5) while the same-sign repeats overlap almost completely, i.e. the sign is carried by
       WHICH cells fire rather than by HOW MANY.
  FALSIFIER: the two signs' activated sets overlap as much as the repeats do, which would mean the
       motor output is genuinely sign-blind and only polarity data could rescue it.
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
DIRECTION = OUT / "sensory_direction_lookup.json"
DT_MS = 0.5
STEPS = 400
SCALE = 0.2
MAGNITUDE = 1.0
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    n = int(built["n"])
    roots = np.asarray(built["tier"]["root_ids"])

    z = np.load(NPZ, allow_pickle=True)
    r2i = {int(r): i for i, r in enumerate(np.asarray(z["root_ids"]))}
    flow = np.asarray(z["ann_flow"]).astype(str)
    sup = np.asarray(z["ann_super_class"]).astype(str)
    cls = np.asarray(z["ann_class"]).astype(str)
    motor_local = np.array(sorted(
        li for li, r in enumerate(roots)
        if (lambda ri: ri is not None and (flow[ri] == "efferent"
                                           or "motor" in sup[ri].lower()
                                           or "motor" in cls[ri].lower()))(r2i.get(int(r)))), dtype=int)

    dirs = json.loads(DIRECTION.read_text())["direction_by_root"]
    groups, _ = channel_index_map(n, roots)
    enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
    p = ConductanceParams()

    pos_local = [li for li, r in enumerate(roots)
                 if dirs.get(str(int(r)), {}).get("signed_direction", 0.0) > 0]
    neg_local = [li for li, r in enumerate(roots)
                 if dirs.get(str(int(r)), {}).get("signed_direction", 0.0) < 0]

    We = sp.csr_matrix((built["edges"]["We"].data * SCALE, built["edges"]["We"].indices,
                        built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
    Wi = sp.csr_matrix((built["edges"]["Wi"].data * SCALE, built["edges"]["Wi"].indices,
                        built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)

    def run(dev):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        c = np.zeros(n)
        if dev != 0.0:
            amp = 3e-1 * min(abs(dev), 1.0)
            tgt = pos_local if dev > 0 else neg_local
            if tgt:
                c[np.asarray(tgt)] += amp
        for _ in range(STEPS):
            net.step(current_nA=c)
        return net.spike_count[motor_local].astype(float).copy()

    counts = {"pos": run(+MAGNITUDE), "neg": run(-MAGNITUDE), "zero": run(0.0),
              "pos_repeat": run(+MAGNITUDE)}

    def jaccard(a, b):
        A, B = set(np.flatnonzero(a > 0).tolist()), set(np.flatnonzero(b > 0).tolist())
        if not (A | B):
            return None
        return len(A & B) / len(A | B)

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    pairs = {
        "pos_vs_neg": (counts["pos"], counts["neg"]),
        "pos_vs_zero": (counts["pos"], counts["zero"]),
        "pos_vs_pos_repeat": (counts["pos"], counts["pos_repeat"]),
        "neg_vs_zero": (counts["neg"], counts["zero"]),
    }
    metrics = {k: {"jaccard": jaccard(a, b), "pearson": corr(a, b),
                   "n_active_a": int((a > 0).sum()), "n_active_b": int((b > 0).sum()),
                   "total_a": float(a.sum()), "total_b": float(b.sum())}
               for k, (a, b) in pairs.items()}

    j_same = metrics["pos_vs_pos_repeat"]["jaccard"]
    j_diff = metrics["pos_vs_neg"]["jaccard"]
    h16 = bool(j_diff is not None and j_diff < 0.5 and j_same is not None and j_same > 0.9)

    cp.write_result(
        OUT / "neural_rebuild_26_sign_pattern.json",
        question=("Is the sign carried by WHICH motor neurons fire rather than by how many, so a "
                  "decoder could read it without transmission-polarity data?"),
        status="PASS",
        measured={
            "n_motor_neurons": int(motor_local.size),
            "n_pos_group": len(pos_local), "n_neg_group": len(neg_local),
            "metrics": metrics,
            "control": {"same_stimulus_twice_jaccard": j_same,
                        "note": "the model is deterministic (step 19 measured bit-identical "
                                "trials), so this control must be exactly 1.0; if it is not, the "
                                "comparison of the two signs is not interpretable"},
            "predictions": {"H16": {"supported": h16,
                                    "falsifier": "the two signs' activated sets overlap as much as "
                                                 "the repeats do"}},
            "verdict": {
                "sign_present_as_population_pattern": bool(j_diff is not None and j_diff < 0.5),
                "sign_present_only_as_magnitude": bool(j_diff is not None and j_diff >= 0.5),
                "reading": ("if the signs' activated sets overlap strongly, the motor output is "
                            "sign-blind at the pattern level too, and the missing transmission "
                            "polarity is the binding gap; if they are largely disjoint, the sign is "
                            "recoverable from the pattern and polarity is needed only to label which "
                            "pattern is corrective"),
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_26_sign_pattern.py"),
            "read_only": True,
            "partition_source": "outputs/neural_rebuild/sensory_direction_lookup.json "
                                "(data-derived per-joint flexor/extensor projection)",
            "scale_factor": SCALE, "magnitude": MAGNITUDE, "dt_ms": DT_MS, "steps": STEPS,
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "A population pattern that separates the signs does not make it CORRECTIVE: polarity is "
            "still needed to know whether the pattern pushes the joint back or further.",
            "The partition is derived from NAMES (motor-neuron action labels), and a synapse count "
            "carries no transmission sign, so 'projects to the flexor pool' is not 'excites it'.",
            "No body, no muscles, no forward integration.",
        ],
    )
    for k, v in metrics.items():
        print(f"  {k:22s} jaccard {v['jaccard']}  corr {v['pearson']}  "
              f"active {v['n_active_a']}/{v['n_active_b']}  total {v['total_a']:.0f}/{v['total_b']:.0f}")
    print(f"  H16 (sign present as pattern): {h16}")
    print("wrote", OUT / "neural_rebuild_26_sign_pattern.json")


if __name__ == "__main__":
    main()
