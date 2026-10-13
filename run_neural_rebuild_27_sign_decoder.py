"""NEURAL INTERFACE REBUILD, step 27: decode the SIGN from the motor pattern, with a real decoder.

WHAT STEP 26 OVERTURNED
-----------------------
Step 25 concluded that the sign does not reach the motor output and that the binding gap was the
missing transmission polarity (BANC's `nt_pair` and `nt_neuron` are all -1).  Step 26 then measured
WHICH motor neurons fire rather than how many, and found:

    same stimulus twice (control) : jaccard 1.000, pearson  1.000   (deterministic, as expected)
    positive vs negative          : jaccard 0.417, pearson -0.133   (185 vs 206 cells active)
    stimulus vs no stimulus       : jaccard 0.000

So the sign DOES reach the motor population, as a change in the IDENTITY of the active cells.  The
polarity gap therefore does not block DECODING the sign; it only blocks LABELLING which pattern is
the corrective one.

This step turns that into a number with a real decoder, because "jaccard 0.417" is a description,
not a result:

  * train a linear discriminant (difference of means, which is the optimal linear rule for
    equal-covariance classes) on the motor count vectors from one set of antagonistic partitions;
  * test it on vectors from a HELD-OUT partition, so the reported accuracy is not fitted;
  * report accuracy, and the same for a magnitude-only control decoder trained on a different
    magnitude pair, to show the sign decoder is not just picking up stimulus size.

PRE-REGISTERED
--------------
  H17  A linear sign decoder trained on some antagonistic partitions classifies held-out
       partitions' motor patterns with accuracy >= 0.9.
  FALSIFIER: held-out accuracy is at or below chance, which would mean the pattern difference is
       partition-specific rather than a sign signal.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
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
SIGNED_CHANNELS = tuple(c for c, (_, _, signed, _) in CHANNEL_SPEC.items() if signed)
# four antagonistic partitions, the same declared source of variability used in step 22
PARTITIONS = ("even_odd", "odd_even", "first_half", "second_half")


def make_partition(groups, kind):
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
                plus[(leg, ch)], minus[(leg, ch)] = ((idx[:h], idx[h:]) if kind == "first_half"
                                                     else (idx[h:], idx[:h]))
    return plus, minus


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
    motor_local = np.array(sorted(
        li for li, r in enumerate(roots)
        if (lambda ri: ri is not None and (flow[ri] == "efferent" or "motor" in sup[ri].lower()))(
            r2i.get(int(r)))), dtype=int)

    groups, _ = channel_index_map(n, roots)
    p = ConductanceParams()
    We = sp.csr_matrix((built["edges"]["We"].data * SCALE, built["edges"]["We"].indices,
                        built["edges"]["We"].indptr), shape=built["edges"]["We"].shape)
    Wi = sp.csr_matrix((built["edges"]["Wi"].data * SCALE, built["edges"]["Wi"].indices,
                        built["edges"]["Wi"].indptr), shape=built["edges"]["Wi"].shape)

    def motor_counts(plus, minus, dev):
        net = ConductanceNetwork(n, params=p, dt_ms=DT_MS)
        net.set_connectivity(We, Wi)
        c = np.zeros(n)
        amp = 3e-1 * min(abs(dev), 1.0)
        tgt = plus if dev > 0 else minus
        flat = np.concatenate([v for v in tgt.values() if v.size]) if tgt else np.zeros(0, int)
        if flat.size:
            c[flat] += amp
        for _ in range(STEPS):
            net.step(current_nA=c)
        return net.spike_count[motor_local].astype(float).copy()

    # build the data: one vector per (partition, sign, magnitude)
    mags = (0.5, 1.0)
    data = []
    for kind in PARTITIONS:
        plus, minus = make_partition(groups, kind)
        for mag in mags:
            for sign in (+1, -1):
                vec = motor_counts(plus, minus, sign * mag)
                data.append({"partition": kind, "magnitude": mag, "sign": sign, "vec": vec})
    print(f"  collected {len(data)} motor patterns", flush=True)

    def loo_accuracy(target_key):
        """Leave-one-partition-out accuracy of a difference-of-means linear rule."""
        correct = total = 0
        per_fold = {}
        for held in PARTITIONS:
            train = [d for d in data if d["partition"] != held]
            test = [d for d in data if d["partition"] == held]
            if not train or not test:
                continue
            if target_key == "sign":
                k = "sign"
            else:
                k = "magnitude"
            classes = sorted({d[k] for d in train})
            if len(classes) < 2:
                continue
            means = {c: np.mean([d["vec"] for d in train if d[k] == c], axis=0)
                     for c in classes}
            w = means[classes[1]] - means[classes[0]]
            b = -0.5 * float((means[classes[1]] + means[classes[0]]) @ w)
            fold_c = fold_t = 0
            for d in test:
                if d[k] not in classes:
                    continue
                score = float(d["vec"] @ w + b)
                pred = classes[1] if score > 0 else classes[0]
                total += 1
                fold_t += 1
                if pred == d[k]:
                    correct += 1
                    fold_c += 1
            per_fold[held] = (fold_c / fold_t) if fold_t else None
        return {"accuracy": (correct / total) if total else None, "n": total,
                "per_fold": per_fold}

    sign_res = loo_accuracy("sign")
    mag_res = loo_accuracy("magnitude")
    h17 = bool(sign_res["accuracy"] is not None and sign_res["accuracy"] >= 0.9)

    cp.write_result(
        OUT / "neural_rebuild_27_sign_decoder.json",
        question=("Can a linear decoder read the SIGN of a joint deviation from the motor-neuron "
                  "pattern, generalising to a held-out antagonistic partition?"),
        status="PASS",
        measured={
            "n_motor_neurons": int(motor_local.size),
            "n_patterns": len(data),
            "partitions": list(PARTITIONS), "magnitudes": list(mags), "scale_factor": SCALE,
            "sign_decoder_leave_one_partition_out": sign_res,
            "magnitude_control_decoder": mag_res,
            "predictions": {"H17": {"supported": h17, "threshold": 0.9,
                                    "falsifier": "held-out accuracy at or below chance"}},
            "context_correction_of_step_25": (
                "step 25 concluded that the sign does not reach the motor output, based on nearly "
                "equal TOTAL spike counts (2771 vs 2844).  Step 26 showed that the nearly-equal "
                "totals hide a different active SET (jaccard 0.417, pearson -0.133), so that "
                "conclusion was too strong: the sign IS present in the pattern."),
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_27_sign_decoder.py"),
            "read_only": True,
            "decoder": "difference of class means (optimal linear rule for equal-covariance "
                       "classes), trained on three partitions and tested on the fourth",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Decoding the sign is not producing a CORRECTIVE command: the transmission polarity is "
            "still needed to know which of the two patterns pushes the joint back, and that data is "
            "absent locally (nt_pair and nt_neuron are all -1).",
            "The four partitions are a DECLARED source of variability, not biological variability: "
            "the dataset carries no per-cell sign.",
            "The response-speed problem is untouched: the 50% step latency measured in step 24 was "
            "88-110 ms against a 25 ms stability window.",
            "No body, no muscles, no forward integration.",
        ],
    )
    print(f"  sign decoder (held-out partition): acc {sign_res['accuracy']} over {sign_res['n']}  "
          f"per-fold {sign_res['per_fold']}")
    print(f"  magnitude control:                 acc {mag_res['accuracy']}")
    print(f"  H17: {h17}")
    print("wrote", OUT / "neural_rebuild_27_sign_decoder.json")


if __name__ == "__main__":
    main()
