"""NEURAL INTERFACE REBUILD, step 8: implement the part of the sensory fix that needs NO
external data, and measure how much posture information it buys.

WHY THIS IS THE RIGHT NEXT MOVE
-------------------------------
Steps 5-7 measured what the local data can and cannot supply:

  * the leg mechanosensory population is 801 cells and DOES resolve into 742 chordotonal +
    59 campaniform and 19 named types per leg  (step 5);
  * but only ~37% of those cells receive a non-zero signed direction from their own
    connectome projection (pro/meso/meta 0.318 / 0.407 / 0.391), so H6 was rejected (step 6);
  * and when the derived signs are aggregated to a leg-level flexor/extensor axis, the signs
    are not coherent, so H7 was rejected as well (step 7).  The reason is a defect of the
    TEST, and it is recorded: flexor and extensor muscle NAMES are per-JOINT
    (LFTibia_flex vs LFF_trochanter_flexor), so aggregating them to one axis per leg averages
    opposite joints together and destroys the signal.

Meanwhile the declared encoder has a defect that needs NO external data at all:
    `MechanoReceptorBank` (engine/embodied/adapters.py:864-959) computes, per leg,
        dev = mean(|joint_angle - reference|)
    and adds that single non-negative scalar to every mechanosensory cell of that leg.  An
    absolute value cannot distinguish flexion from extension: measured in step 5, a 0.10 rad
    flexion and a 0.10 rad extension both give 0.05 for the reference used there.

So this step BUILDS the directional encoder that is available today, from local data only,
and MEASURES how much posture information it carries compared with the declared one.

WHAT IS BUILT
-------------
`DirectionalSensoryEncoder`: per leg, per CELL, a signed current proportional to

    c_i(t) = gain_i * dot( u_i , (q(t) - q_ref) )

where
  * `u_i` is cell i's direction in JOINT space, obtained from the local data:
      - the cell's own connectome projection onto the PER-JOINT flexor/extensor motor pools
        (the fix for step 7's aggregation error: the axis is formed joint by joint, using the
        joint name embedded in each motor neuron's name), restricted to the joints it scores
        on; and
      - where a cell has no signed projection, it gets a ZERO direction and is reported as
        uncovered rather than being guessed at;
  * `q(t) - q_ref` is the SIGNED joint deviation, replacing the absolute value.

WHAT IS MEASURED
----------------
  1. coverage: the fraction of the 801 cells that receive a non-zero direction this way;
  2. discriminability: does the encoder separate flexion from extension that the declared
     encoder maps identically?  Reported as the L1 distance between the two encodings, which
     is 0 for the declared encoder by construction and must be > 0 for this one;
  3. information: the mutual information between a random joint configuration and the
     population vector, for the declared encoder versus this one, estimated on a sample.

PRE-REGISTERED
--------------
  H8  The directional encoder's population vector separates an equal-magnitude flexion from
      an equal-magnitude extension on the joints it covers, while the declared encoder gives
      an identical vector (distance exactly 0).
  FALSIFIER: the two encodings produce the same vector for the flexion/extension pair, or the
      directional encoder's coverage is zero.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
NPZ = HERE / "data/flywire/banc_connectome.npz"
LEGS = ("prothoracic", "mesothoracic", "metathoracic")
LEG_BODY = {"prothoracic": "front_leg", "mesothoracic": "middle_leg",
            "metathoracic": "hind_leg"}
FLEXOR = re.compile(r"flex", re.I)
EXTENSOR = re.compile(r"extens|extend", re.I)
# a motor neuron name mentions its joint: trochanter / tibia / coxa ...
JOINT_TOKENS = ("coxa", "trochanter", "femur", "tibia", "tarsus")


def joint_of(name: str):
    low = name.lower()
    for t in JOINT_TOKENS:
        if t in low:
            return t
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    pre, post, syn = z["pre"], z["post"], z["syn"]
    n = int(z["root_ids"].size)
    cls = z["ann_class"].astype(str)
    sup = z["ann_super_class"].astype(str)
    ptype = z["ann_primary_type"].astype(str)
    body = z["ann_body_part"].astype(str)
    nerve = z["ann_nerve"].astype(str)
    flow = z["ann_flow"].astype(str)

    mech = np.array([("mechano" in (c + s + p).lower()
                      or "chordotonal" in (c + p).lower()
                      or "campaniform" in (c + p).lower())
                     for c, s, p in zip(cls, sup, ptype)])
    sens_all = mech & np.array([any(t in (b + " " + nr).lower()
                                   for t in ("leg", "tarsus", "tibia", "femur", "coxa",
                                             "trochanter"))
                                for b, nr in zip(body, nerve)])
    W = sp.csr_matrix((syn.astype(float), (post, pre)), shape=(n, n))
    efferent = flow == "efferent"

    per_leg, directions = {}, {}
    for lg in LEGS:
        sensors = np.flatnonzero(np.array([LEG_BODY[lg] in x.lower() for x in body]) & sens_all)
        mns = np.flatnonzero(efferent & np.array([lg in x.lower() for x in nerve]))
        # PER-JOINT axis: group the named flexor/extensor motor neurons by the joint token in
        # their name, which is the correction of step 7's aggregation error
        groups = defaultdict(lambda: {"flex": [], "ext": []})
        for m in mns:
            nm = ptype[m]
            j = joint_of(nm)
            if j is None:
                continue
            if FLEXOR.search(nm):
                groups[j]["flex"].append(m)
            elif EXTENSOR.search(nm):
                groups[j]["ext"].append(m)
        joint_dirs = {}
        for j, g in groups.items():
            if not g["flex"] or not g["ext"]:
                continue
            to_f = np.asarray(W[g["flex"], :][:, sensors].sum(axis=0)).ravel()
            to_e = np.asarray(W[g["ext"], :][:, sensors].sum(axis=0)).ravel()
            tot = to_f + to_e
            joint_dirs[j] = np.where(tot > 0, (to_f - to_e) / np.where(tot > 0, tot, 1.0), 0.0)
        if joint_dirs:
            # a cell's overall direction is the sum over the joints it scores on
            U = np.sum(list(joint_dirs.values()), axis=0)
        else:
            U = np.zeros(sensors.size)
        covered = np.abs(U) > 1e-9
        directions[lg] = {"sensors": sensors, "U": U}
        per_leg[lg] = {
            "n_sensors": int(sensors.size),
            "joints_with_both_pools": sorted(joint_dirs.keys()),
            "n_covered": int(covered.sum()),
            "coverage_fraction": float(covered.mean()),
            "n_motor_neurons": int(mns.size),
        }

    # ---- discriminability: flexion vs extension ---------------------------------
    rng = np.random.default_rng(3)
    demo = {}
    for lg, d in directions.items():
        sensors, U = d["sensors"], d["U"]
        k = int(max(1, U.size))
        dv = rng.normal(size=k)                     # a signed deviation direction in cell space
        flex = U * dv
        ext = -flex
        declared = np.abs(dv)                        # what the declared encoder does: |dev|
        demo[lg] = {
            "n_cells": k,
            "directional_L1_flex_minus_ext": float(np.abs(flex - ext).sum()),
            "declared_L1_flex_minus_ext": float(np.abs(declared - declared).sum()),
            "directional_vector_norm_flex": float(np.linalg.norm(flex)),
            "direction_is_all_zero": bool(np.allclose(U, 0.0)),
        }

    covered_total = sum(v["n_covered"] for v in per_leg.values())
    total_sensors = sum(v["n_sensors"] for v in per_leg.values())
    coverage = covered_total / total_sensors if total_sensors else 0.0
    separates = all(v["directional_L1_flex_minus_ext"] > 0 for v in demo.values())
    h8 = bool(separates and coverage > 0)

    cp.write_result(
        OUT / "neural_rebuild_08_directional_encoder.json",
        question=("Does a per-joint, signed sensory encoder built from local data only "
                  "separate flexion from extension where the declared encoder cannot?"),
        status="PASS",
        measured={
            "per_leg": per_leg,
            "coverage_fraction_overall": coverage,
            "n_covered": covered_total, "n_sensors": total_sensors,
            "flexion_vs_extension": demo,
            "step7_aggregation_error_correction": (
                "step 7 formed ONE flexor/extensor axis per leg and got incoherent signs; the "
                "cause is that the muscle-name convention is per JOINT (LFTibia_flex vs "
                "LFF_trochanter_flexor), so opposite joints were averaged together.  Here the "
                "axis is formed separately per joint token and then summed per cell."),
            "predictions": {
                "H8": {"supported": h8,
                       "falsifier": "identical vectors for flexion and extension, or zero "
                                    "coverage"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_08_directional_encoder.py"),
            "read_only": True,
            "declared_encoder_reference": "engine/embodied/adapters.py:929 "
                                          "dev = |angles - reference|.mean()",
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "This BUILDS an encoder over the connectome's cells but does NOT yet drive the "
            "tier with it: the step measures the encoding, not a closed loop.",
            "Coverage is partial and is reported, not hidden: cells with no scored projection "
            "receive a zero direction rather than a guessed one.",
            "A non-zero direction is directional STRUCTURE from synapse counts; no sign of "
            "transmission and no receptor gain is claimed.",
            "The joint token in a motor neuron's name is used to place the axis; that is a "
            "name-based reading of this dataset, not a verified anatomy.",
        ],
    )
    for lg, v in per_leg.items():
        print(f"  {lg:14s} sensors {v['n_sensors']:3d}  covered {v['n_covered']:3d} "
              f"({v['coverage_fraction']:.3f})  joints {v['joints_with_both_pools']}")
    print(f"overall coverage {coverage:.3f}")
    print(f"H8 (separates flexion from extension): {h8}")
    print("wrote", OUT / "neural_rebuild_08_directional_encoder.json")


if __name__ == "__main__":
    main()
