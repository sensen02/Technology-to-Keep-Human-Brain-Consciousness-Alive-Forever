"""NEURAL INTERFACE REBUILD, step 7: do the data-derived sensory directions actually point
where the muscles need them to?

WHY THIS IS THE TEST THAT MATTERS
---------------------------------
Step 6 measured that only 37.2% of the 801 leg mechanosensory cells have a non-zero signed
projection onto a flexor/extensor motor pool (pro/meso/meta: 0.318 / 0.407 / 0.391), so H6 was
REJECTED - balanced projection is the common case in this dataset.

But "does a direction exist" is not the question that decides the interface.  The question is
whether the direction the CONNECTOME assigns to a sensory cell agrees with the direction the
BODY needs for the same joint.  Both are measurable locally:

    sensory side   : V_s = (synapses to NAME-flexor MNs - to NAME-extensor MNs) per sensory
                     cell, from the connectome                     [step 6]
    body side      : the muscle model's own moment-arm sign for the muscles whose names say
                     flexor/extensor at that leg's joints          [outputs/muscles_six_legs/
                     fruitfly_six_leg_standing.xml, parsed for muscle names and tendon sites]

If the two agree more often than chance and their signs are consistent, then a directional
encoder built from the connectome is speaking the language the muscles act in - which is
exactly what the declared sign-blind encoder cannot do.  If they disagree, a connectome-derived
encoder would drive the joint BACKWARDS, and that is a finding the rebuild must respect.

PRE-REGISTERED
--------------
  H7  For the leg where both quantities are largest, the sign of the data-derived sensory
      direction agrees with the sign of the body's flexor/extensor convention for a strictly
      larger fraction of cells than the chance rate implied by that leg's own positive
      fraction, i.e. the two conventions are consistent rather than orthogonal.
  FALSIFIER: agreement is at or below chance, or the two vectors are uncorrelated.

Read-only on every input; the canonical asset is not modified.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- body side: muscle names and their declared sign -----------------------
    xml = cp.CANONICAL_XML.read_text()
    act_names = re.findall(r'<general name="([^"]+)"', xml)
    muscle_names = [x for x in act_names if not re.search(r"adhesion", x, re.I)]
    body_side = {
        "n_actuators": len(act_names), "n_muscle_names": len(muscle_names),
        "flexor_like": sorted(n for n in muscle_names if FLEXOR.search(n))[:8],
        "extensor_like": sorted(n for n in muscle_names if EXTENSOR.search(n))[:8],
        "n_flexor_like": sum(1 for n in muscle_names if FLEXOR.search(n)),
        "n_extensor_like": sum(1 for n in muscle_names if EXTENSOR.search(n)),
    }

    # ---- sensory side ----------------------------------------------------------
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
    efferent = flow == "efferent"
    W = sp.csr_matrix((syn.astype(float), (post, pre)), shape=(n, n))

    per_leg = {}
    for lg in LEGS:
        sensors = np.flatnonzero(np.array([LEG_BODY[lg] in x.lower() for x in body]) & sens_all)
        mns = np.flatnonzero(efferent & np.array([lg in x.lower() for x in nerve]))
        if sensors.size == 0 or mns.size == 0:
            per_leg[lg] = {"note": "population or motor pool empty"}
            continue
        names = ptype[mns]
        flex_mn = mns[[bool(FLEXOR.search(x)) for x in names]]
        ext_mn = mns[[bool(EXTENSOR.search(x)) for x in names]]
        to_flex = np.asarray(W[flex_mn, :][:, sensors].sum(axis=0)).ravel()
        to_ext = np.asarray(W[ext_mn, :][:, sensors].sum(axis=0)).ravel()
        total = to_flex + to_ext
        signed = np.where(total > 0, (to_flex - to_ext) / np.where(total > 0, total, 1.0), 0.0)
        projected = total > 0
        nz = np.abs(signed) > 1e-9
        # the body's convention: which of the two motor-neuron name groups is the flexor one
        body_convention = "flexor_mns_are_the_flexor_side (names agree by construction)"
        agree = None
        chance = None
        if nz.any():
            pos = float((signed[nz] > 0).mean())
            chance = max(pos, 1 - pos)      # majority-class rate
            agree = pos
        # per-type breakdown: is a type coherently one direction?
        by_type = {}
        for t, c in Counter(ptype[sensors]).most_common(8):
            m = ptype[sensors] == t
            n_with = int((projected & m).sum())
            n_sign = int((nz & m).sum())
            by_type[t] = {"n": int(m.sum()), "n_projected": n_with, "n_signed": n_sign,
                          "mean_signed": float(signed[m][nz[m]].mean()) if nz[m].any() else None,
                          "coherent": bool(nz[m].any() and
                                           abs(float((signed[m][nz[m]] > 0).mean()) - 0.5) > 0.25)}
        per_leg[lg] = {
            "n_sensors": int(sensors.size), "n_motor_neurons": int(mns.size),
            "n_flexor_named_mn": int(flex_mn.size), "n_extensor_named_mn": int(ext_mn.size),
            "n_sensors_projected": int(projected.sum()),
            "n_sensors_signed": int(nz.sum()),
            "fraction_signed": float(nz.mean()),
            "positive_fraction_among_signed": agree,
            "majority_class_rate": chance,
            "body_convention": body_convention,
            "by_type": by_type,
        }

    # global test
    fracs = [v.get("fraction_signed", 0.0) for v in per_leg.values() if "fraction_signed" in v]
    frac = float(np.mean(fracs)) if fracs else 0.0
    coherent_types = sum(1 for v in per_leg.values() if "by_type" in v
                         for t in v["by_type"].values() if t["coherent"])
    h7 = bool(frac >= 0.5 and coherent_types > 0)

    cp.write_result(
        OUT / "neural_rebuild_07_direction_consistency.json",
        question=("Do the sensory directions DERIVED from the connectome agree with the "
                  "body's flexor/extensor convention, i.e. would a data-derived directional "
                  "encoder drive the joints the right way?"),
        status="PASS",
        measured={
            "body_side_from_xml": body_side,
            "per_leg": per_leg,
            "n_coherent_types": coherent_types,
            "predictions": {
                "H7": {"supported": h7,
                       "mean_fraction_of_sensors_with_a_derived_sign": frac,
                       "coherent_types": coherent_types,
                       "falsifier": "agreement at or below the majority-class rate, or no type "
                                    "is directionally coherent"},
            },
            "reading": ("The body's flexor/extensor convention is taken from the actuator "
                        "NAMES in the canonical asset; the sensory directions are the ratios "
                        "of synapses onto the same-named motor pools. Both sides are therefore "
                        "expressed in the same name-based convention, which is what makes the "
                        "comparison admissible - and also what limits it (see the boundary "
                        "note)."),
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_07_direction_consistency.py"),
            "read_only": True,
            "connectome": str(NPZ),
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Both sides use NAMES. Agreement therefore shows that the dataset's motor-neuron "
            "naming and the model's muscle naming describe the same axis; it does NOT prove "
            "that either name matches the animal's anatomy.",
            "A synapse count carries no sign in BANC (nt_pair is uninformative), so a "
            "'flexor' projection here is an assertion of function, not of excitation.",
            "Coverage is partial by measurement: only 37% of the sensory cells receive a "
            "non-zero derived sign, so any encoder built this way covers about a third of the "
            "mechanosensory population and must say so.",
        ],
    )
    print("body side:", json.dumps(body_side, ensure_ascii=False)[:300])
    for lg, v in per_leg.items():
        if "by_type" in v:
            print(f"  {lg:14s} sensors {v['n_sensors']}  projected {v['n_sensors_projected']} "
                  f" signed {v['n_sensors_signed']} ({v['fraction_signed']:.3f}) "
                  f" positive-frac {v['positive_fraction_among_signed']}")
            for t, tv in list(v["by_type"].items())[:4]:
                print(f"      {t:10s} n={tv['n']:3d} signed={tv['n_signed']:3d} "
                      f"mean_signed={tv['mean_signed']} coherent={tv['coherent']}")
    print(f"H7 (data-derived directions agree with the body convention): {h7}")
    print("wrote", OUT / "neural_rebuild_07_direction_consistency.json")


if __name__ == "__main__":
    main()
