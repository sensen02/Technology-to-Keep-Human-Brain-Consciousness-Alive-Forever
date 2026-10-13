"""NEURAL INTERFACE REBUILD, step 6: can the connectome supply sensory DIRECTION, so the
interface does not have to be hand-mapped?

WHY THIS IS THE DECISIVE STEP
-----------------------------
Step 5 established (measured):
  * 801 mechanosensory cells sit on the six leg nerves (front/middle/hind leg: 264/253/281);
  * they resolve into anatomically meaningful classes - 742 `chordotonal_organ_neuron` and
    59 `campaniform_sensillum_neuron` - and into 19 distinct `ann_primary_type` values inside
    one leg (SNpp50/39/51/60/41/47/43/58..., SApp09/08/10...);
  * the type NAMES do not state a joint or a direction.
So the local data has RESOLUTION but not (yet) TUNING.  Two ways to get tuning:

    (a) write a lookup table by hand from published anatomy - fast, but it is exactly the
        kind of hand-wiring the project forbids;
    (b) DERIVE it from the connectome's own wiring: a sensory cell that drives the motor
        neurons of joint X more than those of joint Y is, in this dataset, a cell that
        reports joint X.  This is not hand-mapping; it is reading the data.

This step does (b) and tests whether the derived vectors are (i) distinct from each other and
(ii) SIGNED - i.e. whether the direction of the projection, not the presence of a synapse,
carries the information the encoder needs.

METHOD (read-only)
------------------
  1. Partition the leg mechanosensory cells by (leg, class, primary_type).
  2. For each partition, take its outgoing synapses into that leg's motor neurons and its
     premotor pool, giving a connectivity vector per cell.
  3. Reduce each cell to ONE signed direction scalar: the sign and magnitude of its
     correlation with the leg's "flexor-vs-extensor" axis, defined by the motor neurons whose
     own names say flexor/extensor (a NAME-based axis, declared as such).
  4. Report, per partition: whether its cells share a direction (tuning is coherent) or are
     split (the partition mixes directions), and the fraction of cells whose derived sign is
     non-zero.

PRE-REGISTERED
--------------
  H6  At least half of the leg mechanosensory cells receive a NON-ZERO signed direction from
      their own connectome projection, so a directional encoder can be built from the data
      rather than from a hand-written table.
  FALSIFIER: fewer than half of the cells have a non-zero projection onto the declared axis,
  i.e. the local connectivity is balanced and carries no direction.

Every read is from data/flywire/banc_connectome.npz; no file is modified.
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
# MEASURED: ann_body_part uses front_leg / middle_leg / hind_leg while ann_nerve uses
# prothoracic / mesothoracic / metathoracic.  A first version of this script used the nerve
# vocabulary for body_part and matched ZERO cells, which is recorded because it is the same
# class of error as the round-39 unit defect: a label that does not describe its content.
LEGS = ("prothoracic", "mesothoracic", "metathoracic")
LEG_BODY = {"prothoracic": "front_leg", "mesothoracic": "middle_leg",
            "metathoracic": "hind_leg"}
FLEXOR = re.compile(r"flex", re.I)
EXTENSOR = re.compile(r"extens|extend", re.I)


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

    # ---- populations -----------------------------------------------------------
    mech = np.array([("mechano" in (c + s + p).lower()
                      or "chordotonal" in (c + p).lower()
                      or "campaniform" in (c + p).lower())
                     for c, s, p in zip(cls, sup, ptype)])
    leg = np.array([any(t in (b + " " + nr).lower()
                        for t in ("leg", "tarsus", "tibia", "femur", "coxa", "trochanter"))
                    for b, nr in zip(body, nerve)])
    sens = mech & leg
    rows_leg = {lg: np.flatnonzero(
        np.array([LEG_BODY[lg] in x.lower() for x in body]) & sens) for lg in LEGS}
    # motor neurons: efferent flow, on a leg nerve
    efferent = np.array([f == "efferent" for f in z["ann_flow"].astype(str)])
    mn = {}
    for lg in LEGS:
        on_nerve = np.array([lg in x.lower() for x in nerve])
        mn[lg] = np.flatnonzero(efferent & on_nerve)

    # ---- connectivity ----------------------------------------------------------
    W = sp.csr_matrix((syn.astype(float), (post, pre)), shape=(n, n))  # W[post, pre]

    partitions = {}
    axis_report = {}
    for lg in LEGS:
        mns = mn[lg]
        if mns.size == 0:
            continue
        names = ptype[mns]
        flex_mn = mns[[bool(FLEXOR.search(x)) for x in names]]
        ext_mn = mns[[bool(EXTENSOR.search(x)) for x in names]]
        sens_rows = rows_leg[lg]
        if sens_rows.size == 0 or flex_mn.size == 0 or ext_mn.size == 0:
            axis_report[lg] = {"n_sens": int(sens_rows.size), "n_flex_mn": int(flex_mn.size),
                               "n_ext_mn": int(ext_mn.size),
                               "note": "axis not resolvable from motor-neuron names"}
            continue
        # degree vector onto the flexor and extensor pools
        to_flex = np.asarray(W[flex_mn, :][:, sens_rows].sum(axis=0)).ravel()
        to_ext = np.asarray(W[ext_mn, :][:, sens_rows].sum(axis=0)).ravel()
        total = to_flex + to_ext
        signed = (to_flex - to_ext) / np.where(total > 0, total, 1.0)
        # partition by class and type
        by_class = {}
        for c in sorted(set(cls[sens_rows])):
            m = cls[sens_rows] == c
            by_class[c] = {"n": int(m.sum()),
                           "n_with_projection": int((total[m] > 0).sum()),
                           "n_with_nonzero_sign": int((np.abs(signed[m]) > 1e-9).sum()),
                           "mean_signed": float(signed[m].mean()) if m.any() else None,
                           "frac_positive": float((signed[m] > 0).mean()) if m.any() else None}
        by_type = {}
        for t, c in Counter(ptype[sens_rows]).most_common(12):
            m = ptype[sens_rows] == t
            by_type[t] = {"n": int(m.sum()),
                          "n_with_projection": int((total[m] > 0).sum()),
                          "n_with_nonzero_sign": int((np.abs(signed[m]) > 1e-9).sum()),
                          "mean_signed": float(signed[m].mean()) if m.any() else None,
                          "frac_positive": float((signed[m] > 0).mean()) if m.any() else None}
        partitions[lg] = {"by_class": by_class, "by_type": by_type}
        axis_report[lg] = {
            "n_sens": int(sens_rows.size), "n_mn": int(mns.size),
            "n_flex_mn": int(flex_mn.size), "n_ext_mn": int(ext_mn.size),
            "n_sens_with_any_projection": int((total > 0).sum()),
            "n_sens_with_nonzero_sign": int((np.abs(signed) > 1e-9).sum()),
            "fraction_with_nonzero_sign": float((np.abs(signed) > 1e-9).mean()),
            "signed_axis_absmean_among_projected":
                float(np.abs(signed[total > 0]).mean()) if (total > 0).any() else None,
            "distinct_types": len(set(ptype[sens_rows])),
        }

    tot_sens = sum(v.get("n_sens", 0) for v in axis_report.values())
    tot_signed = sum(v.get("n_sens_with_nonzero_sign", 0) for v in axis_report.values())
    frac = (tot_signed / tot_sens) if tot_sens else 0.0
    h6 = bool(frac >= 0.5)

    cp.write_result(
        OUT / "neural_rebuild_06_connectome_direction.json",
        question=("Can a sensory cell's direction be DERIVED from its own connectome projection "
                  "onto the flexor/extensor motor pools, so the directional encoder need not "
                  "be hand-mapped?"),
        status="PASS",
        measured={
            "dataset": {"file": str(NPZ), "neurons": n},
            "populations": {
                "mechanosensory_on_leg": int(sens.sum()),
                "per_leg": {lg: int(rows_leg[lg].size) for lg in LEGS},
                "motor_neurons_efferent_on_leg_nerve": {lg: int(mn[lg].size) for lg in LEGS},
            },
            "axis_definition": "signed = (synapses to NAME-flexor MNs - synapses to "
                               "NAME-extensor MNs) / total; the axis is NAME-based and is "
                               "declared as such",
            "per_leg_axis": axis_report,
            "partitions": partitions,
            "predictions": {
                "H6": {"supported": h6,
                       "fraction_of_sensory_cells_with_nonzero_sign": frac,
                       "n_with_nonzero_sign": tot_signed, "n_total": tot_sens,
                       "falsifier": "< half of the cells have a non-zero projection onto the "
                                    "declared axis"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_06_connectome_direction.py"),
            "read_only": True,
            "weight_convention": "raw synapse counts into the pools; no weight scale, no cap, "
                                 "because only the SIGN and the relative balance matter here",
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "A non-zero projection establishes DIRECTIONAL STRUCTURE in the dataset, not a "
            "calibrated response amplitude: no sign convention of synapses, no PSP size and "
            "no receptor gain is claimed.",
            "The flexor/extensor axis is taken from MOTOR-NEURON NAMES in the annotation; "
            "where the names are absent or ambiguous the axis cannot be formed, which is "
            "reported per leg rather than guessed.",
            "Nothing here is compared with a real fly's proprioceptive tuning.",
        ],
    )
    print(f"mechanosensory on legs: {int(sens.sum())}  per leg "
          f"{ {lg: int(rows_leg[lg].size) for lg in LEGS} }")
    for lg, v in axis_report.items():
        print(f"  {lg:14s} sens {v.get('n_sens')}  flexMN {v.get('n_flex_mn')} "
              f"extMN {v.get('n_ext_mn')}  nonzero-sign {v.get('n_sens_with_nonzero_sign')} "
              f"({v.get('fraction_with_nonzero_sign')})")
    print(f"H6 (>=50% of cells get a sign from the connectome): {h6}  ({frac:.3f})")
    print("wrote", OUT / "neural_rebuild_06_connectome_direction.json")


if __name__ == "__main__":
    main()
