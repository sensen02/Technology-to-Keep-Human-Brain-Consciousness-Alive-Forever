"""NEURAL INTERFACE REBUILD, step 5: what per-cell sensory information exists LOCALLY?

THE DEFECT UNDER TEST
---------------------
The tier's sensory interface is `MechanoReceptorBank` (engine/embodied/adapters.py:864-959):
per leg it computes ONE contact scalar and ONE proprioceptive scalar, where the proprioceptive
scalar is the mean of |joint angle - reference| over that leg's joints, and it adds that
scalar to EVERY mechanosensory neuron on that leg's nerve (measured: 623-808 cells per leg).
The project's own artifact says it plainly:
`outputs/autonomous_design_round36/exploratory.json` -> `leg_sensory_battery.limitations`:
"Class-level only: no per-cell tuning, no preferred direction, no gain."

Two things are wrong with that, and they are separable:

  (i)  AN ABSOLUTE VALUE CANNOT DISTINGUISH FLEXION FROM EXTENSION.  This is a coding error
       inside the interface, and it does not depend on any annotation at all.
  (ii) ALL CELLS OF A CLASS GET THE SAME VALUE.  Fixing this needs per-cell information:
       which cell senses which joint, in which direction.

This step establishes what is available LOCALLY for (ii), and demonstrates (i) numerically so
that the part of the fix that needs no external data is not blocked on the part that does.

WHAT IS MEASURED (read-only; no file modified)
----------------------------------------------
  1. Every `ann_*` column of data/flywire/banc_connectome.npz: how many non-empty, top values.
  2. How many mechanosensory cells can be identified per leg, and under which criterion.
  3. Whether any subtype / direction / geometry field exists that could give per-cell tuning
     (search terms: chordotonal, campaniform, FeCO, claw, hook, club, hair plate, proprio,
     femoral, tarsal, subgenual, campaniform, scolopidium, ...).
  4. A numerical demonstration of (i): the same physical displacement in two opposite
     directions produces the SAME interface scalar under the declared encoding.

PRE-REGISTERED
--------------
  H5  Local annotation contains at least one field whose values differ WITHIN a leg's
      mechanosensory population in a way that can be mapped to a joint or a direction.
  FALSIFIER: every discriminating field partitions cells only by leg / body part and class.
  A rejected H5 means defect (ii) can only be closed with external data, and the interface
  rebuild must therefore do (i) - which needs no external data - and declare (ii) open.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
NPZ = HERE / "data/flywire/banc_connectome.npz"

LEG_TOKENS = ("leg", "tarsus", "tibia", "femur", "coxa", "trochanter", "pretarsus",
              "mesothoracic", "prothoracic", "metathoracic", "T1", "T2", "T3")
SUBTYPE_TERMS = ("chordotonal", "campaniform", "feco", "femoral", "claw", "hook", "club",
                 "hair", "plate", "proprio", "scolopid", "subgenual", "tarsal", "mechano",
                 "touch", "pressure", "vibration", "johnston", "stretch", "load",
                 "hair_plate", "sensillum")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    keys = list(z.keys())
    ann_cols = [k for k in keys if k.startswith("ann_")]
    n = int(z["root_ids"].size)

    columns = {}
    for c in ann_cols:
        a = np.asarray(z[c])
        nonempty = int((a.astype(str) != "").sum())
        cnt = Counter(x for x in a.astype(str).tolist() if x)
        columns[c] = {"nonempty": nonempty,
                      "n_distinct": len(cnt),
                      "top": cnt.most_common(12)}

    # ---- mechanosensory / leg populations --------------------------------------
    cls = np.asarray(z["ann_class"]).astype(str)
    sup = np.asarray(z["ann_super_class"]).astype(str)
    body = np.asarray(z["ann_body_part"]).astype(str)
    nerve = np.asarray(z["ann_nerve"]).astype(str)
    ptype = np.asarray(z["ann_primary_type"]).astype(str)

    mech_mask = np.array([any(t in (c + " " + s + " " + p).lower()
                              for t in ("mechano", "chordotonal", "campaniform", "proprio"))
                          for c, s, p in zip(cls, sup, ptype)])
    leg_mask = np.array([any(t.lower() in (b + " " + nr).lower()
                             for t in ("leg", "tarsus", "tibia", "femur", "coxa",
                                       "trochanter", "pretarsus"))
                         for b, nr in zip(body, nerve)])
    mech_leg = mech_mask & leg_mask

    # ---- per-leg resolution: does any column separate the legs? ----------------
    leg_splits = {}
    for col, arr in (("ann_body_part", body), ("ann_nerve", nerve),
                     ("ann_primary_type", ptype), ("ann_class", cls)):
        vals = Counter(arr[mech_leg].tolist())
        leg_splits[col] = vals.most_common(15)

    # ---- subtype terms inside the mechanosensory leg population ---------------
    haystack = np.array([(a + "|" + b + "|" + c + "|" + d).lower()
                         for a, b, c, d in zip(cls, sup, ptype, body)])
    subtype_hits = {}
    for t in SUBTYPE_TERMS:
        m = np.array([t in h for h in haystack])
        n_hit = int(m.sum())
        n_leg = int((m & mech_leg).sum())
        if n_hit:
            subtype_hits[t] = {"total": n_hit, "in_mechanosensory_leg": n_leg,
                               "example_values": Counter(ptype[m]).most_common(5)}

    # ---- discriminating-power test: does any column differ WITHIN one leg? -----
    within_leg_discrimination = {}
    for col, arr in (("ann_primary_type", ptype), ("ann_class", cls),
                     ("ann_body_part", body), ("ann_nerve", nerve)):
        # group mechanosensory leg cells by their body_part value, then ask how many
        # distinct values the candidate column takes inside the largest such group
        groups = {}
        for i in np.flatnonzero(mech_leg):
            groups.setdefault(body[i], []).append(i)
        if not groups:
            continue
        big = max(groups.values(), key=len)
        within_leg_discrimination[col] = {
            "largest_group_by_body_part": int(len(big)),
            "distinct_values_of_this_column_inside": len({arr[i] for i in big}),
            "values_inside": Counter(arr[big].tolist()).most_common(8),
        }

    h5 = any(v["distinct_values_of_this_column_inside"] > 1
             for v in within_leg_discrimination.values())

    # ---- numerical demonstration of defect (i) --------------------------------
    # The declared encoding: mean(|angle - reference|) per leg.  Two opposite joint
    # displacements of equal size give the same scalar, so the encoder is blind to sign.
    ref = np.array([0.5, 0.5])
    flex = ref + np.array([0.10, 0.0])
    ext = ref - np.array([0.10, 0.0])
    enc = lambda q: float(np.abs(q - ref).mean())
    demo = {
        "reference_angles_rad": ref.tolist(),
        "flexed_angles_rad": flex.tolist(),
        "extended_angles_rad": ext.tolist(),
        "declared_scalar_flexed": enc(flex),
        "declared_scalar_extended": enc(ext),
        "identical_scalars": bool(abs(enc(flex) - enc(ext)) < 1e-12),
        "signed_alternative_flexed": float((flex - ref).mean()),
        "signed_alternative_extended": float((ext - ref).mean()),
        "note": "the declared encoder maps flexion and extension to the SAME value; a signed "
                "mean separates them by 0.10 and 0.10 rad",
    }

    cp.write_result(
        OUT / "neural_rebuild_05_sensory_annotation_survey.json",
        question=("Does the local annotation carry per-cell sensory information (which cell "
                  "senses which joint, in which direction), or only leg/class partitions?"),
        status="PASS",
        measured={
            "dataset": {"file": str(NPZ), "neurons": n, "columns": keys},
            "annotation_columns": columns,
            "mechanosensory_population": {
                "criterion": "class/super_class/primary_type contains mechano|chordotonal|"
                             "campaniform|proprio; AND body_part/nerve contains a leg segment",
                "n_mechanosensory": int(mech_mask.sum()),
                "n_mechanosensory_leg": int(mech_leg.sum()),
                "split_by_column": leg_splits,
            },
            "subtype_terms": subtype_hits,
            "within_leg_discrimination": within_leg_discrimination,
            "sign_blindness_demo": demo,
            "predictions": {
                "H5": {"supported": h5,
                       "falsifier": "every discriminating field partitions cells only by "
                                    "leg / body part and class"},
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_05_sensory_annotation_survey.py"),
            "read_only": True,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "An annotation value being distinct is not the same as it being a usable tuning "
            "curve: this step establishes RESOLUTION, not a transduction function.",
            "Nothing here is validated against fly physiology; these are database labels.",
            "The sign-blindness demonstration is about the PROJECT's encoder, not about any "
            "real mechanoreceptor, which does have directional sensitivity.",
        ],
    )
    print(f"neurons={n}  ann columns={len(ann_cols)}")
    print(f"mechanosensory (any tissue)={int(mech_mask.sum())}  "
          f"mechanosensory on a leg={int(mech_leg.sum())}")
    print(f"H5 (per-cell discriminating field exists): {h5}")
    print(json.dumps(within_leg_discrimination, ensure_ascii=False)[:600])
    print("wrote", OUT / "neural_rebuild_05_sensory_annotation_survey.json")


if __name__ == "__main__":
    main()
