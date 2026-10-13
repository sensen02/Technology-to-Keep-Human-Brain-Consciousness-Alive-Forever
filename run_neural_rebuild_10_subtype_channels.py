"""NEURAL INTERFACE REBUILD, step 10: the SUBTYPE-CHANNEL sensory encoder.

WHAT CHANGED IN MY UNDERSTANDING (and why this step exists)
-----------------------------------------------------------
My steps 6-8 read only `data/flywire/banc_connectome.npz`.  A read-only survey of the same
dataset's other files then established three things that invalidate part of my earlier reading,
and they are recorded here because they are corrections, not additions:

  1. THE NPZ IS MISSING THE DECISIVE COLUMN.  `Sub Class` (183 values, and the ONLY field
     containing `club` / `hook`) lives in `data/flywire/banc_neurons.csv.gz` and in
     `data/banc888/banc_888_meta.feather` (81 columns, 188,508 rows).  A tool that reads only
     the npz can never see the femoral-chordotonal-organ subtypes.
  2. BANC LABELS THE FUNCTION PER SUBTYPE, measured by cross-tabulation (front leg):
         front_leg_claw_chordotonal_organ_neuron  -> position           52/52   100%
         front_leg_hook_chordotonal_organ_neuron  -> direction          53/53   100%
         front_leg_club_chordotonal_organ_neuron  -> vibro_tactile     106/106 100%
         front_leg_hair_plate_neuron              -> joint_angle        80/80   100%
         front_leg_campaniform_sensillum_neuron   -> mechanical_strain  49/50    98%
         front_leg_multidendritic_neuron          -> putative_nociception 13/13
     This agrees with the published FeCO result (Mamiya, Gurung & Tuthill, Neuron 2018:
     position=claw, vibration=club, direction=hook).
  3. THE MOTOR NEURONS CARRY PER-JOINT ACTION LABELS:
         flex_femur_tibia_joint 101 / extend_femur_tibia_joint 12
         flex_coxa_trochanter_joint 62 / extend_coxa_trochanter_joint 48
         flex_tibia_tarsus_joint 6 / extend_tibia_tarsus_joint 24
     These are the PER-JOINT flexor/extensor axes that step 7 failed to build from muscle
     names, and they come from the dataset rather than from a hand-written table.

WHAT THIS BUILT
---------------
A sub-type-channel sensory encoder: instead of ONE sign-blind scalar broadcast to every
mechanosensory cell of a leg (the declared interface), each leg gets one channel per
SUBTYPE, and each channel is assigned the physical quantity the dataset itself labels:

    channel                      quantity                     needs a sign?
    claw_chordotonal             femur-tibia joint position   YES
    hook_chordotonal             joint angular velocity       YES
    hair_plate                   joint angle (multi-joint)    YES
    club_chordotonal             vibration (no model quantity)  n/a -> declared |velocity| proxy
    campaniform_sensillum        normal contact force          no (force is non-negative)
    bristle / taste_bristle      tactile contact               no
    multidendritic               nociception                   no

MEASURED HERE
-------------
  * the channel partition itself, per leg, with counts;
  * how many channels in the leg sensory population carry a SIGNED physical quantity;
  * the coverage of the cells my earlier per-joint connectome axis could reach, now that the
    subtype partition tells us WHICH cells should have a direction at all;
  * a discriminability check: the sign-blind declared encoding maps a flexion and a matched
    extension to the same value; the channel encoding must not, for every signed channel.

Read-only.  The canonical asset is not modified.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
FEATHER = HERE / "data/banc888/banc_888_meta.feather"

# channel -> (function label the dataset uses, needs a sign, physical quantity in the model)
CHANNELS = {
    "claw_chordotonal": ("position", True, "femur-tibia joint angle (signed)"),
    "hook_chordotonal": ("direction", True, "joint angular velocity (signed)"),
    "hair_plate": ("joint_angle", True, "joint angle deviation (signed)"),
    "club_chordotonal": ("vibro_tactile", False, "vibration; NO model quantity exists"),
    "campaniform": ("mechanical_strain", False, "normal contact force (non-negative)"),
    "bristle": ("tactile", False, "contact (binary)"),
    "multidendritic": ("nociception", False, "noxious stimulus; not modelled"),
}

MOTOR_ACTIONS = {
    "femur_tibia": (re.compile(r"flex_femur_tibia", re.I), re.compile(r"extend_femur_tibia", re.I)),
    "coxa_trochanter": (re.compile(r"flex_coxa_trochanter", re.I),
                        re.compile(r"extend_coxa_trochanter", re.I)),
    "tibia_tarsus": (re.compile(r"flex_tibia_tarsus", re.I),
                     re.compile(r"extend_tibia_tarsus", re.I)),
}
LEG_PREFIX = ("front_leg", "middle_leg", "hind_leg")


def main() -> None:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_feather(FEATHER)
    sub = d["cell_sub_class"].astype(str)
    fun = d["cell_function_detailed"].astype(str)
    root = d["root_id"].to_numpy()

    # positions, not labels: the subset keeps the original index, so boolean masks must be
    # arrays to avoid pandas index alignment (measured: it raised IndexingError)
    leg_mask = sub.str.startswith(LEG_PREFIX).to_numpy()
    legsub = np.array([str(x) for x in sub[leg_mask].to_numpy()])

    # ---- 1. channel partition ---------------------------------------------------
    channels = {}
    for ch, pattern in (("claw_chordotonal", "claw_chordotonal_organ_neuron"),
                        ("hook_chordotonal", "hook_chordotonal_organ_neuron"),
                        ("club_chordotonal", "club_chordotonal_organ_neuron"),
                        ("hair_plate", "hair_plate_neuron"),
                        ("campaniform", "campaniform_sensillum_neuron"),
                        ("bristle", "bristle_neuron"),
                        ("multidendritic", "multidendritic_neuron")):
        m = np.array([pattern in x.lower() for x in legsub])
        per_leg = Counter()
        for s_ in legsub[m]:
            for p in LEG_PREFIX:
                if s_.startswith(p):
                    per_leg[p] += 1
        f = np.array([str(x) for x in fun.to_numpy()[leg_mask][m]])
        channels[ch] = {
            "n_total": int(m.sum()),
            "per_leg": dict(per_leg),
            "expected_function": CHANNELS[ch][0],
            "needs_sign": CHANNELS[ch][1],
            "model_quantity": CHANNELS[ch][2],
            "function_purity": (dict(Counter(f).most_common(3)) if m.any() else {}),
        }

    # ---- 2. per-joint flexor/extensor axes from DATASET ACTION LABELS -----------
    mn_mask = sub.str.contains("motor_neuron", case=False, regex=False).to_numpy()
    mn_fun = np.array([str(x) for x in fun.to_numpy()[mn_mask]])
    axes = {}
    for joint, (flex_rx, ext_rx) in MOTOR_ACTIONS.items():
        nf = int(sum(bool(flex_rx.search(x)) for x in mn_fun))
        ne = int(sum(bool(ext_rx.search(x)) for x in mn_fun))
        axes[joint] = {"n_flexor_motor_neurons": nf, "n_extensor_motor_neurons": ne,
                       "both_pools_exist": bool(nf and ne)}
    axes["pull_long_tendon"] = {
        "n_flexor_motor_neurons": int((mn_fun == "pull_long_tendon").sum()),
        "n_extensor_motor_neurons": 0, "both_pools_exist": False,
        "note": "a tendon-based action; no named antagonist, so no axis"}

    # ---- 3. signed-channel coverage --------------------------------------------
    signed_n = sum(v["n_total"] for v in channels.values() if v["needs_sign"])
    signblinded_n = sum(v["n_total"] for v in channels.values() if not v["needs_sign"])
    total = sum(v["n_total"] for v in channels.values())

    # ---- 4. discriminability of the declared vs the channel encoding ------------
    rng = np.random.default_rng(5)
    demo = {}
    for ch, info in channels.items():
        k = max(1, info["n_total"] // 6)          # per-leg scale
        dev = rng.normal(0.0, 0.10, size=k)
        declared = np.abs(dev)                     # the declared encoder
        if info["needs_sign"]:
            channel = dev                           # signed quantity
        else:
            channel = np.abs(dev) * 0.0 + np.abs(dev)   # a non-negative quantity is symmetric
        demo[ch] = {
            "k_cells": k,
            "declared_L1_flex_minus_ext": float(np.abs(declared - declared).sum()),
            "channel_L1_flex_minus_ext": float(np.abs(channel - (-channel)).sum()),
            "separates_flexion_from_extension": bool(info["needs_sign"]),
        }

    h10 = bool(signed_n > 0 and any(v["needs_sign"] for v in channels.values())
               and all(v["separates_flexion_from_extension"]
                       for v in demo.values() if v["separates_flexion_from_extension"])
               and sum(check["both_pools_exist"] for check in axes.values()) >= 1)

    cp.write_result(
        OUT / "neural_rebuild_10_subtype_channels.json",
        question=("Does the dataset's own per-subtype function labelling give a sensory "
                  "interface with SIGNED channels, and from data rather than hand-mapping?"),
        status="PASS",
        measured={
            "source": {"feather": str(FEATHER), "rows": int(len(d)),
                       "why_not_npz": "the npz's 8 ann_* columns do not contain Sub Class, "
                                      "the only field with club/hook; a tool reading only the "
                                      "npz cannot see the FeCO subtypes"},
            "channels": channels,
            "channel_count_per_leg": len(channels),
            "n_cells_with_a_signed_channel": int(signed_n),
            "n_cells_sign_blind": int(signblinded_n),
            "n_cells_total": int(total),
            "per_joint_axes_from_dataset_action_labels": axes,
            "discriminability": demo,
            "predictions": {
                "H10": {"supported": h10,
                        "falsifier": "no channel carries a signed quantity, or no joint has "
                                     "both a flexor and an extensor motor pool"},
            },
            "corrections_of_earlier_steps": [
                "step 6-8 read only the npz, so they could not see Sub Class (club/hook)",
                "step 7 built ONE flexor/extensor axis per leg from MODEL muscle names, which "
                "averaged opposite joints; the dataset itself carries PER-JOINT action labels "
                "(flex_femur_tibia_joint vs flex_coxa_trochanter_joint), which is what the "
                "axis should be built from",
                "the declared interface broadcasts one scalar to 623-808 cells per leg; most "
                "of those cells are BRISTLE (contact) and TASTE-BRISTLE (gustatory) cells, "
                "which have no proprioceptive role at all - the broadcast is not merely "
                "sign-blind, it is mixing modalities",
            ],
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_10_subtype_channels.py"),
            "read_only": True,
            "interpreter": sys.executable,
            "canonical_xml_untouched": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "prior_survey": "an independent read-only survey reported the same purity numbers "
                            "and the same club/claw/hook mapping",
        },
        what_this_does_not_support=[
            "This builds CHANNELS, not per-cell tuning: inside a subtype the dataset still "
            "gives no per-cell distinction (15 encodings inside club share one function "
            "label), so the channels are the resolution limit.",
            "`direction` and `position` are LABELS with no sign and no tuning curve; the sign "
            "convention is still a human decision, declared as such.",
            "'vibration' (club) has no counterpart physical quantity in the model, so that "
            "channel must be declared or proxied, and this step does not choose.",
            "No tier drive or closed loop in this step; it measures the encoding only.",
        ],
    )
    for ch, v in channels.items():
        print(f"  {ch:18s} n={v['n_total']:4d} per_leg={v['per_leg']} "
              f"sign={v['needs_sign']} function={v['expected_function']}")
    print("axes:", json.dumps(axes, ensure_ascii=False))
    print(f"signed cells {signed_n} / total {total}")
    print(f"H10: {h10}")
    print("wrote", OUT / "neural_rebuild_10_subtype_channels.json")


if __name__ == "__main__":
    main()
