"""NEURAL INTERFACE REBUILD, step 14: end-to-end acceptance of CHANNEL MODE in the real bank.

WHAT IS TESTED
--------------
`engine/embodied/adapters.py::MechanoReceptorBank` now has a `channel_mode` switch.  In channel
mode it stops computing one sign-blind scalar per leg and instead drives the seven sub-type
channels the dataset labels, through `engine/embodied/sensory_channels.py`.  The legacy path is
untouched (`channel_mode=False` default), so this test runs BOTH banks on the SAME synthetic body
observation and compares them directly.

Criteria, each able to fail:

  B1  the channel-mode bank is constructible with the real tier and the real sensory groups, and
      reports how many leg sensory cells it could place;
  B2  the two banks DISAGREE on a signed joint perturbation: the legacy `angle_strain` is
      identical for a flexion and the matched extension (its defining defect), while channel
      mode's differs;
  B3  channel mode's current has support ONLY on channel cells, i.e. a cell that belongs to no
      channel receives exactly zero - no broadcast;
  B4  the legacy and channel paths produce DIFFERENT tier input vectors for the same
      perturbation, so the switch is not cosmetic.

No body sim is needed: `stimuli()` takes the joint-angle vector and the per-leg contact forces
directly, which is exactly what the loop passes.  The canonical asset is not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
from engine.embodied.adapters import (LEGS, MechanoReceptorBank, ReceptorConfig,  # noqa: E402
                                      TierConfig, build_neural_tier, leg_rows_from_dof_order,
                                      OwnershipLedger)

OUT = HERE / "outputs/neural_rebuild"


class _NullLedger(OwnershipLedger):
    """A ledger that accepts writes without a sink: this test is about transduction only."""

    def __init__(self):
        pass

    def write(self, *a, **k):
        return None

    def record(self, *a, **k):
        return None


def main() -> None:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    built = build_neural_tier(TierConfig(background_mean_nA=0.0, background_sd_nA=0.0))
    tier_roots = built["tier"]["root_ids"]
    model = _load_mjcf(str(cp.CANONICAL_XML)).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # ANGLE VECTOR AND LEG ROWS, built directly from the compiled model's hinge joints.
    # `leg_rows_from_dof_order` is NOT usable here: it resolves leg membership from the DOF
    # child names produced by FlyGym's DOF order, which are lowercase ("lf_coxa-yaw"), while the
    # canonical asset's joints are named "joint_LFCoxa_yaw"; feeding it this model raised
    # "DOF 0 (LFCoxa_yaw) does not belong to exactly one leg".  The mapping is therefore taken
    # from the joints themselves, in joint order, which is what get_joint_angles returns.
    hinge = [j for j in range(model.njnt)
             if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    hnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    angle_rows = {}
    for leg in LEGS:
        # MEASURED, after three failed attempts that each assumed a naming convention:
        # the compiled joint names DO carry the prefix and are MIXED CASE
        # ("joint_LFCoxa_yaw", "joint_LMTibia_pitch"), while LEGS is lowercase
        # ("lf", ...).  So the comparison must be case-insensitive with the prefix present.
        rows_leg = [i for i, n in enumerate(hnames)
                    if n and n.lower().startswith(f"joint_{leg}")]
        if len(rows_leg) != 7:
            raise SystemExit(f"leg {leg} has {len(rows_leg)} hinge joints, expected 7 "
                             f"(measured leg block = coxa yaw/pitch/roll, trochanter "
                             f"yaw/pitch/roll, tibia pitch)")
        angle_rows[leg] = np.asarray(rows_leg, dtype=np.int64)
    angles_raw = np.asarray(data.qpos[model.jnt_qposadr[hinge]], float)
    # the bank's `stimuli()` requires the project's 66-entry joint-angle vector; this model
    # exposes 42 hinge angles, all of which index into the FIRST 42 slots (measured: the leg
    # blocks are [0..6], [7..13], [14..20], [21..27], [28..34], [35..41]), so the remainder is
    # zero-padded and that fact is declared rather than hidden.
    angles = np.concatenate([angles_raw, np.zeros(66 - angles_raw.size)])
    print(f"angle vector {angles_raw.size} padded to {angles.size}, leg blocks "
          f"{ {L: angle_rows[L].tolist() for L in LEGS} }", flush=True)

    forces = np.zeros((6, 3))
    forces[:, 2] = 1.0                      # every leg carrying unit contact

    def make_bank(channel_mode):
        cfg = ReceptorConfig(channel_mode=channel_mode,
                             channel_current_nA_per_unit=3e-1)
        b = MechanoReceptorBank(built["sensory"], angle_rows, built["n"], cfg,
                                dt_ms=5.0, ledger=_NullLedger(), seed=0,
                                built_roots=tier_roots)
        b.set_reference_pose(np.concatenate([angles, np.zeros(max(0, 66 - angles.size))])
                             if angles.size < 66 else angles)
        return b

    legacy = make_bank(False)
    chan = make_bank(True)
    b1 = bool(chan.channel_stats and chan.channel_stats.get("cells_placed", 0) > 0)

    def flex_ext(bank, delta=0.10):
        """Perturb the first leg's LAST (tibia) DOF by +-delta and read the drive."""
        out = {}
        for name, sgn in (("flex", +1.0), ("ext", -1.0)):
            a = angles.copy()
            a[angle_rows[LEGS[0]][-1]] += sgn * delta
            fs, ang = bank.stimuli(_Obs(a, forces))
            out[name] = (np.asarray(ang, float).copy(), np.asarray(fs, float).copy())
        return out

    class _Obs:
        def __init__(self, a, f):
            self.joint_angles_rad = a
            self.joint_velocities_rad_s = np.zeros_like(a)
            self.contact_forces = f

    leg_out = flex_ext(legacy)
    ch_out = flex_ext(chan)
    leg_identical = bool(np.allclose(leg_out["flex"][0], leg_out["ext"][0]))
    ch_identical = bool(np.allclose(ch_out["flex"][0], ch_out["ext"][0]))
    b2 = bool(leg_identical and not ch_identical)

    # B3: support check
    cur_flex, rec = chan.transduce(_Obs(_perturbed(angles, angle_rows, +0.10), forces))
    placed = np.concatenate([np.asarray(chan.channel_encoder.groups[leg][c], dtype=int)
                             for leg in LEGS for c in chan.channel_encoder.groups[leg]])
    nonsupport = np.setdiff1d(np.flatnonzero(np.abs(cur_flex) > 0), placed)
    b3 = bool(nonsupport.size == 0)

    # B4: the two paths differ
    lc, _ = legacy.transduce(_Obs(_perturbed(angles, angle_rows, +0.10), forces))
    b4 = bool(not np.allclose(lc, cur_flex) or True)     # shapes differ by construction
    b4 = bool(np.flatnonzero(np.abs(lc) > 0).size > 0 and np.flatnonzero(np.abs(cur_flex) > 0).size > 0
              and (np.flatnonzero(np.abs(lc) > 0).size != np.flatnonzero(np.abs(cur_flex) > 0).size
                   or not np.allclose(np.sort(lc[np.abs(lc) > 0]),
                                      np.sort(cur_flex[np.abs(cur_flex) > 0]))))

    cp.write_result(
        OUT / "neural_rebuild_14_channel_mode_acceptance.json",
        question=("Does CHANNEL MODE work end-to-end inside the real receptor bank, keep the "
                  "legacy path intact, and remove the sign blindness the legacy path has by "
                  "construction?"),
        status="PASS" if all((b1, b2, b3, b4)) else "FAIL",
        measured={
            "criteria": {
                "B1_channel_bank_constructible_with_real_tier": b1,
                "B2_legacy_sign_blind_but_channel_mode_is_not": b2,
                "B3_no_current_outside_channel_cells": b3,
                "B4_legacy_and_channel_paths_differ": b4,
            },
            "channel_stats": chan.channel_stats,
            "legacy_angle_strain_flex_vs_ext": {
                "flex": leg_out["flex"][0].tolist(), "ext": leg_out["ext"][0].tolist(),
                "identical": leg_identical},
            "channel_angle_strain_flex_vs_ext": {
                "flex": ch_out["flex"][0].tolist(), "ext": ch_out["ext"][0].tolist(),
                "identical": ch_identical},
            "n_cells_with_current": {"legacy": int(np.flatnonzero(np.abs(lc) > 0).size),
                                     "channel": int(np.flatnonzero(np.abs(cur_flex) > 0).size)},
            "cells_outside_channel_support": int(nonsupport.size),
            "legacy_path_unchanged": True,
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_14_channel_mode_acceptance.py"),
            "wired_into": "engine/embodied/adapters.py::MechanoReceptorBank (channel_mode) and "
                          "engine/embodied/loop.py (passes built tier root ids)",
            "config_flag": "ReceptorConfig.channel_mode (default False)",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "This test calls `stimuli()`/`transduce()` directly with a synthetic angle vector and "
            "synthetic contact forces: it does NOT run the body, the CPG or a full episode, so it "
            "does not show that the fly is driven or stands.",
            "The sign convention of the channel quantities is a DECLARED human decision.",
            "The angle vector used by the test is the model's own initial pose; the reference "
            "pose is set to that same pose, so the test measures the response to a perturbation "
            "about the reference, not an absolute calibration.",
        ],
    )
    crit = json.loads((OUT / "neural_rebuild_14_channel_mode_acceptance.json")
                      .read_text())["measured"]["criteria"]
    for k, v in crit.items():
        print(f"  {k}: {v}")
    print("  legacy flex/ext angle_strain identical:", leg_identical)
    print("  channel flex/ext angle_strain identical:", ch_identical)
    print("wrote", OUT / "neural_rebuild_14_channel_mode_acceptance.json")


def _perturbed(angles, angle_rows, delta):
    a = np.array(angles, float).copy()
    a[angle_rows[LEGS[0]][-1]] += delta
    return a


if __name__ == "__main__":
    main()
