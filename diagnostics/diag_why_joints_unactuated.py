#!/usr/bin/env python3
"""WHY CAN THOSE JOINTS NOT BE ACTUATED?  "NO MUSCLE" AND "A MUSCLE THAT DOES NOTHING" ARE DIFFERENT.

THE QUESTION.  The previous round measured each leg joint's reachable torque interval from all 90
Hill muscles at full output, and found 7 of the 8 largest posture requirements outside it --
RFCoxa_yaw needs 5.443 and can reach [-0.027, 0.001], RMCoxa_yaw needs 4.685 and can reach
[-0.009, 0.000].  A reachable interval of nearly zero has two completely different explanations, and
they need completely different fixes:

  NO MUSCLE      nothing spans the joint, so no activation pattern could ever move it.  The fix is to
                 add or re-attach a muscle, i.e. a modelling change to the actuator set.
  A MUSCLE THAT  something spans the joint but produces no force at this pose -- because its length
  DOES NOTHING   is outside its `lengthrange` (13 of 90 muscles are, measured) or because its moment
                 arm about that axis is ~0.  The fix is different in each case: a lengthrange fix
                 versus a pose/attachment fix.

So this file enumerates, for every leg joint, the actuators that touch it, and for each one reports
the three things that decide whether it can contribute: its moment arm about that axis, its length
against its own `lengthrange`, and the force it actually produces at ctrl=1.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_why_joints_unactuated.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
#: The joints the previous round found unreachable, with their requirements.
TARGETS = {"RFCoxa_yaw": 5.443, "RMCoxa_yaw": 4.685, "LMCoxa_yaw": -5.336,
           "LFCoxa_yaw": -5.202, "RMTrochanter_yaw": 3.990}


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "why_joints_unactuated.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    q_key = np.asarray(d.qpos, dtype=float).copy()
    d.qpos[:] = q_key
    d.qvel[:] = 0.0
    d.ctrl[:] = 1.0
    mujoco.mj_forward(m, d)

    ra = np.asarray(d.moment_rowadr)
    val = np.asarray(d.actuator_moment)
    ci = np.asarray(d.moment_colind)
    af = np.asarray(d.actuator_force)
    ln = np.asarray(d.actuator_length)
    lr = np.asarray(m.actuator_lengthrange)
    aname = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) for i in range(m.nu)]

    # dof -> leg joint name
    dof_joint = {}
    for j in range(m.njnt):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
        if nm.startswith("joint_"):
            dof_joint[int(m.jnt_dofadr[j])] = nm.replace("joint_", "")
    # actuators touching each dof, with their moment arm and force
    per_dof: dict[int, list[dict]] = {}
    for i in range(m.nu):
        s = int(ra[i])
        e = int(ra[i + 1]) if i + 1 < len(ra) else len(val)
        for k in range(s, e):
            c = int(ci[k])
            per_dof.setdefault(c, []).append({
                "actuator": aname[i], "moment_arm": float(val[k]),
                "force_at_ctrl1": float(af[i]), "length": float(ln[i]),
                "lengthrange": [float(lr[i, 0]), float(lr[i, 1])],
                "in_lengthrange": bool(lr[i, 0] <= ln[i] <= lr[i, 1]),
                "contributes": abs(float(val[k]) * float(af[i])),
            })

    print("WHY THESE JOINTS CANNOT BE ACTUATED\n")
    results = {}
    for nm, need in TARGETS.items():
        dof = next((dd for dd, jn in dof_joint.items() if jn == nm), None)
        rows = per_dof.get(dof, []) if dof is not None else []
        total = sum(r["contributes"] for r in rows)
        print(f"--- {nm}  (need {need:+.3f}, reachable |tau| {total:.4f})")
        if not rows:
            print("    NO MUSCLE touches this joint at all")
            results[nm] = {"need": need, "reachable": 0.0, "n_actuators": 0,
                           "diagnosis": "NO_MUSCLE", "actuators": []}
            continue
        for r in sorted(rows, key=lambda r: -r["contributes"]):
            why = []
            if abs(r["moment_arm"]) < 1e-6:
                why.append("moment arm ~0")
            if not r["in_lengthrange"]:
                why.append("OUTSIDE lengthrange")
            if abs(r["force_at_ctrl1"]) < 1e-6:
                why.append("zero force at ctrl=1")
            print(f"    {r['actuator']:28s} arm {r['moment_arm']:+.4f}  "
                  f"len {r['length']:.4f} in [{r['lengthrange'][0]:.4f},{r['lengthrange'][1]:.4f}]"
                  f"{'' if r['in_lengthrange'] else ' OUT'}  "
                  f"force {r['force_at_ctrl1']:+.4f}  -> {r['contributes']:.4f}"
                  + ("   <- " + "; ".join(why) if why else ""))
        zero_arm = sum(1 for r in rows if abs(r["moment_arm"]) < 1e-6)
        out_len = sum(1 for r in rows if not r["in_lengthrange"])
        zero_f = sum(1 for r in rows if abs(r["force_at_ctrl1"]) < 1e-6)
        if total < 1e-6:
            diag = ("MUSCLE_PRESENT_BUT_NO_MOMENT_ARM" if zero_arm else
                    "MUSCLE_PRESENT_BUT_OUTSIDE_LENGTHRANGE" if out_len else
                    "MUSCLE_PRESENT_BUT_NO_FORCE" if zero_f else "MUSCLE_PRESENT_BUT_INSUFFICIENT")
        else:
            diag = "PARTIALLY_ACTUATED"
        print(f"    -> {len(rows)} actuator(s): {zero_arm} with ~0 arm, {out_len} outside "
              f"lengthrange, {zero_f} with zero force")
        print(f"    -> DIAGNOSIS: {diag}")
        results[nm] = {"need": need, "reachable": float(total), "n_actuators": len(rows),
                       "n_zero_moment_arm": zero_arm, "n_outside_lengthrange": out_len,
                       "n_zero_force": zero_f, "diagnosis": diag,
                       "actuators": sorted(rows, key=lambda r: -r["contributes"])}
        print()

    # global picture: how many of the 90 actuators are dead at this pose, and why
    dead_arm = int(sum(1 for i in range(m.nu)
                       if all(abs(float(val[k])) < 1e-6 for k in
                              range(int(ra[i]), int(ra[i + 1]) if i + 1 < len(ra) else len(val)))))
    dead_force = int((np.abs(af) < 1e-6).sum())
    dead_len = int(((ln < lr[:, 0]) | (ln > lr[:, 1])).sum())
    print(f"model-wide at this pose: {m.nu} actuators; "
          f"{dead_force} produce zero force, {dead_len} are outside their lengthrange, "
          f"{dead_arm} have no moment arm on any dof")
    n_leg_dofs = sum(1 for dd in dof_joint)
    no_act = sum(1 for dd in dof_joint if not per_dof.get(dd))
    print(f"leg joints: {n_leg_dofs}; joints with NO actuator at all: {no_act}")

    doc = {
        "overall": "PASS", "status": "PASS",
        "targets": results,
        "model_wide": {"n_actuators": int(m.nu), "zero_force": dead_force,
                       "outside_lengthrange": dead_len, "no_moment_arm_anywhere": dead_arm,
                       "leg_joints": n_leg_dofs, "leg_joints_with_no_actuator": no_act},
        "units": "model torque units for moment arms x force, model force units for force (1 unit = "
                 "1 micro-newton), radians for the requirement",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml at its keyframe with ctrl=1; the sparse "
                       "actuator_moment structure gives which actuators touch which dof, "
                       "actuator_length against actuator_lengthrange gives the working range"),
        "what_this_does_not_establish": (
            "whether re-attaching or re-scaling a muscle would make the fly stand -- this file "
            "diagnoses the CAUSE of the unreachability, not a fix.  It also does not say the shipped "
            "attachment is WRONG: the muscles and their lengthranges are model properties taken from "
            "FlyMimic, and nothing in this project measures a real fly's muscle attachment or working "
            "range.  Everything here is simulator output."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
