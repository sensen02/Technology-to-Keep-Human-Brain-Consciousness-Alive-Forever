#!/usr/bin/env python3
"""IF THE LEGS COULD HOLD THEIR POSTURE, WOULD THE BODY STAND -- AND HOW HARD WOULD THAT BE?

WHY THIS EXPERIMENT.  Two of the goal's three named candidates are now measured and excluded:

  * JOINT PASSIVE STIFFNESS.  All 37 sprung joints carry a uniform 0.4, and with the body's free
    joint held the legs sag about 1.08 rad at the TROCHANTER joints until they rest on their JOINT
    LIMITS.  A sweep of that stiffness from 0.4 to 256 raises the thorax monotonically
    (0.6448 -> 1.7112 mm) but takes the feet from 6 down to 1: a stiffer skeleton does not stand, it
    TIPS.  So no single value of it works.
  * THE CONTACT MODEL.  `solref = (0.02, 1.0)` means a 20 ms contact time constant against a 0.1 ms
    timestep.  Stiffening it by 100x (0.0002 s) moved the settled height by 4.6 % and changed nothing
    qualitatively: no setting stood.

What remains is whether the POSE can be supported at all.  That cannot be answered by tuning a scalar,
so it is asked directly: give the leg joints an explicit posture spring holding them at the keyframe,
leave the free joint completely free, and sweep the spring's stiffness.  The question is whether there
is a stiffness at which the body stands, and if so how large the requirement is.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT MEAN.  A posture spring is NOT a proposed fix and is not a
controller: this project forbids hand-authored posture standing in for neural control, and the spring
here is a MEASUREMENT DEVICE that answers "is the pose supportable".  If some stiffness makes it
stand, the finding is "the skeleton can carry the body once its joints hold their angles", and the
number is the requirement that muscle tone would have to meet -- which is then a target for the
connectome loop, not a substitute for it.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_posture_hold_sweep.py
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
STAND_Z_MM = 1.2
BODY_BW_MAX = 1e-3


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "posture_hold_sweep.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    gnd = {g for g in range(m.ngeom)
           if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    # every hinge joint of the fly's legs, with its qpos address and its keyframe angle
    leg_joints = []
    for j in range(m.njnt):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
        if nm.startswith("joint_") and nm[6:8] in LEGS:
            leg_joints.append((j, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), nm))
    print(f"leg hinge joints: {len(leg_joints)}; one body weight = {mw:.4f} model force units")
    print(f"standing criterion (the project's own): body load < {BODY_BW_MAX}, six feet down, "
          f"thorax z > {STAND_Z_MM} mm\n")

    def run(k_hold: float):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        qk = np.asarray(d.qpos, dtype=float).copy()
        z0 = float(d.xpos[th][2])
        up0 = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        for _ in range(int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))):
            if k_hold > 0.0:
                d.qfrc_applied[:] = 0.0
                for _j, adr, dof, _nm in leg_joints:
                    d.qfrc_applied[dof] = -float(k_hold) * (float(d.qpos[adr]) - float(qk[adr]))
            mujoco.mj_step(m, d)
        legs, body, feet = 0.0, 0.0, set()
        res = np.zeros(6)
        for c in range(int(d.ncon)):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d, c, res)
            f = abs(float(res[0]))
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                legs += f
                feet.add(o[:2])
            else:
                body += f
        up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d.xpos[th][2])
        # how far the joints drifted from the keyframe, which is what the spring was fighting
        drift = [abs(float(d.qpos[adr]) - float(qk[adr])) for _j, adr, _d, _n in leg_joints]
        return {"k_hold": k_hold, "thorax_z_mm": z, "sank_mm": z0 - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "leg_bw": legs / mw, "body_bw": body / mw,
                "max_joint_drift_rad": float(max(drift)), "mean_joint_drift_rad": float(np.mean(drift)),
                "stood": bool(body / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    rows = []
    print(f"  {'k_hold':>10} {'thorax_z':>10} {'sank':>8} {'tilt':>8} {'feet':>5} {'body_bw':>9} "
          f"{'max drift':>10} {'stood':>6}")
    for k in (0.0, 0.4, 2.0, 10.0, 50.0, 200.0, 1000.0, 5000.0):
        r = run(k)
        rows.append(r)
        print(f"  {k:>10.1f} {r['thorax_z_mm']:>10.4f} {r['sank_mm']:>8.4f} {r['tilt_deg']:>8.2f} "
              f"{r['n_feet']:>5d} {r['body_bw']:>9.4f} {r['max_joint_drift_rad']:>10.4f} "
              f"{str(r['stood']):>6}")

    stood = [r for r in rows if r["stood"]]
    kmin = min((r["k_hold"] for r in stood), default=None)
    print(f"\nsettings that STOOD: {[r['k_hold'] for r in stood] if stood else 'NONE'}")
    if kmin is not None:
        print(f"smallest posture stiffness that stood: {kmin:.1f} model torque units per rad "
              f"({kmin / 0.4:.0f}x the shipped joint stiffness of 0.4)")
    else:
        best = max(rows, key=lambda r: (r["n_feet"], r["thorax_z_mm"], -r["body_bw"]))
        print(f"best: k_hold {best['k_hold']:.1f} -> z {best['thorax_z_mm']:.4f} mm, "
              f"tilt {best['tilt_deg']:.2f} deg, {best['n_feet']} feet, body {best['body_bw']:.4f} BW")
    verdict = ("POSE_SUPPORTABLE_WITH_TONE" if stood else "POSE_NOT_SUPPORTABLE_BY_TONE_ALONE")
    print(f"\nVERDICT: {verdict}")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "excluded_so_far": {
            "joint_passive_stiffness": (
                "uniform 0.4 on 37 joints; legs sag ~1.08 rad at the trochanter to their JOINT LIMITS "
                "under their own weight; sweeping 0.4 -> 256 raises the thorax 0.6448 -> 1.7112 mm but "
                "takes the feet from 6 to 1, so a stiffer skeleton TIPS rather than stands"),
            "contact_model_solref": (
                "shipped (0.02, 1.0) = 20 ms time constant against a 0.1 ms timestep; stiffening it "
                "100x moves the settled height by 4.6 % and changes nothing qualitatively"),
        },
        "method": ("a posture spring -k*(q - q_keyframe) applied to every leg hinge joint as "
                   "qfrc_applied, with the FREE JOINT COMPLETELY FREE; this is a measurement device, "
                   "not a controller and not a proposed fix"),
        "n_leg_joints": len(leg_joints), "one_body_weight_model_units": mw,
        "rows": rows, "settings_that_stood": [r["k_hold"] for r in stood],
        "smallest_k_that_stood": kmin,
        "shipped_joint_stiffness": 0.4,
        "ratio_to_shipped": (kmin / 0.4) if kmin else None,
        "units": ("mm, degrees, radians, and model torque units per radian; one MODEL FORCE UNIT is "
                  "1 micro-newton"),
        "provenance": ("MuJoCo rollouts of fruitfly_six_leg_standing.xml from its own keyframe, free "
                       "joint untouched, posture spring applied through qfrc_applied, "
                       f"settle {args.settle_ms} ms"),
        "what_this_does_not_establish": (
            "that muscle tone of this size is available, or that a fly does this.  The posture spring "
            "is artificial, holds every joint at the SHIPPED keyframe pose -- which is itself a "
            "hand-authored model property -- and cannot generate the corrective torques a real animal "
            "uses.  A stiffness that makes the body stand is a REQUIREMENT on the skeleton, not a "
            "finding about muscle: whether the 90 Hill muscles, driven by motor-neuron firing alone, "
            "can meet it is a separate and unmeasured question.  Nor is a body that holds still for "
            "0.5 s a fly that stands: nothing here walks."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
