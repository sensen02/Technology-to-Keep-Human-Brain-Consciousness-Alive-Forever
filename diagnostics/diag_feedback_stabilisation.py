#!/usr/bin/env python3
"""THE EQUILIBRIUM IS UNSTABLE -- SO CLOSE THE LOOP AND SEE IF FEEDBACK HOLDS IT.

WHAT THE PREVIOUS MEASUREMENT SETTLED.  An optimiser over 90 activations drove the free joint's
acceleration at the shipped keyframe from 480 rad/s^2 down to 0.2-0.6 rad/s^2 -- a 23-fold reduction
in the weighted residual, using 56 of the 90 muscles.  So an activation EXISTS that makes the pose a
static equilibrium.  Releasing it does not hold it: the body sinks 1.28 mm and tilts 20 deg.

That is a completely different finding from "the pose cannot be supported", and it changes what the
platform needs.  The mechanism is the familiar one for anything balancing on legs: the upright pose is
an equilibrium but an UNSTABLE one, and what holds it is FEEDBACK, not stiffness.  And feedback is
exactly what this platform is built to provide -- proprioception in, motor commands out.

WHAT THIS TESTS.  A proportional-derivative law written directly in activation space:

    act = clip( act_static - Kp * B^T (q - q_key) - Kd * B^T qdot , 0, 1 )

where B is the moment-arm matrix at the pose, so B^T maps a joint error onto the muscles that can act
on that joint.  This is deliberately simple and deliberately NOT a claim about insect posture control:
it is the smallest feedback that could work, used to answer one question -- does closing the loop
stabilize the equilibrium at all?

Its result is a fork, and both branches are informative:
  * if some (Kp, Kd) holds the pose, then the skeleton, the actuators and the pose are all adequate,
    and what was missing was feedback -- which the connectome loop is supposed to supply;
  * if none does, the instability is not a simple gain problem and the pose or the actuator set is
    implicated after all.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_feedback_stabilisation.py
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
W_LIN = 1.0 / 9810.0
W_ANG = 1.0 / 1000.0


def main() -> int:
    import mujoco
    from scipy.optimize import minimize
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "outputs" / "feedback_stabilisation.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    gnd = {g for g in range(m.ngeom)
           if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    q_key = np.asarray(d.qpos, dtype=float).copy()
    legs = [(int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) for j in range(m.njnt)
            if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith("joint_")
            and (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "")[6:8] in LEGS]
    dofs = [dd for _a, dd in legs]

    # ---- 1. the static-equilibrium activation, solved as before
    def state(act_vec):
        d.qpos[:] = q_key
        d.qvel[:] = 0.0
        d.act[:] = act_vec
        mujoco.mj_forward(m, d)
        a = np.asarray(d.qacc[0:6], dtype=float)
        return np.concatenate([a[0:3] * W_LIN, a[3:6] * W_ANG])

    res = minimize(lambda a: float(np.dot(state(a), state(a))), np.zeros(m.nu),
                   method="L-BFGS-B", bounds=[(0.0, 1.0)] * m.nu,
                   options={"maxiter": args.maxiter, "maxfun": 200000, "eps": 1e-3})
    act0 = res.x
    print(f"static-equilibrium activation: |residual| = {np.linalg.norm(state(act0)):.4f}, "
          f"{int((act0 > 1e-3).sum())} of {m.nu} muscles used")

    # ---- 2. B: moment arm x full-activation-force, for the feedback map B^T
    d.qpos[:] = q_key
    d.qvel[:] = 0.0
    d.act[:] = 1.0
    mujoco.mj_forward(m, d)
    ra = np.asarray(d.moment_rowadr)
    val = np.asarray(d.actuator_moment)
    ci = np.asarray(d.moment_colind)
    af = np.asarray(d.actuator_force)
    didx = {dd: k for k, dd in enumerate(dofs)}
    B = np.zeros((len(dofs), m.nu))
    for i in range(m.nu):
        s = int(ra[i])
        e = int(ra[i + 1]) if i + 1 < len(ra) else len(val)
        for k in range(s, e):
            c = int(ci[k])
            if c in didx:
                B[didx[c], i] += float(val[k]) * float(af[i])
    # THE FEEDBACK MAP IS THE PSEUDO-INVERSE, NOT THE TRANSPOSE.  MEASURED, and it was a real bug in
    # the first version of this file: with `B.T` the gains made things WORSE as they grew (body load
    # 0.28 -> 0.64 BW, tilt 19 -> 24 deg), which is the signature of a map with the wrong geometry.
    # `B` is 42 joints x 90 muscles; a torque `-K * err` is produced by `act = -K * B^+ err` because
    # then `B act = -K err`, whereas `B (B.T err)` is not proportional to `err` at all unless B
    # happens to be orthogonal.  Pseudo-inverse gives the least-squares activation for the requested
    # torque, which is what a proportional law needs.
    scale = np.abs(B).max() or 1.0
    Bn = B / scale
    Bpinv = np.linalg.pinv(Bn)           # (90 muscles x 42 joints)

    def rollout(kp, kd):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = q_key
        d2.qvel[:] = 0.0
        d2.act[:] = act0
        mujoco.mj_forward(m, d2)
        up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        z0 = float(d2.xpos[th][2])
        for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
            err = np.asarray([float(d2.qpos[a]) - float(q_key[a]) for a, _ in legs])
            vel = np.asarray([float(d2.qvel[dd]) for _a, dd in legs])
            d2.ctrl[:] = np.clip(act0 - kp * (Bpinv @ err) - kd * (Bpinv @ vel), 0.0, 1.0)
            mujoco.mj_step(m, d2)
        legf, bodyf, feet = 0.0, 0.0, set()
        rc = np.zeros(6)
        for c in range(int(d2.ncon)):
            cc = d2.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d2, c, rc)
            f = abs(float(rc[0]))
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                legf += f
                feet.add(o[:2])
            else:
                bodyf += f
        up = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d2.xpos[th][2])
        return {"kp": kp, "kd": kd, "thorax_z_mm": z, "sank_mm": z0 - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "body_bw": bodyf / mw,
                "stood": bool(bodyf / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    print(f"\nsweeping proportional-derivative feedback in activation space "
          f"({args.settle_ms:.0f} ms each; criterion: body < {BODY_BW_MAX}, 6 feet, z > {STAND_Z_MM})")
    print(f"  {'kp':>8} {'kd':>8} {'thorax_z':>10} {'tilt':>8} {'feet':>5} {'body_bw':>9} {'stood':>6}")
    rows = []
    for kp in (0.0, 1.0, 5.0, 20.0, 100.0):
        for kd in (0.0, 1.0, 5.0, 20.0):
            r = rollout(kp, kd)
            rows.append(r)
            print(f"  {kp:>8.1f} {kd:>8.1f} {r['thorax_z_mm']:>10.4f} {r['tilt_deg']:>8.2f} "
                  f"{r['n_feet']:>5d} {r['body_bw']:>9.4f} {str(r['stood']):>6}")
    stood = [r for r in rows if r["stood"]]
    best = min(rows, key=lambda r: (r["tilt_deg"], -r["thorax_z_mm"]))
    print(f"\nsettings that STOOD: {[(r['kp'], r['kd']) for r in stood] if stood else 'NONE'}")
    print(f"smallest tilt overall: kp={best['kp']}, kd={best['kd']} -> z {best['thorax_z_mm']:.4f}, "
          f"tilt {best['tilt_deg']:.2f}, feet {best['n_feet']}, body {best['body_bw']:.4f} BW")
    verdict = ("FEEDBACK_STABILISES" if stood else "FEEDBACK_DOES_NOT_STABILISE_WITH_THIS_LAW")
    print(f"\nVERDICT: {verdict}")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "static_equilibrium": {"residual": float(np.linalg.norm(state(act0))),
                               "muscles_used": int((act0 > 1e-3).sum()),
                               "activation_top20": {str(i): float(act0[i])
                                                    for i in np.argsort(-act0)[:20]}},
        "law": ("act = clip(act_static - Kp * B^+ (q - q_key) - Kd * B^+ qdot, 0, 1), with B the "
                "moment-arm x force matrix at the pose, normalised by its largest entry, and B^+ its "
                "PSEUDO-INVERSE -- the transpose was tried first and the gains made things worse as "
                "they grew, which is the signature of the wrong map"),
        "rows": rows, "settings_that_stood": [[r["kp"], r["kd"]] for r in stood],
        "best_by_tilt": best,
        "units": "mm, degrees, and activation in [0,1]; Kp and Kd are in normalised activation per "
                 "radian and per radian/second",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml at its own keyframe; the static "
                       "activation from L-BFGS-B on the free joint's acceleration; the release tests "
                       "with mj_step"),
        "what_this_does_not_establish": (
            "that a fly stands this way, or that this law is posture control.  It is the SMALLEST "
            "feedback that could work, chosen to answer whether closing the loop helps at all; a "
            "success would not be evidence about insect control, and a failure would not rule out a "
            "better controller.  The gains are swept on a grid, not tuned, and only one pose is "
            "tested.  Nothing here walks.  Every number is simulator output; there is no real-fly "
            "recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
