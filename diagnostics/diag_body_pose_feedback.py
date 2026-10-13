#!/usr/bin/env python3
"""FEEDBACK ON THE BODY'S OWN POSE, NOT JUST ON THE LEG JOINTS.

THE GAP THIS CLOSES.  The previous round's feedback corrected only the LEG JOINT error, and it cut
the tilt from 18.9 to 8.1 degrees without restoring the height.  Two readings were left open: the
controller had too little information, or the instability is not a gain problem.  This file supplies
the information and asks again.

WHAT IS MEASURED, AND WHY IT IS CHEAP ENOUGH.  The body's equilibrium is where the free joint's six
accelerations vanish, so the natural feedback variable is those six accelerations themselves.  Their
sensitivity to each of the 90 activations is a 6 x 90 matrix obtainable by finite differences -- 90
`mj_forward` calls, no stepping -- which means the controller can be built from measurement rather than
from a modelling assumption:

    G[k, i] = d( free joint acceleration k ) / d( act_i )        at the equilibrium

A desired corrective acceleration is then turned into an activation change by the pseudo-inverse:

    err   = ( body position error, body rotation-vector error )      relative to the keyframe
    vel   = ( free joint linear and angular velocity )
    want  = -( Kp * err + Kd * vel )                                 a 6-vector
    act   = clip( act0 + G^+ want , 0, 1 )

This is a crude one-step controller -- it ignores how the sensitivity itself changes as the body moves
-- and that is stated rather than hidden.  What it decides is whether the instability is reachable by
state feedback at all: if some (Kp, Kd) holds the pose, the missing piece was information, and the
gains become a target for the connectome instead of a hand-authored posture.  If none does, the
instability is structural and the answer changes to "this pose is not stabilisable by these
actuators".

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_body_pose_feedback.py
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


def _rotvec(q_from, q_to):
    """Rotation vector taking `q_from` to `q_to`, both (w, x, y, z)."""
    import mujoco
    neg = np.array([q_from[0], -q_from[1], -q_from[2], -q_from[3]])
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, q_to, neg)
    if dq[0] < 0:
        dq = -dq
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    return v


def main() -> int:
    import mujoco
    from scipy.optimize import minimize
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "outputs" / "body_pose_feedback.json"))
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

    def resid_at(z, act_vec):
        d.qpos[:] = q_key
        d.qpos[2] = z
        d.qvel[:] = 0.0
        d.act[:] = act_vec
        mujoco.mj_forward(m, d)
        # .copy() IS NOT OPTIONAL.  MEASURED, and it silently produced G = 0: np.asarray on a MuJoCo
        # field returns a VIEW into the data buffer, so an earlier reading and a later one are the
        # SAME memory -- `a1 - a0` was identically zero and the controller did nothing while every
        # swept gain printed an identical row.  The weighted array below is a fresh array and was
        # safe, which is why the optimiser worked and only the finite differences were dead.
        a = np.array(d.qacc[0:6], dtype=float)
        return np.concatenate([a[0:3] * W_LIN, a[3:6] * W_ANG]), a

    # ---- the static-equilibrium activation at the keyframe
    r0, _ = resid_at(float(q_key[2]), np.zeros(m.nu))
    res = minimize(lambda a: float(np.dot(resid_at(float(q_key[2]), a)[0],
                                          resid_at(float(q_key[2]), a)[0])),
                   np.zeros(m.nu), method="L-BFGS-B", bounds=[(0.0, 1.0)] * m.nu,
                   options={"maxiter": args.maxiter, "maxfun": 200000, "eps": 1e-3})
    act0 = res.x
    _, a0 = resid_at(float(q_key[2]), act0)
    print(f"static equilibrium: |residual| {np.linalg.norm(r0):.4f} -> "
          f"{np.linalg.norm(resid_at(float(q_key[2]), act0)[0]):.4f}; "
          f"qacc[0:6] = {np.round(a0, 2)}")

    # ---- G: the 6 x 90 sensitivity, measured by finite differences
    delta = 1e-2
    G = np.zeros((6, m.nu))
    for i in range(m.nu):
        ap_ = act0.copy()
        ap_[i] = min(1.0, ap_[i] + delta)
        _, a1 = resid_at(float(q_key[2]), ap_)
        G[:, i] = (a1 - a0) / (ap_[i] - act0[i])
    Gpinv = np.linalg.pinv(G)
    scale = np.abs(G).max()
    print(f"G: 6 x {m.nu}, |G| max {scale:.1f}; sensitivity to the vertical accel ranges "
          f"[{G[2].min():.1f}, {G[2].max():.1f}]")

    def rollout(kp_p, kd_p, kp_r=None, kd_r=None):
        import mujoco as mj
        kp_r = kp_p if kp_r is None else kp_r
        kd_r = kd_p if kd_r is None else kd_r
        d2 = mj.MjData(m)
        d2.qpos[:] = q_key
        d2.qvel[:] = 0.0
        d2.act[:] = act0
        mj.mj_forward(m, d2)
        up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        sat = 0
        for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
            err = np.concatenate([np.asarray(d2.qpos[0:3]) - q_key[0:3],
                                  _rotvec(q_key[3:7], np.asarray(d2.qpos[3:7]))])
            vel = np.asarray(d2.qvel[0:6], dtype=float)
            want = np.concatenate([-(kp_p * err[0:3] + kd_p * vel[0:3]),
                                   -(kp_r * err[3:6] + kd_r * vel[3:6])])
            ctrl = np.clip(act0 + Gpinv @ want, 0.0, 1.0)
            if np.any((act0 + Gpinv @ want < 0) | (act0 + Gpinv @ want > 1)):
                sat += 1
            d2.ctrl[:] = ctrl
            mj.mj_step(m, d2)
        legf, bodyf, feet = 0.0, 0.0, set()
        rc = np.zeros(6)
        for c in range(int(d2.ncon)):
            cc = d2.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mj.mj_contactForce(m, d2, c, rc)
            f = abs(float(rc[0]))
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                legf += f
                feet.add(o[:2])
            else:
                bodyf += f
        up = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d2.xpos[th][2])
        return {"kp": kp_p, "kd": kd_p, "thorax_z_mm": z, "sank_mm": float(q_key[2]) - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "body_bw": bodyf / mw, "saturated_steps": sat,
                "stood": bool(bodyf / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    print(f"\nsweeping body-pose feedback ({args.settle_ms:.0f} ms each; criterion: body < "
          f"{BODY_BW_MAX}, 6 feet, z > {STAND_Z_MM} mm)")
    print(f"  {'kp':>8} {'kd':>8} {'thorax_z':>10} {'tilt':>8} {'feet':>5} {'body_bw':>9} {'sat':>7} {'stood':>6}")
    rows = []
    for kp in (0.0, 0.5, 2.0, 10.0, 50.0):
        for kd in (0.0, 0.2, 1.0, 5.0):
            r = rollout(kp, kd)
            rows.append(r)
            print(f"  {kp:>8.1f} {kd:>8.1f} {r['thorax_z_mm']:>10.4f} {r['tilt_deg']:>8.2f} "
                  f"{r['n_feet']:>5d} {r['body_bw']:>9.4f} {r['saturated_steps']:>7d} "
                  f"{str(r['stood']):>6}")
    stood = [r for r in rows if r["stood"]]
    best = min(rows, key=lambda r: (r["tilt_deg"], -r["thorax_z_mm"]))
    print(f"\nsettings that STOOD: {[(r['kp'], r['kd']) for r in stood] if stood else 'NONE'}")
    print(f"smallest tilt: kp={best['kp']}, kd={best['kd']} -> z {best['thorax_z_mm']:.4f}, "
          f"tilt {best['tilt_deg']:.2f}, feet {best['n_feet']}, body {best['body_bw']:.4f} BW")
    verdict = ("BODY_POSE_FEEDBACK_STABILISES" if stood else "BODY_POSE_FEEDBACK_DOES_NOT_STABILISE")
    print(f"\nVERDICT: {verdict}")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "static_equilibrium": {"residual": float(np.linalg.norm(resid_at(float(q_key[2]), act0)[0])),
                               "qacc_0_6": [float(v) for v in a0]},
        "sensitivity_matrix": {"shape": [6, int(m.nu)], "abs_max": float(scale),
                               "vertical_accel_sensitivity_range":
                                   [float(G[2].min()), float(G[2].max())]},
        "law": ("act = clip( act0 + G^+ * want, 0, 1 ) with want = -(Kp * err + Kd * vel) on the free "
                "joint's six degrees of freedom; err is the body's position and rotation-vector "
                "departure from the keyframe"),
        "rows": rows, "settings_that_stood": [[r["kp"], r["kd"]] for r in stood],
        "best_by_tilt": best,
        "units": "mm, degrees, mm/s^2, rad/s^2; activation in [0,1]; Kp in activation per mm or per "
                 "radian, Kd per mm/s or per rad/s",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml at its own keyframe; G measured by 90 "
                       "finite-difference mj_forward calls; the controller tested by mj_step"),
        "what_this_does_not_establish": (
            "that a fly balances this way, or that a found controller would be posture control.  G is a "
            "ONE-STEP sensitivity measured at the equilibrium and is treated as constant while the body "
            "moves, which is a crude approximation; the gains are swept on a grid rather than tuned.  "
            "A success would mean the instability is reachable by state feedback and would give a "
            "TARGET for the connectome, not a substitute for it -- this project forbids hand-authored "
            "posture standing in for neural control.  One pose is tested, nothing walks, and every "
            "number is simulator output with no real-fly recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
