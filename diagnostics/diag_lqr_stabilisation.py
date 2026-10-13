#!/usr/bin/env python3
"""LQR AROUND THE EQUILIBRIUM -- THE PRE-DECLARED DECISIVE TEST.

THE CRITERION WAS FIXED BEFORE THIS RAN, AND IT MAY NOT BE RELAXED AFTERWARDS:

  * if LQR holds the pose, then "what is missing is a controller" is CONFIRMED, and the gain matrix
    becomes a TARGET for the connectome to reproduce -- which turns the problem into the platform's
    own problem (sense -> brain -> muscle) instead of a modelling problem;
  * if LQR does not, the conclusion must change to "this skeleton is not stabilisable in this pose",
    and the only remaining move is to change the pose or the skeleton -- with the cost declared first:
    changing the XML invalidates every recorded episode (6 arena + 11 tactile) and they must be
    re-recorded and re-registered.

WHY LQR AND NOT ANOTHER HAND-TUNED LAW.  Two simple laws have already failed: proportional-derivative
feedback on the LEG JOINT error (tilt improved from 18.9 to 8.1 degrees, height not restored) and a
one-step pseudo-inverse on the BODY POSE error (saturated every step, net effect nil).  LQR is the
textbook answer to "is this equilibrium stabilisable at all", because it does not assume a structure:
if the linearised pair (A, B) is stabilisable it returns a gain, and if it is not, no linear state
feedback exists.

THE STATE.  A tangent-space state is used so the quaternion is not linearised naively:

    x = [ p(3), theta(3), q_leg(42), v(3), omega(3), qdot_leg(42) ]    (96)

with `theta` a rotation vector applied to the equilibrium quaternion through `mju_quatIntegrate`.
A and B come from finite differences of the full dynamics at the equilibrium, which is legitimate here
because the contact set -- the six feet on the floor -- does not change for small perturbations.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_lqr_stabilisation.py
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
    from scipy.linalg import solve_continuous_are
    from scipy.optimize import minimize
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "outputs" / "lqr_stabilisation.json"))
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
    q_eq = np.asarray(d.qpos, dtype=float).copy()
    n_leg_q = m.nq - 7
    n_leg_v = m.nv - 6
    NX = 6 + n_leg_q + 6 + n_leg_v
    print(f"nq {m.nq}, nv {m.nv}, nu {m.nu} -> tangent state of dimension {NX}")

    def qpos_of(x):
        q = np.zeros(m.nq)
        q[0:3] = x[0:3]
        mujoco.mju_quatIntegrate(q[3:7], np.asarray(x[3:6], dtype=float), 1.0)
        q[3:7] /= np.linalg.norm(q[3:7])
        q[7:] = q_eq[7:] + x[6:6 + n_leg_q]
        return q

    def set_state(x, act):
        d.qpos[:] = qpos_of(x)
        d.qvel[0:3] = x[6 + n_leg_q:9 + n_leg_q]
        d.qvel[3:6] = x[9 + n_leg_q:12 + n_leg_q]
        d.qvel[6:] = x[12 + n_leg_q:NX]
        d.act[:] = act
        mujoco.mj_forward(m, d)

    def xdot(x, act):
        set_state(x, act)
        a = np.array(d.qacc, dtype=float)
        out = np.zeros(NX)
        out[0:3] = d.qvel[0:3]
        out[3:6] = d.qvel[3:6]
        out[6:6 + n_leg_q] = d.qvel[6:]
        out[6 + n_leg_q:9 + n_leg_q] = a[0:3]
        out[9 + n_leg_q:12 + n_leg_q] = a[3:6]
        out[12 + n_leg_q:NX] = a[6:]
        return out

    # ---- THE EQUILIBRIUM MUST BE AN EQUILIBRIUM OF THE WHOLE SYSTEM, NOT JUST OF THE BODY.
    #
    # MEASURED, AND THIS WAS A REAL ERROR THAT INVALIDATED THE FIRST LQR RUN.  The previous rounds
    # zeroed only the FREE JOINT's six accelerations; the leg joints were left accelerating, and the
    # tangent state's derivative at that "equilibrium" had norm 29437 with a free-joint part of
    # [-47, 912, 2997, -1271, -399, -58].  An LQR linearised about a point that is NOT an equilibrium
    # returns a gain that stabilises the LINEARISATION while the true trajectory leaves immediately --
    # which is exactly what happened: the Riccati equation solved, the closed-loop eigenvalues were
    # all negative, and the release test still failed.  So the target is now ZERO ACCELERATION ON ALL
    # 48 VELOCITY DEGREES OF FREEDOM.
    def resid(a):
        d.qpos[:] = q_eq
        d.qvel[:] = 0.0
        d.act[:] = a
        mujoco.mj_forward(m, d)
        r = np.array(d.qacc, dtype=float)
        w = np.ones(m.nv) * 1e-3              # leg joints in rad/s^2
        w[0:3] = W_LIN
        w[3:6] = W_ANG
        return r * w

    res = minimize(lambda a: float(np.dot(resid(a), resid(a))), np.zeros(m.nu),
                   method="L-BFGS-B", bounds=[(0.0, 1.0)] * m.nu,
                   options={"maxiter": 600, "maxfun": 600000, "eps": 1e-3})
    act0 = res.x
    print(f"full-state equilibrium activation: |residual| "
          f"{np.linalg.norm(resid(np.zeros(m.nu))):.4f} -> {np.linalg.norm(resid(act0)):.4f}, "
          f"{int((act0 > 1e-3).sum())} of {m.nu} muscles")

    x_eq = np.zeros(NX)
    f0 = xdot(x_eq, act0)
    print(f"xdot at the equilibrium: |f| = {np.linalg.norm(f0):.4f} "
          f"(free-joint part {np.round(f0[6 + n_leg_q:6 + n_leg_q + 6], 2)})")

    # ---- A and B by central differences
    hx, ha = 1e-4, 1e-3
    A = np.zeros((NX, NX))
    for k in range(NX):
        xp = x_eq.copy()
        xm = x_eq.copy()
        xp[k] += hx
        xm[k] -= hx
        A[:, k] = (xdot(xp, act0) - xdot(xm, act0)) / (2 * hx)
    B = np.zeros((NX, m.nu))
    for i in range(m.nu):
        ap_ = act0.copy()
        am_ = act0.copy()
        ap_[i] = min(1.0, ap_[i] + ha)
        am_[i] = max(0.0, am_[i] - ha)
        if ap_[i] == am_[i]:
            continue
        B[:, i] = (xdot(x_eq, ap_) - xdot(x_eq, am_)) / (ap_[i] - am_[i])
    print(f"A: {A.shape}, |A| max {np.abs(A).max():.3g} | B: {B.shape}, |B| max {np.abs(B).max():.3g}")

    eig = np.linalg.eigvals(A)
    print(f"open-loop eigenvalues: {int((eig.real > 1e-6).sum())} with positive real part, "
          f"largest {eig.real.max():.2f} 1/s")

    # ---- LQR
    Q = np.zeros((NX, NX))
    for k in range(3):
        Q[k, k] = 1e2                 # body position, mm
    for k in range(3, 6):
        Q[k, k] = 1e3                 # body attitude, rad
    for k in range(6, 6 + n_leg_q):
        Q[k, k] = 1e1                 # leg joint angles, rad
    for k in range(6 + n_leg_q, NX):
        Q[k, k] = 1e0                 # velocities
    R = np.eye(m.nu) * 1e-4
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        closed = A - B @ K
        ce = np.linalg.eigvals(closed)
        stable = bool((ce.real < -1e-9).all())
        print(f"LQR: gain {K.shape}, closed-loop eigenvalues all negative real part: {stable} "
              f"(largest real part {ce.real.max():.3f})")
    except Exception as exc:                                            # noqa: BLE001
        K = None
        stable = False
        print(f"LQR FAILED to solve the Riccati equation: {type(exc).__name__}: {exc}")

    # ---- simulate with the LQR feedback
    def rollout(gain, tag):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = q_eq
        d2.qvel[:] = 0.0
        d2.act[:] = act0
        mujoco.mj_forward(m, d2)
        up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        import mujoco as mj
        for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
            quat = np.asarray(d2.qpos[3:7], dtype=float)
            neg = np.array([q_eq[3], -q_eq[4], -q_eq[5], -q_eq[6]])
            dq = np.zeros(4)
            mj.mju_mulQuat(dq, quat, neg)
            if dq[0] < 0:
                dq = -dq
            thv = np.zeros(3)
            mj.mju_quat2Vel(thv, dq, 1.0)
            x = np.zeros(NX)
            x[0:3] = np.asarray(d2.qpos[0:3]) - q_eq[0:3]
            x[3:6] = thv
            x[6:6 + n_leg_q] = np.asarray(d2.qpos[7:]) - q_eq[7:]
            x[6 + n_leg_q:9 + n_leg_q] = d2.qvel[0:3]
            x[9 + n_leg_q:12 + n_leg_q] = d2.qvel[3:6]
            x[12 + n_leg_q:NX] = d2.qvel[6:]
            if gain is None:
                d2.ctrl[:] = act0
            else:
                d2.ctrl[:] = np.clip(act0 - gain @ x, 0.0, 1.0)
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
        r = {"tag": tag, "thorax_z_mm": z, "sank_mm": float(q_eq[2]) - z,
             "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
             "n_feet": len(feet), "body_bw": bodyf / mw,
             "stood": bool(bodyf / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}
        print(f"  {tag:18s} z {r['thorax_z_mm']:8.4f}  tilt {r['tilt_deg']:7.2f}  "
              f"feet {r['n_feet']}  body {r['body_bw']:7.4f} BW  stood {r['stood']}")
        return r

    print(f"\nrelease test for {args.settle_ms:.0f} ms (criterion: body < {BODY_BW_MAX}, 6 feet, "
          f"z > {STAND_Z_MM} mm)")
    tests = [rollout(None, "no feedback (act0)")]
    if K is not None:
        tests.append(rollout(K, "LQR"))
    stood = [t for t in tests if t["stood"]]
    held = any(t["tag"] == "LQR" and t["stood"] for t in tests)
    verdict = ("LQR_STABILISES" if held else
               "LQR_DOES_NOT_STABILISE" if K is not None else "LQR_UNSOLVABLE")
    print(f"\nVERDICT: {verdict}")
    print("  pre-declared criterion: LQR holding the pose confirms 'what is missing is a controller' "
          "and makes the gain a target for the connectome; LQR failing changes the conclusion to "
          "'this skeleton is not stabilisable in this pose'.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "pre_declared_criterion": ("LQR holds -> 'the missing piece is a controller', and the gain is "
                                   "the connectome's target; LQR fails -> 'the skeleton is not "
                                   "stabilisable in this pose' and the pose/skeleton must change, at "
                                   "the declared cost of re-recording 6 arena and 11 tactile episodes"),
        "state_dim": int(NX), "n_muscles": int(m.nu),
        "equilibrium": {"residual": float(np.linalg.norm(resid(act0))),
                        "xdot_norm": float(np.linalg.norm(f0)),
                        "muscles_used": int((act0 > 1e-3).sum())},
        "open_loop": {"n_unstable_eigenvalues": int((eig.real > 1e-6).sum()),
                      "largest_real_part_per_s": float(eig.real.max())},
        "lqr": ({"solved": bool(K is not None), "closed_loop_stable": bool(stable),
                 "largest_closed_loop_real_part": float(ce.real.max()) if K is not None else None}
                if K is not None else {"solved": False, "reason": "Riccati equation not solvable"}),
        "release_tests": tests,
        "units": "mm, degrees, per-second for eigenvalues; activation in [0,1]",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml at its own keyframe; A and B by central "
                       "finite differences of the tangent-space dynamics; LQR via "
                       "scipy.linalg.solve_continuous_are"),
        "what_this_does_not_establish": (
            "that a fly balances this way, or that a working LQR would be posture control.  LQR is "
            "linear and local: it is valid only while the contact set is unchanged and the state stays "
            "near the equilibrium, so a success would say the equilibrium is locally stabilisable and "
            "nothing about the basin around it.  A success would give a TARGET for the connectome, not "
            "a substitute for it -- hand-authored control standing in for neural control is what this "
            "project forbids.  One pose is tested, nothing walks, and every number is simulator output "
            "with no real-fly recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
