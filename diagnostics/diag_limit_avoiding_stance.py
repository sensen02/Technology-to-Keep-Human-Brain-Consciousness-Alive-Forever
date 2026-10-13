#!/usr/bin/env python3
"""DOES A POSE THAT AVOIDS THE JOINT LIMITS FIX ANYTHING?  THE PRE-DECLARED TEST.

THE EVIDENCE THAT PROMPTED IT, ALL MEASURED:

  * the shipped keyframe is NOT the stance the model's own builder solved: 0 of 24 common joints agree
    (mean |delta| 0.617 rad) and the free-joint height differs by 0.52 mm;
  * at the shipped keyframe, `joint_RFCoxa_pitch` sits 0.027 rad from its UPPER LIMIT, and that one
    joint owns 7.29 of the 11.49 weighted residual when the optimiser tries to bring the WHOLE state
    to rest;
  * the builder's own comment says the solved stance holds each leg at 70-85 % of its reach "so no
    joint sits at a limit", and that no-joint-at-a-limit property is exactly what the shipped pose
    lacks.

So the hypothesis is concrete: the pose is at a joint stop, and a pose that avoids the stops will have
a usable full-state equilibrium, and LQR will then hold it.

FALSIFICATION CONDITION, FIXED BEFORE THE RUN:

  * if, on the limit-avoiding stance, the full-state equilibrium residual comes down (below 1) AND the
    LQR release test holds, then "the pose was the problem" is CONFIRMED, and the pose change is worth
    its declared cost -- re-recording the 6 arena and 11 tactile episodes built on this XML;
  * if the residual still cannot come down (above 1) or LQR still does not hold, then "changing the
    pose fixes it" is FALSIFIED, and the honest conclusion becomes that this skeleton cannot be
    statically supported at this scale and mass, whatever the pose.

Both poses are put through the SAME battery so the comparison is like for like: joint-limit distances,
then the full-state equilibrium solve, then LQR, then the release test.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_limit_avoiding_stance.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
STANCE = HERE / "outputs" / "stance_solution.json"
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
    ap.add_argument("--out", default=str(HERE / "outputs" / "limit_avoiding_stance.json"))
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
    q_shipped = np.asarray(d.qpos, dtype=float).copy()

    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    jid = {norm(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))[6:]): j
           for j in range(m.njnt)
           if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith("joint_")}

    # ---- build the limit-avoiding pose: the builder's solved angles, at the height that puts the
    #      feet on the floor
    solved = json.loads(STANCE.read_text())["stance_joint_angles"]
    q_alt = q_shipped.copy()
    applied = 0
    for k, v in solved.items():
        n = norm(k)
        if n in jid:
            q_alt[int(m.jnt_qposadr[jid[n]])] = float(v)
            applied += 1
    best = None
    for z in np.arange(0.4, 2.6, 0.002):
        d.qpos[:] = q_alt
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        lo = min(float(d.geom_xpos[g][2]) for g in range(m.ngeom) if names[g][:2] in LEGS)
        if best is None or abs(lo) < abs(best[0]):
            best = (lo, z)
    q_alt[2] = best[1]
    print(f"applied {applied} of {len(solved)} solved angles; feet touch at free z = {best[1]:.4f} mm "
          f"(lowest foot {best[0]:+.5f})")
    print(f"shipped keyframe free z = {q_shipped[2]:.4f} mm\n")

    legs = [(int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]),
             str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)).replace("joint_", ""))
            for j in range(m.njnt)
            if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith("joint_")
            and (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "")[6:8] in LEGS]
    n_leg_q = m.nq - 7
    n_leg_v = m.nv - 6
    NX = 6 + n_leg_q + 6 + n_leg_v

    def limit_report(q, tag):
        rows = []
        for j in range(m.njnt):
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
            if not nm.startswith("joint_") or nm[6:8] not in LEGS:
                continue
            if not bool(m.jnt_limited[j]):
                continue
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            qq = float(q[int(m.jnt_qposadr[j])])
            rows.append({"joint": nm.replace("joint_", ""), "q": qq, "lo": lo, "hi": hi,
                         "to_lo": qq - lo, "to_hi": hi - qq})
        rows.sort(key=lambda r: min(r["to_lo"], r["to_hi"]))
        print(f"  {tag}: closest 5 joints to a stop")
        for r in rows[:5]:
            side = "lo" if r["to_lo"] < r["to_hi"] else "hi"
            print(f"     {r['joint']:24s} q {r['q']:+.4f}  [{r['lo']:+.4f},{r['hi']:+.4f}]  "
                  f"to {side} {min(r['to_lo'], r['to_hi']):+.4f}")
        return rows

    print("1. joint-limit distances")
    lim_ship = limit_report(q_shipped, "shipped ")
    lim_alt = limit_report(q_alt, "solved  ")
    print(f"  smallest margin: shipped {min(min(r['to_lo'], r['to_hi']) for r in lim_ship):.4f} rad, "
          f"solved {min(min(r['to_lo'], r['to_hi']) for r in lim_alt):.4f} rad")

    # ---- the shared battery, run on both poses
    def battery(q_pose, tag):
        def wresid(a, q):
            d.qpos[:] = q
            d.qvel[:] = 0.0
            d.act[:] = a
            mujoco.mj_forward(m, d)
            r = np.array(d.qacc, dtype=float)
            w = np.ones(m.nv) * 1e-3
            w[0:3] = W_LIN
            w[3:6] = W_ANG
            return r * w

        r0 = float(np.linalg.norm(wresid(np.zeros(m.nu), q_pose)))
        res = minimize(lambda a: float(np.dot(wresid(a, q_pose), wresid(a, q_pose))),
                       np.zeros(m.nu), method="L-BFGS-B", bounds=[(0.0, 1.0)] * m.nu,
                       options={"maxiter": 600, "maxfun": 600000, "eps": 1e-3})
        act0 = res.x
        r1 = float(np.linalg.norm(wresid(act0, q_pose)))
        print(f"\n2. [{tag}] full-state equilibrium: |residual| {r0:.4f} -> {r1:.4f} "
              f"({100 * r1 / r0:.0f} % of the no-activation value), "
              f"{int((act0 > 1e-3).sum())} of {m.nu} muscles")

        d.qpos[:] = q_pose
        d.qvel[:] = 0.0
        d.act[:] = act0
        mujoco.mj_forward(m, d)
        a = np.array(d.qacc, dtype=float)
        worst = int(np.argmax(np.abs(a)))
        print(f"   worst dof qacc {a[worst]:+.1f} at dof {worst}")

        # ---- release test, with and without LQR
        def rollout(gain, label):
            d2 = mujoco.MjData(m)
            d2.qpos[:] = q_pose
            d2.qvel[:] = 0.0
            d2.act[:] = act0
            mujoco.mj_forward(m, d2)
            up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
            for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
                if gain is None:
                    d2.ctrl[:] = act0
                else:
                    quat = np.asarray(d2.qpos[3:7], dtype=float)
                    neg = np.array([q_pose[3], -q_pose[4], -q_pose[5], -q_pose[6]])
                    dq = np.zeros(4)
                    mujoco.mju_mulQuat(dq, quat, neg)
                    if dq[0] < 0:
                        dq = -dq
                    thv = np.zeros(3)
                    mujoco.mju_quat2Vel(thv, dq, 1.0)
                    x = np.zeros(NX)
                    x[0:3] = np.asarray(d2.qpos[0:3]) - q_pose[0:3]
                    x[3:6] = thv
                    x[6:6 + n_leg_q] = np.asarray(d2.qpos[7:]) - q_pose[7:]
                    x[6 + n_leg_q:9 + n_leg_q] = d2.qvel[0:3]
                    x[9 + n_leg_q:12 + n_leg_q] = d2.qvel[3:6]
                    x[12 + n_leg_q:NX] = d2.qvel[6:]
                    d2.ctrl[:] = np.clip(act0 - gain @ x, 0.0, 1.0)
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
            r = {"tag": f"{tag}/{label}", "thorax_z_mm": z,
                 "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                 "n_feet": len(feet), "body_bw": bodyf / mw,
                 "stood": bool(bodyf / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}
            print(f"   release {label:14s} z {r['thorax_z_mm']:8.4f}  tilt {r['tilt_deg']:7.2f}  "
                  f"feet {r['n_feet']}  body {r['body_bw']:7.4f} BW  stood {r['stood']}")
            return r

        tests = [rollout(None, "no feedback")]

        # ---- LQR around this pose
        from scipy.linalg import solve_continuous_are

        def qpos_of(x):
            q = np.zeros(m.nq)
            q[0:3] = x[0:3]
            mujoco.mju_quatIntegrate(q[3:7], np.asarray(x[3:6], dtype=float), 1.0)
            q[3:7] /= np.linalg.norm(q[3:7])
            q[7:] = q_pose[7:] + x[6:6 + n_leg_q]
            return q

        def xdot(x, act):
            d.qpos[:] = qpos_of(x)
            d.qvel[0:3] = x[6 + n_leg_q:9 + n_leg_q]
            d.qvel[3:6] = x[9 + n_leg_q:12 + n_leg_q]
            d.qvel[6:] = x[12 + n_leg_q:NX]
            d.act[:] = act
            mujoco.mj_forward(m, d)
            aa = np.array(d.qacc, dtype=float)
            out = np.zeros(NX)
            out[0:3] = d.qvel[0:3]
            out[3:6] = d.qvel[3:6]
            out[6:6 + n_leg_q] = d.qvel[6:]
            out[6 + n_leg_q:9 + n_leg_q] = aa[0:3]
            out[9 + n_leg_q:12 + n_leg_q] = aa[3:6]
            out[12 + n_leg_q:NX] = aa[6:]
            return out

        x_eq = np.zeros(NX)
        hx, ha = 1e-4, 1e-3
        A = np.zeros((NX, NX))
        f0 = xdot(x_eq, act0)
        for k in range(NX):
            xp, xm = x_eq.copy(), x_eq.copy()
            xp[k] += hx
            xm[k] -= hx
            A[:, k] = (xdot(xp, act0) - xdot(xm, act0)) / (2 * hx)
        Bm = np.zeros((NX, m.nu))
        for i in range(m.nu):
            ap_ = act0.copy()
            am_ = act0.copy()
            ap_[i] = min(1.0, ap_[i] + ha)
            am_[i] = max(0.0, am_[i] - ha)
            if ap_[i] == am_[i]:
                continue
            Bm[:, i] = (xdot(x_eq, ap_) - xdot(x_eq, am_)) / (ap_[i] - am_[i])
        Q = np.zeros((NX, NX))
        for k in range(3):
            Q[k, k] = 1e2
        for k in range(3, 6):
            Q[k, k] = 1e3
        for k in range(6, 6 + n_leg_q):
            Q[k, k] = 1e1
        for k in range(6 + n_leg_q, NX):
            Q[k, k] = 1e0
        lqr = {"solved": False}
        try:
            P = solve_continuous_are(A, Bm, Q, np.eye(m.nu) * 1e-4)
            K = np.linalg.solve(np.eye(m.nu) * 1e-4, Bm.T @ P)
            ce = np.linalg.eigvals(A - Bm @ K)
            lqr = {"solved": True, "closed_loop_stable": bool((ce.real < -1e-9).all()),
                   "largest_closed_loop_real_part": float(ce.real.max())}
            print(f"   LQR: solved, closed-loop stable {lqr['closed_loop_stable']} "
                  f"(largest real part {ce.real.max():.3f})")
            tests.append(rollout(K, "LQR"))
        except Exception as exc:                                        # noqa: BLE001
            lqr = {"solved": False, "why": f"{type(exc).__name__}: {exc}"}
            print(f"   LQR failed: {lqr['why']}")
        return {"tag": tag, "free_z_mm": float(q_pose[2]),
                "residual_no_activation": r0, "residual_optimised": r1,
                "residual_ratio": r1 / r0 if r0 else None,
                "muscles_used": int((act0 > 1e-3).sum()),
                "worst_dof": worst, "worst_qacc": float(a[worst]),
                "xdot_norm": float(np.linalg.norm(f0)), "lqr": lqr, "release_tests": tests}

    ship = battery(q_shipped, "shipped")
    alt = battery(q_alt, "solved")

    ok = (alt["residual_optimised"] < 1.0
          and any(t["tag"].endswith("LQR") and t["stood"] for t in alt["release_tests"]))
    verdict = "LIMIT_AVOIDING_STANCE_FIXES_IT" if ok else "CHANGING_THE_POSE_DOES_NOT_FIX_IT"
    print(f"\nVERDICT: {verdict}")
    print("  pre-declared: a limit-avoiding stance with a workable full-state equilibrium AND a "
          "holding LQR confirms the pose was the problem and justifies re-recording; otherwise "
          "'changing the pose fixes it' is FALSIFIED and the skeleton cannot be statically supported "
          "whatever the pose.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "pre_declared_criterion": (
            "on the limit-avoiding stance, a full-state residual below 1 AND a holding LQR confirm "
            "that the pose was the problem and justify the declared cost of re-recording 6 arena and "
            "11 tactile episodes; otherwise 'changing the pose fixes it' is FALSIFIED"),
        "applied_solved_angles": applied, "n_solved_angles": len(solved),
        "joint_limit_margins": {
            "shipped_min_rad": float(min(min(r["to_lo"], r["to_hi"]) for r in lim_ship)),
            "solved_min_rad": float(min(min(r["to_lo"], r["to_hi"]) for r in lim_alt)),
            "shipped_closest": lim_ship[:3], "solved_closest": lim_alt[:3]},
        "shipped": ship, "solved_stance": alt,
        "units": "mm, degrees, radians, per-second for eigenvalues; one MODEL FORCE UNIT is 1 "
                 "micro-newton",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml; the alterative pose is the builder's "
                       "own solved stance from outputs/stance_solution.json, placed at the height "
                       "where the feet touch; both poses put through the same battery"),
        "what_this_does_not_establish": (
            "that a fly stands, or that the solved stance is anatomically right -- it is the builder's "
            "own declaration, described in that script as NOT measured for this specimen.  A falsified "
            "result here would say the pose is not the obstacle; it would NOT say the model is "
            "biologically wrong, and it would not rule out a better pose or a nonlinear controller.  "
            "The equilibrium solve targets zero acceleration on all 48 velocity degrees of freedom at "
            "a fixed pose with a CONSTANT activation; LQR is local and linear and valid only while the "
            "contact set is unchanged.  Nothing here walks, and every number is simulator output with "
            "no real-fly recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
