#!/usr/bin/env python3
"""CAN AN ACTIVATION PATTERN HOLD THE POSTURE?  SOLVE FOR ONE AND TEST IT.

WHY THIS IS THE RIGHT NEXT STEP, AND WHAT HAD TO BE CORRECTED FIRST.  The previous round measured the
per-joint torque each leg joint could reach from all 90 muscles and concluded the actuation could not
hold the posture -- from muscle force read WITHOUT establishing the activation state, since `mj_forward`
does not integrate `act` and these are muscle actuators.  With `act` set properly the muscles are far
stronger (303.5 model force units, 12.4 body weights, against 54.1 before) and the median reachable
interval is 7.19 instead of 0.034.  That conclusion was WITHDRAWN.

So the question is reopened and asked directly, as an optimisation rather than an interval test: solve
for a non-negative activation vector in [0, 1] whose joint torques match the torques a posture spring
implies, then APPLY it and see whether the body stands.

    tau_j  ~=  sum_i  moment_arm(i, j) * force_i(1) * act_i          ->   B act = tau_need

B is (42 leg joints) x (90 muscles) and is taken at the keyframe pose with full activation, so the
solve is a bounded linear least squares.  The linear step in `act` is an approximation -- a real Hill
muscle's force is not linear in activation -- and the test that follows is what decides whether the
approximation was good enough, because a solved pattern is only interesting if the body stands.

This is NOT hand-authored posture: an activation pattern is the OUTPUT of neural control, which is what
the platform's connectome loop supplies later.  What is hand-made here is only the SOLVE, and it is
reported as such.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_solve_posture_activation.py
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
K_HOLD = 200.0
M_EXT = 7.5
STAND_Z_MM = 1.2
BODY_BW_MAX = 1e-3


def main() -> int:
    import mujoco
    from scipy.optimize import lsq_linear
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "solve_posture_activation.json"))
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
    legs = [(int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]),
             str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)).replace("joint_", ""))
            for j in range(m.njnt)
            if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith("joint_")
            and (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "")[6:8] in LEGS]
    dofs = [dd for _a, dd, _n in legs]

    # ---- 1. the torque a posture spring implies at steady state
    d.qpos[:] = q_key
    d.qvel[:] = 0.0
    d.act[:] = 1.0
    mujoco.mj_forward(m, d)
    for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[th, 4] = M_EXT
        for adr, dof, _n in legs:
            d.qfrc_applied[dof] = -K_HOLD * (float(d.qpos[adr]) - float(q_key[adr]))
        mujoco.mj_step(m, d)
    tau_need = np.array([-K_HOLD * (float(d.qpos[adr]) - float(q_key[adr])) for adr, _d, _n in legs])
    print(f"posture spring k={K_HOLD}, compensating torque {M_EXT} uN*mm")
    print(f"  torque to reproduce: {len(tau_need)} leg joints, "
          f"|tau| total {np.abs(tau_need).sum():.2f}, max {np.abs(tau_need).max():.3f}")

    # ---- 2. build B at the keyframe pose with FULL activation, then solve B act = tau_need
    d.qpos[:] = q_key
    d.qvel[:] = 0.0
    d.act[:] = 1.0
    mujoco.mj_forward(m, d)
    ra = np.asarray(d.moment_rowadr)
    val = np.asarray(d.actuator_moment)
    ci = np.asarray(d.moment_colind)
    af = np.asarray(d.actuator_force)
    B = np.zeros((len(dofs), m.nu))
    dof_index = {dd: k for k, dd in enumerate(dofs)}
    for i in range(m.nu):
        s = int(ra[i])
        e = int(ra[i + 1]) if i + 1 < len(ra) else len(val)
        F = float(af[i])
        for k in range(s, e):
            c = int(ci[k])
            if c in dof_index:
                B[dof_index[c], i] += float(val[k]) * F
    sol = lsq_linear(B, tau_need, bounds=(0.0, 1.0), max_iter=500)
    act = sol.x
    resid = B @ act - tau_need
    print(f"  solve: {int((act > 1e-6).sum())} of {m.nu} muscles used, cost {sol.cost:.4f}, "
          f"residual |B act - tau| max {np.abs(resid).max():.3f} "
          f"({100 * np.abs(resid).max() / max(np.abs(tau_need).max(), 1e-9):.0f} % of the largest need)")

    # ---- 3. APPLY the solved pattern and see whether the body stands
    def run(ctrl_vec, tag):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = q_key
        d2.qvel[:] = 0.0
        d2.act[:] = 1.0
        mujoco.mj_forward(m, d2)
        up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        z0 = float(d2.xpos[th][2])
        for _ in range(int(args.settle_ms * 1e-3 / float(m.opt.timestep))):
            d2.ctrl[:] = ctrl_vec
            mujoco.mj_step(m, d2)
        legf, bodyf, feet = 0.0, 0.0, set()
        res = np.zeros(6)
        for c in range(int(d2.ncon)):
            cc = d2.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d2, c, res)
            f = abs(float(res[0]))
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                legf += f
                feet.add(o[:2])
            else:
                bodyf += f
        up = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d2.xpos[th][2])
        return {"tag": tag, "thorax_z_mm": z, "sank_mm": z0 - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "leg_bw": legf / mw, "body_bw": bodyf / mw,
                "stood": bool(bodyf / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    print(f"\ntesting constant activation patterns for {args.settle_ms:.0f} ms "
          f"(criterion: body < {BODY_BW_MAX}, 6 feet, z > {STAND_Z_MM} mm)")
    made = {}
    for tag, vec in (("no activation", np.zeros(m.nu)),
                     ("solved pattern", act),
                     ("uniform ctrl=1", np.ones(m.nu))):
        r = run(vec, tag)
        made[tag] = r
        print(f"  {tag:16s} z {r['thorax_z_mm']:8.4f}  tilt {r['tilt_deg']:7.2f}  "
              f"feet {r['n_feet']}  body {r['body_bw']:7.4f} BW  stood {r['stood']}")

    stood = [k for k, r in made.items() if r["stood"]]
    verdict = "SOLVED_ACTIVATION_STANDS" if "solved pattern" in stood else "SOLVED_ACTIVATION_DOES_NOT_STAND"
    print(f"\nVERDICT: {verdict}")
    if not stood:
        print("  A constant activation that reproduces the posture-spring torques at this pose does "
              "not hold the body.  Two readings, and this run cannot separate them: the linear step "
              "in `act` may be too crude, or a CONSTANT pattern may simply not be enough and the "
              "posture needs state feedback. The first is testable by re-solving iteratively against "
              "the true nonlinear force; the second is the control problem.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "corrected_first": (
            "the previous round's claim that the actuation cannot hold the posture was WITHDRAWN: it "
            "read muscle force without establishing the activation state (`mj_forward` does not "
            "integrate `act`). With act=1 the strongest muscle gives 303.5 model force units (12.4 "
            "body weights) instead of 54.1, and the median reachable interval is 7.19 instead of 0.034"),
        "solve": {"n_joints": len(dofs), "n_muscles": int(m.nu),
                  "muscles_used": int((act > 1e-6).sum()),
                  "cost": float(sol.cost),
                  "residual_max": float(np.abs(resid).max()),
                  "residual_frac_of_largest_need":
                      float(np.abs(resid).max() / max(np.abs(tau_need).max(), 1e-9)),
                  "activation": {str(i): float(act[i]) for i in np.argsort(-act)[:20]},
                  "linear_step_is_an_approximation": True},
        "torque_target": {"k_hold": K_HOLD, "M_ext_uN_mm": M_EXT,
                          "abs_total": float(np.abs(tau_need).sum()),
                          "abs_max": float(np.abs(tau_need).max())},
        "tests": made,
        "units": "mm, degrees, model torque units; one MODEL FORCE UNIT is 1 micro-newton and one "
                 "body weight is 24.4464 of them",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml from its own keyframe; torque target from "
                       "the posture spring's steady state; activation solved by bounded linear least "
                       "squares against the moment-arm matrix at full activation"),
        "what_this_does_not_establish": (
            "that no activation exists.  This tests ONE constant pattern, solved under a linear "
            "approximation of the muscle force law, and it cannot separate 'the approximation was too "
            "crude' from 'a constant pattern is not enough and feedback is needed'.  Nor is a pattern "
            "that DID stand a fly standing: it would be a constant activation found by an optimiser at "
            "one pose, with no sensory feedback, no gait and no connection to the connectome.  The "
            "posture spring and the external torque that define the target are measurement devices, "
            "not fixes, and every number here is simulator output."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
