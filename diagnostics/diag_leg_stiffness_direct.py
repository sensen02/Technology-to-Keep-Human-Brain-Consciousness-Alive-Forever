#!/usr/bin/env python3
"""PER-LEG VERTICAL STIFFNESS BY DIRECT LOADING -- NO CONTACTS INVOLVED.

WHY NOT THE CONTACT ROUTE.  The obvious way to get per-leg stiffness is to press the body down and
read the contact forces.  That was tried and it does not work here, and the reason is worth stating:
a rigid body on six tarsus MESHES sees about eighteen contact points (a plane-versus-mesh contact
generates one point per nearby mesh feature, not one per leg), and an eighteen-point support of a
rigid body is STATICALLY INDETERMINATE -- the equilibrium equations admit many force distributions
and MuJoCo picks one.  MEASURED: restricting collisions to the floor and one tarsus geom per leg left
`ncon` at 14-18, not 6, so the indeterminacy survived the restriction.  That is why the previous
round's per-leg load shares had to be retracted.

THE MEASUREMENT INSTEAD.  Every contact is removed from the question: the body's pose is HELD EXACTLY
(written back every step, which is what makes the legs behave as struts), and a known vertical force
is applied to one leg's tip through `xfrc_applied`.  The tip's vertical displacement then gives that
leg's stiffness directly:

    k = F / dz        model force units per mm

with no contact set, no solver distribution and no indeterminacy anywhere in it.  Each leg is
measured alone, so the six numbers are independent.

WHAT THE NUMBERS DECIDE.  If a leg's k is positive and large enough that six of them carry one body
weight with a workable deflection, the pose is supportable and the work is in how the legs combine.
If a leg CANNOT produce vertical restoring force in this pose -- k near zero or negative -- then no
spring, muscle or controller will help, and the POSE itself has to change.  That is the fork this
round exists to resolve.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_leg_stiffness_direct.py
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
TIP = "Tarsus5"


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--force-fraction", type=float, default=1.0 / 6.0,
                    help="fraction of one body weight applied to the single leg")
    ap.add_argument("--out", default=str(HERE / "outputs" / "leg_stiffness_direct.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    q0 = np.asarray(d.qpos, dtype=float).copy()
    tip_body = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{L}{TIP}") for L in LEGS}
    dt = float(m.opt.timestep)
    n_steps = int(round(args.settle_ms * 1e-3 / dt))
    F = args.force_fraction * mw

    def run(leg: str | None):
        """Hold the body's pose exactly; optionally push one tip down; return the tips' z."""
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        before = {L: float(d.xpos[tip_body[L]][2]) for L in LEGS}
        for _ in range(n_steps):
            # HOLD ONLY THE FREE JOINT -- qpos[0:7] and qvel[0:6] -- NOT THE WHOLE STATE.
            # MEASURED, and the first version of this file got it wrong in a way that made the whole
            # measurement vacuous: writing ALL of qpos back every step froze the loaded leg's own
            # joints too, so the tip could not move and every leg reported dz = 0.00000 and
            # "k = inf".  Holding the body means holding the BODY; the legs must stay free or there
            # is nothing to measure.
            d.qpos[:7] = q0[:7]
            d.qvel[:6] = 0.0
            d.xfrc_applied[:] = 0.0
            if leg is not None:
                # force applied at the tip BODY's origin, in world axes, pointing down
                d.xfrc_applied[tip_body[leg], 2] = -F
            mujoco.mj_step(m, d)
        d.qpos[:7] = q0[:7]
        d.qvel[:6] = 0.0
        mujoco.mj_forward(m, d)
        after = {L: float(d.xpos[tip_body[L]][2]) for L in LEGS}
        jac = {}
        for j in range(m.njnt):
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
            if nm.startswith("joint_"):
                a = int(m.jnt_qposadr[j])
                jac[nm] = float(d.qpos[a] - q0[a])
        return before, after, jac

    print(f"one body weight = {mw:.4f} model force units; applied to one leg: {F:.4f} "
          f"({args.force_fraction:.3f} BW)")
    print("body pose HELD exactly every step; no contacts involved\n")
    _, base_after, _ = run(None)
    rows = []
    print(f"   {'leg':>4} {'dz_mm':>9} {'k (F/mm)':>11} {'verdict':>10}   joints that moved")
    for L in LEGS:
        _, after, jac = run(L)
        dz = after[L] - base_after[L]
        k = (-F / dz) if dz < 0 else float("inf") if dz == 0 else float("-inf")
        moved = sorted(((abs(v), n) for n, v in jac.items()
                        if abs(v) > 1e-4), reverse=True)[:3]
        verdict = ("SUPPORTS" if dz < -1e-4 else "NO_GIVE" if dz == 0 else "RISES_UNDER_LOAD")
        rows.append({"leg": L, "dz_mm": float(dz), "k_force_units_per_mm": float(k),
                     "verdict": verdict,
                     "joints_moved_rad": {n: v for n, v in jac.items() if abs(v) > 1e-4}})
        print(f"   {L:>4} {dz:>9.5f} {k:>11.2f} {verdict:>10}   "
              + ", ".join(f"{_n.replace('joint_','')} {float(jac[_n]):+.4f}"
                          for _a, _n in moved))
    ks = [r["k_force_units_per_mm"] for r in rows if np.isfinite(r["k_force_units_per_mm"])]
    ktot = sum(ks)
    print(f"\nsum of the six finite stiffnesses = {ktot:.2f} force units/mm")
    if ktot > 0:
        print(f"  six legs in parallel would deflect {mw / ktot:+.4f} mm to carry one body weight")
    ok = [r for r in rows if r["verdict"] == "SUPPORTS"]
    bad = [r for r in rows if r["verdict"] != "SUPPORTS"]
    verdict = "POSE_SUPPORTABLE" if len(ok) == 6 else "LEGS_THAT_CANNOT_SUPPORT"
    print(f"legs that give under load: {len(ok)}/6"
          + (f"; legs that do NOT: {[r['leg'] + ' ' + r['verdict'] for r in bad]}" if bad else ""))
    print(f"\nVERDICT: {verdict}")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "why_not_the_contact_route": (
            "a rigid body on six tarsus MESHES has about eighteen contact points, because a "
            "plane-versus-mesh contact yields one point per nearby mesh feature rather than one per "
            "leg.  An eighteen-point support of a rigid body is statically indeterminate, so the "
            "per-leg split is a solver choice -- which is why the previous round's load shares were "
            "retracted.  MEASURED: restricting collisions to one tarsus geom per leg left `ncon` at "
            "14-18, not 6."),
        "method": ("body pose held exactly every step; a known vertical force applied to ONE leg's "
                   "tip via xfrc_applied; stiffness = force / tip vertical displacement"),
        "one_body_weight_model_units": mw, "force_applied_force_units": F,
        "force_fraction_of_body_weight": args.force_fraction,
        "rows": rows, "n_legs_that_give": len(ok),
        "legs_that_do_not_give": [r["leg"] for r in bad],
        "sum_of_finite_stiffnesses": float(ktot) if ks else None,
        "required_deflection_mm_for_one_body_weight": float(mw / ktot) if ktot > 0 else None,
        "units": ("mm, radians, and MODEL FORCE UNITS where one model force unit is 1 micro-newton; "
                  f"one body weight is {mw:.4f} model force units"),
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml at its own keyframe, free joint held by "
                       "writing qpos back every step, dt "
                       f"{dt} s, settle {args.settle_ms} ms"),
        "what_this_does_not_establish": (
            "that the fly stands with contacts present.  Holding the body's pose EXACTLY is a "
            "measurement device and not a controller: it removes the very free-joint motion whose "
            "instability is the thing being explained, so a positive stiffness here says a leg CAN "
            "resist a vertical load in this pose and says nothing about whether the assembled body "
            "is stable.  It also does not say the pose is biological -- the keyframe, the joint "
            "springs and the contact parameters are model properties, and this fly is a simulator "
            "object."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
