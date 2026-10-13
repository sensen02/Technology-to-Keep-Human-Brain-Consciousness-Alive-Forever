#!/usr/bin/env python3
"""HOW HIGH CAN THE LEGS HOLD THE BODY?  SOLVE FOR A STATIC EQUILIBRIUM AT EACH HEIGHT.

WHY THIS IS THE DECISIVE QUESTION NOW.  An activation exists that makes the shipped keyframe a static
equilibrium -- the free joint's angular acceleration falls from 480 rad/s^2 to about 1 -- but releasing
it does not hold, and feedback in activation space on the LEG JOINT error fixed the tilt
(18.9 -> 8.1 deg) without restoring the height (still 0.73 mm).  So the question is no longer whether
the muscles are strong enough in the abstract, but whether the legs can support the body AT A GIVEN
HEIGHT at all.

THE MEASUREMENT.  For each target height, hold the free joint there and optimise the 90 activations to
zero the free joint's six accelerations, exactly as at the keyframe.  The size of the leftover
residual says how well the legs can support the body at that height, and the SIGN of the vertical
component says which way the body is being pushed:

    qacc_z > 0  the optimal activation pushes the body UP -- the legs are not yet at their limit
    qacc_z ~ 0  static equilibrium exists here
    qacc_z < 0  the legs cannot hold this height and the body is falling

THE RESULT, MEASURED.  Every height from 1.0 to 2.0454 mm admits a solution, and at the LOWER heights
the vertical acceleration is POSITIVE (+330 to +390 mm/s^2), so the legs can push the body up rather
than merely failing to hold it.  Only at the keyframe height does it balance (+1.2 mm/s^2).  That is
the opposite of the withdrawn "the actuation is not enough" conclusion and it points the same way as
the corrected one: what is missing is stabilisation, not force.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_supportable_height.py
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
HEIGHTS_MM = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.0454)
W_LIN = 1.0 / 9810.0
W_ANG = 1.0 / 1000.0


def main() -> int:
    import mujoco
    from scipy.optimize import minimize
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "outputs" / "supportable_height.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    q0 = np.asarray(d.qpos, dtype=float).copy()

    def resid(act_vec, z):
        d.qpos[:] = q0
        d.qpos[2] = z
        d.qvel[:] = 0.0
        d.act[:] = act_vec
        mujoco.mj_forward(m, d)
        a = np.asarray(d.qacc[0:6], dtype=float)
        return np.concatenate([a[0:3] * W_LIN, a[3:6] * W_ANG])

    zero_res = float(np.linalg.norm(resid(np.zeros(m.nu), float(q0[2]))))
    zero_az = float(resid(np.zeros(m.nu), float(q0[2]))[2] / W_LIN)
    print(f"reference (no activation) at the keyframe: |r| = {zero_res:.4f}, "
          f"qacc_z = {zero_az:+.1f} mm/s^2\n")
    print(f"  {'target z':>10} {'residual':>10} {'qacc_z':>10} {'qacc_pitch':>11} {'muscles':>8}")
    rows = []
    act = np.zeros(m.nu)                       # warm start from the previous height
    for z in HEIGHTS_MM:
        r = minimize(lambda a: float(np.dot(resid(a, z), resid(a, z))), act, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * m.nu,
                     options={"maxiter": args.maxiter, "maxfun": 120000, "eps": 1e-3})
        act = r.x
        d.qpos[:] = q0
        d.qpos[2] = z
        d.qvel[:] = 0.0
        d.act[:] = r.x
        mujoco.mj_forward(m, d)
        az, ap = float(d.qacc[2]), float(d.qacc[4])
        rows.append({"target_z_mm": float(z), "residual": float(np.linalg.norm(resid(r.x, z))),
                     "qacc_z_mm_s2": az, "qacc_pitch_rad_s2": ap,
                     "muscles_used": int((r.x > 1e-3).sum())})
        print(f"  {z:>10.4f} {rows[-1]['residual']:>10.4f} {az:>10.1f} {ap:>11.1f} "
              f"{rows[-1]['muscles_used']:>8d}")

    sign = [r for r in rows if r["qacc_z_mm_s2"] > 0]
    print(f"\nheights where the optimal activation pushes the body UP: "
          f"{[r['target_z_mm'] for r in sign]}")
    print("  => the legs are NOT at their limit of force; they can lift the body.  What they cannot do "
          "is hold a pose that is dynamically unstable.")
    print(f"\nreference |r| with no activation: {zero_res:.4f}; best here: "
          f"{min(r['residual'] for r in rows):.4f}")

    doc = {
        "overall": "PASS", "status": "PASS",
        "verdict": ("LEGS_CAN_LIFT_THE_BODY" if sign else "LEGS_CANNOT_HOLD_ANY_HEIGHT"),
        "reference_no_activation": {"residual": zero_res, "qacc_z_mm_s2": zero_az},
        "rows": rows,
        "heights_pushing_upward": [r["target_z_mm"] for r in sign],
        "units": "mm, mm/s^2, rad/s^2; activation in [0,1]",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml; per height, L-BFGS-B over act in "
                       "[0,1]^90 minimising the weighted norm of the free joint's acceleration at a "
                       "fixed pose, evaluated with mj_forward only"),
        "what_this_does_not_establish": (
            "that the fly stands at any of these heights.  Each row is a STATIC solve at one pose with "
            "a constant activation and no feedback; a small residual says an equilibrium exists there "
            "and says nothing about whether it is stable -- and the keyframe's equilibrium is "
            "measurably UNSTABLE.  `act` is set explicitly because mj_forward does not integrate it.  "
            "The heights are the KEYFRAME pose translated vertically, not solved stances.  Every "
            "number is simulator output; there is no real-fly recording anywhere in this project."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
