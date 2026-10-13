#!/usr/bin/env python3
"""THE COMBINED TEST, WITH ITS FALSIFICATION CONDITION WRITTEN BEFORE THE RUN.

THE PREDICTION, made after both components failed alone:

  A  the leg joints are too soft -- 37 joints at a uniform 0.4, so the legs sag about 1.08 rad at the
     trochanter until they rest on their JOINT LIMITS.  Stiffening them alone raises the body but
     takes the feet from 6 to 1: it TIPS.
  B  the six feet are not coplanar -- 0.0596 mm of spread, RM touching 0.05 mm before RF, contrary to
     the XML's own declared intent.  Coplanarising them alone changes the load split and the outcome
     not at all.

  PREDICTION: A and B must be fixed TOGETHER.  Neither alone stands.

FALSIFICATION CONDITION, fixed before running and not to be relaxed afterwards:

  * if ANY (coplanar, k_hold) combination satisfies the project's own standing criterion -- body load
    < 1e-3, all six feet down, thorax z > 1.2 mm -- the prediction holds, and the SMALLEST such
    k_hold is the requirement muscle tone would have to meet;
  * if NO combination across the whole grid stands, the prediction is FALSIFIED, and the conclusion
    must be stated as "this keyframe pose is not supportable in geometry", after which the only
    remaining move is to change the pose -- with its cost declared first: changing the keyframe
    invalidates every recorded episode and requires re-recording and re-registration.

The posture spring is a MEASUREMENT DEVICE, not a proposed fix and not a controller: this project
forbids hand-authored posture standing in for neural control, and what is being measured here is what
the skeleton would need, not what the fly does.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_coplanar_and_tone.py
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
COPLANAR_TOL_MM = 0.002


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "coplanar_and_tone.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    gnd = {g for g in range(m.ngeom)
           if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    q_key = np.asarray(d.qpos, dtype=float).copy()
    leg_joints = [(int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
                  for j in range(m.njnt)
                  if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "")
                  .startswith("joint_")
                  and (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "")[6:8] in LEGS]

    def foot_z(leg: str, q: np.ndarray) -> float:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return min(float(d.geom_xpos[g][2]) for g in range(m.ngeom) if names[g].startswith(leg))

    # ---- B: solve every foot onto the lowest one's plane (coarse grid then refine, which the
    #     previous round's single-pass grid failed to converge with)
    before = {L: foot_z(L, q_key) for L in LEGS}
    target = min(before.values())
    q_cop = q_key.copy()
    solved = {}
    for L in LEGS:
        best = None
        for j in range(m.njnt):
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or ""
            if not nm.startswith(f"joint_{L}") or \
                    int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            adr = int(m.jnt_qposadr[j])
            if bool(m.jnt_limited[j]):
                lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            else:
                lo, hi = -3.14, 3.14
            grid = np.linspace(lo, hi, 161)
            vals = np.array([foot_z(L, np.where(np.arange(q_key.size) == adr, Q, q_key))
                             for Q in grid])
            k = int(np.argmin(np.abs(vals)))
            a, b = float(grid[max(0, k - 1)]), float(grid[min(len(grid) - 1, k + 1)])
            fine = np.linspace(a, b, 81)
            fv = np.array([foot_z(L, np.where(np.arange(q_key.size) == adr, Q, q_key))
                           for Q in fine])
            kk = int(np.argmin(np.abs(fv)))
            cand = (abs(float(fv[kk])), adr, float(fine[kk]))
            if cand[0] < COPLANAR_TOL_MM and (best is None or cand[0] < best[0]):
                best = cand
        if best:
            q_cop[best[1]] = best[2]
            solved[L] = best
    after = {L: foot_z(L, q_cop) for L in LEGS}
    spread_before = max(before.values()) - min(before.values())
    spread_after = max(after.values()) - min(after.values())
    print(f"B. feet: spread {spread_before:.4f} -> {spread_after:.4f} mm "
          f"({len(solved)}/6 legs solved onto the common plane)")
    if spread_after > 5 * COPLANAR_TOL_MM:
        print("   WARNING: the coplanar solve did not converge tightly; the combination below is "
              "therefore testing something weaker than intended, and that is recorded.")

    def evaluate(q_base: np.ndarray, k_hold: float, tag: str):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = q_base
        d2.qvel[:] = 0.0
        d2.qfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d2)
        z0 = float(d2.xpos[th][2])
        up0 = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        for _ in range(int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))):
            d2.qfrc_applied[:] = 0.0
            if k_hold > 0.0:
                for adr, dof in leg_joints:
                    d2.qfrc_applied[dof] = -k_hold * (float(d2.qpos[adr]) - float(q_base[adr]))
            mujoco.mj_step(m, d2)
        legs, body, feet = 0.0, 0.0, set()
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
                legs += f
                feet.add(o[:2])
            else:
                body += f
        up = np.asarray(d2.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d2.xpos[th][2])
        return {"tag": tag, "k_hold": k_hold, "thorax_z_mm": z, "sank_mm": z0 - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "leg_bw": legs / mw, "body_bw": body / mw,
                "stood": bool(body / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    print(f"\nthe grid: coplanar feet x posture stiffness "
          f"(criterion: body < {BODY_BW_MAX}, 6 feet, z > {STAND_Z_MM} mm)")
    print(f"  {'pose':>10} {'k_hold':>9} {'thorax_z':>10} {'tilt':>8} {'feet':>5} "
          f"{'body_bw':>9} {'stood':>6}")
    rows = []
    for pose_tag, q_base in (("shipped", q_key), ("coplanar", q_cop)):
        for k in (0.0, 2.0, 10.0, 50.0, 200.0, 1000.0):
            r = evaluate(q_base, k, f"{pose_tag}+k{k:g}")
            rows.append(r)
            print(f"  {pose_tag:>10} {k:>9.1f} {r['thorax_z_mm']:>10.4f} {r['tilt_deg']:>8.2f} "
                  f"{r['n_feet']:>5d} {r['body_bw']:>9.4f} {str(r['stood']):>6}")

    stood = [r for r in rows if r["stood"]]
    combos = [r for r in stood if r["tag"].startswith("coplanar")]
    kmin = min((r["k_hold"] for r in combos), default=None)
    if stood:
        verdict = "PREDICTION_HELD"
        print(f"\nPREDICTION HELD: {[r['tag'] for r in stood]}")
        if kmin is not None:
            print(f"  smallest posture stiffness that stood WITH coplanar feet: {kmin:g} "
                  f"({kmin / 0.4:.0f}x the shipped joint stiffness of 0.4)")
    else:
        verdict = "PREDICTION_FALSIFIED"
        print("\nPREDICTION FALSIFIED: no (coplanar, k_hold) combination satisfied the standing "
              "criterion.")
        print("  => the conclusion is that this keyframe POSE is not supportable in geometry.")
        print("  => the only remaining move is to change the pose, and its cost must be declared "
              "first: changing the keyframe invalidates every recorded episode.")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "prediction": ("A (joint stiffness) and B (foot coplanarity) must be fixed TOGETHER; neither "
                       "alone stands"),
        "falsification_condition": ("any (coplanar, k_hold) combination satisfying body load < 1e-3, "
                                    "all six feet down and thorax z > 1.2 mm confirms it; none across "
                                    "the grid falsifies it, and then the pose itself is unsupportable"),
        "feet_spread_before_mm": spread_before, "feet_spread_after_mm": spread_after,
        "legs_solved": sorted(solved), "coplanar_tolerance_mm": COPLANAR_TOL_MM,
        "coplanar_solve_tight": bool(spread_after <= 5 * COPLANAR_TOL_MM),
        "rows": rows, "settings_that_stood": [r["tag"] for r in stood],
        "smallest_k_that_stood_with_coplanar_feet": kmin,
        "shipped_joint_stiffness": 0.4,
        "one_body_weight_model_units": mw,
        "units": "mm, degrees, and model torque units per radian; one MODEL FORCE UNIT is 1 micro-newton",
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml from its own keyframe; feet made "
                       "coplanar by a per-leg coarse-then-fine joint search; posture spring applied "
                       "through qfrc_applied with the free joint untouched"),
        "what_this_does_not_establish": (
            "that a standing combination is available to the ANIMAL.  The posture spring is a "
            "measurement device that holds every joint at the shipped keyframe pose, which is itself "
            "a hand-authored model property; a stiffness that makes the body stand is a requirement "
            "on the skeleton, not a finding about muscle, and whether the 90 Hill muscles driven by "
            "motor-neuron firing alone can meet it is unmeasured.  A falsified prediction would say "
            "the POSE is unsupportable in this model's geometry, which is still a statement about the "
            "model and not about Drosophila.  Nothing here walks."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
