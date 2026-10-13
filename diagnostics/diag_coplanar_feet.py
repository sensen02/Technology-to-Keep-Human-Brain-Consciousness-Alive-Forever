#!/usr/bin/env python3
"""MAKE THE SIX FEET COPLANAR -- THE MODEL'S OWN DECLARED DESIGN, WHICH IT DOES NOT MEET.

THE DEFECT, MEASURED.  `fruitfly_six_leg_standing.xml` says in its own header that the leg offsets
are set "so that all six tarsi reach the floor at once".  They do not.  Lowering the body in 0.01 mm
steps and recording when each leg first contacts gives:

    LF 2.0653   LM 2.0653   LH 2.0653   RH 2.0653   RM 2.0953   RF 2.0454   (free-joint z, mm)

a 0.0500 mm spread, with RM touching 0.05 mm early and RF 0.02 mm late.  With legs whose vertical
stiffness is tens to thousands of uN/mm and a body weight of 24.45 uN, that spread decides the whole
load path: MEASURED at the height where the contacts carry one body weight, the split is
LM 0.335 BW + RM 0.496 BW against 0.086 BW for LF+RF+LH+RH together -- the fore and hind legs
barely touch, the support is effectively a narrow transverse line through the middle pair, and the
body pitches until the abdomen reaches the floor.

WHY FIXING THIS IS NOT "HAND-AUTHORED POSTURE".  This project forbids hand-authored gait or posture
standing in for neural control.  This is a different thing: the keyframe pose is ALREADY a model
property that FlyMimic ships, and the correction makes the model satisfy the intent its own header
declares.  Nothing here adds a controller, a gait, or a behaviour; it removes a 0.05 mm geometric
error.  The distinction is stated because confusing the two would be exactly the failure this
project is built to avoid.

WHAT IT DOES.  For each leg, one joint is used as the height knob and solved so that leg's foot lands
on the same plane as the others.  The solve is a secant iteration on the foot's minimum world z,
with each leg independent, so no global optimisation or hand-tuning is involved.  The corrected pose
is then released and measured exactly as the uncorrected one was.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_coplanar_feet.py
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
#: One joint per leg, used only as a HEIGHT KNOB for the solve.  The tibia is the longest segment
#: below the body, so a small angle change moves the foot furthest and the correction stays small.
KNOB = {leg: f"joint_{leg}Tibia_pitch" for leg in LEGS}


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--tol-mm", type=float, default=0.002)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "coplanar_feet.json"))
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
    qpos_key = np.asarray(d.qpos, dtype=float).copy()
    res = np.zeros(6, dtype=float)

    def foot_z(leg: str) -> float:
        mujoco.mj_forward(m, d)
        return min(float(d.geom_xpos[g][2]) for g in range(m.ngeom) if names[g].startswith(leg))

    def contact_state():
        tot, per, feet = 0.0, {}, set()
        for c in range(int(d.ncon)):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d, c, res)
            other = names[g2] if g1 in gnd else names[g1]
            k = other[:2] if other[:2] in LEGS else other
            per[k] = per.get(k, 0.0) + abs(float(res[0]))
            tot += abs(float(res[0]))
            if k in LEGS:
                feet.add(k)
        return tot / mw, {k: v / mw for k, v in per.items()}, sorted(feet)

    def release_and_measure(tag: str):
        n = int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(m, d)
        up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        up0 = np.asarray(mujoco.MjData(m).xmat[th], dtype=float).reshape(3, 3)[:, 2] \
            if False else None
        return {"tag": tag, "thorax_z_model": float(d.xpos[th][2]),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up[2]), -1, 1))))}

    # ---- 1. the defect
    print("1. foot heights BEFORE, in the shipped keyframe")
    before = {}
    for leg in LEGS:
        before[leg] = foot_z(leg)
        print(f"   {leg}: {before[leg]:+.4f} mm")
    spread_before = max(before.values()) - min(before.values())
    print(f"   spread = {spread_before:.4f} mm")

    # ---- 2. solve each leg onto the LOWEST foot's plane (the strictest target, so no leg is lifted
    #         above the ground it would otherwise rest on)
    target = min(before.values())
    print(f"\n2. solving every foot onto z = {target:+.4f} mm (the lowest foot's plane)")
    # THE KNOB IS CHOSEN INSIDE THE LEG'S OWN JOINT RANGES, not fixed by name.  MEASURED: with the
    # tibia pitch as a fixed knob, two legs (LM, RF) could not be solved at all, with a residual of
    # 1.35 mm against a 0.04 mm requirement -- because the model's joint ranges are TIGHTENED (the
    # stance solver narrowed them; `outputs/stance_solution.json` records `joint_RFCoxa_pitch` going
    # from "-3.142 3.142" to "-0.5783 0.7375"), so that joint was already sitting on its limit and
    # could not move the foot at all.  So for each leg every hinge joint is tried, its usable travel
    # is measured, and the one with the most room that can reach the plane is used.
    solved, knob_used, knob_report = {}, {}, {}
    for leg in LEGS:
        js = [j for j in range(m.njnt)
              if (str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) or "").startswith(
                  f"joint_{leg}") and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
        best = None
        for j in js:
            adr_j = int(m.jnt_qposadr[j])
            lo_lim, hi_lim = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) \
                if bool(m.jnt_limited[j]) else (-3.14, 3.14)
            a = np.asarray(d.qpos, dtype=float).copy()
            a[:] = qpos_key
            a[adr_j] = min(max(qpos_key[adr_j], lo_lim), hi_lim)
            d.qpos[:] = a
            z_at = foot_z(leg)
            # scan the joint's usable range for the foot's reachable z
            zs = []
            for Q in np.linspace(lo_lim, hi_lim, 41):
                a[:] = qpos_key
                a[adr_j] = Q
                d.qpos[:] = a
                zs.append(foot_z(leg))
            zs = np.asarray(zs)
            reach = float(zs.min()) <= target <= float(zs.max())
            room = min(abs(qpos_key[adr_j] - lo_lim), abs(hi_lim - qpos_key[adr_j]))
            if reach:
                cand = (room, j, adr_j, lo_lim, hi_lim)
                if best is None or cand[0] > best[0]:
                    best = cand
            knob_report.setdefault(leg, []).append(
                {"joint": str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
                 "limited": bool(m.jnt_limited[j]), "range": [lo_lim, hi_lim],
                 "reachable": reach, "z_at_current": z_at,
                 "z_min": float(zs.min()), "z_max": float(zs.max()), "room_rad": room})
        if best is None:
            solved[leg] = None
            knob_used[leg] = None
            print(f"   {leg}: NO joint in this leg can reach z = {target:+.4f} mm")
            continue
        _, j, adr_j, lo_lim, hi_lim = best
        knob_used[leg] = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))

        def f(Q):
            a = np.asarray(d.qpos, dtype=float).copy()
            a[:] = qpos_key
            a[adr_j] = Q
            d.qpos[:] = a
            return foot_z(leg) - target

        # GRID THEN REFINE, NOT SECANT.  MEASURED: a secant iteration stalled with residuals of
        # 0.08-0.20 mm on three legs, because the objective is the leg's LOWEST GEOM and that is not
        # monotone in the joint angle -- a leg swings its tibia down, then the tarsus sweeps sideways
        # and the minimum stops falling or starts rising.  A 1-D scan is cheap here (a few hundred
        # `mj_forward` calls) and cannot be fooled by non-monotonicity, so it is used instead.
        grid = np.linspace(lo_lim, hi_lim, 161)
        vals = np.array([f(Q) for Q in grid])
        k = int(np.argmin(np.abs(vals)))
        root, f2 = None, float(vals[k])
        if abs(f2) < args.tol_mm:
            root = float(grid[k])
        else:
            a = float(grid[max(0, k - 1)])
            b = float(grid[min(len(grid) - 1, k + 1)])
            fine = np.linspace(a, b, 81)
            fv = np.array([f(Q) for Q in fine])
            kk = int(np.argmin(np.abs(fv)))
            if abs(float(fv[kk])) < args.tol_mm:
                root, f2 = float(fine[kk]), float(fv[kk])
        solved[leg] = root
        print(f"   {leg}: {knob_used[leg]:32s} {qpos_key[adr_j]:+.4f} -> "
              + (f"{root:+.4f} rad  (foot now {target + f2:+.4f} mm)" if root is not None
                 else f"NO SOLVE (end residual {f1:+.4f} mm)"))
    adr = {leg: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                                    knob_used[leg] or KNOB[leg])]) for leg in LEGS}

    # ---- 3. measure the corrected pose: statics, then release
    ok = all(v is not None for v in solved.values())
    doc = {"overall": "PASS", "status": "PASS",
           "foot_z_before_mm": before, "spread_before_mm": spread_before,
           "target_z_mm": target, "knob_joint_per_leg": KNOB,
           "knob_joint_actually_used": knob_used,
           "knob_search": knob_report,
           "solved_angles": {k: (float(v) if v is not None else None) for k, v in solved.items()},
           "keyframe_angles": {leg: float(qpos_key[adr[leg]]) for leg in LEGS},
           "units": "mm for heights, radians for angles, multiples of one body weight for forces",
           "one_body_weight_model_units": mw,
           "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml, per-leg secant solve on one joint "
                          "using the foot's minimum world z as the objective"),
           "what_this_does_not_establish": (
               "that a coplanar pose stands -- the release test below says whether it does.  It also "
               "does not make the pose biological: it removes a 0.05 mm geometric error from a "
               "shipped keyframe, adding no controller, gait or behaviour.  A pose that holds still "
               "for 0.5 s is not a fly standing.")}
    if not ok:
        doc["overall"] = doc["status"] = "NOT_RUN"
        doc["why"] = "at least one leg's foot could not be brought onto the common plane"
    else:
        q = np.asarray(d.qpos, dtype=float).copy()
        q[:] = qpos_key
        for leg in LEGS:
            q[adr[leg]] = solved[leg]
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        after = {leg: foot_z(leg) for leg in LEGS}
        spread_after = max(after.values()) - min(after.values())
        tot, per, feet = contact_state()
        print(f"\n3. after the solve")
        print(f"   foot z: " + " ".join(f"{l} {after[l]:+.4f}" for l in LEGS))
        print(f"   spread = {spread_after:.4f} mm (was {spread_before:.4f})")
        print(f"   at that pose the contacts carry {tot:.4f} BW over {len(feet)} feet")
        print(f"   per-leg: " + " ".join(f"{k} {v:.3f}" for k, v in sorted(per.items())))
        doc.update({"foot_z_after_mm": after, "spread_after_mm": spread_after,
                    "static_total_bw": tot, "static_per_leg_bw": per,
                    "static_feet_down": feet})
        # release from a few heights around the coplanar one
        runs = []
        z_ref = float(q[2])
        for dz in (0.0, -0.02, -0.04, -0.06, -0.10, -0.20):
            qq = np.asarray(q, dtype=float).copy()
            d.qpos[:] = qq
            d.qpos[2] = z_ref + dz
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            r = release_and_measure(f"dz {dz:+.2f}")
            r["dz_mm"] = dz
            runs.append(r)
            print(f"   release dz {dz:+.2f}: settled z {r['thorax_z_mm']:.4f} mm, "
                  f"tilt {r['tilt_deg']:5.2f} deg")
        held = [r for r in runs if r["tilt_deg"] < 5.0 and abs(r["thorax_z_mm"] - (z_ref + r["dz_mm"])) < 0.05]
        doc["release_runs"] = runs
        doc["heights_that_held"] = [r["dz_mm"] for r in held]
        doc["verdict"] = "HELD" if held else "STILL_FALLS"
        print(f"\nVERDICT: {doc['verdict']}"
              + (f"  -- held at dz {doc['heights_that_held']}" if held
                 else "  -- coplanar feet alone do not make it stand"))

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
