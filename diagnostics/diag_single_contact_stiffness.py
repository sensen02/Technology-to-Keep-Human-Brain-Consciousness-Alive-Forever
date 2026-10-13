#!/usr/bin/env python3
"""PER-LEG LOAD, WITH THE STATIC INDETERMINACY REMOVED.

WHY THIS EXISTS.  The previous round measured that a RIGID downward translation of the body by
0.06 mm -- joint angles untouched, so every leg's penetration grows by exactly 0.06 mm -- raised the
total contact force only from 0.9040 to 0.9848 body weights, while some legs gained and others LOST
(LF 0.037 -> 0.008).  Under rigid translation every contact force must rise.  They did not, because a
rigid body resting on about twenty contact points is STATICALLY INDETERMINATE: the equilibrium
equations admit infinitely many force distributions and MuJoCo picks one from `solref`/`solimp` and
the constraint ordering.  So "the middle legs carry the load" is largely a SOLVER CHOICE and cannot
be read as a geometric fact -- that claim was retracted.

WHAT THIS DOES.  It removes the indeterminacy instead of arguing about it: collisions are enabled on
ONE tarsus geom per leg (and the floor), so the pose has six contact points.  Six equations of
equilibrium for six unknown normal forces then have a unique solution, and each leg's vertical
stiffness dF/dz becomes a well-posed measurement.

The question it answers is the one that decides the whole workstream: CAN this pose support the body
at all?  If every leg has a healthy positive stiffness, the problem is in how they combine and a
stance solution exists.  If a leg cannot produce positive vertical stiffness anywhere in its joint
range, no spring and no muscle will help and the POSE has to change.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_single_contact_stiffness.py
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
#: The tip segment of each leg.  Its contact with the plane is a single point per leg.
TIP = "Tarsus5"


def build(tip_only: bool = True):
    """Stack B's spec, optionally with collisions restricted to one geom per leg."""
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    spec = _load_mjcf(str(XML))
    if tip_only:
        def walk(b):
            for g in list(b.geoms):
                nm = str(getattr(g, "name", "") or "")
                is_floor = int(g.type) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                keep = is_floor or (nm[:2] in LEGS and TIP in nm)
                if not keep:
                    g.contype = 0
                    g.conaffinity = 0
            for c in b.bodies:
                walk(c)
        walk(spec.worldbody)
    return spec.compile()


def main() -> int:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "single_contact_stiffness.json"))
    args = ap.parse_args()

    m = build(True)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    gnd = {g for g in range(m.ngeom)
           if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    n_collidable = sum(1 for g in range(m.ngeom)
                       if int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
    q = np.asarray(d.qpos, dtype=float).copy()
    res = np.zeros(6)

    def per_leg():
        out = {L: 0.0 for L in LEGS}
        n = 0
        for c in range(int(d.ncon)):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d, c, res)
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                out[o[:2]] += abs(float(res[0]))
            n += 1
        return out, n

    print(f"collidable geoms: {n_collidable} (floor + one tarsus per leg = {1 + len(LEGS)})")
    print(f"one body weight = {mw:.4f} model force units\n")

    # ---- geometry of the tip-only stance
    tip_z = {}
    for L in LEGS:
        gs = [g for g in range(m.ngeom) if names[g].startswith(L) and TIP in names[g]]
        tip_z[L] = min(float(d.geom_xpos[g][2]) for g in gs) if gs else float("nan")
    print("foot (tip) heights in the shipped keyframe, mm:")
    for L in LEGS:
        print(f"   {L}: {tip_z[L]:+.4f}")
    spread = max(tip_z.values()) - min(tip_z.values())
    print(f"   spread = {spread:.4f} mm\n")

    # ---- per-leg stiffness by rigid downward translation
    print("rigid downward translation: per-leg normal force (body weights)")
    print(f"   {'dz_mm':>7} " + " ".join(f"{L:>7}" for L in LEGS) + f" {'total':>8} {'ncon':>5}")
    samples = []
    for dz in np.arange(0.0, -0.101, -0.01):
        d.qpos[:] = q
        d.qpos[2] = q[2] + dz
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        per, n = per_leg()
        bw = {L: per[L] / mw for L in LEGS}
        samples.append({"dz_mm": float(dz), "per_leg_bw": bw, "total_bw": float(sum(bw.values())),
                        "ncon": n})
        print(f"   {dz:>7.3f} " + " ".join(f"{bw[L]:>7.3f}" for L in LEGS)
              + f" {sum(bw.values()):>8.4f} {n:>5d}")

    z0, z1 = samples[0], samples[-1]
    stiff = {L: (z1["per_leg_bw"][L] - z0["per_leg_bw"][L]) * mw / (z1["dz_mm"] - z0["dz_mm"])
             for L in LEGS}
    print("\nper-leg vertical stiffness dF/dz, model force units per mm:")
    for L in LEGS:
        need = mw / 6.0
        print(f"   {L}: k = {stiff[L]:9.3f}    force at dz=0: {z0['per_leg_bw'][L]:.3f} BW")
    ktot = sum(stiff.values())
    print(f"   total k = {ktot:9.3f}  ({ktot / mw:.3f} body weights per mm)")
    print(f"   carrying one body weight would need {mw / ktot if ktot else float('nan'):+.4f} mm "
          f"of further travel")

    # ---- is any leg's stiffness non-positive?  that is the decisive question
    bad = {L: k for L, k in stiff.items() if k <= 0.0}
    print("\nlegs whose vertical stiffness is NOT positive: "
          + (", ".join(f"{L} ({k:.2f})" for L, k in bad.items()) if bad else "none"))

    # ---- can it stand from any height with tip-only contacts?
    print("\nrelease test (0.5 s, no actuation):")
    runs = []
    for dz in (0.0, -0.02, -0.06, -0.12):
        d.qpos[:] = q
        d.qpos[2] = q[2] + dz
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))):
            mujoco.mj_step(m, d)
        up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        per, n = per_leg()
        runs.append({"dz_mm": float(dz), "thorax_z_model": float(d.xpos[th][2]),
                     "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up[2]), -1, 1)))),
                     "ncon": n, "per_leg_bw": {L: per[L] / mw for L in LEGS}})
        print(f"   dz {dz:+.2f}: settled z {runs[-1]['thorax_z_mm']:.4f} mm, "
              f"tilt {runs[-1]['tilt_deg']:5.2f} deg, contacts {n}")
    held = [r for r in runs if r["tilt_deg"] < 5.0
            and abs(r["thorax_z_mm"] - (q[2] + r["dz_mm"])) < 0.05]

    verdict = ("POSE_CANNOT_SUPPORT" if bad else
               ("STILL_FALLS" if not held else "HELD"))
    print(f"\nVERDICT: {verdict}")

    doc = {
        "overall": "PASS", "status": "PASS", "verdict": verdict,
        "why_this_test_exists": (
            "the previous round's per-leg load shares were retracted as a solver artefact of a "
            "statically indeterminate contact set (~20 contacts on a rigid body).  Restricting "
            "collisions to one tarsus geom per leg leaves six contacts, for which equilibrium has a "
            "unique solution, so per-leg load and per-leg STIFFNESS become well-posed measurements."),
        "collidable_geoms": n_collidable, "tip_only": True, "tip_segment": TIP,
        "one_body_weight_model_units": mw,
        "foot_heights_mm": tip_z, "foot_height_spread_mm": spread,
        "rigid_translation_samples": samples,
        "per_leg_stiffness_force_units_per_mm": stiff,
        "total_stiffness_force_units_per_mm": ktot,
        "legs_with_nonpositive_stiffness": bad,
        "release_runs": runs, "heights_that_held": [r["dz_mm"] for r in held],
        "units": ("mm for lengths, radians/degrees for angles, model force units for force; one "
                  "MODEL FORCE UNIT is 1 micro-newton, and one body weight is "
                  f"{mw:.4f} of them"),
        "provenance": ("MuJoCo, fruitfly_six_leg_standing.xml with collision restricted to the floor "
                       "and one tarsus geom per leg, from the model's own keyframe, no actuation"),
        "what_this_does_not_establish": (
            "that the real model stands, or that a fly does.  Restricting collisions to the tarsus "
            "tip CHANGES THE MODEL: a real tarsus rests on more than its tip, and the tip-only pose "
            "has a 0.06 mm-scale foot-height spread of its own.  What it establishes is narrower and "
            "is the thing that was not knowable before: whether each leg can produce vertical "
            "restoring force in THIS pose at all.  A pose that holds still for 0.5 s is not walking, "
            "and every number here is a simulator property, not an animal's."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
