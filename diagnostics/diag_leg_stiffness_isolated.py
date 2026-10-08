#!/usr/bin/env python3
"""MEASURE A LEG'S OWN VERTICAL STIFFNESS, WITH THE BODY HELD STILL.

WHY THE PREVIOUS TEST COULD NOT ANSWER THIS.  Applying one sixth of body weight downward at one foot
while the fly stands on all six asks the legs to support 1.167 body weights, so the fly simply falls
until its belly reaches the floor.  The "sag" that came out of it -- 1.33 mm, and IDENTICAL for every
contact time constant from 0.02 s to 0.0005 s even though the contact stiffness changed 1600x -- was
therefore not a compliance at all.  It was the distance from the seated height down to the height at
which the body rests on the ground (2.03 - 0.69 = 1.34 mm).  So that test measured nothing about the
leg, and the earlier conclusion that the contact was to blame was also wrong.

WHAT ISOLATES THE LEG.  Gravity is switched off, so the fly has no weight of its own, and then an equal
and opposite pair is applied: a downward force at ONE foot and the same force upward at the thorax.
The net force on the animal is zero, so the body does not accelerate, and the only thing that can give
is the loaded leg.  Its vertical displacement under that force is its vertical stiffness -- measured
like pressing on one leg of a table with the table held down.

This is a plant measurement: no activation is computed or imposed anywhere.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
SIXTH = 1.0 / 6.0


def measure(stiff, mass_mg, tc):
    m = _load_mjcf(str(XML)).compile()
    if mass_mg:
        cur = float(sum(m.body_mass)) * 1000.0
        f = mass_mg / cur
        for b in range(m.nbody):
            m.body_mass[b] *= f
            m.body_inertia[b] *= f
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
    for g in range(m.ngeom):
        m.geom_solref[g, 0] = tc
        m.geom_solref[g, 1] = 1.0
    m.opt.gravity[:] = 0.0                       # no body weight: the applied pair is the whole load
    d = mujoco.MjData(m)
    mw = mass_mg * 1e-3 * 9801.0 if mass_mg else float(sum(m.body_mass)) * 9801.0
    F = SIXTH * mw
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = {}
    for leg in LEGS:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.act[:] = 0.0; d.ctrl[:] = 0.0; d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
        gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}Tarsus5_geom")
        z0 = float(d.xpos[tb][2])
        body_z0 = float(d.xpos[th][2])
        for _ in range(4000):                    # 400 ms
            d.xfrc_applied[:] = 0.0
            d.xfrc_applied[tb, 2] = -F           # push this foot down
            d.xfrc_applied[th, 2] = +F           # and hold the thorax up by the same amount
            mujoco.mj_step(m, d)
        dz_foot = float(d.xpos[tb][2]) - z0
        dz_body = abs(float(d.xpos[th][2]) - body_z0)
        contact = False
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 == fid or cc.geom2 == fid:
                other = cc.geom2 if cc.geom1 == fid else cc.geom1
                if other == gb:
                    contact = True
        out[leg] = {"foot_displacement_mm": dz_foot, "body_movement_mm": dz_body,
                    "vertical_stiffness_uN_per_mm": float(F / dz_foot) if abs(dz_foot) > 1e-9 else None,
                    "foot_still_on_floor": bool(contact)}
    return out, F


def main():
    print("=== isolated leg stiffness: gravity off, equal-and-opposite force pair ===")
    print("(foot pushed down by 1/6 body weight, thorax pulled up by the same, body held still)\n")
    res = {"cases": []}
    for stiff in (0.4, 2.0, 8.0, 16.0):
        for mass_mg, tc in ((1.0, 0.001),):
            rows, F = measure(stiff, mass_mg, tc)
            print(f"--- joint stiffness {stiff}, mass {mass_mg} mg, contact timeconst {tc}s "
                  f"(force = {F:.4f} uN) ---")
            print(f"  {'leg':<5}{'foot moves (mm)':>17}{'body moves (mm)':>18}"
                  f"{'vertical stiffness (uN/mm)':>30}{'on floor':>10}")
            for leg in LEGS:
                r = rows[leg]
                k = r["vertical_stiffness_uN_per_mm"]
                print(f"  {leg:<5}{r['foot_displacement_mm']:>17.4f}{r['body_movement_mm']:>18.4f}"
                      f"{(f'{k:.1f}' if k else 'inf'):>30}"
                      f"{('yes' if r['foot_still_on_floor'] else 'NO'):>10}")
            ks = [rows[l]["vertical_stiffness_uN_per_mm"] for l in LEGS
                  if rows[l]["vertical_stiffness_uN_per_mm"]]
            print(f"  median vertical stiffness = "
                  f"{np.median(ks) if ks else float('nan'):.1f} uN/mm")
            print(f"  a leg must carry 1/6 body weight = {F:.4f} uN; to keep its deflection under "
                  f"0.1 mm it needs a stiffness of {10*F:.1f} uN/mm\n")
            res["cases"].append({"joint_stiffness": stiff, "mass_mg": mass_mg, "contact_tc": tc,
                                 "force_uN": F, "legs": rows})
    (HERE / "outputs" / "leg_stiffness_isolated.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {HERE / 'outputs' / 'leg_stiffness_isolated.json'}")


if __name__ == "__main__":
    main()
