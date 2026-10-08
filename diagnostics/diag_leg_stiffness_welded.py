#!/usr/bin/env python3
"""A LEG'S VERTICAL STIFFNESS, MEASURED WITH THE BODY WELDED IN PLACE.

TWO EARLIER ATTEMPTS AT THIS MEASUREMENT WERE BOTH ILL-POSED, AND BOTH ARE RECORDED SO THEY ARE NOT
REPEATED:
  * pushing one sixth of body weight down on one foot while the fly stood on all six asked the legs to
    carry 1.167 body weights, so the fly fell until its belly hit the floor -- and the "sag" that came
    out, 1.33 mm, was IDENTICAL for contact time constants from 0.02 s to 0.0005 s even though that
    changed the contact stiffness 1600x, which proves it was not a compliance at all but simply the
    distance from the seated height down to where the body rests (2.03 - 0.69 = 1.34 mm);
  * switching gravity off and applying an equal-and-opposite pair at the foot and the thorax made the
    fly TUMBLE AWAY, because a pair of equal and opposite forces at two different points is a COUPLE
    and nothing was holding the body.  The foot and the body both moved 14-19 mm.

The fix is to hold the body still and then load the foot.  The body is held by making the free joint's
springs very stiff about the stance pose, so the legs carry the fly's weight as usual and the applied
foot force is a pure extra load on that one leg.  The measured quantity is then the incremental
deflection of the foot per unit of extra load: an ordinary stiffness test, with the plate bolted down.
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


def probe(stiff, mass_mg, weld_k=None, total_mass=True):
    m = _load_mjcf(str(XML)).compile()
    if mass_mg:
        f = mass_mg / (float(sum(m.body_mass)) * 1000.0)
        for b in range(m.nbody):
            m.body_mass[b] *= f
            m.body_inertia[b] *= f
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    a = int(m.jnt_qposadr[jf])
    va = int(m.jnt_dofadr[jf])
    # HOLD THE BODY BY FREEZING IT, NOT BY A STIFF SPRING.  MEASURED: a free-joint spring of 1e7 made
    # the solver so slow it did not finish, and a softer one would not have been an exact hold.  Writing
    # the stance pose and zero velocity back into the free joint's own dofs after every step pins the
    # body exactly while the legs articulate and the contacts resolve normally.
    body_qpos = np.array(m.key_qpos[0][a:a + 7], dtype=float)
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    F = SIXTH * mw
    out = {}
    for leg in LEGS:
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
        def settle(force, n=4000):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.act[:] = 0.0; d.ctrl[:] = 0.0; d.xfrc_applied[:] = 0.0
            mujoco.mj_forward(m, d)
            for _ in range(n):
                d.xfrc_applied[:] = 0.0
                d.xfrc_applied[tb, 2] = -force
                mujoco.mj_step(m, d)
                d.qpos[a:a + 7] = body_qpos          # the exact weld
                d.qvel[va:va + 6] = 0.0
            return float(d.xpos[tb][2]), float(d.xpos[th][2])
        z0, b0 = settle(0.0)
        z1, b1 = settle(F)
        dz = z1 - z0
        out[leg] = {"foot_z_unloaded": z0, "foot_z_loaded": z1,
                    "foot_deflection_mm": dz, "body_movement_mm": abs(b1 - b0),
                    "vertical_stiffness_uN_per_mm": float(F / abs(dz)) if abs(dz) > 1e-9 else None,
                    "direction": "compressed (foot moved down)" if dz < 0 else "extended"}
    return out, F, mw


def main():
    print("=== leg vertical stiffness, body welded, extra load = 1/6 body weight ===")
    res = {"cases": []}
    for mass_mg in (2.49427, 1.0):
        for stiff in (0.4, 4.0, 16.0):
            rows, F, mw = probe(stiff, mass_mg, weld_k=1e7)
            ks = [r["vertical_stiffness_uN_per_mm"] for r in rows.values()
                  if r["vertical_stiffness_uN_per_mm"]]
            defl = [r["foot_deflection_mm"] for r in rows.values()]
            bodym = [r["body_movement_mm"] for r in rows.values()]
            print(f"\n--- mass {mass_mg} mg, joint stiffness {stiff} (extra load {F:.4f} uN, "
                  f"body moved {np.mean(bodym):.4f} mm) ---")
            print(f"  {'leg':<5}{'foot deflection (mm)':>22}{'stiffness (uN/mm)':>21}")
            for leg in LEGS:
                r = rows[leg]
                k = r["vertical_stiffness_uN_per_mm"]
                print(f"  {leg:<5}{r['foot_deflection_mm']:>22.4f}"
                      f"{(f'{k:.1f}' if k else 'inf'):>21}")
            print(f"  median stiffness {np.median(ks) if ks else float('nan'):.1f} uN/mm; "
                  f"median deflection {np.median(defl):.4f} mm")
            res["cases"].append({"mass_mg": mass_mg, "joint_stiffness": stiff, "force_uN": F,
                                 "body_movement_mm": float(np.mean(bodym)), "legs": rows})
    print("\n  READ THIS WAY: the deflection should be a small fraction of a millimetre for the leg to be")
    print("  a usable strut; a deflection of order a millimetre means the leg folds instead of carrying.")
    (HERE / "outputs" / "leg_stiffness_welded.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'leg_stiffness_welded.json'}")


if __name__ == "__main__":
    main()
