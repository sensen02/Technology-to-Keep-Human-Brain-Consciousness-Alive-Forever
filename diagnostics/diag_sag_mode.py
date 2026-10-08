#!/usr/bin/env python3
"""WHICH FREEDOM LETS THE BODY SINK?  Lock the free joint's parts one at a time.

THE CONTRADICTION BEING RESOLVED, both MEASURED:
  * with the body's pose written back every step (an exact hold), a leg deflects only 0.004-0.03 mm
    under one sixth of body weight -- the legs are stiff struts, stiffness 50-4500 uN/mm;
  * with the body free it sinks 1.38 mm (2.0293 -> 0.6465 mm) until its belly reaches the floor, even
    though all six feet stay in contact.
A 40-to-400-fold difference means the leg is not being loaded the same way in the two cases.  The body
has six degrees of freedom, and a rigid body settles along its SOFTEST direction, which need not be
plain vertical compression.  So the free joint is restricted one part at a time:
    translation only  -- rotation frozen, so the body cannot tilt
    rotation only     -- translation frozen, so the body can only tilt
    both free         -- the real case
and the resulting movement says which is responsible.  No activation anywhere.
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


def run(mode, mass_mg=None, dur=1.5):
    m = _load_mjcf(str(XML)).compile()
    if mass_mg:
        f = mass_mg / (float(sum(m.body_mass)) * 1000.0)
        for b in range(m.nbody):
            m.body_mass[b] *= f
            m.body_inertia[b] *= f
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    a, va = int(m.jnt_qposadr[jf]), int(m.jnt_dofadr[jf])
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[th][2])
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    q_rot = np.array(m.key_qpos[0][a + 3:a + 7], dtype=float)
    q_pos = np.array(m.key_qpos[0][a:a + 3], dtype=float)
    for i in range(int(dur / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if mode in ("translation_only", "both"):
            d.qpos[a + 3:a + 7] = q_rot          # freeze rotation
            d.qvel[va + 3:va + 6] = 0.0
        if mode == "rotation_only":
            d.qpos[a:a + 3] = q_pos              # freeze translation
            d.qvel[va:va + 3] = 0.0
    up = d.xmat[th].reshape(3, 3)[:, 2]
    leg = body = 0.0; legs = set()
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        o = cc.geom2 if cc.geom1 == fid else cc.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o])) or ""
        f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
        if b[:2] in LEGS:
            leg += abs(f[0]); legs.add(b[:2])
        else:
            body += abs(f[0])
    return {"mode": mode, "mass_mg": mass_mg or float(sum(m.body_mass)) * 1000,
            "sag_mm": z0 - float(d.xpos[th][2]), "tilt_deg": float(np.degrees(
                np.arccos(np.clip(up @ up0, -1, 1)))),
            "thorax_z": float(d.xpos[th][2]), "leg_bw": leg / mw, "body_bw": body / mw,
            "legs_down": len(legs)}


def main():
    rows = []
    print(f"{'mode':<18}{'mass (mg)':>11}{'sag (mm)':>10}{'tilt (deg)':>12}{'thorax z':>11}"
          f"{'tarsi bw':>10}{'body bw':>10}")
    for mass in (2.49427, 1.0):
        for mode in ("both", "translation_only", "rotation_only"):
            r = run(mode, mass)
            rows.append(r)
            print(f"{mode:<18}{r['mass_mg']:>11.4f}{r['sag_mm']:>10.4f}{r['tilt_deg']:>12.2f}"
                  f"{r['thorax_z']:>11.4f}{r['leg_bw']:>10.4f}{r['body_bw']:>10.4f}")
    print("\n  both free        = the real fly")
    print("  translation only = the body may only go up and down (rotation frozen)")
    print("  rotation only    = the body may only tilt (translation frozen)")
    (HERE / "outputs" / "sag_mode.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'sag_mode.json'}")


if __name__ == "__main__":
    main()
