#!/usr/bin/env python3
"""JOINT STIFFNESS, RE-SWEPT NOW THAT THE DECLARED LIMITS ARE ACTUALLY ENFORCED.

The earlier stiffness sweep was run against a model in which the joints could travel THROUGH their
anatomical ranges, so raising the stiffness only pushed them further past their stops -- which is why
8x more stiffness bought only 25% less sag.  With the limits enforced that confound is gone, so the
sweep is repeated and is finally interpretable.

The body's rotation is frozen so the measurement is about the vertical direction alone, the run is
taken to rest, and STANDING means at rest with all six feet down and no non-leg body part touching.
This is a PLANT measurement: not one activation is computed or imposed.
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


def run(stiff, mass_mg, dur=4.0, freeze_rot=True):
    m = _load_mjcf(str(XML)).compile()
    f = mass_mg / (float(sum(m.body_mass)) * 1000.0)
    for b in range(m.nbody):
        m.body_mass[b] *= f
        m.body_inertia[b] *= f
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    a, va = int(m.jnt_qposadr[jf]), int(m.jnt_dofadr[jf])
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[th][2])
    qr = np.array(m.key_qpos[0][a + 3:a + 7], dtype=float)
    for i in range(int(dur / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if freeze_rot:
            d.qpos[a + 3:a + 7] = qr
            d.qvel[va + 3:va + 6] = 0.0
    leg = body = 0.0; legs = set(); nb = set()
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        o = cc.geom2 if cc.geom1 == fid else cc.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o])) or ""
        fr = np.zeros(6); mujoco.mj_contactForce(m, d, c, fr)
        if b[:2] in LEGS:
            leg += abs(fr[0]); legs.add(b[:2])
        else:
            body += abs(fr[0]); nb.add(b)
    worst = 1e9; wn = ""
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not nm.startswith("joint_") or not m.jnt_limited[j]:
            continue
        q = float(d.qpos[int(m.jnt_qposadr[j])]); lo, hi = m.jnt_range[j]
        span = hi - lo
        if span > 0:
            pos = min((q - lo) / span, (hi - q) / span)
            if pos < worst:
                worst, wn = pos, nm
    vz = abs(float(d.qvel[va + 2]))
    at_rest = vz < 0.5
    stands = bool(at_rest and body < 1e-3 and len(legs) == 6)
    return {"stiffness": stiff, "mass_mg": mass_mg, "seated_z": z0,
            "thorax_z": float(d.xpos[th][2]), "sag_mm": z0 - float(d.xpos[th][2]),
            "vz": vz, "at_rest": bool(at_rest), "leg_bw": leg / mw, "body_bw": body / mw,
            "legs_down": len(legs), "nonleg": sorted(nb),
            "worst_joint": wn, "worst_joint_margin": worst if worst < 1e9 else None,
            "stands": stands}


def main():
    out = {"rows": []}
    print("limits ENFORCED, 4 s to rest")
    print(f"{'stiff':>7}{'mass mg':>10}{'seated z':>10}{'settled z':>11}{'sag':>8}{'|vz|':>8}"
          f"{'tarsi bw':>10}{'body bw':>9}{'legs':>6}{'worst joint %':>15}  verdict")
    FREEZE = os.environ.get("FREEZE_ROT", "1") == "1"
    print(f"body rotation frozen: {FREEZE}   (free = the real fly, which may also tip)")
    for mass in (2.49427, 1.0):
        for stiff in (0.4, 2.0, 4.0, 8.0, 16.0, 32.0):
            r = run(stiff, mass, freeze_rot=FREEZE)
            out["rows"].append(r)
            w = f"{100*r['worst_joint_margin']:.0f}" if r["worst_joint_margin"] is not None else "-"
            print(f"{stiff:>7.1f}{mass:>10.5f}{r['seated_z']:>10.4f}{r['thorax_z']:>11.4f}"
                  f"{r['sag_mm']:>8.4f}{r['vz']:>8.4f}{r['leg_bw']:>10.4f}{r['body_bw']:>9.4f}"
                  f"{r['legs_down']:>6}{w:>15}  "
                  f"{'*** STANDS ***' if r['stands'] else 'collapsed'}")
        print()
    (HERE / "outputs" / "stiffness_after_limits.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE / 'outputs' / 'stiffness_after_limits.json'}")


if __name__ == "__main__":
    main()
