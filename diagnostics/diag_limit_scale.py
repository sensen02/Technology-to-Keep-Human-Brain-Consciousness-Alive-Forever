#!/usr/bin/env python3
"""THE JOINT LIMITS ARE SOFT TOO, AND THAT IS WHAT COLLAPSES THE FLY.

MEASURED, and it is the mechanism behind every sag in this project: with the body restricted to pure
vertical motion, the fly sinks 1.39 mm and the WORST JOINT IN THE MODEL ENDS UP 110% OF ITS RANGE
BEYOND ITS LIMIT (joint_RHCoxa_yaw at -110%, joint_RFCoxa_roll at -60%, joint_LHCoxa_yaw at -27%).
A joint cannot be outside its own range unless the limit is soft -- and MuJoCo's limit constraints use
the same metre-scale default as the contacts, solref = (0.02 s, 1.0).  So under load the legs simply
travel THROUGH their anatomical ranges and the leg folds.

The tell-tale is the mass sweep: at 0.25 mg with stiffness 16 the joint overshoots by only 4% and the
sag is 0.0275 mm -- the fly stands.  Every heavier case overshoots further and sags further.

This scales the joint-limit solver parameters to the model's own clock and reports the overshoot, the
sag and whether the fly stands.  Nothing here imposes any activation: it makes the limits the source
model declared actually hold.
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


def run(lim_tc, stiff=0.4, mass_mg=2.49427, freeze_rot=True, contact_tc=None, dur=1.5):
    m = _load_mjcf(str(XML)).compile()
    f = mass_mg / (float(sum(m.body_mass)) * 1000.0)
    for b in range(m.nbody):
        m.body_mass[b] *= f
        m.body_inertia[b] *= f
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
            if lim_tc is not None:
                m.jnt_solref[j, 0] = lim_tc      # the LIMIT's solver time constant
                m.jnt_solref[j, 1] = 1.0
    if contact_tc is not None:
        for g in range(m.ngeom):
            m.geom_solref[g, 0] = contact_tc
            m.geom_solref[g, 1] = 1.0
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
    leg = body = 0.0; legs = set()
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
            body += abs(fr[0])
    worst, wn = 1e9, ""
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not nm.startswith("joint_"):
            continue
        q = float(d.qpos[int(m.jnt_qposadr[j])]); lo, hi = m.jnt_range[j]
        span = hi - lo
        if span <= 0:
            continue
        pos = min((q - lo) / span, (hi - q) / span)
        if pos < worst:
            worst, wn = pos, nm
    return {"limit_timeconst": lim_tc, "joint_stiffness": stiff, "mass_mg": mass_mg,
            "contact_timeconst": contact_tc, "sag_mm": z0 - float(d.xpos[th][2]),
            "thorax_z": float(d.xpos[th][2]), "leg_bw": leg / mw, "body_bw": body / mw,
            "legs_down": len(legs), "worst_joint": wn, "worst_joint_margin": worst,
            "stands": bool(body < 1e-3 and len(legs) == 6 and d.xpos[th][2] > 1.6),
            "seated_z": z0}


def main():
    out = {"rows": []}
    print("=== scale the JOINT-LIMIT solver time constant (seated height 2.0293 mm; 'stands' needs")
    print("    no body contact, six feet down, thorax above 1.6 mm) ===")
    print(f"{'limit tc':>10}{'stiff':>7}{'mass mg':>10}{'sag mm':>9}{'thorax z':>10}"
          f"{'worst joint margin':>24}{'legs':>6}{'body bw':>9}  verdict")
    for tc in (None, 0.02, 0.005, 0.002, 0.001, 0.0005, 0.0002):
        for mass in (2.49427, 1.0):
            r = run(tc, 0.4, mass)
            out["rows"].append(r)
            tag = "default" if tc is None else f"{tc:g}"
            print(f"{tag:>10}{0.4:>7.1f}{mass:>10.5f}{r['sag_mm']:>9.4f}{r['thorax_z']:>10.4f}"
                  f"{f'{r[chr(119)+chr(111)+chr(114)+chr(115)+chr(116)+chr(95)+chr(106)+chr(111)+chr(105)+chr(110)+chr(116)]} {100*r[chr(119)+chr(111)+chr(114)+chr(115)+chr(116)+chr(95)+chr(106)+chr(111)+chr(105)+chr(110)+chr(116)+chr(95)+chr(109)+chr(97)+chr(114)+chr(103)+chr(105)+chr(110)]:.0f}%':>24}"
                  f"{r['legs_down']:>6}{r['body_bw']:>9.4f}  "
                  f"{'*** STANDS ***' if r['stands'] else 'collapsed'}")
    (HERE / "outputs" / "limit_scale.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'limit_scale.json'}")


if __name__ == "__main__":
    main()
