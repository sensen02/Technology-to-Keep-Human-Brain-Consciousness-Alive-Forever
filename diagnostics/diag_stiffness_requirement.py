#!/usr/bin/env python3
"""HOW MUCH SUPPORT IS MISSING?  MEASURE THE POSTURAL-TONE REQUIREMENT.

The standing model, free joint, no activation at all, collapses: the thorax sinks 0.66 mm and 53%
of the weight ends up on the thorax, abdomen, wing and proboscis.  Two sweeps say how far off the
passive skeleton is, and what motor-neuron-driven muscle tone would have to supply.

SWEEP A -- joint spring stiffness.  NOT a proposed fix: raising the springs would be exactly the
hand-authored posture the project forbids.  It is a MEASUREMENT of the torque scale required to
carry the body, expressed in the model's own parameter.

SWEEP B -- uniform muscle activation.  Also not a controller: a constant ctrl on every muscle is not
behaviour and is not proposed as one.  It measures how much muscle tone is needed, which is the
number the connectome loop has to reproduce from motor-neuron firing alone.

A "stand" here means: no body part other than the legs touches the floor, and the thorax holds the
height at which the tarsi were seated (1.3067 mm).
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
SEATED_Z = 1.3067          # MEASURED by tools_stand_and_ground.py: the height at which tarsi seated
DUR = 2.0


def probe(m, d, mw):
    """settle, then report height, tilt, and the split of load between tarsi and body."""
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(int(DUR / m.opt.timestep)):
        mujoco.mj_step(m, d)
    up = d.xmat[th].reshape(3, 3)[:, 2]
    leg = body = 0.0
    legs = set()
    for c in range(d.ncon):
        cc = d.contact[c]
        if cc.geom1 != fid and cc.geom2 != fid:
            continue
        other = cc.geom2 if cc.geom1 == fid else cc.geom1
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
        f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
        if b[:2] in LEGS:
            leg += abs(f[0]); legs.add(b[:2])
        else:
            body += abs(f[0])
    return {"thorax_z": float(d.xpos[th][2]), "up_z": float(up[2]),
            "legs_down": len(legs), "leg_bw": leg / mw, "body_bw": body / mw,
            "standing": bool(body < 1e-3 and len(legs) == 6)}


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    base_stiff = None
    for j in range(m.njnt):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "") == "joint_LFCoxa_pitch":
            base_stiff = float(m.jnt_stiffness[j])
    print(f"one body weight = {mw:.4f} force units;  seated thorax height = {SEATED_Z} mm")
    print(f"model's own leg-joint stiffness = {base_stiff}")

    out = {"stiffness_sweep": [], "activation_sweep": [], "seated_thorax_z_mm": SEATED_Z,
           "one_body_weight": mw}
    print(f"\n=== SWEEP A: leg-joint spring stiffness (no activation) ===")
    print(f"{'stiffness':>11}{'thorax z':>11}{'up_z':>9}{'legs':>6}{'tarsi bw':>10}{'body bw':>10}  verdict")
    import copy
    saves = {}
    for k, s in enumerate([0.4, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]):
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith("joint_"):
                m.jnt_stiffness[j] = s
        r = probe(m, d, mw)
        r["stiffness"] = s
        out["stiffness_sweep"].append(r)
        print(f"{s:>11.1f}{r['thorax_z']:>11.4f}{r['up_z']:>9.4f}{r['legs_down']:>6}"
              f"{r['leg_bw']:>10.4f}{r['body_bw']:>10.4f}  "
              f"{'STANDS' if r['standing'] else 'collapsed'}")
    # restore
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = base_stiff

    print(f"\n=== SWEEP B: uniform muscle activation, stiffness at the model's own {base_stiff} ===")
    print(f"{'ctrl':>8}{'thorax z':>11}{'up_z':>9}{'legs':>6}{'tarsi bw':>10}{'body bw':>10}  verdict")
    for a in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for i in range(int(DUR / m.opt.timestep)):
            d.ctrl[:] = a
            mujoco.mj_step(m, d)
        up = d.xmat[th].reshape(3, 3)[:, 2]
        leg = body = 0.0; legs = set()
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 != fid and cc.geom2 != fid:
                continue
            other = cc.geom2 if cc.geom1 == fid else cc.geom1
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
            f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
            if b[:2] in LEGS:
                leg += abs(f[0]); legs.add(b[:2])
            else:
                body += abs(f[0])
        r = {"ctrl": a, "thorax_z": float(d.xpos[th][2]), "up_z": float(up[2]),
             "legs_down": len(legs), "leg_bw": leg / mw, "body_bw": body / mw,
             "standing": bool(body < 1e-3 and len(legs) == 6)}
        out["activation_sweep"].append(r)
        print(f"{a:>8.3f}{r['thorax_z']:>11.4f}{r['up_z']:>9.4f}{r['legs_down']:>6}"
              f"{r['leg_bw']:>10.4f}{r['body_bw']:>10.4f}  "
              f"{'STANDS' if r['standing'] else 'collapsed'}")
    (HERE / "outputs" / "postural_tone_requirement.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'postural_tone_requirement.json'}")


if __name__ == "__main__":
    main()
