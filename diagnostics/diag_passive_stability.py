#!/usr/bin/env python3
"""IS THE TIPPING REAL, OR DID I KICK THE FLY BY STARTING IT IN PENETRATION?

The stance solve seats the tarsi with about 0.003 mm of penetration, and the springs sit exactly at
their reference.  If the first step then pushes the feet out of the floor, a stiff model gets an
impulse at t=0 and can tip over for a reason that has nothing to do with whether a passive stance is
stable.  This separates the two by starting the fly slightly ABOVE the floor and letting it settle
onto its feet, across stiffness and start height.

A passive stance is called STABLE only if, at the end, all six legs are down, no body part touches
the floor, and the body is still within 10 degrees of upright.
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
DUR = float(os.environ.get("STAB_DUR", "3.0"))


def run(m, d, stiff, lift, mw, damping=0.02, dur=DUR):
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
            m.dof_damping[int(m.jnt_dofadr[j])] = damping
    mujoco.mj_resetDataKeyframe(m, d, 0)
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")])
    d.qpos[adr + 2] += lift
    mujoco.mj_forward(m, d)
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(int(dur / m.opt.timestep)):
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
    tilt = float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1))))
    stable = bool(body < 1e-3 and len(legs) == 6 and tilt < 10.0)
    return {"stiffness": stiff, "lift_mm": lift, "thorax_z": float(d.xpos[th][2]),
            "tilt_deg": tilt, "legs_down": len(legs), "leg_bw": leg / mw, "body_bw": body / mw,
            "stable": stable}


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    print(f"one body weight = {mw:.4f} force units; duration {DUR}s")
    rows = []
    print(f"\n{'stiff':>7}{'lift':>7}{'thorax z':>10}{'tilt':>8}{'legs':>6}{'tarsi bw':>10}"
          f"{'body bw':>10}  verdict")
    for stiff in (0.4, 2.0, 8.0, 32.0, 128.0):
        for lift in (0.0, 0.05, 0.2):
            r = run(m, d, stiff, lift, mw)
            rows.append(r)
            print(f"{stiff:>7.1f}{lift:>7.2f}{r['thorax_z']:>10.4f}{r['tilt_deg']:>8.1f}"
                  f"{r['legs_down']:>6}{r['leg_bw']:>10.4f}{r['body_bw']:>10.4f}  "
                  f"{'STABLE' if r['stable'] else 'collapsed'}")
    print(f"\n=== DOES DAMPING RESCUE IT?  stiffness 32, damping swept ===")
    print(f"{'damping':>9}{'thorax z':>10}{'tilt':>8}{'legs':>6}{'body bw':>10}  verdict")
    for damp in (0.02, 0.2, 1.0, 5.0):
        r = run(m, d, 32.0, 0.05, mw, damping=damp)
        r["damping"] = damp
        rows.append(r)
        print(f"{damp:>9.2f}{r['thorax_z']:>10.4f}{r['tilt_deg']:>8.1f}{r['legs_down']:>6}"
              f"{r['body_bw']:>10.4f}  {'STABLE' if r['stable'] else 'collapsed'}")
    any_stable = any(r["stable"] for r in rows)
    print(f"\nany passive configuration stands: {any_stable}")
    (HERE / "outputs" / "passive_stability.json").write_text(json.dumps(
        {"rows": rows, "any_stable": bool(any_stable), "duration_s": DUR}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'passive_stability.json'}")


if __name__ == "__main__":
    main()
