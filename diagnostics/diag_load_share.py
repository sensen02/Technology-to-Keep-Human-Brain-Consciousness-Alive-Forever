#!/usr/bin/env python3
"""IS THE LOAD SHARED EVENLY BY THE SIX LEGS?  A measurement of the PLANT, no solver involved.

WHY: with leg-joint stiffness raised, the fly does not fold but ROCKS -- at stiffness 16 the passive
run leaves the fly sitting back on A5, A6 and the left wing with only four legs down, and two of the
three legs off the floor are on the SAME side.  A load that is not shared evenly between corresponding
left and right legs will roll the body, so this measures the load share directly.

The stance's foot POSITIONS are symmetric by construction but its joint ANGLES are not, because the
model's right-leg geometry is only approximately the mirror of the left's (measured coxa offset
0.008-0.018 mm).  This asks what that costs in load.
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
PAIRS = (("LF", "RF"), ("LM", "RM"), ("LH", "RH"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    stiff = float(os.environ.get("LOAD_STIFF", "0.4"))
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    tarsus = {}
    for leg in LEGS:
        tarsus[leg] = [g for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(leg)
                       and "Tarsus" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    out = {"stiffness": stiff, "samples": []}
    print(f"one body weight = {mw:.4f}; leg-joint stiffness {stiff}")
    for tag, n_steps in (("at t=0 (as seated)", 0), ("after 50 ms", 500), ("after 200 ms", 2000)):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        for _ in range(n_steps):
            mujoco.mj_step(m, d)
        share = {leg: 0.0 for leg in LEGS}
        body = 0.0
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 != fid and cc.geom2 != fid:
                continue
            other = cc.geom2 if cc.geom1 == fid else cc.geom1
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
            f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
            for leg in LEGS:
                if other in tarsus[leg]:
                    share[leg] += abs(f[0])
                    break
            else:
                body += abs(f[0])
        up = d.xmat[th].reshape(3, 3)[:, 2]
        print(f"\n  {tag}: thorax {d.xpos[th][2]:.4f} mm, up_z {up[2]:+.4f}, "
              f"body on floor {body/mw:.4f} bw")
        print(f"    load share (bw): " + " ".join(f"{l}={share[l]/mw:.4f}" for l in LEGS))
        for a, b_ in PAIRS:
            tot = share[a] + share[b_]
            print(f"    {a}/{b_}: {share[a]/mw:.4f} / {share[b_]/mw:.4f}  -> imbalance "
                  f"{abs(share[a]-share[b_])/mw:.4f} bw "
                  f"({100*abs(share[a]-share[b_])/max(tot,1e-9):.0f}% of the pair's load)")
        out["samples"].append({"when": tag, "thorax_z": float(d.xpos[th][2]), "up_z": float(up[2]),
                               "body_bw": body / mw,
                               "share_bw": {l: share[l] / mw for l in LEGS}})
    (HERE / "outputs" / "load_share.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'load_share.json'}")


if __name__ == "__main__":
    main()
