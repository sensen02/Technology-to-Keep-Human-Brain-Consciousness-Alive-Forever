#!/usr/bin/env python3
"""WHICH JOINT ABSORBS THE LOAD?  Read every joint's deflection under a known foot force.

WHY THIS MEASUREMENT.  The load sharing is grossly uneven -- at the seated instant the middle legs
carry 70.5% of the body weight, the forelegs 4.5% and the hind legs 15.4% -- and that is what tips the
fly over, because the fore-aft support degenerates to a line under the middle legs.  The leg's share of
the load is set by its VERTICAL STIFFNESS, which was measured as 4.9 body weights per mm for a middle
leg against 0.025 for a foreleg: a factor of 200.

A leg's vertical stiffness is not a property of its pose alone; it is the sum of its joints' resistance,
each contributing  k_j / arm_j^2  where arm_j is the moment arm a vertical foot force has about that
joint's axis.  So the question of WHICH joint makes a foreleg soft is answerable by reading the joints
themselves.  This welds the body in place, applies a known vertical force at one foot, and reports each
joint's angular deflection and the share of the foot's vertical displacement it accounts for.

A PLANT measurement: no activation is computed or imposed.
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
JR = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"), ("Trochanter", "yaw"),
      ("Trochanter", "pitch"), ("Trochanter", "roll"), ("Tibia", "pitch"))


def main():
    m = _load_mjcf(str(XML)).compile()
    stiff = float(os.environ.get("DEFL_STIFF", "0.4"))
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    F = mw / 6.0
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    a, va = int(m.jnt_qposadr[jf]), int(m.jnt_dofadr[jf])
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    out = {"force_uN": F, "stiffness": stiff, "legs": {}}
    print(f"one sixth of body weight = {F:.4f} uN; joint stiffness {stiff}; body welded in place")
    for leg in LEGS:
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
        def settle(force, n=6000):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.act[:] = 0.0; d.ctrl[:] = 0.0; d.xfrc_applied[:] = 0.0
            mujoco.mj_forward(m, d)
            def read_joints():
                out_q = {}
                for _s, _js in JR:
                    _j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"joint_{leg}{_s}_{_js}")
                    out_q[f"{_s}_{_js}"] = float(d.qpos[int(m.jnt_qposadr[_j])])
                return out_q
            q0 = read_joints()
            z0 = float(d.xpos[tb][2])
            bq = np.array(m.key_qpos[0][a:a + 7], dtype=float)
            for _ in range(n):
                d.xfrc_applied[:] = 0.0
                d.xfrc_applied[tb, 2] = -force
                mujoco.mj_step(m, d)
                d.qpos[a:a + 7] = bq
                d.qvel[va:va + 6] = 0.0
            q1 = read_joints()
            return q0, q1, z0, float(d.xpos[tb][2])
        q0f, q1f, z0, z1 = settle(F)
        travel = {f"{s}_{js}": q1f[f"{s}_{js}"] - q0f[f"{s}_{js}"] for s, js in JR}
        # the share of the foot's vertical drop that each joint's rotation accounts for:
        # the foot's vertical velocity from joint j alone is arm_j * qdot_j
        arms = {}
        for s, js in JR:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{leg}{s}_{js}")
            p = np.asarray(d.xanchor[jid], float)
            ax = np.asarray(d.xaxis[jid], float)
            c = np.asarray(d.xpos[tb], float)
            arms[f"{s}_{js}"] = float(np.dot(ax, np.cross(c - p, np.array([0.0, 0.0, -1.0]))))
        contrib = {k: abs(arms[k] * travel[k]) for k in travel}
        tot = sum(contrib.values())
        out["legs"][leg] = {"foot_drop_mm": z0 - z1, "joint_travel_rad": travel,
                            "moment_arm_mm": arms, "vertical_contribution": contrib,
                            "sum_of_contributions": tot}
        print(f"\n  {leg}: foot drops {z0 - z1:.4f} mm under {F:.4f} uN "
              f"(stiffness {F/max(z0-z1,1e-12):.1f} uN/mm)")
        print(f"    {'joint':<20}{'moment arm':>12}{'travel (rad)':>15}{'arm*travel':>13}"
              f"{'share %':>10}")
        for k in sorted(contrib, key=lambda k: -contrib[k]):
            print(f"    {k:<20}{arms[k]:>12.4f}{travel[k]:>15.4f}{contrib[k]:>13.4f}"
                  f"{100*contrib[k]/max(tot,1e-12):>10.1f}")
    (HERE / "outputs" / "joint_deflection.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'joint_deflection.json'}")


if __name__ == "__main__":
    main()
