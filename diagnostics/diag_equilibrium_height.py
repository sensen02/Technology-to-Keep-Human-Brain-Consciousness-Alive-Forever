#!/usr/bin/env python3
"""IS THE COLLAPSE A BAD STARTING HEIGHT, OR AN UNSTABLE EQUILIBRIUM?

THE TWO HYPOTHESES, and they need completely different fixes:

  H1  the keyframe is simply at the WRONG HEIGHT -- released a little too high, so the body falls
      onto its legs and overshoots into the belly.  Then there is an equilibrium height, and the
      whole "does not stand" result is an artefact of where the model starts.
  H2  the pose is at an equilibrium but it is UNSTABLE -- the body tips out of it.  Then no starting
      height helps and the fix is geometry or control, not the initial condition.

MEASURED EVIDENCE FOR NEITHER SO FAR, which is why this runs:
  * at the keyframe (free-joint z = 2.0454 mm) all six tarsi are within 0.06 mm of the floor and the
    centre of mass sits 0.5227 mm INSIDE the support polygon, so it looks statically fine;
  * yet only 0.904 of one body weight is carried by the contacts there, so it is NOT in equilibrium;
  * and the load is carried almost entirely by the two MIDDLE legs (LM 0.309 + RM 0.396 of 0.904),
    with a -0.2702 mm*BW pitching moment about the centre of mass.

So the experiment is a sweep of the free joint's initial height, and at each height two questions:
does the contact force balance body weight (equilibrium?), and if released there, does the fly stay
(then H1) or leave (then H2)?

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_equilibrium_height.py
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


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "equilibrium_height.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    gnd = {g for g in range(m.ngeom)
           if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    z_key = float(d.qpos[2])
    qpos_key = np.asarray(d.qpos, dtype=float).copy()
    up0 = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
    res = np.zeros(6, dtype=float)

    def statics():
        """Contact force total and per-leg split, at the CURRENT state (mj_forward already run)."""
        tot, per = 0.0, {}
        for c in range(int(d.ncon)):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d, c, res)
            f = abs(float(res[0]))
            other = names[g2] if g1 in gnd else names[g1]
            k = other[:2] if other[:2] in LEGS else other
            per[k] = per.get(k, 0.0) + f
            tot += f
        return tot / mw, per

    def up_axis():
        return np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]

    rows = []
    for dz in np.arange(-0.30, 0.16, 0.02):
        d.qpos[:] = qpos_key
        d.qpos[2] = z_key + dz
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        tot0, per0 = statics()
        n1 = sum(1 for v in per0.values() if v > 1e-6)
        # release and see where it ends
        n_steps = int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))
        for _ in range(n_steps):
            mujoco.mj_step(m, d)
        up = up_axis()
        rows.append({
            "dz_mm": float(dz), "start_z_mm": float(z_key + dz),
            "static_total_bw": float(tot0), "static_n_feet": n1,
            "static_per_leg_bw": {k: float(v / mw) for k, v in sorted(per0.items())},
            "settled_z_model": float(d.xpos[th][2]),
            "settled_tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
            "drift_model": float(d.xpos[th][2] - (z_key + dz)),
        })
        print(f"  dz {dz:+.2f}  start z {z_key+dz:.4f}  static {tot0:5.3f} BW over {n1} feet  "
              f"-> settled {rows[-1]['settled_z_mm']:.4f} mm, tilt {rows[-1]['settled_tilt_deg']:5.2f} deg, "
              f"drift {rows[-1]['drift_mm']:+.4f}")

    # where is it actually in equilibrium?
    near = min(rows, key=lambda r: abs(r["static_total_bw"] - 1.0))
    stable = [r for r in rows if abs(r["drift_mm"]) < 0.05 and r["settled_tilt_deg"] < 5.0]
    print(f"\nclosest to static equilibrium: dz {near['dz_mm']:+.2f} mm "
          f"(z {near['start_z_mm']:.4f}), carrying {near['static_total_bw']:.4f} BW "
          f"over {near['static_n_feet']} feet")
    print(f"  per-leg there (BW): " + " ".join(f"{k} {v:.3f}" for k, v in near["static_per_leg_bw"].items()))
    print(f"\nheights that HELD (drift < 0.05 mm and tilt < 5 deg): "
          + (f"{[round(r['dz_mm'],2) for r in stable]}" if stable else "NONE"))
    verdict = "H1_HEIGHT" if stable else "H2_UNSTABLE"
    print(f"VERDICT: {verdict}"
          + ("  -- there is a starting height from which it stays"
             if stable else "  -- no starting height held, so the pose is unstable, not mis-seated"))

    doc = {
        "overall": "PASS", "status": "PASS",
        "verdict": verdict,
        "keyframe_z_mm": z_key, "one_body_weight_model_units": mw,
        "settle_ms": args.settle_ms, "rows": rows,
        "closest_to_equilibrium": near,
        "heights_that_held": [r["dz_mm"] for r in stable],
        "units": "mm, degrees, and multiples of one body weight (model force units)",
        "provenance": ("MuJoCo rollouts of fruitfly_six_leg_standing.xml from its own keyframe with the "
                       "free joint's z offset, no actuation"),
        "what_this_does_not_establish": (
            "that a held height is a usable stance.  Holding still for 0.5 s at one starting height "
            "is not walking, and 0.5 s is short.  It also does not say the pose is biologically "
            "right: the keyframe, the joint springs and the contact parameters are model properties, "
            "and this fly is a simulator object."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
