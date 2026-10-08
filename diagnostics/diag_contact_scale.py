#!/usr/bin/env python3
"""THE CONTACT PARAMETERS ARE SCALED FOR A METRE-SIZE ROBOT, NOT FOR A MILLIGRAM FLY.

MEASURED, and it is the compliance that all the earlier searches were chasing: raising the leg-joint
stiffness 40x improved the sag by only 36% (1.36 -> 0.87 mm), where a linear spring system would give
40x.  So the compliance is not in the joints.  It is in the CONTACT.

MuJoCo's default solref is (0.02 s, 1.0), and the contact stiffness goes as m_eff / timeconst^2.  With
this model's units (mass in grams, length in mm) and a single-leg effective mass of about 0.416 mg, that
gives a contact stiffness of about 1.04 force units per mm -- so ONE SIXTH of body weight (4.07 units)
indents the contact by roughly 3.9 mm, against a seated thorax height of only 2.03 mm.  The fly is
standing on a soft mattress.

This sweeps the contact time constant and reports the indentation and the sag directly.  The stiffness
scales as 1/timeconst^2, so going from 0.02 s to 1e-3 s multiplies it by 400.  MuJoCo requires the time
constant to be longer than about two timesteps; the timestep here is 1e-4 s.
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


def main():
    out = {"sweep": []}
    print(f"{'timeconst':>11}{'k_contact':>12}{'predicted indent':>18}"
          f"{'measured sag':>14}{'thorax z':>11}{'up_z':>9}{'foot in contact':>17}")
    for tc in (0.02, 0.005, 0.002, 0.001, 0.0005):
        m = _load_mjcf(str(XML)).compile()
        for g in range(m.ngeom):
            m.geom_solref[g, 0] = tc
            m.geom_solref[g, 1] = 1.0
        d = mujoco.MjData(m)
        mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
        meff = float(sum(m.body_mass)) / 6.0
        k = meff / (tc ** 2)
        predicted = SIXTH * mw / k
        th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        sags, ups, touches = [], [], 0
        for leg in LEGS:
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.act[:] = 0.0; d.ctrl[:] = 0.0; d.xfrc_applied[:] = 0.0
            mujoco.mj_forward(m, d)
            z0 = float(d.xpos[th][2])
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")
            for _ in range(2000):
                d.xfrc_applied[b, :] = 0.0
                d.xfrc_applied[b, 2] = -SIXTH * mw
                mujoco.mj_step(m, d)
            sag = z0 - float(d.xpos[th][2])
            ups.append(float(d.xmat[th].reshape(3, 3)[:, 2][2]))
            sags.append(sag)
            for c in range(d.ncon):
                cc = d.contact[c]
                if cc.geom1 == fid or cc.geom2 == fid:
                    other = cc.geom2 if cc.geom1 == fid else cc.geom1
                    if other == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                  f"{leg}Tarsus5_geom"):
                        touches += 1
                        break
        row = {"timeconst_s": tc, "k_contact": k, "predicted_indent_mm": predicted,
               "mean_sag_mm": float(np.mean(sags)), "max_sag_mm": float(np.max(sags)),
               "final_thorax_z": float(d.xpos[th][2]), "min_up_z": float(np.min(ups)),
               "legs_keeping_contact": touches,
               "sag_per_leg": {l: float(s) for l, s in zip(LEGS, sags)}}
        out["sweep"].append(row)
        stable = " (UNSTABLE)" if not np.all(np.isfinite(sags)) else ""
        print(f"{tc:>11.4f}{k:>12.2f}{predicted:>18.4f}{np.mean(sags):>14.4f}"
              f"{row['final_thorax_z']:>11.4f}{row['min_up_z']:>9.4f}"
              f"{str(touches) + '/6':>17}{stable}")
    print("\n  target: the sag should be a small fraction of the 2.03 mm seated height, and all six feet")
    print("  should keep contact when loaded.  The time constant must stay above about 2 timesteps")
    print("  (2e-4 s here) or the solver goes unstable.")
    (HERE / "outputs" / "contact_scale.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'contact_scale.json'}")


if __name__ == "__main__":
    main()
