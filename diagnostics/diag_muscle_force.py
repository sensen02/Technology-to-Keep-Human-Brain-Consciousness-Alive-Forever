#!/usr/bin/env python3
"""DO THE MUSCLES ACTUALLY PULL?  Measure force at full activation against one body weight.

MEASURED RESULT THAT PROMPTED THIS: activating each of the 90 muscles at level 0.3, one at a time,
raised the thorax by NO measurable amount in every single case -- so before blaming the wiring or the
stance, check the obvious: how much force does a muscle actually produce, and is it operating inside
the length range its own parameters declare?
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    print(f"one body weight = {mw:.4f} force units (uN-equivalent)")
    # settle first, so the muscle lengths are the loaded ones
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"\nsettled thorax z = {d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Thorax')][2]:.4f} mm")
    print(f"{'actuator':<44}{'L':>9}{'Lrange':>17}{'in?':>5}{'F@ctrl=1':>11}{'F/bw':>9}")
    rows = []
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        d.ctrl[:] = 0.0
        d.ctrl[i] = 1.0
        # step a little so the activation dynamics catch up, then read the force
        for _ in range(200):
            mujoco.mj_step(m, d)
        F = float(abs(d.actuator_force[i]))
        L = float(d.actuator_length[i])
        lo, hi = (float(x) for x in m.actuator_lengthrange[i]) if m.actuator_lengthrange.any() else (
            float("nan"), float("nan"))
        ins = (lo <= L <= hi) if np.isfinite(lo) else False
        rows.append((F, nm, L, lo, hi, ins))
        if i < 12 or i % 15 == 0:
            print(f"{nm:<44}{L:>9.4f}{f'[{lo:.3f},{hi:.3f}]':>17}{str(ins):>5}{F:>11.4f}{F/mw:>9.5f}")
    rows.sort(reverse=True)
    F = np.array([r[0] for r in rows])
    print(f"\nforces at full activation: max {F.max():.4f} ({F.max()/mw:.5f} bw), "
          f"median {np.median(F):.4f} ({np.median(F)/mw:.6f} bw), min {F.min():.4f}")
    print(f"  actuators producing more than 1% of body weight: {(F > 0.01*mw).sum()} / {m.nu}")
    print(f"  actuators producing more than 10% of body weight: {(F > 0.10*mw).sum()} / {m.nu}")
    print(f"  actuators operating inside their declared lengthrange: {sum(1 for r in rows if r[5])} / {m.nu}")
    print(f"\nstrongest five: " + ", ".join(f"{r[1]}={r[0]/mw:.4f}bw" for r in rows[:5]))
    print("  (for reference, holding the fly up needs about 1.0 body weight in total, i.e. roughly "
          "0.17 bw per leg)")


if __name__ == "__main__":
    main()
