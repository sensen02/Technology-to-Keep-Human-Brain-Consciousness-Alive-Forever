#!/usr/bin/env python3
"""CLEAR SHOTS OF THE STANDING STANCE -- tendons hidden, wider camera.

The first attempt was unusable for verification: the camera sat 5.2 mm away and the 90 tendons were
drawn as blue lines, so the render was a knot of blue over the body and the legs could not be seen at
all.  Verification needs the legs, so the tendon/muscle visualisation is switched off here and the
camera pulled back.
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

XML = Path(os.environ.get("STANDING_XML",
                          HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"))
SHOTS = HERE / "outputs" / "standing_shots"
SHOTS.mkdir(parents=True, exist_ok=True)
DUR = float(os.environ.get("SHOT_DUR", "0.0"))
TAG = os.environ.get("SHOT_TAG", "stance")


def main():
    m = _load_mjcf(str(XML)).compile()
    # optional plant calibration for the shot, so the rendered state is the one being reported
    _mass = float(os.environ.get("SHOT_MASS_MG", "0"))
    if _mass:
        _f = _mass / (float(sum(m.body_mass)) * 1000.0)
        for _b in range(m.nbody):
            m.body_mass[_b] *= _f
            m.body_inertia[_b] *= _f
    _st = float(os.environ.get("SHOT_STIFF", "0"))
    if _st:
        for _j in range(m.njnt):
            _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _j) or ""
            if _nm.startswith("joint_"):
                m.jnt_stiffness[_j] = _st
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    for _ in range(int(DUR / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"rendering at t={DUR}s: thorax z={d.xpos[th][2]:.4f} mm")
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for flag in ("mjVIS_TENDON", "mjVIS_ACTUATOR", "mjVIS_ACTIVATION", "mjVIS_JOINT",
                 "mjVIS_CONTACTPOINT", "mjVIS_CONTACTFORCE", "mjVIS_COM", "mjVIS_CAMERA",
                 "mjVIS_LIGHT", "mjVIS_SELECT"):
        try:
            opt.flags[getattr(mujoco.mjtVisFlag, flag)] = 0
        except Exception:
            pass
    r = mujoco.Renderer(m, 640, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [-0.4, 0.0, 0.7]
    views = [("wide_persp", 135.0, -25.0, 8.0), ("wide_side", 90.0, -12.0, 8.0),
             ("wide_front", 180.0, -12.0, 8.0), ("wide_top", 135.0, -75.0, 8.0),
             ("close_legs", 135.0, -12.0, 4.5)]
    from PIL import Image
    for name, az, el, dist in views:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam, opt)
        Image.fromarray(r.render()).save(SHOTS / f"{TAG}_{name}.png")
        print(f"wrote {SHOTS / f'{TAG}_{name}.png'}")
    r.close()


if __name__ == "__main__":
    main()
