#!/usr/bin/env python3
"""LOOK AT THE STANDING FLY, AND NAME EVERY CONTACT IT RESTS ON.

The number "1.000 body weight through the legs" is not enough: the fly could be supported by its
tarsi, or its belly could be on the floor with the legs merely touching.  This separates the two by
naming every contact, and renders the settled pose from three angles.
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
SHOTS = HERE / "outputs" / "standing_shots"
SHOTS.mkdir(parents=True, exist_ok=True)
DUR = float(os.environ.get("STAND_DUR", "2.0"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
    leg_stiff = None
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm == "joint_LFCoxa_pitch":
            leg_stiff = float(m.jnt_stiffness[j]); leg_damp = float(m.dof_damping[m.jnt_dofadr[j]])
            leg_range = m.jnt_range[j]
    print(f"leg joint armature/stiffness: stiffness={leg_stiff} damping={leg_damp} "
          f"(one body weight = {mw:.4f} force units)")
    # a rough stiffness scale: how much thorax drop does the leg-spring set buy?
    print(f"  free-joint dofs: {[ (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '') for j in range(7)]}")

    for i in range(int(DUR / m.opt.timestep)):
        mujoco.mj_step(m, d)
    up = d.xmat[th].reshape(3, 3)[:, 2]
    print(f"\nsettled after {DUR}s: thorax z={d.xpos[th][2]:.4f} mm  "
          f"tilt={np.degrees(np.arccos(np.clip(up @ up0, -1, 1))):.1f} deg")
    print(f"{'geom1':<22}{'geom2':<22}{'dist mm':>10}{'|Fn|':>10}{'bw':>9}")
    per_body, legs, other = {}, set(), []
    for c in range(d.ncon):
        cc = d.contact[c]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, cc.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, cc.geom2) or ""
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[cc.geom1])) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[cc.geom2])) or ""
        f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
        fn = abs(f[0])
        print(f"{g1:<22}{g2:<22}{cc.dist:>10.4f}{fn:>10.3f}{fn/mw:>9.4f}")
        for a, b in ((b1, b2), (b2, b1)):
            if b == "world":
                per_body[a] = per_body.get(a, 0.0) + fn
                if a[:2] in ("LF", "RF", "LM", "RM", "LH", "RH"):
                    legs.add(a[:2])
                else:
                    other.append(a)
    print(f"\nsupport by body: " + ", ".join(f"{k}={v/mw:.4f}bw" for k, v in sorted(per_body.items())))
    print(f"legs carrying load: {sorted(legs)}  ({len(legs)}/6)")
    nonleg_bw = sum(v for k, v in per_body.items()
                    if k[:2] not in ("LF", "RF", "LM", "RM", "LH", "RH")) / mw
    leg_bw = sum(v for k, v in per_body.items()
                 if k[:2] in ("LF", "RF", "LM", "RM", "LH", "RH")) / mw
    print(f"\nSUPPORT SPLIT: tarsi {leg_bw/max(leg_bw+nonleg_bw,1e-9)*100:.1f}% of the load, "
          f"BODY ON THE FLOOR {nonleg_bw/max(leg_bw+nonleg_bw,1e-9)*100:.1f}% "
          f"({leg_bw:.4f} bw vs {nonleg_bw:.4f} bw)")
    if other:
        print(f"*** NON-LEG BODIES ON THE FLOOR: {sorted(set(other))} -- this fly is lying down, "
              f"not standing ***")
    else:
        print("no non-leg body touches the floor -- the legs alone carry the fly")
    print(f"thorax height above floor = {d.xpos[th][2]:.4f} mm; body length is about 2.87 mm")

    # ---- centre of mass against the support polygon of the tarsus contacts
    com = np.asarray(d.subtree_com[0]).copy()
    pts = []
    for c in range(d.ncon):
        cc = d.contact[c]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[cc.geom1])) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[cc.geom2])) or ""
        if b2 == "world" and b1[:2] in ("LF", "RF", "LM", "RM", "LH", "RH"):
            pts.append(np.asarray(cc.pos)[:2])
    if len(pts) >= 3:
        P = np.array(pts)
        from scipy.spatial import ConvexHull
        try:
            h = ConvexHull(P)
            # is the COM's projection inside?
            inside = True
            for eq in h.equations:
                if eq[0] * com[0] + eq[1] * com[1] + eq[2] > 1e-9:
                    inside = False
            print(f"COM projection {np.round(com[:2],4)} vs support polygon of {len(P)} contact points "
                  f"(area {h.volume:.4f} mm^2): {'INSIDE' if inside else 'OUTSIDE <-- would topple'}")
        except Exception as e:
            print(f"convex hull failed: {e}")
    print(f"COM = {np.round(com,4)} mm  (z above floor {com[2]:.4f})")

    # ---- render
    r = mujoco.Renderer(m, 640, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.6]
    views = [("perspective", 130.0, -22.0, 5.2), ("front", 180.0, -8.0, 4.6),
             ("side", 90.0, -8.0, 4.6), ("top", 130.0, -80.0, 5.2)]
    for name, az, el, dist in views:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam)
        img = r.render()
        from PIL import Image
        p = SHOTS / f"standing_{name}.png"
        Image.fromarray(img).save(p)
        print(f"wrote {p}")
    r.close()


if __name__ == "__main__":
    main()
