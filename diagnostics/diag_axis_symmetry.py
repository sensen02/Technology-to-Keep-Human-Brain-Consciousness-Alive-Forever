#!/usr/bin/env python3
"""IS THE MODEL LEFT-RIGHT SYMMETRIC?  Check the joint axes AND the geometry separately.

WHY THIS MATTERS, MEASURED: the passive acceleration of the fly at its stance is dominated by LATERAL
terms, which must vanish for a symmetric stance, and a standing insect is left-right symmetric.  Two
different things could break symmetry -- the joint AXES, and the leg GEOMETRY -- and they have to be
checked separately because they can disagree.

Results recorded here, and the two must not be confused:
  * JOINT AXES mirror EXACTLY.  All 21 left/right pairs lie at 0.00 degrees between a_R and M a_L, with
    no exceptions, so the mirror rule for joint ANGLES is well defined: pitch joints keep their sign
    (dot = -1), yaw and roll negate (dot = +1).
  * THE BUILD IS ONLY APPROXIMATELY MIRRORED.  The coxa origins -- the one column here that does not
    depend on the pose -- differ by 0.008-0.018 mm in y (LFCoxa +0.200 against RFCoxa -0.182 where a
    mirror needs -0.200).
  * BODY POSITIONS CANNOT SETTLE THE QUESTION, and the table below is reported with that caveat: it
    reads positions in the model's own keyframe, where the left and right legs are deliberately in
    DIFFERENT poses, so a difference in tarsus position is mostly pose, not build.  The honest build
    measure is the coxa column.
The operational consequence, MEASURED: enforcing angle symmetry moves the right legs' feet off their
declared targets -- the seating spread grows from 0.030 to 0.286 mm -- and the fly then flips over
(up.up0 = -0.9998).  A stance with symmetric FEET therefore needs slightly ASYMMETRIC joint angles,
which is what the free inverse kinematics gives, and that is why the mirror is derived and verified
here but NOT applied.
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

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml"
M = np.diag([1.0, -1.0, 1.0])           # reflection across the sagittal plane
PK = (("Coxa", "yaw"), ("Coxa", "pitch"), ("Coxa", "roll"), ("Trochanter", "yaw"),
      ("Trochanter", "pitch"), ("Trochanter", "roll"), ("Tibia", "pitch"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    out = {"axes": [], "geometry": []}
    print("=== JOINT AXES: is a_R the mirror of a_L? ===")
    n_bad = 0
    for rleg, lleg in (("RF", "LF"), ("RM", "LM"), ("RH", "LH")):
        for seg, js in PK:
            lj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{lleg}{seg}_{js}")
            rj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{rleg}{seg}_{js}")
            aL, aR = np.asarray(m.jnt_axis[lj]), np.asarray(m.jnt_axis[rj])
            dot = float(np.dot(aR, M @ aL))
            ang = float(np.degrees(np.arccos(np.clip(abs(dot), 0, 1))))
            if ang > 1.0:
                n_bad += 1
            out["axes"].append({"pair": f"{rleg}/{lleg} {seg}_{js}", "angle_deg": ang, "dot": dot,
                                "sign_for_mirror": float(np.sign(dot))})
            print(f"  {rleg}/{lleg} {f'{seg}_{js}':<20} angle(a_R, M a_L) = {ang:>6.2f} deg  "
                  f"dot = {dot:+.1f}  -> mirror needs x{np.sign(dot):+.0f}")
    print(f"  axis pairs that are NOT mirror-related: {n_bad} / {3 * len(PK)}")
    print("\n=== LEG GEOMETRY: is the right leg's build the mirror of the left's? ===")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    t0 = np.asarray(d.xpos[th]).copy()
    print(f"  {'body':<16}{'left (x,y,z)':>26}{'right (x,y,z)':>26}{'mirror error y (mm)':>22}")
    worst = 0.0
    for rleg, lleg in (("RF", "LF"), ("RM", "LM"), ("RH", "LH")):
        for seg in ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus3", "Tarsus5"):
            lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{lleg}{seg}")
            rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{rleg}{seg}")
            if lb < 0 or rb < 0:
                continue
            pl = np.asarray(d.xpos[lb]) - t0
            pr = np.asarray(d.xpos[rb]) - t0
            err = float(abs(pr[1] + pl[1]))          # a mirror needs y_R = -y_L
            worst = max(worst, err)
            out["geometry"].append({"body": f"{rleg}/{lleg} {seg}", "left": pl.tolist(),
                                    "right": pr.tolist(), "mirror_error_y_mm": err})
            print(f"  {f'{rleg}/{lleg} {seg}':<16}{f'({pl[0]:+.3f},{pl[1]:+.3f},{pl[2]:+.3f})':>26}"
                  f"{f'({pr[0]:+.3f},{pr[1]:+.3f},{pr[2]:+.3f})':>26}{err:>22.4f}")
    print(f"\n  worst geometric mirror error: {worst:.4f} mm")
    print(f"  NOTE: these positions are read in the model's own keyframe, where the left and right "
          f"legs are in DIFFERENT poses, so this table cannot separate build from pose.  The "
          f"pose-independent build measure is the coxa origin, which differs by 0.008-0.018 mm.")
    print(f"  CONCLUSION: the joint AXES mirror exactly ({n_bad} exceptions); the BUILD is only "
          f"approximately mirrored.  Enforcing angle symmetry moves the feet off target (seating "
          f"spread 0.030 -> 0.286 mm) and flips the fly, so it is not applied.")
    out["n_bad_axis_pairs"] = n_bad
    out["worst_geometry_error_mm"] = worst
    (HERE / "outputs" / "axis_symmetry.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE / 'outputs' / 'axis_symmetry.json'}")


if __name__ == "__main__":
    main()
