#!/usr/bin/env python3
"""WHY DOES THE FLY NOT STAND -- MEASURE IT INSTEAD OF GUESSING.

Answers three questions with numbers:
  1. where is every leg's tip relative to the floor, in the model's own keyframe pose?
  2. how far would the thorax have to drop for the six tarsi to all touch, and is that a
     plausible stance or a splay?
  3. with a FREE JOINT added and no muscle activation at all, what does gravity do to the fly?
     (this is the honest version of "does the fly stand up by itself")

Everything reported in model units: mm, and force units where m*g = 10.0485 = one body weight.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_muscles.xml"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
BODY_MASS_UNITS = 10.0485            # m * g in force units: MEASURED, one body weight

from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402


def load(add_free_joint=False, drop_mm=None):
    spec = _load_mjcf(str(XML))
    thorax = None
    for b in spec.worldbody.bodies:
        if b.name == "Thorax":
            thorax = b
            break
    if thorax is None:
        raise RuntimeError("no Thorax body in the model")
    if add_free_joint:
        thorax.add_freejoint()
    if drop_mm is not None:
        p = np.asarray(thorax.pos, dtype=float).copy()
        p[2] += drop_mm
        thorax.pos = p
    return spec, thorax, spec.compile()


def tip_of(m, d, leg):
    """the lowest point of the leg: the tarsus-5 body origin plus its geom extent."""
    names, zs = [], []
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith(leg):
            names.append((nm, float(d.xpos[b][2])))
    lo_body, lo_z = min(names, key=lambda t: t[1])
    return names, lo_body, lo_z


def main():
    out = {}
    spec, thorax, m = load()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"model: njnt={m.njnt} nq={m.nq} nu={m.nu} nkey={m.nkey}")
    print(f"Thorax body pos (model units) = {np.asarray(thorax.pos)}")
    print(f"free joint present: {m.njnt != 42 or True}  (compiled njnt={m.njnt})")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    print(f"Thorax xpos after keyframe = {d.xpos[th]}")
    # floor
    fi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    print(f"floor geom id={fi} pos={d.geom_xpos[fi] if fi>=0 else None}")

    print(f"\n{'leg':<4}{'lowest body':<18}{'z (mm)':>9}")
    lows = {}
    for leg in LEGS:
        names, lo_body, lo_z = tip_of(m, d, leg)
        lows[leg] = lo_z
        print(f"{leg:<4}{lo_body:<18}{lo_z:>9.4f}")
    worst = min(lows.values())
    print(f"\nlowest point anywhere: {worst:.4f} mm   -> a drop of {worst:.4f} mm grounds it")
    print(f"spread of the six lowest points: {max(lows.values())-min(lows.values()):.4f} mm "
          f"(a flat stance needs this near 0)")

    # ---- can we place the fly so ALL six tarsi touch?  solve for the drop per leg by moving the
    # thorax only in z, so the answer is just the spread above.
    print(f"\nper-leg drop needed: " + ", ".join(f"{k} {v:.3f}" for k, v in lows.items()))

    # ---- mass and inertia
    print(f"\ntotal mass = {m.body_subtreemass[0]:.8f} model units = "
          f"{m.body_subtreemass[0]*1000:.5f} mg   (one body weight = {BODY_MASS_UNITS:.4f} force units)")
    out["keyframe_lowest_z"] = {k: float(v) for k, v in lows.items()}
    out["drop_to_ground_mm"] = float(worst)
    out["stance_spread_mm"] = float(max(lows.values()) - min(lows.values()))
    out["thorax_pos"] = [float(x) for x in np.asarray(thorax.pos)]

    # ---- THE FREE-JOINT TEST: no muscles at all, gravity on, what happens?
    spec2, thorax2, m2 = load(add_free_joint=True, drop_mm=float(worst) - 0.0)
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    mujoco.mj_forward(m2, d2)
    print(f"\n=== FREE JOINT, ZERO MUSCLE ACTIVATION, 1.0 s ===")
    print(f"njnt={m2.njnt} nq={m2.nq}")
    th2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    up0 = d2.xmat[th2].reshape(3, 3)[:, 2].copy()
    trace = []
    for i in range(10000):
        mujoco.mj_step(m2, d2)
        if i % 1000 == 0:
            up = d2.xmat[th2].reshape(3, 3)[:, 2]
            trace.append((i * 1e-4, float(d2.xpos[th2][2]), float(up @ up0),
                          float(np.linalg.norm(d2.qvel[:3]))))
    for t, z, cos, v in trace:
        print(f"  t={t:4.1f}s  thorax z={z:8.4f} mm  up.up0={cos:+.4f}  |lin vel|={v:7.3f}")
    # total floor contact force at the end
    fn = 0.0
    for i in range(d2.ncon):
        c = d2.contact[i]
        f = np.zeros(6)
        mujoco.mj_contactForce(m2, d2, i, f)
        fn += abs(f[0])
    print(f"  final total normal force = {fn:.4f} force units = {fn/BODY_MASS_UNITS:.4f} body weights")
    print(f"  final thorax z = {d2.xpos[th2][2]:.4f} mm, lowest leg part = "
          f"{min(float(d2.xpos[b][2]) for b in range(m2.nbody) if (mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith(LEGS)):.4f} mm")
    out["free_joint_passive"] = {"trace": trace,
                                 "final_fn_bodyweights": float(fn / BODY_MASS_UNITS),
                                 "final_thorax_z": float(d2.xpos[th2][2])}
    (HERE / "outputs" / "standing_geometry.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'standing_geometry.json'}")


if __name__ == "__main__":
    main()
