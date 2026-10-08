#!/usr/bin/env python3
"""CAN EACH LEG CARRY ONE SIXTH OF THE WEIGHT?  Load each foot directly and measure the deflection.

The previous measurement (how much the load share changes when the thorax is nudged) mixes in the
contact and the other legs.  This asks the question straight: apply a KNOWN vertical force at ONE leg's
tarsus, with zero activation, and see whether the body holds or that leg folds.  Nothing here computes
or imposes any activation -- it is a load test on the plant, like pressing on a table leg.

If a leg's stiffness under load is near zero, the leg is not a strut and no amount of motor-neuron
drive through THAT leg can hold the body up.
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
    m = _load_mjcf(str(XML)).compile()
    # THE LEG-JOINT STIFFNESS IS THE PARAMETER UNDER TEST.  MEASURED with the shipped 0.4: pushing one
    # sixth of body weight onto a single foot sags the fly by 1.34-1.84 mm, i.e. the leg gives way
    # entirely (the seated thorax height is only 2.08 mm).  This sweep finds the value at which a leg
    # can actually carry its share, WITHOUT the fly being able to stand with zero activation.
    _stiff = float(os.environ.get("LOAD_STIFF", "0.4"))
    for _j in range(m.njnt):
        _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _j) or ""
        if _nm.startswith("joint_"):
            m.jnt_stiffness[_j] = _stiff
    # MASS: THE MODEL IS 2.44x TOO HEAVY, AND THAT SCALES EVERY LOAD IN THIS PROJECT.
    # MEASURED: the total is 2.49427 mg, but a real Drosophila is about 1.0 mg (and FlyGym's own
    # walking flybody model is 1.02431 mg).  The mass here comes from the OpenSim conversion computing
    # mesh volumes at density 1000, so it is a model artefact rather than a measured animal.  Every
    # sag and every joint torque is proportional to it, so correcting it is a calibration of the plant
    # with a real biological basis -- not a fudge to make an answer come out.
    MASS_MG = float(os.environ.get("MASS_MG", "0")) or None
    if MASS_MG:
        cur = float(sum(m.body_mass)) * 1000.0
        f = MASS_MG / cur
        for b in range(m.nbody):
            m.body_mass[b] *= f
            m.body_inertia[b] *= f
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    print(f"total mass = {1000*float(sum(m.body_mass)):.5f} mg; one body weight = {mw:.4f}; "
          f"one sixth = {SIXTH*mw:.4f} force units")
    print(f"the model's own leg-joint stiffness = {m.jnt_stiffness[1]:.3f} uN*mm/rad")

    def rest(dur=0.0):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.act[:] = 0.0
        d.ctrl[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(int(dur / m.opt.timestep)):
            mujoco.mj_step(m, d)

    def body_of(leg):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}Tarsus5")

    rest()
    z0 = float(d.xpos[th][2])
    print(f"\nrest thorax z = {z0:.4f} mm")
    print(f"\n=== push DOWN on ONE foot at a time, by one sixth of body weight, then hold 200 ms ===")
    print(f"  {'leg':<5}{'thorax z after':>17}{'sag (mm)':>11}{'up_z':>9}"
          f"{'that leg still in contact?':>30}")
    out = {"mw": mw, "stiffness": float(m.jnt_stiffness[1]), "rest_thorax_z": z0, "legs": []}
    for leg in LEGS:
        rest()
        b = body_of(leg)
        for _ in range(2000):                      # 200 ms with the load applied to that foot
            d.xfrc_applied[b, :] = 0.0
            d.xfrc_applied[b, 2] = -SIXTH * mw     # downward force at that tarsus
            mujoco.mj_step(m, d)
        z = float(d.xpos[th][2])
        up = d.xmat[th].reshape(3, 3)[:, 2]
        touched = False
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 != fid and cc.geom2 != fid:
                continue
            other = cc.geom2 if cc.geom1 == fid else cc.geom1
            if other in (b,):
                touched = True
        out["legs"].append({"leg": leg, "thorax_z": z, "sag_mm": z0 - z,
                            "up_z": float(up[2]), "still_in_contact": bool(touched)})
        print(f"  {leg:<5}{z:>17.4f}{z0 - z:>11.4f}{up[2]:>9.4f}"
              f"{('yes' if touched else 'NO -- the foot left the floor'):>30}")
    sags = [r["sag_mm"] for r in out["legs"]]
    print(f"\n  sag per leg: " + ", ".join(f"{r['leg']} {r['sag_mm']:.4f}" for r in out["legs"]))
    print(f"  spread between the most and least yielding leg: {max(sags)-min(sags):.4f} mm")
    print(f"  MEAN SAG AT STIFFNESS {m.jnt_stiffness[1]:.2f}: {np.mean(sags):.4f} mm")
    print(f"  for reference, the seated thorax height is {z0:.4f} mm, so a sag of that size means the "
          f"leg gave way entirely")
    (HERE / "outputs" / "leg_load_test.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'leg_load_test.json'}")


if __name__ == "__main__":
    main()
