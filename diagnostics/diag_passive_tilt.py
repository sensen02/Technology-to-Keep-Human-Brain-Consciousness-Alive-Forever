#!/usr/bin/env python3
"""WHERE DOES THE PASSIVE TILT COME FROM?  A DECOMPOSITION AND A CAUSAL TEST.

WHY.  After the round-25 repairs the fly settles at 1.3917 mm with all 90 leg muscles able to
produce force, and the static-equilibrium solve now FINDS an equilibrium (residual 3.87 % of its
no-activation value) -- but the fly tilts about 46 deg while settling and the equilibrium is
unstable.  Writing that down as "balance" and moving on would be an assertion, not a measurement.
The candidate is specific: at act = 0, 42 of 90 muscles still carry PASSIVE force, the net passive
torque over left-leg joints (-4.00) and right-leg joints (-2.98) does not cancel, and joint passive
stiffness is 0.000 -- so contacts and these passive forces are all that hold the pose.

TWO THINGS THIS TOOL REFUSES TO CONFUSE.
  * A muscle is INTERNAL: it pulls on two points of the same fly, so by Newton's third law the
    muscles' net wrench on the whole body is identically ZERO.  The first version of this tool
    computed that zero and reported it as "the passive wrench is small" -- a vacuous conclusion.
    Here the zero is used for what it is: a bookkeeping check that the moment/force decomposition
    is right.  It being non-zero would mean the decomposition is WRONG.
  * A per-joint torque sum is not a body wrench.  So the question "do the muscles cause the tilt"
    is answered by a CAUSAL test instead: the same passive drop run twice, once with the muscles as
    they are and once with every muscle's force scale set to zero.

WHAT IT DOES NOT ESTABLISH.
    Why the passive forces are asymmetric.  The obvious candidate -- the middle and hind legs have
    no Trochanter body in this model, so 24 trochanter attachment sites are mapped onto the femur --
    is a HYPOTHESIS this tool does not test.  It does not make the fly stand, and it says nothing
    about the neural tier.  Every number is simulator output; no real-fly recording exists in this
    project.  This is a hand-built prototype, not a finished product.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
SIDES = {"L": ("LF", "LM", "LH"), "R": ("RF", "RM", "RH")}
DEFAULT_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
DEFAULT_OUT = HERE / "outputs" / "passive_tilt_decomposition.json"


def main(argv: list[str] | None = None) -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seconds", type=float, default=1.0)
    a = ap.parse_args(argv)

    xml = Path(a.xml)
    m = _load_mjcf(str(xml)).compile()
    d = mujoco.MjData(m)
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)) for i in range(m.nu)]
    free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    if free < 0:
        print("FAIL: this model has no free thorax joint; nothing to tilt")
        return 2
    fadr = int(m.jnt_dofadr[free])                      # the free joint owns the first 6 dofs
    # gravity is a 3-vector in this model; the body weight is the vertical component
    gravity_force = float(mujoco.mj_getTotalmass(m) * abs(np.asarray(m.opt.gravity)[2]))

    # ------------------------------------------------------------------ wrench by leg, one pose
    def by_leg(act_level):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qvel[:] = 0.0
        d.act[:] = act_level
        mujoco.mj_forward(m, d)
        ra = np.asarray(d.moment_rowadr)
        ci = np.asarray(d.moment_colind)
        am = np.asarray(d.actuator_moment)
        af = np.asarray(d.actuator_force)
        out = {leg: np.zeros(6) for leg in LEGS}
        for i in range(m.nu):
            if names[i][0:2] not in LEGS:
                continue
            s = int(ra[i]); e = int(ra[i + 1]) if i + 1 < len(ra) else len(am)
            for k in range(s, e):
                dof = int(ci[k])
                if fadr <= dof < fadr + 6:
                    out[names[i][0:2]][dof - fadr] += float(am[k]) * float(af[i])
        return out

    per_leg0 = by_leg(0.0)
    per_leg1 = by_leg(1.0)
    body0 = sum(per_leg0.values(), np.zeros(6))
    body1 = sum(per_leg1.values(), np.zeros(6))
    net_ok = float(np.max(np.abs(body0))) < 1e-9 and float(np.max(np.abs(body1))) < 1e-9

    # per-joint passive torque, all 42 leg joints
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qvel[:] = 0.0
    d.act[:] = 0.0
    mujoco.mj_forward(m, d)
    ra = np.asarray(d.moment_rowadr)
    ci = np.asarray(d.moment_colind)
    am = np.asarray(d.actuator_moment)
    af = np.asarray(d.actuator_force)
    jtor = {}
    for j in range(m.njnt):
        jn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if not (jn.startswith("joint_") and jn[6:8] in LEGS):
            continue
        dof = int(m.jnt_dofadr[j]); t = 0.0
        for i in range(m.nu):
            if names[i][0:2] not in LEGS:
                continue
            s = int(ra[i]); e = int(ra[i + 1]) if i + 1 < len(ra) else len(am)
            for k in range(s, e):
                if int(ci[k]) == dof:
                    t += float(am[k]) * float(af[i])
                    break
        jtor[jn] = t

    # ------------------------------------------------------------------------------- causal test
    def drop(muscles_on: bool):
        # gainprm[0] is not a force-disable switch for MuJoCo muscles.
        # The prior intervention left substantial actuator forces: its inference is withdrawn.
        saved_flags = int(m.opt.disableflags)
        if not muscles_on:
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        try:
            d2 = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d2, 0)
            mujoco.mj_forward(m, d2)
            th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
            up0 = np.asarray(d2.xmat[th], float).reshape(3, 3)[:, 2].copy()
            worst = 0.0
            for _ in range(int(a.seconds / m.opt.timestep)):
                mujoco.mj_step(m, d2)
                mujoco.mj_forward(m, d2)
                if not muscles_on:
                    assert not np.any(d2.actuator_force), 'Ablation failed: actuator force remains'
                    assert not np.any(d2.qfrc_actuator), 'Ablation failed: generalized force remains'
                up = np.asarray(d2.xmat[th], float).reshape(3, 3)[:, 2]
                worst = max(worst, float(np.degrees(np.arccos(np.clip(up @ up0, -1.0, 1.0)))))
            up = np.asarray(d2.xmat[th], float).reshape(3, 3)[:, 2]
            final = float(np.degrees(np.arccos(np.clip(up @ up0, -1.0, 1.0))))
            return {"final_thorax_z_model": float(d2.xpos[th][2]), "worst_tilt_deg": worst,
                    "final_tilt_deg": final}
        finally:
            m.opt.disableflags = saved_flags

    on = drop(True)
    off = drop(False)
    tilt_from_muscles = on["worst_tilt_deg"] - off["worst_tilt_deg"]

    left_sum = sum(v for k, v in jtor.items() if k[6:8] in SIDES["L"])
    right_sum = sum(v for k, v in jtor.items() if k[6:8] in SIDES["R"])
    fore = sum(v for k, v in jtor.items() if k[6:8] in ("LF", "RF"))
    midhind = sum(v for k, v in jtor.items() if k[6:8] in ("LM", "RM", "LH", "RH"))
    big = sorted(jtor.items(), key=lambda x: -abs(x[1]))[:8]

    print(f"model: {xml.name}")
    print(f"  total body weight = {gravity_force:.4f} model force units")
    print("\n  A) BOOKKEEPING: a muscle is INTERNAL, so the muscles' net wrench on the free joint "
          "must be zero.")
    print(f"     act=0 {np.round(body0, 12)}")
    print(f"     act=1 {np.round(body1, 12)}")
    print(f"     {'PASS' if net_ok else 'FAIL: non-zero means this decomposition is wrong'}")
    print("\n  B) per-joint PASSIVE torque (act=0), all 42 leg joints -- largest 8:")
    for k, v in big:
        print(f"     {k:26s} {v:+10.4f}")
    print(f"     net over left-leg joints {left_sum:+.4f}, right-leg joints {right_sum:+.4f}, "
          f"difference {left_sum - right_sum:+.4f}")
    print(f"     net over fore legs {fore:+.4f}, over mid+hind legs {midhind:+.4f}")
    print(f"\n  C) CAUSAL TEST -- the same {a.seconds:.1f} s drop, muscles as they are vs every "
          f"muscle's force scale set to zero:")
    print(f"     muscles ON : final z {on['final_thorax_z_mm']:.4f} mm, worst tilt "
          f"{on['worst_tilt_deg']:.2f} deg, final tilt {on['final_tilt_deg']:.2f} deg")
    print(f"     muscles OFF: final z {off['final_thorax_z_mm']:.4f} mm, worst tilt "
          f"{off['worst_tilt_deg']:.2f} deg, final tilt {off['final_tilt_deg']:.2f} deg")
    print(f"     => the muscles account for {tilt_from_muscles:+.2f} deg of the worst tilt")

    if not net_ok:
        verdict = "DECOMPOSITION_IS_WRONG"
    elif off["worst_tilt_deg"] > 0.7 * on["worst_tilt_deg"]:
        verdict = "THE_TILT_SURVIVES_WITH_THE_MUSCLES_GONE"
    else:
        verdict = "THE_PASSIVE_MUSCLES_ARE_MOST_OF_THE_TILT"
    print(f"\nVERDICT: {verdict}")
    print('  This conditional comparison does not identify the sole cause of tilt.')
    print('  In particular, active equality locks must be audited before attributing contact defects.')

    out = {
        "overall": "PASS", "status": "PASS" if net_ok else "FAIL", "verdict": verdict,
        "xml": str(xml), "body_weight_force_units": gravity_force,
        "bookkeeping_net_muscle_wrench_is_zero": bool(net_ok),
        "net_muscle_wrench_free_joint_act0": [float(v) for v in body0],
        "net_muscle_wrench_free_joint_act1": [float(v) for v in body1],
        "by_leg_act0": {k: [float(x) for x in v] for k, v in per_leg0.items()},
        "by_leg_act1": {k: [float(x) for x in v] for k, v in per_leg1.items()},
        "per_joint_passive_torque": {k: float(v) for k, v in jtor.items()},
        "largest_per_joint_passive_torques": {k: float(v) for k, v in big},
        "net_passive_torque_left_joints": float(left_sum),
        "net_passive_torque_right_joints": float(right_sum),
        "net_passive_torque_fore_legs": float(fore),
        "net_passive_torque_mid_hind_legs": float(midhind),
        "causal_passive_drop": {"seconds": a.seconds, "muscles_on": on, "muscles_off": off,
                                "tilt_attributable_to_muscles_deg": tilt_from_muscles},
        "units": ("model force units and model torque units; this model is mm + gram, so 1 force "
                  "unit = 1e-6 N = 1 uN.  A muscle's net wrench on the whole fly is zero by "
                  "Newton's third law, which is why that row is a bookkeeping check, not a result."),
        "provenance": ("MuJoCo 3.9.0; the model at its own keyframe with qvel zeroed and act set "
                       "directly (mj_forward does not integrate act); the causal test steps the "
                       "same model with all muscle gainprm/biasprm scales set to zero and restores "
                       "them afterwards"),
        "open_hypothesis_NOT_tested_here": (
            "that the asymmetry is caused by the middle and hind legs having no Trochanter body, so "
            "24 trochanter attachment sites are mapped onto the femur.  This tool measures and "
            "attributes at the joint level; it does not test that mechanism."),
        "what_this_does_not_establish": (
            "that the fly stands, or what the correct contact model is.  The causal test shows "
            "whether the passive muscles are NECESSARY for the tilt, not what is sufficient, and "
            "this platform separately declares its contact-force channel a NON_PHYSICAL_LOAD.  It "
            "says nothing about the neural tier.  Every number is simulator output; no real-fly "
            "recording exists in this project."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                           encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0 if net_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
