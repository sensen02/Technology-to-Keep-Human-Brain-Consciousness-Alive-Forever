#!/usr/bin/env python3
"""CAN A STEADY, EXTENSOR-BIASED MUSCLE PATTERN HOLD THE FLY UP?  (a probe, not behaviour)

The acceleration probe says the extensors push the body up (trochanter extensors +1620 mm/s^2, tibia
extensors +1329, while the trochanter flexors push it down by -1474).  Netting +0.3 g of upward
acceleration is not the same as standing, so this holds each candidate pattern for 3 seconds and asks
what actually happens.

STANDS means: at the end, all six legs are down, no non-leg body part touches the floor, and the
thorax is at least 1.2 mm up (the solved stance is 1.72 mm and the collapsed height is 0.65 mm).

THIS IS A PROBE.  These activations are written by hand and are NOT proposed as behaviour; they answer
whether the target the connectome has to reach is reachable at all.
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
from muscle_wiring import actuator_to_mn_type  # noqa: E402

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
DUR = float(os.environ.get("HOLD_DUR", "3.0"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    a2t = actuator_to_mn_type(verbose=False)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    groups = {}
    for i, nm in enumerate(names):
        t = a2t.get(nm)
        key = t[1] if isinstance(t, tuple) else "UNMATCHED"
        groups.setdefault(key, []).append(i)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")

    def hold_idx(idx, level, dur=DUR, preactivate=False, tag=""):
        # PRE-ACTIVATION MATTERS, AND IT IS NOT A TRICK.  MEASURED: with the muscles switched on at
        # the same instant the run starts, even the strongest extensor pattern fails to hold the fly
        # (thorax 0.655 mm, body still on the floor, at every level from 0.05 to 1.0).  The reason is
        # a race: MuJoCo's muscle activation rises through a first-order filter with a 10-40 ms time
        # constant, while the passive stance sags to the floor in about 100 ms, and once the legs have
        # folded, extending the tibia pushes the tarsus sideways along the floor instead of lifting --
        # measured separately as 0 of 90 muscles being able to raise the collapsed body.
        # An animal is never in that situation: its muscles are already active when it bears load.  So
        # the meaningful question is whether a steady pattern MAINTAINS the stance, which is tested by
        # setting the activation state directly before the first step.
        idx = sorted(set(idx))
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        if preactivate:
            d.act[:] = 0.0
            for i in idx:
                d.act[i] = level
            mujoco.mj_forward(m, d)
        up0 = d.xmat[th].reshape(3, 3)[:, 2].copy()
        for _ in range(int(dur / m.opt.timestep)):
            d.ctrl[:] = 0.0
            for i in idx:
                d.ctrl[i] = level
            mujoco.mj_step(m, d)
        up = d.xmat[th].reshape(3, 3)[:, 2]
        leg = body = 0.0; legs = set(); others = set()
        for c in range(d.ncon):
            cc = d.contact[c]
            if cc.geom1 != fid and cc.geom2 != fid:
                continue
            other = cc.geom2 if cc.geom1 == fid else cc.geom1
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])) or ""
            f = np.zeros(6); mujoco.mj_contactForce(m, d, c, f)
            if b[:2] in LEGS:
                leg += abs(f[0]); legs.add(b[:2])
            else:
                body += abs(f[0]); others.add(b)
        return {"tag": tag, "level": level, "n_actuators": len(idx),
                "thorax_z": float(d.xpos[th][2]),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))),
                "legs_down": len(legs), "leg_bw": leg / mw, "body_bw": body / mw,
                "nonleg": sorted(others),
                "stood": bool(body < 1e-3 and len(legs) == 6 and d.xpos[th][2] > 1.2)}

    # BUILD THE PATTERN FROM THE MEASURED LIFT DIRECTIONS, NOT FROM A NAMED SET.
    # MEASURED, and it is why naming the extensors failed: with the coxa springs at 0.4 (essentially
    # free), a torque applied at the TIBIA is transmitted up the chain and simply swings the whole leg
    # about the coxa -- the leg rotates instead of the body rising, and the extensor pattern holds the
    # thorax at 0.655 mm from every level 0.05 to 1.0, exactly as if nothing were active.  A serial
    # chain with a compliant base cannot be lifted by actuating one joint of it; every joint in the
    # chain has to be held.  So the pattern is the set of muscles MEASURED to push the body upward
    # (positive vertical acceleration), which spans the coxa, trochanter and tibia.
    import json as _json
    lift = _json.loads((HERE / "outputs" / "lift_direction.json").read_text())
    ranked = [r["actuator"] for r in lift["per_actuator"] if r["dz_a"] > 0]
    idx_of = {nm: i for i, nm in enumerate(names)}
    print(f"muscles measured to push the body up: {len(ranked)} of {m.nu}")
    rows = []
    print(f"\n{'pattern':<34}{'level':>7}{'thr z':>9}{'tilt':>8}{'legs':>6}{'body bw':>10}  verdict")
    for n in (len(ranked), 24, 16, 8):
        idx = [idx_of[nm] for nm in ranked[:n]]
        for lvl in (0.1, 0.25, 0.5, 1.0):
            r = hold_idx(idx, lvl, preactivate=True, tag=f"top {n} lifters [pre-act]")
            rows.append(r)
            print(f"{f'top {n} lifters (of {len(ranked)})':<34}{lvl:>7.2f}{r['thorax_z']:>9.4f}"
                  f"{r['tilt_deg']:>8.1f}{r['legs_down']:>6}{r['body_bw']:>10.4f}  "
                  f"{'*** STANDS ***' if r['stood'] else 'collapsed'}")
    # and, for contrast, the same pattern WITHOUT pre-activation
    idx = [idx_of[nm] for nm in ranked]
    for lvl in (0.5,):
        r = hold_idx(idx, lvl, preactivate=False, tag="top lifters [cold start]")
        rows.append(r)
        print(f"{'all lifters (cold start)':<34}{lvl:>7.2f}{r['thorax_z']:>9.4f}{r['tilt_deg']:>8.1f}"
              f"{r['legs_down']:>6}{r['body_bw']:>10.4f}  {'*** STANDS ***' if r['stood'] else 'collapsed'}")
    winners = [r for r in rows if r["stood"]]
    print(f"\n=== VERDICT ===")
    if winners:
        best = max(winners, key=lambda r: r["thorax_z"])
        print(f"  YES -- a steady anti-gravity pattern holds the fly up: {best['tag']} at level "
              f"{best['level']} gives thorax z {best['thorax_z']:.4f} mm, tilt {best['tilt_deg']:.1f} "
              f"deg, {best['legs_down']}/6 legs, no body contact")
    else:
        print("  NO -- no steady pattern tested holds the fly up; the best leaves the body on the "
              "floor.  Standing would then require FEEDBACK, not tone.")
    (HERE / "outputs" / "tone_hold.json").write_text(json.dumps(
        {"rows": rows, "mw": mw, "duration_s": DUR,
         "any_stood": bool(winners)}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'tone_hold.json'}")


if __name__ == "__main__":
    main()
