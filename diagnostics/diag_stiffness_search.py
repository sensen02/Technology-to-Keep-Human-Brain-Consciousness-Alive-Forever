#!/usr/bin/env python3
"""FIND A LOAD-BEARING SKELETON, SUBJECT TO A HARD CONSTRAINT.

WHERE THIS COMES FROM.  Four independent measurements say the fly cannot stand in the shipped model:
no passive stiffness or damping stands it, 0 of 90 muscles can lift the collapsed body, no steady
hand-set pattern maintains the stance, and the connectome loop presses it down at every gain and
reference rate.  The root cause is mechanical -- the passive joint stiffness of 0.4 uN*mm/rad was
fitted by the source paper for a TETHERED model in which only one leg moves, so as a load-bearing
skeleton it is about two orders of magnitude too weak, and with an essentially free coxa any torque
applied at the tibia travels up the chain and swings the whole leg about the coxa instead of raising
the body.

THE CONSTRAINT, AND IT IS HARD.  Raising the stiffness until the fly stands by itself would destroy the
entire claim of this project, because then the posture would be produced by the springs and not by the
motor neurons.  So a configuration is accepted only if BOTH hold:

    (a) with ZERO muscle activation the fly does NOT stand  -- no body part on the floor, 6 legs down
        and the thorax near the seated 1.72 mm is a REJECT;
    (b) some muscle pattern derived from a MEASUREMENT ON THAT SAME CONFIGURATION does hold the stance.

The pattern is never chosen by hand.  At each configuration the vertical acceleration imparted at the
stance is measured per muscle (the well-posed probe: the fly is still standing when it is read), the
muscles with a positive contribution are taken as the anti-gravity set, and that set is then held.
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
SEATED_MM = 1.7209            # MEASURED by tools_stand_and_ground.py
PASSIVE_S = 2.0
HOLD_S = 2.0
LEVELS = (0.2, 0.5, 1.0)


def setup(m, d):
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    return mw, fid, th, jf


def set_params(m, stiff, damp):
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("joint_"):
            m.jnt_stiffness[j] = stiff
            m.dof_damping[int(m.jnt_dofadr[j])] = damp


def reset_stance(m, d):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.act[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)


def state(m, d, mw, fid, legs_geoms):
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    leg = body = 0.0
    legs, others = set(), set()
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
    return {"thorax_z": float(d.xpos[th][2]), "leg_bw": leg / mw, "body_bw": body / mw,
            "legs_down": sorted(legs), "nonleg": sorted(others)}


def stands(s):
    return bool(s["body_bw"] < 1e-3 and len(s["legs_down"]) == 6 and s["thorax_z"] > 1.35)


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw, fid, th, jf = setup(m, d)
    v0 = int(m.jnt_dofadr[jf])
    out = {"grid": [], "seated_mm": SEATED_MM,
           "passive_seconds": PASSIVE_S, "hold_seconds": HOLD_S, "levels": list(LEVELS)}
    print(f"one body weight = {mw:.4f}; seated thorax height = {SEATED_MM} mm; "
          f"a STAND needs no body contact, 6 legs down, thorax > 1.35 mm")
    for stiff in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        for damp in (0.02, 0.2):
            set_params(m, stiff, damp)
            print(f"\n=== stiffness {stiff} damping {damp} ===")
            # ---- (a) PASSIVE.  This must NOT stand, or the configuration is rejected outright.
            reset_stance(m, d)
            for _ in range(int(PASSIVE_S / m.opt.timestep)):
                mujoco.mj_step(m, d)
            pas = state(m, d, mw, fid, None)
            pas_stands = stands(pas)
            print(f"  (a) passive 2 s: thorax {pas['thorax_z']:.4f} mm, body {pas['body_bw']:.4f} bw, "
                  f"{len(pas['legs_down'])}/6 legs  -> "
                  + ("REJECT: stands with zero activation, so the posture would be the springs"
                     if pas_stands else "collapsed, OK"))
            if pas_stands:
                out["grid"].append({"stiffness": stiff, "damping": damp, "passive": pas,
                                    "rejected": "stands with zero activation"})
                continue
            # ---- the anti-gravity set, MEASURED at the stance of THIS configuration
            reset_stance(m, d)
            base_a = float(d.qacc[v0 + 2])
            lift = []
            for i in range(m.nu):
                reset_stance(m, d)
                d.ctrl[i] = 1.0
                d.act[i] = 1.0
                mujoco.mj_forward(m, d)
                a = float(d.qacc[v0 + 2])
                if a - base_a > 0:
                    lift.append(i)
            print(f"  anti-gravity set at this configuration: {len(lift)} of {m.nu} muscles")
            # ---- (b) hold the measured anti-gravity set
            best = None
            for lvl in LEVELS:
                reset_stance(m, d)
                d.act[:] = 0.0
                for i in lift:
                    d.act[i] = lvl
                mujoco.mj_forward(m, d)
                for _ in range(int(HOLD_S / m.opt.timestep)):
                    d.ctrl[:] = 0.0
                    for i in lift:
                        d.ctrl[i] = lvl
                    mujoco.mj_step(m, d)
                s = state(m, d, mw, fid, None)
                ok = stands(s)
                print(f"  (b) anti-gravity set held at {lvl}: thorax {s['thorax_z']:.4f} mm, "
                      f"body {s['body_bw']:.4f} bw, {len(s['legs_down'])}/6 legs  -> "
                      f"{'*** STANDS ***' if ok else 'collapsed'}")
                if ok and (best is None or s["thorax_z"] > best["thorax_z"]):
                    best = dict(s); best["level"] = lvl
            out["grid"].append({"stiffness": stiff, "damping": damp, "passive": pas,
                                "n_antigravity": len(lift), "antigravity": lift,
                                "best_stand": best})
    winners = [g for g in out["grid"] if g.get("best_stand")]
    print("\n=== ACCEPTED CONFIGURATIONS (collapses passively AND stands under muscle drive) ===")
    if winners:
        for g in sorted(winners, key=lambda g: -g["best_stand"]["thorax_z"]):
            b = g["best_stand"]
            print(f"  stiffness {g['stiffness']:<5} damping {g['damping']:<5} "
                  f"level {b['level']}: thorax {b['thorax_z']:.4f} mm, "
                  f"tarsi {b['leg_bw']:.3f} bw, {len(b['legs_down'])}/6 legs, "
                  f"passive would have been {g['passive']['thorax_z']:.4f} mm")
    else:
        print("  NONE.  No (stiffness, damping) here both collapses passively AND stands under a "
              "measured muscle pattern.")
    out["accepted"] = len(winners)
    (HERE / "outputs" / "stiffness_search.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'outputs' / 'stiffness_search.json'}")


if __name__ == "__main__":
    main()
