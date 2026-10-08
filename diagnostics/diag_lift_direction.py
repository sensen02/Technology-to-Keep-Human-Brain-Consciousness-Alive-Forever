#!/usr/bin/env python3
"""WHICH MUSCLES PUSH THE BODY UP -- probed by VERTICAL ACCELERATION at the stance.

WHY THE PREVIOUS PROBE WAS ILL-POSED, MEASURED: activating each muscle and reading the thorax height
after a second said 0 of 90 muscles raises the body.  But the muscles are not weak -- at full
activation the median muscle produces 0.83 body weights of tendon force and the tibia extensors reach
12.6 body weights -- and the fly starts each probe AT the stance (thorax 1.72 mm), so the force is
there.  The problem is the measurement: once the fly has sagged onto its belly, extending the tibia
pushes the tarsus SIDEWAYS along the floor, which does not raise the body at all.  The signal was
flat for a geometric reason, not a physiological one, and ranking muscles by a flat signal produced a
meaningless "top-k".

The well-posed probe is the vertical acceleration imparted in the FIRST instant, while the fly is
still standing in its stance:

    a_z = qacc of the free joint's vertical dof, one step after applying the activation

Muscles that push the body up give a less negative a_z.  Free fall is -9801; the passive stance is
whatever the contacts and springs give.

THIS IS A PROBE.  Setting ctrl by hand is not behaviour and is not proposed as one; it measures which
muscle can lift the body, which is the target the connectome has to reach.
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
LEVEL = float(os.environ.get("LIFT_LEVEL", "0.5"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "thorax_free")
    v0 = int(m.jnt_dofadr[jf])            # first dof of the free joint = world x, +2 = z
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

    def probe(idx, level):
        # SET THE ACTIVATION STATE DIRECTLY, do not rely on ctrl.  MEASURED, and it made the first
        # version of this probe report EXACTLY zero change for all 90 muscles: MuJoCo's muscle
        # dyntype drives its activation state through a first-order filter with dynprm time constants
        # of 0.01-0.04 s, so ONE step (0.1 ms) after setting ctrl leaves act still at 0.0 and the
        # active force exactly zero.  "No muscle lifts the body" was an artefact of the probe clock,
        # not a physiological result.  With d.act set explicitly, the acceleration is read with the
        # muscle already at its requested activation.
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = 0.0
        d.act[:] = 0.0
        for i in idx:
            d.ctrl[i] = level
            d.act[i] = level
        mujoco.mj_forward(m, d)
        return float(d.qacc[v0 + 2]), float(d.xpos[th][2])

    base_acc, base_z = probe([], LEVEL)
    print(f"one body weight = {mw:.4f}; stance thorax z = {base_z:.4f} mm")
    print(f"passive vertical acceleration at the stance = {base_acc:+.2f} mm/s^2 "
          f"(free fall would be {m.opt.gravity[2]:+.0f})")
    rows = []
    for i in range(m.nu):
        a, _z = probe([i], LEVEL)
        rows.append({"actuator": names[i], "a_z": a, "dz_a": a - base_acc})
    rows.sort(key=lambda r: -r["dz_a"])
    print(f"\n{'actuator':<46}{'a_z':>12}{'d(a_z)':>12}")
    for r in rows[:12]:
        print(f"{r['actuator']:<46}{r['a_z']:>12.1f}{r['dz_a']:>+12.1f}")
    print("  ...")
    for r in rows[-4:]:
        print(f"{r['actuator']:<46}{r['a_z']:>12.1f}{r['dz_a']:>+12.1f}")
    lifters = [r for r in rows if r["dz_a"] > 0]
    print(f"\nmuscles with a POSITIVE lift contribution: {len(lifters)} / {m.nu}")
    print(f"  strongest lifters: "
          + ", ".join(f"{r['actuator']}({r['dz_a']:+.0f})" for r in rows[:6]))
    print(f"  strongest depressors: "
          + ", ".join(f"{r['actuator']}({r['dz_a']:+.0f})" for r in rows[-6:]))

    # ---- does the natural extensor set lift the whole body?  group by motor-neuron type.
    from muscle_wiring import actuator_to_mn_type
    a2t = actuator_to_mn_type(verbose=False)
    groups = {}
    for i, nm in enumerate(names):
        t = a2t.get(nm)
        key = t[1] if isinstance(t, tuple) else str(t)
        groups.setdefault(key, []).append(i)
    print(f"\n=== by motor-neuron TYPE ({len(groups)} types), all six legs together ===")
    print(f"{'type':<34}{'n':>3}{'a_z':>11}{'d(a_z)':>11}")
    gres = []
    for t, idx in sorted(groups.items()):
        a, _z = probe(idx, LEVEL)
        gres.append({"type": t, "n": len(idx), "a_z": a, "dz_a": a - base_acc})
    gres.sort(key=lambda r: -r["dz_a"])
    for r in gres:
        print(f"{r['type']:<34}{r['n']:>3}{r['a_z']:>11.1f}{r['dz_a']:>+11.1f}")
    top = [r["type"] for r in gres if r["dz_a"] > 0]
    print(f"\ntypes that lift: {top or 'NONE'}")
    # try the lifters together, with the antagonists suppressed
    if top:
        idx = sorted({i for t in top for i in groups[t]})
        for lvl in (LEVEL, 1.0):
            a, _z = probe(idx, lvl)
            print(f"  all lifting types together at level {lvl}: d(a_z) = {a - base_acc:+.1f}")
    (HERE / "outputs" / "lift_direction.json").write_text(json.dumps(
        {"baseline_a_z": base_acc, "level": LEVEL, "per_actuator": rows, "per_type": gres,
         "lifting_types": top}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'lift_direction.json'}")


if __name__ == "__main__":
    main()
