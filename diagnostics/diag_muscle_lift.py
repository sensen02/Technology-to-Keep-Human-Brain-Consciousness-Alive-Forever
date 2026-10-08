#!/usr/bin/env python3
"""WHICH MUSCLES CAN LIFT THE BODY?  A FEASIBILITY PROBE, NOT A CONTROLLER.

The passive fly cannot stand at any stiffness or damping, and the connectome-driven loop as wired
does not make it stand either (it drives the fly INTO the floor: body load 0.80 bw against 0.63 with
the motor neurons silenced).  Before trying to fix the loop, this settles a prior question:

    does ANY steady muscle activation pattern hold this fly up at all?

Method: one muscle type at a time, activate every instance of it on all six legs at a fixed level,
and measure what it does to the body.  A type that raises the thorax is an extensor for this stance;
one that lowers it is a flexor.  The combination of the strongest lifters is then tried together.

THIS IS A PROBE.  A constant activation written into ctrl is not behaviour and is not proposed as
one; it answers only whether the target the connectome has to reach is reachable.
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
DUR = float(os.environ.get("LIFT_DUR", "1.0"))
LEVEL = float(os.environ.get("LIFT_LEVEL", "0.3"))


def main():
    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
    a2t = actuator_to_mn_type(verbose=False)
    # GROUP BY THE ACTUATOR, i.e. by the individual MUSCLE.  MEASURED: actuator_to_mn_type returns
    # (leg, motor-neuron type) TUPLES and several distinct muscles share a type (polyneuronal
    # innervation), so grouping by that tuple merged muscles and gave 71 groups for 90 actuators.
    # The unit being probed is one muscle, so the actuator is the key; the type is only a label.
    by_type = {}
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        t = a2t.get(nm)
        lbl = f"{nm}" + (f" [{t[1]}]" if isinstance(t, tuple) else "")
        by_type[lbl] = [i]
    print(f"{len(by_type)} individual muscles over {m.nu} actuators; one body weight = {mw:.4f}")
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def settle(setter, dur=DUR):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        z0 = float(d.xpos[th][2])
        for i in range(int(dur / m.opt.timestep)):
            d.ctrl[:] = 0.0
            setter()
            mujoco.mj_step(m, d)
        leg = body = 0.0
        legs = set()
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
                body += abs(f[0])
        return {"thorax_z": float(d.xpos[th][2]), "body_bw": body / mw, "leg_bw": leg / mw,
                "legs_down": len(legs), "z0": z0,
                "stood": bool(body < 1e-3 and len(legs) == 6 and d.xpos[th][2] > 1.2)}

    base = settle(lambda: None)
    print(f"baseline (no activation): thorax z {base['thorax_z']:.4f} mm, "
          f"body {base['body_bw']:.4f} bw, {base['legs_down']}/6 legs")
    rows = []
    print(f"\n{'muscle [motor-neuron type]':<40}{'thorax z':>10}{'dz':>9}{'body bw':>9}{'legs':>6}")
    for t, idx in sorted(by_type.items()):
        r = settle(lambda idx=idx: d.ctrl.__setitem__(idx, LEVEL))
        r["type"] = t
        r["dz"] = r["thorax_z"] - base["thorax_z"]
        rows.append(r)
        print(f"{t:<30}{r['thorax_z']:>10.4f}{r['dz']:>+9.4f}{r['body_bw']:>9.4f}{r['legs_down']:>6}")
    rows.sort(key=lambda r: -r["dz"])
    lifters = [r for r in rows if r["dz"] > 0.01]
    print(f"\n{len(lifters)} of {len(rows)} muscles RAISE the thorax; strongest: "
          f"{[(r['type'], round(r['dz'], 4)) for r in lifters[:5]] or 'NONE'}")
    # try the strongest lifters together, at the same level and at a higher one
    best = None
    for k in (1, 2, 3, 4, 5, 8):
        names = [r["type"] for r in rows[:k]]
        idx = sorted({i for n in names for i in by_type[n]})
        for lvl in (LEVEL, 0.6, 1.0):
            r = settle(lambda idx=idx, lvl=lvl: d.ctrl.__setitem__(idx, lvl))
            r["types"] = names
            r["level"] = lvl
            print(f"  top {k} types at level {lvl}: thorax z {r['thorax_z']:.4f} mm, "
                  f"body {r['body_bw']:.4f} bw, {r['legs_down']}/6 legs "
                  f"{'STANDS' if r['stood'] else ''}")
            if r["stood"] and (best is None or r["thorax_z"] > best["thorax_z"]):
                best = r
    print(f"\n=== IS A STEADY PATTERN ENOUGH? ===")
    if best:
        print(f"  YES: top {len(best['types'])} extensor types at level {best['level']} holds the fly "
              f"up at thorax z {best['thorax_z']:.4f} mm with {best['legs_down']}/6 legs and no body "
              f"contact ({best['body_bw']:.4f} bw on the body)")
    else:
        print("  NO steady uniform-across-legs pattern found that stands the fly in this search")
    (HERE / "outputs" / "muscle_lift_sensitivity.json").write_text(json.dumps(
        {"baseline": base, "per_type": rows, "level": LEVEL, "duration_s": DUR,
         "best_combination": best}, indent=2))
    print(f"wrote {HERE / 'outputs' / 'muscle_lift_sensitivity.json'}")


if __name__ == "__main__":
    main()
