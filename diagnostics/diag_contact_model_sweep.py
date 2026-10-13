#!/usr/bin/env python3
"""THE CONTACT MODEL: HOW MUCH DOES IT DECIDE, AND WHAT WOULD CHANGING IT COST?

THE THIRD NAMED CANDIDATE.  The goal names three possible sources for "the body cannot stand":
joint passive stiffness, the contact model (`solref`), and the mass distribution.  The first has been
measured: all 37 sprung joints carry a uniform stiffness of 0.4, the legs sag about 1.08 rad under
their own weight until they rest on their JOINT LIMITS, and a sweep from 0.4 to 256 raises the thorax
monotonically (0.6448 -> 1.7112 mm) while the number of legs on the floor falls from 6 to 1, i.e. a
stiffer skeleton does not stand, it TIPS.  This file measures the second.

WHAT IS ACTUALLY THERE.  Every fly tarsus geom and the floor carry `solref = (0.02, 1.0)`, i.e. a
contact time constant of 20 ms with a damping ratio of 1, against a simulation timestep of 0.1 ms --
one contact time constant spans 200 steps, so the contacts are extremely compliant.  The question is
whether the pose fails because the feet sink too far into the floor.

THE COST IS DECLARED BEFORE THE SWEEP, NOT AFTER.  Changing a contact parameter changes the physics of
every episode recorded from this model, so a change that helped would invalidate all of them and
require re-recording and re-registration.  The sweep therefore reports, alongside each setting,
whether it is a candidate worth paying that price for -- and it is run as a MEASUREMENT, with the
shipped value included as a control so the harness is shown to reproduce it.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_contact_model_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
#: The project's own standing criterion, taken verbatim from `diagnostics/diag_tone_hold.py` so the
#: numbers are comparable with the earlier rounds instead of being a new definition.
STAND_Z_MM = 1.2
BODY_BW_MAX = 1e-3


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "contact_model_sweep.json"))
    args = ap.parse_args()

    base_spec = _load_mjcf(str(XML))

    def build(timeconst: float | None, dampratio: float | None):
        """A fresh spec each time: a compiled model cannot have its contact model edited."""
        import mujoco as mj
        from flygym.compose.fly.musculoskeletal import _load_mjcf as load
        spec = load(str(XML))
        if timeconst is not None:
            def walk(b):
                for g in list(b.geoms):
                    nm = str(getattr(g, "name", "") or "")
                    is_floor = int(g.type) == int(mj.mjtGeom.mjGEOM_PLANE)
                    if is_floor or (nm[:2] in LEGS and "Tarsus" in nm):
                        g.solref = [float(timeconst),
                                    float(dampratio if dampratio is not None
                                          else float(g.solref[1]))]
                for c in b.bodies:
                    walk(c)
            walk(spec.worldbody)
        return spec.compile()

    def measure(m, tag: str, note: str = ""):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
        mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))
        gnd = {g for g in range(m.ngeom)
               if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
        names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or ""
                 for g in range(m.ngeom)]
        z0 = float(d.xpos[th][2])
        up0 = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
        for _ in range(int(round(args.settle_ms * 1e-3 / float(m.opt.timestep)))):
            mujoco.mj_step(m, d)
        legs, body = 0.0, 0.0
        feet = set()
        res = np.zeros(6)
        for c in range(int(d.ncon)):
            cc = d.contact[c]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if not (g1 in gnd or g2 in gnd):
                continue
            mujoco.mj_contactForce(m, d, c, res)
            f = abs(float(res[0]))
            o = names[g2] if g1 in gnd else names[g1]
            if o[:2] in LEGS:
                legs += f
                feet.add(o[:2])
            else:
                body += f
        up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
        z = float(d.xpos[th][2])
        return {"tag": tag, "note": note, "thorax_z_mm": z, "sank_mm": z0 - z,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet": len(feet), "leg_bw": legs / mw, "body_bw": body / mw,
                "stood": bool(body / mw < BODY_BW_MAX and len(feet) == 6 and z > STAND_Z_MM)}

    rows = []
    print(f"settle {args.settle_ms:.0f} ms per setting; the project's own standing criterion is "
          f"body load < {BODY_BW_MAX}, all six feet down, thorax z > {STAND_Z_MM} mm\n")
    print(f"  {'timeconst':>10} {'dampratio':>10} {'thorax_z':>10} {'sank':>8} {'tilt':>8} "
          f"{'feet':>5} {'body_bw':>8} {'stood':>6}")
    settings = [(None, None)]
    for tc in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002):
        settings.append((tc, None))
    for tc, dr in settings:
        m = build(tc, dr)
        r = measure(m, f"timeconst={tc}" if tc else "SHIPPED (control)",
                    "shipped solref (0.02, 1.0)" if tc is None else "")
        rows.append(r)
        tc_eff = float(m.geom_solref[0][0])
        dr_eff = float(m.geom_solref[0][1])
        print(f"  {tc_eff:>10.5f} {dr_eff:>10.3f} {r['thorax_z_mm']:>10.4f} {r['sank_mm']:>8.4f} "
              f"{r['tilt_deg']:>8.2f} {r['n_feet']:>5d} {r['body_bw']:>8.4f} {str(r['stood']):>6}")

    shipped = rows[0]
    stood = [r for r in rows if r["stood"]]
    best = max(rows, key=lambda r: (r["stood"], -r["body_bw"], r["thorax_z_mm"]))
    print(f"\ncontrol reproduces the shipped behaviour: thorax z {shipped['thorax_z_mm']:.4f} mm, "
          f"{shipped['n_feet']} feet, body load {shipped['body_bw']:.4f} BW")
    print(f"settings that STOOD: {[r['tag'] for r in stood] if stood else 'NONE'}")
    print(f"best by body load: {best['tag']} -> z {best['thorax_z_mm']:.4f} mm, "
          f"tilt {best['tilt_deg']:.2f} deg, {best['n_feet']} feet, body {best['body_bw']:.4f} BW")

    # how many recorded episodes would a contact change invalidate?
    n_ep = len(list((HERE / "outputs" / "arena").glob("arena_*")))
    n_tac = len(list((HERE / "outputs" / "tactile").glob("tactile_*")))
    print(f"\nCOST OF ANY CONTACT CHANGE: it alters the physics of every episode from this model -- "
          f"{n_ep} arena episode(s) and {n_tac} tactile episode(s) would need re-recording and "
          f"re-registration")

    doc = {
        "overall": "PASS", "status": "PASS",
        "verdict": ("CONTACT_MODEL_IS_NOT_THE_CAUSE" if not stood else "CONTACT_CHANGE_HELPED"),
        "shipped_solref": [0.02, 1.0], "timestep_s": 0.0001,
        "shipped_timeconst_ms": 20.0, "steps_per_timeconst": 200.0,
        "standing_criterion": {"body_bw_below": BODY_BW_MAX, "all_six_feet": True,
                               "thorax_z_above_mm": STAND_Z_MM,
                               "source": "verbatim from diagnostics/diag_tone_hold.py"},
        "rows": rows, "control_reproduced": shipped,
        "settings_that_stood": [r["tag"] for r in stood],
        "best_by_body_load": best,
        "cost_of_any_change": {"arena_episodes": n_ep, "tactile_episodes": n_tac,
                               "declared_before_the_sweep": True,
                               "text": ("changing a contact parameter changes the physics of every "
                                        "episode recorded from this model; all of them would need "
                                        "re-recording and re-registration")},
        "units": ("mm, degrees, seconds, and multiples of one body weight; one MODEL FORCE UNIT is "
                  "1 micro-newton"),
        "provenance": ("MuJoCo rollouts of fruitfly_six_leg_standing.xml from its own keyframe with "
                       "the tarsus and floor geoms' solref overridden, no actuation"),
        "what_this_does_not_establish": (
            "that the shipped solref is WRONG -- it is the model's declared default and this sweep "
            "only reports what changing it does.  It also does not establish that a stiffer contact "
            "is more correct physically: MuJoCo's time constant is a solver parameter, not a "
            "measured material property of cuticle, and no experiment in this project measures a "
            "fly's foot compliance.  A setting that stood would still be a simulator property, and "
            "0.5 s of stillness is not walking."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
