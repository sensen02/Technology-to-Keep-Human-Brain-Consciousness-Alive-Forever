#!/usr/bin/env python3
"""WHY DOES THE STANDING STACK COLLAPSE?  TRACE THE FIRST 100 ms, IN ORDER.

WHAT IS ALREADY MEASURED, so this does not redo it:
  * the neutral keyframe IS a standing pose -- free-joint z = 2.0454 mm and all six tarsi within
    0.06 mm of the floor;
  * it is also STATICALLY STABLE there -- support polygon 2.817 mm^2, centre of mass 0.5227 mm
    INSIDE the nearest edge;
  * holding the body's pose, a leg deflects only 0.004-0.03 mm under one sixth of body weight, so the
    legs are stiff struts when loaded the way that test loads them;
  * yet with the body free the thorax sinks 1.39 mm while all six feet stay down.

A 40x-400x discrepancy between "stiff when held" and "sinks when free" means the leg is NOT being
loaded the same way in the two cases, and the earlier `sag_mode` result rules out the obvious
explanation: freezing ROTATION (translation-only) reproduced the full 1.3918 mm sag with a final tilt
of 0.0 deg, so the sag is not simply "the body tips over".  A final tilt of zero does not prove it
never tipped, though -- it can pitch and come back -- so what is missing is the ORDER of events.

This traces it: every millisecond for the first 100 ms, the thorax height and the body's up-vector,
how many feet are loaded and how hard, and which GEOM is carrying the body's weight when the legs are
not.  The question it answers is "what fails first", and that is what decides whether the fix is
stiffness, geometry, or something else entirely.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_collapse_order.py
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


def main() -> int:
    import mujoco
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=100.0)
    ap.add_argument("--every-ms", type=float, default=1.0)
    ap.add_argument("--out", default=str(HERE / "outputs" / "collapse_order.json"))
    args = ap.parse_args()

    m = _load_mjcf(str(XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
    mw = float(sum(m.body_mass)) * abs(float(m.opt.gravity[2]))     # one body weight, model units
    dt = float(m.opt.timestep)
    n_every = max(1, int(round(args.every_ms * 1e-3 / dt)))
    n_steps = int(round(args.ms * 1e-3 / dt))
    up0 = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()
    geom_names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)) or "" for g in range(m.ngeom)]
    # THE FLOOR IS IDENTIFIED BY ITS GEOM TYPE, NOT BY ITS NAME.  MEASURED, and it is why the first
    # version of this trace reported zero feet loaded even at t=0, where the geometry says all six
    # tarsi are within 0.06 mm of z=0: the floor geom in this model is named "floor", while the
    # detector looked for a "ground"/"rough" PREFIX -- so it found no ground contacts AT ALL and the
    # trace silently described a fly interacting with nothing.
    ground_geoms = {g for g in range(m.ngeom)
                    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
    if not ground_geoms:
        raise RuntimeError("no plane geom in the model; cannot tell floor contacts from self contacts")

    rows = []
    res = np.zeros(6, dtype=float)
    for step in range(n_steps + 1):
        if step % n_every == 0:
            legf, bodyf, by_geom = 0.0, 0.0, {}
            loaded = set()
            for c in range(int(d.ncon)):
                cc = d.contact[c]
                g1, g2 = int(cc.geom1), int(cc.geom2)
                if not (g1 in ground_geoms or g2 in ground_geoms):
                    continue
                mujoco.mj_contactForce(m, d, c, res)
                f = abs(float(res[0]))
                other = geom_names[g2] if g1 in ground_geoms else geom_names[g1]
                if other[:2] in LEGS:
                    legf += f
                    loaded.add(other[:2])
                else:
                    bodyf += f
                    by_geom[other] = by_geom.get(other, 0.0) + f
            up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
            rows.append({
                "t_ms": round(step * dt * 1e3, 3),
                "thorax_z_model": float(d.xpos[th][2]),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(float(up @ up0), -1, 1)))),
                "n_feet_loaded": len(loaded),
                "loaded_feet": sorted(loaded),
                "leg_bw": legf / mw, "body_bw": bodyf / mw,
                "body_contacts": {k: round(v / mw, 4) for k, v in
                                  sorted(by_geom.items(), key=lambda kv: -kv[1])[:4]},
            })
        mujoco.mj_step(m, d)

    # ---- report the ORDER, which is the point
    z0 = rows[0]["thorax_z_mm"]
    first_tilt = next((r for r in rows if r["tilt_deg"] > 5.0), None)
    first_body = next((r for r in rows if r["body_bw"] > 0.05), None)
    first_len_drop = next((r for r in rows if r["n_feet_loaded"] < 6), None)
    peak_body = max(rows, key=lambda r: r["body_bw"])
    final = rows[-1]

    print(f"traced {len(rows)} samples over {args.ms:.0f} ms "
          f"(one body weight = {mw:.4f} model units)")
    print(f"\n  {'t_ms':>7} {'thorax_z':>9} {'tilt':>7} {'feet':>5} {'leg_bw':>8} {'body_bw':>8}"
          f"  worst body contact")
    for r in rows[::max(1, len(rows) // 14)]:
        w = next(iter(r["body_contacts"].items()), ("-", 0.0))
        print(f"  {r['t_ms']:>7.1f} {r['thorax_z_mm']:>9.4f} {r['tilt_deg']:>7.2f} "
              f"{r['n_feet_loaded']:>5d} {r['leg_bw']:>8.3f} {r['body_bw']:>8.3f}  {w[0]} {w[1]:.3f}")

    print("\nORDER OF FAILURE:")
    print(f"  start thorax z = {z0:.4f} mm, all six feet down")
    print(f"  1. first foot LOST:      "
          + (f"t = {first_len_drop['t_ms']:.1f} ms (feet {first_len_drop['loaded_feet']})"
             if first_len_drop else "never in this window"))
    print(f"  2. first BODY contact:   "
          + (f"t = {first_body['t_ms']:.1f} ms, body_bw {first_body['body_bw']:.3f}, "
             f"worst {next(iter(first_body['body_contacts'].items()), ('-', 0))}"
             if first_body else "never in this window"))
    print(f"  3. first TILT over 5 deg:"
          + (f"t = {first_tilt['t_ms']:.1f} ms, tilt {first_tilt['tilt_deg']:.2f} deg"
             if first_tilt else "never in this window"))
    print(f"  peak body load: t = {peak_body['t_ms']:.1f} ms, body_bw {peak_body['body_bw']:.3f}")
    print(f"  end: thorax z {final['thorax_z_mm']:.4f} (sank {z0 - final['thorax_z_mm']:.4f} mm), "
          f"tilt {final['tilt_deg']:.2f} deg, feet {final['n_feet_loaded']}")

    doc = {
        "overall": "PASS", "status": "PASS",
        "question": "what fails first when the standing stack is released",
        "rows": rows, "one_body_weight_model_units": mw,
        "order": {
            "start_thorax_z_mm": z0,
            "first_foot_lost_t_ms": first_len_drop["t_ms"] if first_len_drop else None,
            "first_body_contact_t_ms": first_body["t_ms"] if first_body else None,
            "first_tilt_over_5deg_t_ms": first_tilt["t_ms"] if first_tilt else None,
            "peak_body_bw": peak_body["body_bw"], "peak_body_t_ms": peak_body["t_ms"],
            "end_thorax_z_mm": final["thorax_z_mm"], "end_tilt_deg": final["tilt_deg"],
            "sank_mm": z0 - final["thorax_z_mm"],
        },
        "units": "mm, degrees, and multiples of one body weight (model force units)",
        "provenance": ("one MuJoCo rollout of fruitfly_six_leg_standing.xml from its own keyframe, "
                       "no actuation, "
                       f"dt {dt} s, sampled every {args.every_ms} ms"),
        "what_this_does_not_establish": (
            "WHY it fails -- this is the order of events, not the cause.  It also says nothing about "
            "whether the model is biologically right: the keyframe pose, the joint springs and the "
            "contact parameters are all model properties, and the fly here is a simulator object, "
            "not an animal.  A passing check here would not mean the fly can walk."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
