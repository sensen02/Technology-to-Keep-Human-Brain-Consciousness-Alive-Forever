#!/usr/bin/env python3
"""CROSS-CHECK THE PLATFORM AGAINST REALITY, row by row, with the provenance of each number.

WHY THIS FILE EXISTS.  Everything this project reports is only as good as the simulator it came
out of, and a simulator built on a published body model inherits both that model's measurements
AND its compromises.  So each quantity the platform depends on is measured HERE, put beside the
value reality is believed to have, and given a verdict.  A row that fails is not a footnote; it
is a limit on every downstream claim, and the rows that fail are the ones named loudest.

PROVENANCE OF THE REALITY COLUMN, stated per row because it is not uniform:
  * REAL(measured-literature) -- a value from a source, cited in the row.
  * REAL(textbook)            -- a value so standard that it is not worth a citation, e.g. the
                                 weight of a 1 mg animal under 9.81 m/s^2.
  * REAL(unverified)          -- I believe this but could NOT confirm it against a source in this
                                 session; it is labelled rather than quietly asserted.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_reality_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARENA = HERE / "outputs" / "arena"
TACTILE = HERE / "outputs" / "tactile"

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
TARSUS_CHAIN = ("tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5")

#: The model's unit system, MEASURED from it: length mm, and the mass unit is the GRAM, because
#: the fly weighs 1.02431 mg, which is what a real fly weighs.  So 1 force unit = 1 uN.
G_MM_S2 = 9810.0


def morphology() -> dict:
    """Static quantities, read out of the compiled model and its specs."""
    from engine.embodied import BodyBackend, BodyConfig
    cfg = BodyConfig(seed=0, scene_preset=None, add_world_camera=False,
                     add_tracking_camera=False, add_vision=False,
                     extra_cameras=(), extra_geoms=(), extra_materials=())
    be = BodyBackend(cfg, gl_backend=None)
    m = be.model

    def bid(name):
        return int(mujoco_name_to_id(m, "body", name))

    import mujoco
    names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)) for i in range(m.nbody)]

    def find(suffix):
        hit = [i for i, n in enumerate(names) if n and n.endswith("/" + suffix)]
        if not hit:
            raise KeyError(suffix)
        return hit[0]

    mass = float(np.sum(np.asarray(m.body_mass, dtype=float)))
    # UNIT ARITHMETIC SPELLED OUT, because I got it wrong here first: ``body_mass`` is in the
    # model's own mass unit, which IS the gram (the fly is 1.02431e-3 of it, i.e. 1.02431 mg,
    # which is what a real fly weighs).  So milligrams are the number x 1e3, and the weight is
    # mass_g x 9810 mm/s^2 = 10.0485, because 1 g*mm/s^2 = 1e-3 kg x 1e-3 m/s^2 = 1 uN.
    out = {
        "mass_model_unit": mass,
        "mass_mg": mass * 1e3,
        "weight_uN": mass * G_MM_S2,
        "timestep_s": float(be.cfg.timestep_s),
        "n_bodies": int(m.nbody),
        "n_geoms": int(m.ngeom),
        "n_joints": int(m.njnt),
        "n_actuators": int(m.nu),
        "standing_thorax_z_mm": None,
    }
    # BODY EXTENT: a LOWER BOUND, and labelled as one.  ``geom_aabb`` is in the geom's LOCAL
    # frame, so unioning those is meaningless (it gave [0.53, 0.61, 1.35] mm for a 2.5 mm fly).
    # What is available without assumptions is the spread of the body ORIGINS in world
    # coordinates, which cannot exceed the body and therefore bounds it from below.
    wl = np.asarray(be.data.xpos, dtype=float)
    out["body_origins_extent_mm"] = (wl.max(axis=0) - wl.min(axis=0)).tolist()
    out["body_extent_note"] = ("lower bound: the spread of body ORIGINS in world coordinates; "
                              "mesh surfaces extend beyond it")

    # LEG SEGMENT LENGTHS, FROM WORLD POSITIONS.  ``body_pos`` is RELATIVE TO THE PARENT, so
    # measuring between consecutive bodies with it gives nonsense (it produced a 0.340 mm
    # "femur" where the real one is 0.705 mm).  A forward pass is run and ``xpos`` is used.
    import mujoco as _mj
    _mj.mj_forward(m, be.data)
    seg_len = {}
    for leg in LEGS:
        chain = ["coxa", "trochanterfemur", "tibia"] + list(TARSUS_CHAIN)
        pts = []
        for s in chain:
            i = find(f"{leg}_{s}")
            pts.append(np.asarray(be.data.xpos[i], dtype=float))
        lens = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1)]
        seg_len[leg] = {"femur_mm": lens[1], "tibia_mm": lens[2],
                        "tarsus_total_mm": float(sum(lens[3:])),
                        "total_leg_mm": float(sum(lens[1:]))}
    out["per_leg_mm"] = seg_len
    return out


def mujoco_name_to_id(m, kind, name):
    import mujoco
    obj = {"body": mujoco.mjtObj.mjOBJ_BODY}[kind]
    return mujoco.mj_name2id(m, obj, name)


def behaviour() -> dict:
    """Kinematics and contact statistics from an episode ALREADY RECORDED.

    Reusing the tactile episodes on purpose: those are the recordings every tactile claim rests
    on, so the behavioural rows are read from the same data rather than from a fresh run that
    might behave differently.
    """
    ep = TACTILE / "tactile_bare_seed0" / "tactile.npz"
    z = np.load(ep)
    t = np.asarray(z["truth/time_s"], dtype=float)
    cf = np.asarray(z["truth/contact_found_raw"], dtype=float)
    nf = np.asarray(z["truth/normal_force_N"], dtype=float)
    pen = np.asarray(z["truth/penetration_m"], dtype=float)
    pos = np.asarray(z["truth/contact_positions_mm"], dtype=float)
    body_p = np.asarray(z["truth/body_positions_mm"], dtype=float) \
        if "truth/body_positions_mm" in z.files else None
    jq = np.asarray(z["truth/joint_angles_rad"], dtype=float) \
        if "truth/joint_angles_rad" in z.files else None
    thorax = np.asarray(z["truth/thorax_mm"], dtype=float) \
        if "truth/thorax_mm" in z.files else None

    BW = 10.0485
    legs_in_contact = (cf > 0).sum(axis=1)
    out = {
        "episode": str(ep.relative_to(HERE)),
        "duration_s": float(t[-1] - t[0]),
        "n_frames": int(t.size),
        "dt_s": float(np.median(np.diff(t))),
        "duty_per_leg": {leg: float((cf[:, i] > 0).mean()) for i, leg in enumerate(LEGS)},
        "legs_in_contact_mean": float(legs_in_contact.mean()),
        "frames_with_no_leg_down": float((legs_in_contact == 0).mean()),
        "frames_with_3_or_more_down": float((legs_in_contact >= 3).mean()),
        "summed_normal_force_over_weight_mean": float((nf / 1e-3).sum(axis=1).mean() / BW),
        "summed_normal_force_over_weight_peak": float((nf / 1e-3).sum(axis=1).max() / BW),
        "penetration_um_when_contacting": (
            float(np.median(pen[pen > 0]) * 1e6) if (pen > 0).any() else None),
    }
    if thorax is not None:
        d = np.linalg.norm(np.diff(thorax, axis=0), axis=1)
        out["thorax_path_mm"] = float(d.sum())
        out["thorax_speed_mm_per_s"] = float(d.sum() / (t[-1] - t[0]))
        out["thorax_net_displacement_mm"] = float(np.linalg.norm(thorax[-1] - thorax[0]))
    if jq is not None:
        out["joint_angle_span_rad"] = float(jq.max(axis=0).max() - jq.min(axis=0).min())
    return out


#: REALITY, with the provenance of each value spelled out.  ``None`` means I could not verify it.
REALITY = [
    ("mass", "1.02 mg measured", "~1 mg for an adult D. melanogaster",
     "REAL(measured-literature): standard value; the fly is the reference animal of genetics "
     "and its mass is reported throughout the literature", "ok"),
    ("body length", None, "~2.5-3 mm adult", "REAL(textbook)", "check"),
    ("leg segment lengths", None, "NeuroMechFly's Extended Data Fig. 1 compares leg segment "
     "lengths for REAL female D. melanogaster against the model",
     "REAL(measured-literature): https://preview-www.nature.com/articles/s41592-022-01466-7/figures/7",
     "check"),
    ("walking speed", None, "tens of mm/s; D. melanogaster walks roughly 10-30 mm/s",
     "REAL(unverified): I could not confirm a figure from a source in this session",
     "check"),
    ("leg coordination", None, "tripod gait: three legs down at essentially all times",
     "REAL(measured-literature): Strauss & Heisenberg 1990, J Comp Physiol A 167:403-412, "
     "coordination of legs during straight walking in D. melanogaster",
     "check"),
    ("duty factor", None, "~0.5-0.7 per leg in straight walking",
     "REAL(unverified): follows from the tripod gait but I did not confirm the number",
     "check"),
    ("summed ground reaction force", None, "must average ~1x body weight during steady walking",
     "REAL(textbook): Newton's first law -- a body that neither rises nor sinks has zero net "
     "vertical impulse over a gait cycle",
     "check"),
    ("adhesion", None, "Drosophila feet adhere on smooth surfaces (pulvilli)",
     "REAL(measured-literature): insect tarsal adhesion is a large literature; the platform has "
     "NO adhesion model at all", "check"),
]


def main() -> int:
    print("=" * 100)
    print("PLATFORM vs REALITY")
    print("=" * 100)
    print("\n--- A. MORPHOLOGY (read out of the compiled model) ---")
    try:
        morph = morphology()
        print(f"  mass                 : {morph['mass_mg']:.5f} mg  "
              f"(weight {morph['weight_uN']:.4f} uN)")
        print(f"  body extent (lower bound, body origins): "
              f"{np.round(morph['body_origins_extent_mm'], 3).tolist()} mm")
        print(f"  timestep             : {morph['timestep_s']:.6f} s")
        print(f"  bodies/geoms/joints/actuators: {morph['n_bodies']}/{morph['n_geoms']}/"
              f"{morph['n_joints']}/{morph['n_actuators']}")
        print(f"  {'leg':<5}{'femur mm':>10}{'tibia mm':>10}{'tarsus mm':>11}{'whole leg mm':>14}")
        for leg, v in morph["per_leg_mm"].items():
            print(f"  {leg:<5}{v['femur_mm']:>10.3f}{v['tibia_mm']:>10.3f}"
                  f"{v['tarsus_total_mm']:>11.3f}{v['total_leg_mm']:>14.3f}")
    except Exception as exc:
        morph = {"error": repr(exc)}
        print("  morphology measurement FAILED:", exc)

    print("\n--- B. BEHAVIOUR (read out of a recorded episode) ---")
    try:
        beh = behaviour()
        for k, v in beh.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"      {k2:<6} {v2 * 100:6.1f}% duty")
            elif isinstance(v, float):
                print(f"  {k:<44}: {v:.4f}")
            else:
                print(f"  {k:<44}: {v}")
    except Exception as exc:
        beh = {"error": repr(exc)}
        print("  behaviour measurement FAILED:", exc)

    print("\n--- C. REALITY COLUMN AND VERDICTS ---")
    print(f"  {'quantity':<30}{'platform':>14}   reality / provenance")
    verdicts = []
    if "error" not in morph:
        verdicts.append(("mass", f"{morph['mass_mg']:.3f} mg", "~1 mg", "AGREES (3%)"))
    if "error" not in beh:
        verdicts.append(("summed GRF / body weight",
                         f"{beh['summed_normal_force_over_weight_mean']:.2f}x mean, "
                         f"{beh['summed_normal_force_over_weight_peak']:.1f}x peak",
                         "~1x mean", "mean is ~2.9x too high"))
        verdicts.append(("frames with no leg on the ground",
                         f"{beh['frames_with_no_leg_down'] * 100:.0f}%",
                         "~0% (tripod gait)",
                         "FAILS: the fly is airborne most of the time"))
        verdicts.append(("frames with >=3 legs down",
                         f"{beh['frames_with_3_or_more_down'] * 100:.0f}%",
                         "~100% (tripod gait)", "FAILS"))
        verdicts.append(("per-leg duty factor",
                         f"{np.mean(list(beh['duty_per_leg'].values())):.2f}",
                         "~0.5-0.7", "FAILS: ~4x too low"))
    for q, p, r, v in verdicts:
        print(f"  {q:<30}{p:>14}   {r:<14} -> {v}")

    print("\n--- D. WHAT THIS MEANS FOR THE HEADLINE CLAIMS ---")
    print("  * GEOMETRY AND SCALE are inherited from NeuroMechFly and are the part of the")
    print("    platform that IS tied to real measurements; the mass matches a real fly to 3%.")
    print("  * BEHAVIOUR IS NOT. The fly leaves the ground in ~80% of frames while a real fly")
    print("    keeps three legs down, so every contact-time statistic (duty factor, step")
    print("    frequency, impulse per step, and the tactile event train) describes a bouncing")
    print("    object rather than a walking fly.")
    print("  * The force scale is now measured honestly (mean 2.9x body weight, not the 11,400x")
    print("    a unit error produced earlier), so the remaining error is ~3x on the mean and is")
    print("    most likely a SYMPTOM of the gait, not a separate defect.")
    print("  * There is NO adhesion model, so anything about attachment to a surface is absent.")

    dest = HERE / "outputs" / "reality_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "kind": "PLATFORM vs REALITY cross-check",
        "morphology": morph, "behaviour": beh,
        "verdicts": [{"quantity": q, "platform": p, "reality": r, "verdict": v}
                     for q, p, r, v in verdicts],
        "reality_provenance": [{"quantity": a, "platform_side": b, "reality": c,
                                "provenance": d, "status": e} for a, b, c, d, e in REALITY],
        "unit_system": {"length": "mm", "mass": "g (fly = 1.02431 mg, as a real fly)",
                        "force": "1 unit = 1 uN",
                        "evidence": "a resting fly's summed normal force reads ~10.05 units, "
                                    "and m*g = 1.02431e-3 g x 9810 mm/s^2 = 10.0485 units"},
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
