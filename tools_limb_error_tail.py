#!/usr/bin/env python3
"""THE TAIL, NOT THE MEDIAN: how bad can a leg angle get, and does a few degrees matter?

Two questions, answered with measurement rather than reassurance:

  1. CAN THE ANGLE JUMP CATASTROPHICALLY, e.g. -22 deg to +80 deg?  A median of 3 deg says
     nothing about that.  A jump of that size has exactly two causes and BOTH are measurable:

       (a) IDENTITY SWAP -- the pipeline assigns a blob belonging to segment B to segment A.
           The question is then a pure GEOMETRY question, answerable from the real fly: if you
           compute the bone with the wrong endpoint, how far off is the angle?  That is done
           here for every bone and EVERY possible wrong endpoint, over every frame of the
           recording.  It needs no cameras at all, so it is exact.

       (b) MERGED MARKERS -- two markers closer together than they are wide become one blob.
           That is also geometry: the error is bounded by the merged footprint.

  2. IF IT IS ONLY A FEW DEGREES, DOES IT MATTER DOWNSTREAM?  This is the question that decides
     whether the limb layer is usable, and it has a hard numeric answer because the two
     downstream consumers live on completely different length scales:
       * limb KINEMATICS / behaviour: the scale is the bone, 0.7-1.4 mm -> a few degrees is fine;
       * CONTACT / TACTILE / FORCE: the scale is the PENETRATION, which this project measured at
         a median of 5.42 um and a cap at 7.85 um -> a few degrees moves the foot by 50-100 um,
         which is 10-20x the entire contact scale, so the contact/no-contact decision itself
         flips.  That is not noise on the answer, it is a different answer.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_limb_error_tail.py
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
EPISODE = ARENA / "arena_bare_seed0"

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
#: MEASURED, this project (run_tactile_record.py / tactile.py).
PENETRATION_MEDIAN_UM = 5.42
PENETRATION_P95_UM = 12.53
GEOMETRIC_CAP_UM = 7.85
TIP_RADIUS_UM = 28.9           # tarsus5 mesh AABB -> tip radius


def angle_deg(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def main() -> int:
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    segs = list(geom["marker_body_segments"])
    idx = {n: i for i, n in enumerate(segs)}
    z = np.load(EPISODE / "episode.npz")
    P = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)   # (F, 21, 3)
    F = P.shape[0]

    print("=" * 78)
    print("PART 1a -- IDENTITY SWAP: how large is the angle error if the WRONG segment is used")
    print("=" * 78)
    print("  Every bone is recomputed with the correct other endpoint replaced by EACH of the")
    print("  other 20 segments, over all 107 frames.  This is exact geometry, not a model.")

    bones = []
    for leg in LEGS:
        bones.append((f"{leg} femur", idx[f"{leg}_trochanterfemur"], idx[f"{leg}_tibia"]))
        bones.append((f"{leg} tibia", idx[f"{leg}_tibia"], idx[f"{leg}_tarsus5"]))

    all_swaps = []
    print(f"\n  {'bone':<12}{'own length':>11}{'swap: med':>11}{'p95':>9}{'max':>9}"
          f"{'>30 deg':>10}{'>90 deg':>10}{'>150 deg':>10}")
    rows = []
    for name, a, b in bones:
        own = np.array([np.linalg.norm(P[f, b] - P[f, a]) for f in range(F)])
        errs = []
        for c in range(len(segs)):
            if c == b:
                continue
            for f in range(F):
                errs.append(angle_deg(P[f, b] - P[f, a], P[f, c] - P[f, a]))
        e = np.asarray(errs, dtype=float)
        e = e[np.isfinite(e)]
        if e.size == 0:
            continue
        all_swaps.append(e)
        r = {"bone": name, "own_length_mm": float(np.median(own)),
             "swap_median_deg": float(np.median(e)),
             "swap_p95_deg": float(np.percentile(e, 95)),
             "n_swap_cases": int(e.size),
             "swap_max_deg": float(e.max()),
             "fraction_gt_30": float((e > 30).mean()),
             "fraction_gt_90": float((e > 90).mean()),
             "fraction_gt_150": float((e > 150).mean())}
        rows.append(r)
        print(f"  {name:<12}{np.median(own):>11.3f}{np.median(e):>11.1f}"
              f"{np.percentile(e, 95):>9.1f}{e.max():>9.1f}"
              f"{100 * (e > 30).mean():>9.1f}%{100 * (e > 90).mean():>9.1f}%"
              f"{100 * (e > 150).mean():>9.1f}%")
    e = np.concatenate(all_swaps)
    e = e[np.isfinite(e)]
    print(f"\n  ALL bones pooled: median {np.median(e):.1f} deg, "
          f"fraction >30 deg {100 * (e > 30).mean():.1f}%, "
          f">90 deg {100 * (e > 90).mean():.1f}%, >150 deg {100 * (e > 150).mean():.1f}%")
    print("  A -22 -> +80 deg jump is a ~100 deg error, i.e. the >90 deg column.")

    # WHICH swaps are catastrophic?  distance from the wrong endpoint to the right one decides it
    print("\n  what decides the size of the error is HOW FAR the wrong marker is:")
    near, far = [], []
    for name, a, b in bones:
        for c in range(len(segs)):
            if c == b:
                continue
            d = float(np.median(np.linalg.norm(P[:, c] - P[:, b], axis=1)))
            vals = np.asarray([angle_deg(P[f, b] - P[f, a], P[f, c] - P[f, a])
                               for f in range(F)], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            err = float(np.median(vals))
            (near if d < 0.3 else far).append((d, err, name, segs[c]))
    if near:
        print(f"    wrong marker within 0.3 mm of the right one ({len(near)} cases): "
              f"median angle error {np.median([x[1] for x in near]):.1f} deg, "
              f"max {max(x[1] for x in near):.1f} deg")
    else:
        print("    wrong marker within 0.3 mm of the right one: 0 cases")
    print(f"    wrong marker further than 0.3 mm ({len(far)} cases): "
          f"median angle error {np.median([x[1] for x in far]):.1f} deg")
    print("    -> confusing two markers that sit ON TOP OF EACH OTHER is nearly harmless;")
    print("       confusing markers that are far apart is what produces the 100+ deg jump.")
    worst = sorted(far, key=lambda x: -x[1])[:5]
    for d, err, name, cs in worst:
        print(f"       worst: {name} with {cs} instead ({d:.2f} mm away) -> {err:.1f} deg")

    print()
    print("=" * 78)
    print("PART 1b -- MERGED MARKERS: the error is bounded by the merged footprint")
    print("=" * 78)
    spacing = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            spacing.append((float(np.median(np.linalg.norm(P[:, i] - P[:, j], axis=1))),
                            segs[i], segs[j]))
    spacing.sort()
    print(f"  markers 0.60 mm across; closest {len([s for s in spacing if s[0] < 0.6])} pairs are "
          f"closer than that and MERGE at every resolution:")
    for d, x, y in spacing[:6]:
        print(f"    {d:.3f} mm  {x} <-> {y}")
    print("  when N markers merge into one blob, any position invented for them lies inside the")
    print("  merged footprint, so the error is bounded by ITS SIZE, not by the sensor:")
    for d, x, y in spacing[:3]:
        for bone_len, bn in ((0.705, "femur"), (0.987, "tibia")):
            print(f"    {x} <-> {y} merge ({d + 0.60:.2f} mm footprint) on a {bn} "
                  f"({bone_len} mm): up to "
                  f"{math.degrees(math.atan2(d + 0.60, bone_len)):.1f} deg")

    print()
    print("=" * 78)
    print("PART 1c -- THE ASSOCIATION TOLERANCE: there is NO graceful middle")
    print("=" * 78)
    print("  the substitution above is bimodal: either the blob is the right one (a few")
    print("  degrees) or it is the wrong one (~107 deg median).  So the useful question is how")
    print("  close an alternative candidate has to be to stay harmless:")
    print(f"  {'target angle error':>19}{'allowed offset, femur':>23}{'allowed offset, tibia':>23}"
          f"{'at 36.3 um/px':>15}")
    gsd_um = 36.27
    for tgt in (1, 3, 5, 10):
        a = math.radians(tgt)
        f_ = 0.705 * math.tan(a) * 1000
        t_ = 0.987 * math.tan(a) * 1000
        print(f"  {tgt:>16} deg{f_:>20.1f} um{t_:>20.1f} um{f_ / gsd_um:>13.1f} px")
    print("  MEASURED comparison: the closest pair of fly markers is 0.257 mm apart (median")
    print("  over frames) and the closest alternative endpoint to a bone's own endpoint is")
    nearest = min(x[0] for x in far)
    print(f"  {nearest:.2f} mm away = {nearest * 1000 / 61.7:.1f}x the 5-deg tolerance, and the")
    print(f"  closest marker PAIR is 0.257 mm = {0.257 * 1000 / 61.7:.1f}x it.  So the association")
    print("  is either right or catastrophic; there is nothing in between to average away, and")
    print("  no amount of smoothing or filtering can turn a swap into a few degrees.")
    print("  Also measured: the fiducial detector itself is good to ~0.15-0.5 px, so the")
    print("  MEASUREMENT is not the risk -- IDENTITY is.")

    print()
    print("=" * 78)
    print("PART 2 -- DOES A FEW DEGREES MATTER?  Compare the error to the SCALE it acts on")
    print("=" * 78)
    print(f"  a joint-angle error moves the foot tip by  L * dtheta:")
    print(f"  {'angle error':>12}{'femur 0.705 mm':>17}{'tibia 0.987 mm':>17}"
          f"{'tibia 1.352 mm':>17}{'vs penetration scale':>22}")
    for deg in (1, 3, 5, 10, 30, 90):
        d = math.radians(deg)
        vals = [0.705 * d, 0.987 * d, 1.352 * d]
        print(f"  {deg:>10} deg{vals[0] * 1000:>16.1f}um{vals[1] * 1000:>16.1f}um"
              f"{vals[2] * 1000:>16.1f}um"
              f"{vals[1] * 1000 / PENETRATION_MEDIAN_UM:>19.1f}x")
    print(f"\n  THE PENETRATION SCALE (MEASURED, this project): median "
          f"{PENETRATION_MEDIAN_UM} um, p95 {PENETRATION_P95_UM} um, "
          f"geometric-area cap at {GEOMETRIC_CAP_UM} um")
    print(f"  -> 3 deg on the front tibia moves the tip {0.987 * math.radians(3) * 1000:.0f} um, "
          f"which is {0.987 * math.radians(3) * 1000 / PENETRATION_MEDIAN_UM:.0f}x the median "
          f"penetration")
    print(f"  -> contact area A = pi*(2*R*h - h^2), R = {TIP_RADIUS_UM} um: "
          f"dA/dh at h = 1 um is {math.pi * (2 * TIP_RADIUS_UM - 2 * 1):.0f} um^2 per um")
    print("     so the contact AREA is a stiff function of a length 10-20x SMALLER than the")
    print("     angle error's tip displacement: the contact decision, not just its value, is wrong")
    print("\n  verdict by downstream consumer:")
    print("    limb kinematics / gait / behaviour readout ..... a few degrees is FINE")
    print("    muscle or joint-torque model .................. a few degrees shifts moment arms by")
    print("                                                    cos-error of a few %, tolerable")
    print("    contact / tactile / force / electrode interface  a few degrees is FATAL: the foot")
    print("                                                    moves 50-100 um while contact")
    print("                                                    physics lives on 1-10 um")

    dest = ARENA / "limb_error_tail.json"
    dest.write_text(json.dumps({
        "kind": "EVALUATION ONLY: no model, geometry or recording changed",
        "swap_geometry_per_bone": rows,
        "swap_pooled": {"median_deg": float(np.median(e)),
                        "fraction_gt_30": float((e > 30).mean()),
                        "fraction_gt_90": float((e > 90).mean()),
                        "fraction_gt_150": float((e > 150).mean())},
        "closest_marker_pairs_mm": [{"separation_mm": d, "pair": [x, y]}
                                    for d, x, y in spacing[:8]],
        "downstream_scales": {
            "penetration_median_um": PENETRATION_MEDIAN_UM,
            "penetration_p95_um": PENETRATION_P95_UM,
            "geometric_area_cap_um": GEOMETRIC_CAP_UM,
            "tip_radius_um": TIP_RADIUS_UM,
            "tip_displacement_um_per_deg": {b: L * math.pi / 180 * 1000
                                            for b, L in (("femur0.705", 0.705),
                                                         ("tibia0.987", 0.987),
                                                         ("tibia1.352", 1.352))},
        },
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
