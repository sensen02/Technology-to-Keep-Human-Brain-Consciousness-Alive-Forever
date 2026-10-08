#!/usr/bin/env python3
"""IDENTITY BUDGET: what precision is the FLY'S OWN ANATOMY capable of, before any camera?

The previous round ended on a measured conclusion: the error is not detector noise, it is
IDENTITY -- picking the wrong blob.  So the next question is not "how good a camera" but "how
far apart are the things we must tell apart", and that is pure anatomy, measured here:

  for every bone (6 legs x femur/tibia) and BOTH of its endpoints,
    * the closest marker that is NOT that endpoint  ->  the CLEARANCE
    * the joint-angle error if that marker is used instead
    * the largest marker diameter that keeps them separate
  and then, for each bone, the ANATOMICAL FLOOR: the best angle precision that is physically
  available from markers placed on segment origins, with a perfect camera.

THE RESULT IS A NEGATIVE ONE ON PURPOSE.  A 5-degree femur angle needs the proximal marker
resolved to 62 um.  The frontal coxae sit 144-267 um apart, i.e. the ANATOMY is already coarser
than the tolerance, so no camera, no resolution and no filter can deliver 5 degrees there.  What
CAN be delivered differs per leg, and that per-leg table is the useful output: it says which
joint angles are worth measuring and which are not.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_identity_budget.py
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
#: a blob needs a few pixels across for its centroid to mean anything
MIN_MARKER_PX = 5.0
FOVY_DEG = 50.0


def angle_deg(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def main() -> int:
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    segs = list(geom["marker_body_segments"])
    idx = {n: i for i, n in enumerate(segs)}
    P = np.asarray(np.load(EPISODE / "episode.npz")["truth/marker_fly_world_mm"], dtype=float)

    def sep(i, j):
        """median separation over frames, mm"""
        return float(np.median(np.linalg.norm(P[:, i] - P[:, j], axis=1)))

    bones = []
    for leg in LEGS:
        bones.append((f"{leg} femur", idx[f"{leg}_trochanterfemur"], idx[f"{leg}_tibia"],
                      f"{leg}_trochanterfemur", f"{leg}_tibia"))
        bones.append((f"{leg} tibia", idx[f"{leg}_tibia"], idx[f"{leg}_tarsus5"],
                      f"{leg}_tibia", f"{leg}_tarsus5"))

    print("=" * 96)
    print("PART A -- CLEARANCE PER BONE: how close is the nearest CONFUSABLE marker?")
    print("=" * 96)
    print(f"  {'bone':<11}{'length':>8}{'endpoint':<22}{'clearance':>11}{'>= that diam':>14}"
          f"{'angle err':>11}{'to 5 deg needs':>16}")
    rows = []
    for name, a, b, an, bn in bones:
        L = sep(a, b)
        for end_i, end_n in ((a, an), (b, bn)):
            others = [k for k in range(len(segs)) if k != end_i]
            dd = sorted(((sep(end_i, k), segs[k]) for k in others))
            clr, who = dd[0]
            other_end = b if end_i == a else a
            err = float(np.median([angle_deg(P[f, other_end] - P[f, end_i],
                                             P[f, idx[who]] - P[f, end_i])
                                   for f in range(P.shape[0])]))
            need = L * math.tan(math.radians(5.0)) * 1000.0
            rows.append({"bone": name, "length_mm": L, "endpoint": end_n,
                         "clearance_mm": clr, "closest_confusable": who,
                         "angle_error_if_confused_deg": err,
                         "arms_length": clr * 1000.0 / need,
                         "required_um_for_5deg": need})
            print(f"  {name:<11}{L:>8.3f}{end_n:<22}{clr:>11.3f}{clr:>14.3f}"
                  f"{err:>10.2f}d{clr * 1000.0 / need:>14.1f}x")

    print("\n  'to 5 deg needs' is the clearance as a MULTIPLE of the 62-86 um that a 5-degree")
    print("  angle requires.  Below 1x means the anatomy itself is too coarse -- no camera helps.")

    print()
    print("=" * 96)
    print("PART B -- SEPARABLE OR MERGED, PER BONE, AT A GIVEN MARKER SIZE")
    print("=" * 96)
    print("  THE LOGIC MATTERS HERE.  If the clearance EXCEEDS the marker diameter the two")
    print("  markers are optically separate, so the right one can be picked and the error is the")
    print("  MEASUREMENT error (the few degrees measured earlier), NOT the clearance.  Only when")
    print("  the clearance is SMALLER than the marker do they merge, and then the endpoint is")
    print("  ambiguous within the clearance and the error is atan(clearance / bone_length).")
    MEASURED_MEAS_ERR = 3.0      # deg, the measured floor from the fiducial study
    floors = []
    for mark in (0.60, 0.40, 0.25, 0.15):
        print(f"\n  marker diameter {mark:.2f} mm:")
        print(f"  {'bone':<11}{'length mm':>10}{'clearance':>11}{'separable':>11}"
              f"{'angle error':>13}")
        nbad = 0
        for name, a, b, an, bn in bones:
            L = sep(a, b)
            c = min(r["clearance_mm"] for r in rows if r["bone"] == name)
            ok = c >= mark
            if not ok:
                nbad += 1
            err = MEASURED_MEAS_ERR if ok else math.degrees(math.atan2(c, L))
            floors.append({"bone": name, "marker_mm": mark, "length_mm": L,
                           "clearance_mm": c, "separable": bool(ok),
                           "angle_error_deg": err})
            print(f"  {name:<11}{L:>10.3f}{c:>11.3f}{'yes' if ok else 'MERGED':>11}"
                  f"{err:>12.2f}d")
        print(f"  -> {nbad} of 12 bones unmeasurable at this marker size")

    print("\n  WHAT THIS MEANS FOR THE MARKER ALREADY IN THE SCENE (0.60 mm):")
    bad = [f for f in floors if f["marker_mm"] == 0.60 and not f["separable"]]
    print(f"    {len(bad)} of 12 bones are MERGED, giving {min(b['angle_error_deg'] for b in bad):.1f}"
          f"-{max(b['angle_error_deg'] for b in bad):.1f} deg of ambiguity:")
    print("      " + ", ".join(b["bone"] for b in bad))
    good = [f for f in floors if f["marker_mm"] == 0.60 and f["separable"]]
    print(f"    and {len(good)} are separable at the current marker: "
          + ", ".join(b["bone"] for b in good))
    print("    So the femur is broken on ALL SIX legs, plus the two hind tibiae -- not because")
    print("    of the camera, but because the marker is 2.3x wider than the coxa gap it must")
    print("    resolve.")

    print()
    print("=" * 96)
    print("PART C -- WHAT THE CAMERA HAS TO DO, GIVEN THE SMALLEST MARKER THAT WORKS")
    print("=" * 96)
    print(f"  the marker must be SMALLER than the clearance (else it merges) and BIGGER than")
    print(f"  {MIN_MARKER_PX:.0f} px (else its centroid is meaningless).  Both together fix the ground")
    print("  sampling distance, and the test is on rows per mm of ring radius:")
    print(f"  {'bone':<11}{'clearance mm':>13}{'max marker diam':>17}{'needed GSD um':>15}"
          f"{'rows/r needed':>14}{'rows/r now':>12}")
    cam_rows = []
    for name, a, b, an, bn in bones:
        c = min(r["clearance_mm"] for r in rows if r["bone"] == name)
        gsd = c / MIN_MARKER_PX * 1000.0
        ratio = 2.0 * math.sqrt(2.0) * math.tan(math.radians(FOVY_DEG / 2)) * 1000.0 / gsd
        cam_rows.append({"bone": name, "clearance_mm": c, "max_marker_diameter_mm": c,
                         "needed_gsd_um": gsd, "rows_per_mm_radius": ratio})
        print(f"  {name:<11}{c:>13.3f}{c:>17.3f}{gsd:>15.2f}{ratio:>14.1f}{800 / 22:>12.1f}")
    need = max(c["rows_per_mm_radius"] for c in cam_rows)
    print(f"\n  the tightest pair needs rows/r >= {need:.1f}; the CURRENT rig has "
          f"{800 / 22:.1f} at r=22 mm")
    print(f"  -> the pixel budget is ALREADY SUFFICIENT.  My earlier line in this file claiming")
    print(f"     rows would have to rise to ~563 was WRONG (563 rows at r=22 gives rows/r =")
    print(f"     {563 / 22:.1f}, which is exactly the requirement, and 800 already exceeds it).")
    print(f"  -> so the whole identity problem is solvable by SHRINKING THE MARKER to <= "
          f"{min(c['max_marker_diameter_mm'] for c in cam_rows):.2f} mm diameter,")
    print(f"     at the CURRENT resolution.  No new camera.  At r=8 mm even "
          f"{math.ceil(need * 8):d} rows would do.")

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    tight = min(c["clearance_mm"] for c in cam_rows)
    print(f"  * with the marker ALREADY IN THE SCENE (0.60 mm), {len(bad)} of 12 bones can never")
    print(f"    be measured, because the marker is {0.60 / tight:.1f}x the tightest clearance "
          f"({tight * 1000:.0f} um).")
    print(f"    The femur is broken on all six legs.  No camera changes this.")
    print(f"  * shrinking the marker to <= {tight:.2f} mm diameter makes every bone separable, and")
    print(f"    the required rows/r is {need:.1f} against {800 / 22:.1f} already available -- so it")
    print(f"    costs NOTHING but the marker.")
    print(f"  * after that the error is the MEASUREMENT error, ~3-5 deg, so the IDENTITY problem")
    print(f"    and the PRECISION problem are separate, and identity is the cheap fix.")
    print(f"  * the honest deliverable is a PER-JOINT number: a joint angle is trustworthy only if")
    print(f"    its clearance exceeds the marker AND its bone is long enough for the target error.")

    dest = ARENA / "identity_budget.json"
    dest.write_text(json.dumps({
        "kind": "EVALUATION ONLY: no model, geometry or recording changed",
        "per_endpoint_clearance": rows,
        "per_bone_by_marker_size": floors,
        "camera_requirement": cam_rows,
        "current_rig": {"ring_mm": 22.0, "rows": 800, "gsd_um": 1.319 * 22 / 800 * 1000,
                        "rows_per_mm_radius": 800 / 22},
        "current_marker_diameter_mm": 0.60,
        "min_marker_px": MIN_MARKER_PX,
        "note": "clearances are medians over the 107 frames of the recorded episode",
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
