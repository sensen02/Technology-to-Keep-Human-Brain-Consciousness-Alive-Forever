#!/usr/bin/env python3
"""CAN A LIMB JOINT ANGLE BE RESOLVED?  Monte Carlo on the actual camera geometry.

THE QUESTION, ANSWERED WITH NUMBERS RATHER THAN ARITHMETIC.  A joint angle is the angle of a
bone, and a bone direction comes from two triangulated endpoints.  For a bone of length L whose
endpoints each carry a position error sigma, the direction error is of order
``sqrt(2) * sigma / L``, so what matters is the POSITION ERROR RELATIVE TO THE BONE LENGTH.
The bone lengths are MEASURED from the model (femur 0.705 mm, tibia 0.961 mm), not assumed,
because an earlier estimate of mine (0.35 mm for a femur) was wrong by a factor of two and would
have made the answer pessimistic by the same factor.

For each camera configuration this places the two endpoints of a real bone in the arena,
projects them, adds detector noise, triangulates, and measures the ANGLE error.  It also reports
the fraction of endpoints that fall outside the frame, because a leg that leaves the frame is not
measured at all.

TWO ANSWERS ARE PRINTED, AND THE FLATTERING ONE IS NOT THE ONE TO QUOTE.

  1. NOISE-ONLY.  Detector noise 0.5 px, everything else perfect.  This is an upper bound on
     quality, not a prediction: at the current rig it predicts a 3D point error of ~0.115 mm
     while the MEASURED world-frame error on the fiducial ring is 0.883 mm, 7.6x larger.

  2. CALIBRATED AGAINST THE MEASUREMENT.  The pixel noise is scaled until the predicted 3D error
     equals the measured one, and the angle error is read at that noise level.  Two scalings are
     shown because the truth is unknown: ``fixed`` keeps the measured 0.883 mm absolute at every
     resolution (the honest pessimistic case -- a systematic error does NOT disappear when you
     buy a bigger sensor), and ``res`` scales it as 200/rows (the optimistic case, which assumes
     the residual is pure detector noise).

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_limb_angle_precision.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import PinholeCamera, triangulate  # noqa: E402

#: MEASURED from the compiled model's joint body positions in the neutral walking pose.
FEMUR_MM = 0.705          # shortest femur (lf, rf)
TIBIA_MM = 0.961          # shortest tibia (lf)
TARSUS_CHAIN_MM = 0.961   # tibia origin to tarsus5 origin

#: MEASURED, this project, 60 frames of the six-camera fiducial reconstruction.
MEASURED_3D_MM = 0.883
MEASURED_3D_ROWS = 200

CONFIGS = [
    ("current: r=22 h=22 rows=200", 22.0, 22.0, 200),
    ("r=22 h=22 rows=640", 22.0, 22.0, 640),
    ("r=10 h=10 rows=480", 10.0, 10.0, 480),
    ("r=8 h=8 rows=480", 8.0, 8.0, 480),
    ("r=6 h=6 rows=480", 6.0, 6.0, 480),
    ("r=5 h=5 rows=480", 5.0, 5.0, 480),
    ("r=4 h=4 rows=480", 4.0, 4.0, 480),
    ("r=3 h=3 rows=480", 3.0, 3.0, 480),
    ("r=3 h=3 rows=640", 3.0, 3.0, 640),
]


def ring(radius, height, rows, fovy=50.0, n=6, aspect=1.25):
    W = int(round(rows * aspect))
    cams = []
    for k in range(n):
        az = 2 * math.pi * k / n
        pos = np.array([radius * math.cos(az), radius * math.sin(az), height])
        f = -pos / np.linalg.norm(pos)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        cams.append(PinholeCamera({"name": f"c{k}", "pos_mm": list(pos),
                                   "xyaxes": list(right) + list(up),
                                   "fovy_deg": fovy, "resolution_px": [rows, W]}))
    return cams


def angle_error(cams, bone_mm, noise_px=0.5, n_trials=600, seed=0,
                bone_centre=(0.0, 0.0, 0.6), spread_deg=40.0):
    """Angle error of one bone, as the endpoint-to-endpoint direction, in degrees.

    The bone is placed near the arena centre and given random orientations within
    ``spread_deg`` of horizontal, which is where a walking leg's femur mostly sits.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(bone_centre, dtype=float)
    errs = []
    out_of_frame = 0
    for _ in range(n_trials):
        th = rng.uniform(0, 2 * math.pi)
        ph = np.radians(rng.uniform(-spread_deg, spread_deg))
        u = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th),
                      math.sin(ph)])
        p0 = c - 0.5 * bone_mm * u
        p1 = c + 0.5 * bone_mm * u
        proj = []
        ok = True
        for cam in cams:
            pr = cam.project(np.stack([p0, p1]))
            if not np.isfinite(pr).all():
                ok = False
                break
            if ((pr[:, 0] < 0) | (pr[:, 0] >= cam.H) | (pr[:, 1] < 0) | (pr[:, 1] >= cam.W)).any():
                out_of_frame += 1
            proj.append(pr + rng.normal(0, noise_px, pr.shape))
        if not ok:
            continue
        rec = []
        for k in range(2):
            X = triangulate(cams, [(proj[i][k][0], proj[i][k][1]) for i in range(len(cams))])
            rec.append(X)
        if any(r is None for r in rec):
            continue
        v_true = p1 - p0
        v_rec = np.asarray(rec[1]) - np.asarray(rec[0])
        ct = float(np.dot(v_true, v_rec) / (np.linalg.norm(v_true) * np.linalg.norm(v_rec)))
        errs.append(math.degrees(math.acos(max(-1.0, min(1.0, ct)))))
    e = np.asarray(errs)
    return {"n": int(e.size), "median_deg": float(np.median(e)) if e.size else None,
            "p95_deg": float(np.percentile(e, 95)) if e.size else None,
            "out_of_frame_fraction": out_of_frame / max(1, n_trials * len(cams))}


def _point_error(cams, noise_px, n_trials=150, seed=3, centre=(0.0, 0.0, 0.6)):
    """3D point error at the arena centre for a given pixel noise, in mm (median)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(centre, dtype=float)
    e = []
    for _ in range(n_trials):
        X = c + rng.normal(0, 0.05, 3)
        px = []
        for cam in cams:
            pr = cam.project(X)
            px.append(pr + rng.normal(0, noise_px, pr.shape))
        rec = triangulate(cams, [(px[i][0][0], px[i][0][1]) for i in range(len(cams))])
        if rec is not None:
            e.append(float(np.linalg.norm(np.asarray(rec) - X)))
    return float(np.median(e)) if e else float("inf")


def noise_for_target(cams, target_mm, lo=0.005, hi=400.0, iters=26):
    """Bisect the pixel noise until the predicted 3D error reaches ``target_mm``."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _point_error(cams, mid) < target_mm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    print(f"bone lengths (MEASURED): femur {FEMUR_MM} mm, tibia {TIBIA_MM} mm")
    print("detector noise assumed: 0.5 px (the measured marker centroid noise)")
    print(f"measured 3D world-frame error: {MEASURED_3D_MM} mm at rows={MEASURED_3D_ROWS}\n")

    part1, part2, part3 = [], [], []

    print("=== 1. NOISE-ONLY (optimistic upper bound on quality) ===")
    print(f"{'configuration':<30} {'bone':<8} {'noise px':>9} {'3D err mm':>10} "
          f"{'median deg':>11} {'p95 deg':>9} {'out-of-frame':>13}")
    for label, r, h, rows in CONFIGS:
        cams = ring(r, h, rows)
        p3d = _point_error(cams, 0.5)
        for bone_name, L in (("femur", FEMUR_MM), ("tibia", TIBIA_MM)):
            res = angle_error(cams, L, noise_px=0.5)
            part1.append({"config": label, "bone": bone_name, "bone_mm": L,
                          "radius_mm": r, "height_mm": h, "rows": rows,
                          "scene": "noise_only", "noise_px": 0.5,
                          "predicted_3d_mm": p3d, **res})
            print(f"{label:<30} {bone_name:<8} {0.5:>9.3f} {p3d:>10.4f} "
                  f"{res['median_deg']:>11.3f} {res['p95_deg']:>9.3f} "
                  f"{res['out_of_frame_fraction']:>13.3f}")

    for scene, scale in (("calibrated_fixed", "fixed"), ("calibrated_scaled", "res")):
        print(f"\n=== 2. CALIBRATED (scene={scene}: 3D error "
              f"{'constant 0.883 mm' if scale == 'fixed' else 'scaled as 200/rows'}) ===")
        print(f"{'configuration':<30} {'bone':<8} {'noise px':>9} {'3D err mm':>10} "
              f"{'median deg':>11} {'p95 deg':>9}")
        for label, r, h, rows in CONFIGS:
            cams = ring(r, h, rows)
            target = MEASURED_3D_MM if scale == "fixed" else MEASURED_3D_MM * MEASURED_3D_ROWS / rows
            npx = noise_for_target(cams, target)
            for bone_name, L in (("femur", FEMUR_MM), ("tibia", TIBIA_MM)):
                res = angle_error(cams, L, noise_px=npx)
                part2.append({"config": label, "bone": bone_name, "bone_mm": L,
                              "radius_mm": r, "height_mm": h, "rows": rows,
                              "scene": scene, "noise_px": npx,
                              "predicted_3d_mm": target, **res})
                print(f"{label:<30} {bone_name:<8} {npx:>9.3f} {target:>10.4f} "
                      f"{res['median_deg']:>11.3f} {res['p95_deg']:>9.3f}")

    # What a SUFFICIENT rig would need: the bone half-angle error set by a target angle error.
    print("\n=== 3. WHAT THE RIG WOULD HAVE TO BE (calibrated_fixed target) ===")
    print(f"{'target angle err':>17} {'allowed 3D err mm':>18} {'needed rows @ r=3':>18}")
    for tgt in (1.0, 2.0, 5.0, 10.0, 30.0):
        allowed = tgt * math.radians(1.0) * FEMUR_MM / math.sqrt(2.0)
        cams3 = ring(3.0, 3.0, 480)
        need = noise_for_target(cams3, allowed)
        # 3D error at fixed noise falls as 1/rows, so shrinking the error by 0.5/need
        # costs that factor in rows.
        rows_needed = 480 * 0.5 / need if need > 0 else float("nan")
        part3.append({"target_angle_deg": tgt, "allowed_3d_mm": allowed,
                      "noise_px_at_r3_480": need, "rows_needed_at_r3": rows_needed})
        print(f"{tgt:>17.1f} {allowed:>18.4f} {rows_needed:>18.0f}")

    dest = HERE / "outputs" / "arena" / "limb_angle_precision.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"bone_lengths_mm": {"femur": FEMUR_MM, "tibia": TIBIA_MM,
                             "tarsus_chain": TARSUS_CHAIN_MM},
         "measured_3d_mm": MEASURED_3D_MM, "measured_3d_rows": MEASURED_3D_ROWS,
         "trials": 600, "noise_only": part1, "calibrated": part2,
         "rig_requirement": part3}, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
