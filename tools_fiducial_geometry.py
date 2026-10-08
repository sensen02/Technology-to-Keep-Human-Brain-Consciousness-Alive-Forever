#!/usr/bin/env python3
"""HOW MANY FIDUCIALS, SPREAD HOW WIDE, ARE NEEDED TO CALIBRATE THE ARRAY?

THE QUESTION THIS ANSWERS.  The bundle adjustment's synthetic self-test fails on the arena's
actual constellation and the failure is informative rather than mysterious: with only six
markers in a near-planar arrangement spanning 2.6 mm laterally and 1 mm vertically, a fit
that leaves BOTH the points and the cameras free reaches the injected noise level in
projection (0.36 px) while putting the points 1.03 mm from the truth.  That is a GAUGE
freedom: a rigid motion can be shuffled between the points and the cameras without changing
any projection.  Anchoring the points removes it but then the six near-coplanar points
cannot determine six camera poses, and the recovered rotations are off by 17-30 degrees.

So the question is a DESIGN question -- how many fiducials and how widely spread -- and it
is cheaper to answer it in simulation than to re-record the arena repeatedly.

WHAT IS SWEPT
    n_points       6, 12, 24, 48
    lateral spread 5 mm (the fly's own walking area), 30 mm, 60 mm
    vertical span  2 mm (nearly planar, what the arena has) versus 20 mm (a real 3D volume)
and for each combination: does the fit recover the injected 3-degree pose perturbation, and
how far off are the recovered points?

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_fiducial_geometry.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_bundle import (bundle_adjust_scipy, project,  # noqa: E402
                             rotation_from_rotvec)


def make_cameras(n_cam: int, radius: float, height: float, res=(200, 260)):
    """A ring of cameras, all aimed at the origin.  Columns of R are the camera's axes."""
    K = np.array([[214.45, 0, res[1] / 2.0], [0, 214.45, res[0] / 2.0], [0, 0, 1.0]])
    cams = []
    for k in range(n_cam):
        az = 2 * math.pi * k / n_cam
        pos = np.array([radius * math.cos(az), radius * math.sin(az), height])
        f = -pos / np.linalg.norm(pos)                 # forward, toward the origin
        right = np.cross(f, [0, 0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        R = np.stack([right, up, np.cross(right, up)], axis=1)   # columns = axes
        cams.append({"R": R, "pos_mm": pos, "K": K})
    return cams


def trial(n_pts: int, lateral_mm: float, vertical_mm: float, n_cam: int = 6,
          radius: float = 60.0, height: float = 50.0, noise_px: float = 0.5,
          perturb_deg: float = 3.0, perturb_mm: float = 0.5, seed: int = 0,
          free_points: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    cams = make_cameras(n_cam, radius, height)
    # the camera ring aims at the ORIGIN and the constellation is built with z from 0.5 up,
    # so it is shifted to be centred on the aim point before projecting
    Ks = cams[0]["K"]
    H, W = int(Ks[1, 2] * 2), int(Ks[0, 2] * 2)
    # fiducials in a box centred on the origin: lateral spread x vertical span
    # A REGULAR THREE-TIER LATTICE, not uniform random: measured, a uniform random cloud in
    # a thin box behaves like a nearly-planar set, and the arena's real constellation is
    # deliberately tiered (6 low + 3 mid + 3 high).  The sweep should match the thing being
    # designed.
    X = []
    tiers = max(1, min(6, n_pts // 3 if n_pts >= 3 else 1))
    for t in range(tiers):
        z = 0.5 + t * (max(vertical_mm, 1.0) / max(1, tiers - 1) if tiers > 1 else 0.0)
        ring = max(3, n_pts // tiers)
        for k in range(ring):
            ang = 2 * math.pi * k / ring + t * math.pi / ring
            X.append([lateral_mm / 2 * math.cos(ang), lateral_mm / 2 * math.sin(ang), z])
    X = np.array(X[:n_pts])
    X[:, 2] -= max(vertical_mm, 1.0) / 2.0
    obs, truth_px = [], []
    for ci, c in enumerate(cams):
        Rc = c["R"].T.copy(); Rc[2, :] = -Rc[2, :]
        t = -Rc @ c["pos_mm"]
        for mj in range(n_pts):
            got = project(c["K"], c["R"], t, X[mj])
            if got is None:
                continue
            (r, cc, _d), _ = got
            if not (0 <= r < H and 0 <= cc < W):
                continue
            obs.append((ci, mj, r + rng.normal(0, noise_px), cc + rng.normal(0, noise_px)))
            truth_px.append((ci, mj, r, cc))
    if len(obs) < 12:
        return {"n_obs": len(obs), "usable": False}
    # perturb the poses and the points, then try to recover
    cams_p = []
    for c in cams:
        w = rng.normal(0, math.radians(perturb_deg), 3)
        cams_p.append({"R": c["R"] @ rotation_from_rotvec(w).T,
                       "pos_mm": c["pos_mm"] + rng.normal(0, perturb_mm, 3),
                       "K": c["K"]})
    X0 = X + rng.normal(0, 1.0, X.shape)
    # MULTI-START: the single-start fit was measured to land in a local minimum (more
    # fiducials made it WORSE, which a better-conditioned problem cannot do), so several
    # starts are tried and the best kept.  That is standard for calibration and cheap here.
    best_rep = None
    for si in range(3):
        xs = X + rng.normal(0, 1.0 + 2.0 * si, X.shape)
        r_try = bundle_adjust_scipy(obs, cams_p, xs, free_points=free_points,
                                    free_principal_point=False, max_nfev=2000)
        if r_try.get("ok") and (best_rep is None or r_try["rms_px"] < best_rep["rms_px"]):
            best_rep = r_try
    rep = best_rep if best_rep is not None else {"ok": False, "reason": "no start solved"}
    if not rep.get("ok"):
        return {"n_obs": len(obs), "usable": False, "reason": rep.get("reason")}
    X_rec = np.array(rep["X_mm"])
    if free_points:
        # compare after removing the gauge: a rigid fit of the recovered points to the truth
        from run_arena_reconstruct import fit_rigid
        fit = fit_rigid(X_rec, X)
        pt_err = fit["residual_mm"]
    else:
        pt_err = float(np.sqrt(np.mean(np.sum((X_rec - X) ** 2, axis=1))))
    rot = [float(np.linalg.norm(c["drotvec_deg"])) for c in rep["camera_deltas"]]
    return {"n_obs": len(obs), "usable": True,
            "rms_px": rep["rms_px"], "point_err_mm_after_gauge": pt_err,
            "max_rot_correction_deg": max(rot),
            "mean_rot_correction_deg": float(np.mean(rot))}


def main() -> int:
    table = []
    print(f"{'n_pts':>6} {'lateral':>8} {'vertical':>9} {'n_obs':>6} {'rms_px':>8} "
          f"{'pt_err_mm':>10} {'max_rot_deg':>12}")
    for n_pts in (6, 12, 24, 48):
        for lateral in (5.0, 30.0, 60.0):
            for vertical in (2.0, 20.0):
                r = trial(n_pts, lateral, vertical)
                row = {"n_points": n_pts, "lateral_mm": lateral, "vertical_mm": vertical, **r}
                table.append(row)
                if r.get("usable"):
                    print(f"{n_pts:6d} {lateral:8.0f} {vertical:9.0f} {r['n_obs']:6d} "
                          f"{r['rms_px']:8.3f} {r['point_err_mm_after_gauge']:10.4f} "
                          f"{r['max_rot_correction_deg']:12.2f}")
                else:
                    print(f"{n_pts:6d} {lateral:8.0f} {vertical:9.0f} "
                          f"{r.get('n_obs', 0):6d}   unusable: {r.get('reason', '')}")
    dest = HERE / "outputs" / "arena" / "fiducial_geometry.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"trials": table,
                                "note": ("point_err_mm_after_gauge removes the rigid "
                                         "freedom before comparing, because with coplanar "
                                         "points a fit can be perfect in projection while "
                                         "the points slide"),
                                "injected": {"rotation_deg": 3.0, "translation_mm": 0.5,
                                             "pixel_noise_px": 0.5}},
                               indent=2, sort_keys=True))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
