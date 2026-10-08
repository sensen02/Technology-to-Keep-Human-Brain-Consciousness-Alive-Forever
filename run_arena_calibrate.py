#!/usr/bin/env python3
"""Solve ONLY the per-camera 2D offset, with the 3D poses and the marker positions fixed.

WHY THIS PARAMETERISATION AND NOT A FULL BUNDLE ADJUSTMENT.  Every ingredient is already
verified except one:

  * the 3D camera poses are correct -- the marker image displacements match the analytic ones
    with cos = 1.000 in all three world directions (`tools_axis_probe2.py`);
  * the marker positions are exact -- read back from the compiled model and identically equal
    to the declared positions;
  * the FOCAL LENGTH is now correct -- the renderer's aspect was swapped, giving
    f = (160/2)/tan(25 deg) = 171.6 px where the model had been using 214.5, and with that
    fixed the association residual is a MEDIAN of 0.7 px.

What remains is a constant per-camera 2D offset whose size (about 0.9 mm of world error, and
CONSTANT across 60 frames to 2 %) is the signature of a fixed calibration term rather than
noise.  A 2-parameter-per-camera fit against 12 markers is 12x overdetermined and extremely
well conditioned -- unlike the 12-parameter full bundle adjustment, which was measured to be
ill-conditioned on this constellation.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python run_arena_calibrate.py \\
        --episode outputs/arena/arena_bare_seed0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/arena/arena_bare_seed0")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--gate-px", type=float, default=6.0)
    args = ap.parse_args()

    from PIL import Image
    from scipy import ndimage
    from scipy.optimize import least_squares
    from run_arena_reconstruct import load_cameras_for_episode

    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    decl = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    d = HERE / args.episode
    cams, src = load_cameras_for_episode(geom, d, HERE / "outputs" / "arena"
                                         / "camera_poses.json")

    # ---- one set of observations per camera, from several frames averaged -------------
    # Averaging over frames is legitimate here because EVERYTHING is static: the markers do not
    # move, so the per-frame scatter measures the detector's noise and the mean is the estimate.
    obs = {}
    for cam in cams:
        frames = sorted((d / "frames" / cam.name).glob("*.png"))
        proj = cam.project(decl)
        acc = {k: [] for k in range(len(decl))}
        for fi in range(min(args.frames, len(frames))):
            img = np.asarray(Image.open(frames[fi]).convert("RGB"),
                             dtype=float).mean(axis=2)
            lab, n = ndimage.label(img >= 240.0)
            blobs = []
            for i in range(1, n + 1):
                yy, xx = np.nonzero(lab == i)
                if len(yy) >= 3:
                    blobs.append((float(yy.mean()), float(xx.mean()), int(len(yy))))
            if not blobs:
                continue
            B = np.array([[b[0], b[1]] for b in blobs])
            for k in range(len(decl)):
                dd = np.linalg.norm(B - proj[k], axis=1)
                j = int(np.argmin(dd))
                if dd[j] <= args.gate_px:
                    acc[k].append(B[j])
        obs[cam.name] = {k: (np.mean(v, axis=0) if v else None, len(v))
                         for k, v in acc.items()}

    def residual(p):
        """p has 2 offsets per camera; the model is projection + that offset."""
        out = []
        for ci, cam in enumerate(cams):
            dx, dy = p[2 * ci], p[2 * ci + 1]
            proj = cam.project(decl)
            for k in range(len(decl)):
                m, n_used = obs[cam.name][k]
                if m is None:
                    continue
                pr = proj[k]
                # (col, row) parameterisation to match the pixel axes explicitly
                out.append((pr[1] + dx) - m[1])
                out.append((pr[0] + dy) - m[0])
        return np.asarray(out, dtype=float)

    p0 = np.zeros(2 * len(cams))
    before = residual(p0)
    sol = least_squares(residual, p0, loss="soft_l1", f_scale=3.0, x_scale="jac")
    after = sol.fun
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    report = {
        "episode": d.name, "pose_source": src,
        "frames_used": min(args.frames, len(next(iter(obs.values())) and
                                            (d / "frames" / cams[0].name).glob("*.png").__length_hint__() or 0))
        if False else args.frames,
        "gate_px": args.gate_px,
        "rms_px_before": rms(before), "rms_px_after": rms(after),
        "median_abs_before": float(np.median(np.abs(before))),
        "median_abs_after": float(np.median(np.abs(after))),
        "n_residuals": int(len(after)),
        "per_camera": [],
    }
    for ci, cam in enumerate(cams):
        used = [k for k in range(len(decl)) if obs[cam.name][k][0] is not None]
        per_before = []
        per_after = []
        proj = cam.project(decl)
        for k in used:
            m, _n = obs[cam.name][k]
            per_before.append(float(np.hypot(proj[k][1] - m[1], proj[k][0] - m[0])))
            per_after.append(float(np.hypot(proj[k][1] + sol.x[2 * ci] - m[1],
                                            proj[k][0] + sol.x[2 * ci + 1] - m[0])))
        report["per_camera"].append({
            "name": cam.name, "n_markers_used": len(used),
            "offset_px": [float(sol.x[2 * ci]), float(sol.x[2 * ci + 1])],
            "before_mean_px": float(np.mean(per_before)) if per_before else None,
            "after_mean_px": float(np.mean(per_after)) if per_after else None,
            "after_median_px": float(np.median(per_after)) if per_after else None,
            "after_max_px": float(np.max(per_after)) if per_after else None,
        })
    dest = d / "camera_offsets.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"{d.name}: rms_px before {report['rms_px_before']:.3f} -> after "
          f"{report['rms_px_after']:.3f}  (median |r| {report['median_abs_before']:.2f} -> "
          f"{report['median_abs_after']:.2f})")
    for c in report["per_camera"]:
        print(f"   {c['name']}: offset ({c['offset_px'][0]:+7.2f},{c['offset_px'][1]:+7.2f}) px, "
              f"{c['n_markers_used']:2d} markers, mean {c['before_mean_px']:6.2f} -> "
              f"{c['after_mean_px']:5.2f} px")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
