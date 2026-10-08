#!/usr/bin/env python3
"""On a REAL recorded frame: every bright blob, the projections, and the optimal assignment.

THIS REPLACES GUESSING ABOUT THRESHOLDS.  It prints every blob with its area and peak grey, the
fiducial projections, and the one-to-one assignment that minimises total distance, so the
question "are the markers where the camera model says" is answered by one table.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_blob_vs_proj.py \\
        --episode outputs/arena/arena_bare_seed0 --camera cam0
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def blobs_at(img, thr, min_area):
    from scipy import ndimage
    lab, n = ndimage.label(img >= thr)
    out = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(yy) >= min_area:
            out.append((float(yy.mean()), float(xx.mean()), int(len(yy)),
                        float(img[yy, xx].max())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/arena/arena_bare_seed0")
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--thr", type=float, default=245.0)
    ap.add_argument("--min-area", type=int, default=3)
    args = ap.parse_args()

    from PIL import Image
    from run_arena_reconstruct import load_cameras
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    decl = np.array([m["pos_mm"] for m in geom["markers"]["markers"]])
    cam = next(c for c in load_cameras(geom) if c.name == args.camera)
    d = HERE / args.episode
    frames = sorted((d / "frames" / args.camera).glob("*.png"))
    img = np.asarray(Image.open(frames[args.frame]).convert("RGB"),
                     dtype=float).mean(axis=2)
    H, W = img.shape
    print(f"{args.camera} frame {args.frame}: {W}x{H}  model {cam.W}x{cam.H}  "
          f"principal ({cam.W/2:.0f},{cam.H/2:.0f})  f={cam.fx:.1f}")
    b = blobs_at(img, args.thr, args.min_area)
    print(f"blobs at grey >= {args.thr:.0f} with area >= {args.min_area}: {len(b)}")
    for q in sorted(b, key=lambda q: -q[3]):
        print(f"   row {q[0]:6.1f} col {q[1]:6.1f} area {q[2]:4d} peak {q[3]:5.1f}")
    proj = cam.project(decl)
    print(f"projections ({len(proj)}): {[(round(float(p[0]),1), round(float(p[1]),1)) for p in proj]}")
    if len(b) >= 12:
        # LINEAR SUM ASSIGNMENT, NOT A PERMUTATION SEARCH.  ``permutations(range(n), 12)`` is
        # ~4.8e8 tuples for a typical n and simply hangs; the Hungarian algorithm is O(n^3)
        # and gives the same optimum.  Padding columns of zero cost let the assignment pick
        # which 12 of the blobs to use.
        from scipy.optimize import linear_sum_assignment
        B = np.array([[q[0], q[1]] for q in b])
        cost = np.linalg.norm(B[:, None, :] - np.asarray(proj)[None, :, :], axis=2)
        pad = np.full((len(B), max(0, len(B) - 12)), 0.0)
        full = np.hstack([cost, pad])
        ri, ci = linear_sum_assignment(full)
        chosen = {}
        for r, c in zip(ri, ci):
            if c < 12:
                chosen[c] = r
        if len(chosen) < 12:
            print(f"only {len(chosen)} of 12 markers could be assigned")
        tot = 0.0
        offs = []
        for k in sorted(chosen):
            q = B[chosen[k]]
            dr, dc = q[0] - proj[k][0], q[1] - proj[k][1]
            tot += float(np.hypot(dr, dc))
            offs.append((dr, dc))
            print(f"   mk{k:2d} proj({proj[k][0]:6.1f},{proj[k][1]:6.1f}) "
                  f"blob({q[0]:6.1f},{q[1]:6.1f}) d=({dr:+6.1f},{dc:+6.1f})")
        if offs:
            o = np.array(offs)
            print(f"optimal assignment: total {tot:.1f} px, mean {tot/len(offs):.2f} px")
            print(f"   offset mean ({o[:,0].mean():+.1f},{o[:,1].mean():+.1f}) "
                  f"sd ({o[:,0].std():.1f},{o[:,1].std():.1f})")
    else:
        print("fewer than 12 blobs: the assignment is not attempted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
