#!/usr/bin/env python3
"""Draw the DECLARED fiducials' projections onto a real rendered frame.

THE FASTEST WAY TO FIND A CALIBRATION BUG IS TO LOOK.  If the declared positions project
onto the bright blobs the detector found, the camera model in the reconstruction agrees
with the renderer.  If they do not, the disagreement is visible as an offset, and its
DIRECTION says which end is wrong: a uniform offset in one camera is a pose error, a
scaling about the principal point is a focal-length error, and a mirrored pattern is a
sign error on one axis.

Writes one PNG per camera plus a text report of every projected and detected position.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python tools_arena_calib_check.py --episode outputs/arena/arena_bare_seed0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import load_cameras  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/arena/arena_bare_seed0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="outputs/arena/calib_check")
    args = ap.parse_args()

    d = HERE / args.episode
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    cams = load_cameras(geom)
    declared = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    z = np.load(d / "episode.npz")
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw
    lines = []
    for c in cams:
        # the rendered frame
        fdir = d / "frames" / c.name
        frames = sorted(fdir.glob("*.png"))
        if not frames:
            lines.append(f"{c.name}: no frames")
            continue
        fi = min(args.frame, len(frames) - 1)
        img = Image.open(frames[fi]).convert("RGB")
        W, H = img.size
        # the detected candidates for that frame
        rows = np.asarray(z[f"px/{c.name}/rows"], dtype=float)
        cols = np.asarray(z[f"px/{c.name}/cols"], dtype=float)
        off = np.asarray(z[f"px/{c.name}/offsets"], dtype=np.int64)
        a, b = int(off[fi]), int(off[fi + 1])
        det = np.stack([rows[a:b], cols[a:b]], axis=1) if b > a else np.zeros((0, 2))
        proj = c.project(declared)
        dr = ImageDraw.Draw(img)
        for k, (r, col) in enumerate(proj):
            r, col = float(r), float(col)
            dr.ellipse([col - 6, r - 6, col + 6, r + 6], outline=(255, 0, 0), width=1)
            dr.text((col + 7, r - 5), str(k), fill=(255, 255, 0))
        for r, col in det:
            dr.ellipse([col - 2, r - 2, col + 2, r + 2], outline=(0, 255, 0), width=1)
        dest = out / f"calib_{c.name}.png"
        img.save(dest)
        # nearest detected to each projection
        pairs = []
        for k, (r, col) in enumerate(proj):
            if len(det):
                dd = np.hypot(det[:, 0] - r, det[:, 1] - col)
                j = int(np.argmin(dd))
                pairs.append((k, round(float(r), 1), round(float(col), 1),
                              round(float(det[j, 0]), 1), round(float(det[j, 1]), 1),
                              round(float(dd[j]), 1)))
            else:
                pairs.append((k, round(float(r), 1), round(float(col), 1), None, None, None))
        lines.append(f"{c.name}  image {W}x{H}  detections {len(det)}  "
                     f"focal {c.fx:.1f} px  pos {np.round(c.pos,2).tolist()}")
        lines.append("   idx  proj_row proj_col | det_row det_col | dist_px")
        for p in pairs:
            lines.append("   %3d  %8s %8s | %7s %7s | %s" % p)
        lines.append(f"   wrote {dest}")

    # also compare against the TRUTH marker positions, which removes the renderer from
    # the question: if the truth markers project onto the blobs but the DECLARED ones do
    # not, the world frame of the episode is not the world frame of the declaration.
    tstat = np.load(d / "episode.npz")["truth/marker_static_world_mm"]
    lines.append("")
    lines.append("truth static marker positions, frame %d:" % args.frame)
    lines.append(np.array2string(tstat[min(args.frame, len(tstat) - 1)], precision=3))
    lines.append("declared:")
    lines.append(np.array2string(declared, precision=3))
    lines.append("difference (truth - declared), mm:")
    lines.append(np.array2string(tstat[min(args.frame, len(tstat) - 1)] - declared,
                                 precision=4))
    txt = "\n".join(lines)
    (out / "calib_report.txt").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
