#!/usr/bin/env python3
"""LOOK at a recorded frame with the fiducial projections drawn on it.

WHY THIS HAS NOT BEEN DONE UNTIL NOW, AND WHY IT SHOULD HAVE BEEN.  Every measurement in the
last two rounds went through a DETECTOR, and the detector has been wrong at least twice (it
reported grass highlights as markers; it merged adjacent markers into one component).  Looking
at the frame with the projections drawn on it separates the two possibilities immediately:
either the markers ARE there and the detector missed them, or they are not there at all.

It also prints, for each projection, the pixel value at that location and the value of the
brightest pixel within 6 px, so the answer is a number as well as a picture.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_look_at_recording.py \\
        --episode outputs/arena/arena_bare_seed0 --camera cam0 --frame 0
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
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="outputs/arena/look")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from run_arena_reconstruct import load_cameras
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    decl = np.array([m["pos_mm"] for m in geom["markers"]["markers"]])
    cam = next(c for c in load_cameras(geom) if c.name == args.camera)
    d = HERE / args.episode
    frames = sorted((d / "frames" / args.camera).glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames in {d / 'frames' / args.camera}")
    fi = min(args.frame, len(frames) - 1)
    img = Image.open(frames[fi]).convert("RGB")
    a = np.asarray(img, dtype=float).mean(axis=2)
    H, W = a.shape
    proj = cam.project(decl)
    # the camera model in the reconstruction has the RING resolution; rescale if needed
    print(f"frame {frames[fi].name}  image {W}x{H}  camera model {cam.W}x{cam.H}  "
          f"f={cam.fx:.1f} px")
    print(f"declared markers: {len(decl)}")
    print(f"  idx   pos(mm)               proj(row,col)   grey here   max grey within 6px")
    lines = []
    for k, (r, c) in enumerate(proj):
        r0, c0 = int(round(r)), int(round(c))
        inb = 0 <= r0 < H and 0 <= c0 < W
        here = float(a[r0, c0]) if inb else float("nan")
        y0, x0 = max(0, r0 - 6), max(0, c0 - 6)
        win = a[y0:min(H, r0 + 7), x0:min(W, c0 + 7)]
        mx = float(win.max()) if win.size else float("nan")
        lines.append(f"  {k:3d}   {str(np.round(decl[k],1).tolist()):22s} "
                     f"({r:6.1f},{c:6.1f})   {here:7.1f}    {mx:7.1f}")
    print("\n".join(lines))
    grey = a
    print(f"\nframe stats: min {grey.min():.0f} max {grey.max():.0f} "
          f"median {np.median(grey):.0f}  n>=250 {int((grey>=250).sum())}")
    dr = ImageDraw.Draw(img)
    for k, (r, c) in enumerate(proj):
        dr.ellipse([c - 7, r - 7, c + 7, r + 7], outline=(255, 255, 0), width=1)
        dr.text((c + 8, r - 5), str(k), fill=(255, 255, 0))
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.episode.replace('/', '_')}_{args.camera}_f{fi:05d}.png"
    img.save(dest)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
