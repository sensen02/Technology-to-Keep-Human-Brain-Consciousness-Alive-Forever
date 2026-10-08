#!/usr/bin/env python3
"""Which way do the camera's image axes point?  Answered by a known 3D displacement.

WHY THIS IS THE RIGHT EXPERIMENT.  The analytic projection matched the rendered ROWS to
within a pixel and was wrong in the COLUMNS by tens of pixels on all six cameras.  That
pattern has exactly two possible causes -- a wrong image-X axis, or detections that are not
the markers -- and the earlier probe could not tell them apart because it used 0.15 mm
markers that the grass swamped (62 saturated pixels from grass, 15 from markers).

With 0.8 mm markers the detections are unambiguous, so this probe puts ONE marker at the
image centre's ray and THREE more at known +x, +y and +z offsets from it, and prints where
each lands.  The answer is then read off the numbers instead of argued.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_axis_probe2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def render(pts, camera: str = "cam0", radius: float = 0.8, res=(200, 260),
           preset: str = "meadow_grass"):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    from run_arena_record import MARKER_MATERIAL_NAME
    cams = [c for c in arena_cameras() if c["name"] == camera]
    geoms = tuple({"name": f"p{k}", "type": "sphere", "size": [radius] * 3,
                   "pos": list(p), "rgba": [1, 1, 1, 1],
                   "material": MARKER_MATERIAL_NAME, "contype": 0, "conaffinity": 0}
                  for k, p in enumerate(pts))
    cfg = BodyConfig(scene_preset=preset, add_tracking_camera=False,
                     add_world_camera=False, add_vision=False,
                     extra_cameras=tuple(cams), extra_geoms=geoms,
                     extra_materials=({"name": MARKER_MATERIAL_NAME,
                                       "rgba": [1, 1, 1, 1], "emission": 1.0,
                                       "reflectance": 0.0, "shininess": 0.0,
                                       "specular": 0.0},))
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=res, playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(250):
        be.step()
        if k % 120 == 0:
            be.sim.render_as_needed()
    fr = [np.asarray(f) for f in r.frames.get(camera, [])]
    be.close()
    return np.asarray(fr[-1])[..., :3].astype(float)


def blobs(img, min_area=20):
    from scipy import ndimage
    g = img.mean(axis=2)
    lab, n = ndimage.label(g >= 250.0)
    out = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(yy) >= min_area:
            out.append({"row": float(yy.mean()), "col": float(xx.mean()),
                        "area": int(len(yy))})
    return out


def main() -> int:
    from run_arena_reconstruct import load_cameras
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    cam = load_cameras(geom)[0]
    # a marker on the optical axis at 2 mm, then three one-millimetre displacements
    # THE OFFSETS ARE 4 mm, NOT 1 mm.  MEASURED: at 1 mm separation and 28 mm range the four
    # 0.8 mm markers merge into ONE blob (area 241, i.e. four discs), so the probe returned
    # one detection and told us nothing.  4 mm separates them by about 40 px.
    base = np.array([0.0, 0.0, 2.0])
    pts = [tuple(base),
           tuple(base + np.array([4.0, 0.0, 0.0])),
           tuple(base + np.array([0.0, 4.0, 0.0])),
           tuple(base + np.array([0.0, 0.0, 4.0]))]
    labels = ["origin(0,0,2)", "+x 4mm", "+y 4mm", "+z 4mm"]
    img = render(pts)
    bs = blobs(img)
    proj = cam.project(np.array(pts))
    print(f"cam0 pos {np.round(cam.pos, 2).tolist()}  f {cam.fx:.1f} px  "
          f"principal ({cam.W/2}, {cam.H/2})")
    print(f"detected blobs (area >= 20 px): {len(bs)}   expected 4")
    for b in sorted(bs, key=lambda b: -b["area"]):
        print(f"   blob row {b['row']:7.1f} col {b['col']:7.1f} area {b['area']:4d}")
    print()
    print("  marker        analytic(row,col)   nearest blob(row,col)   d_row   d_col")
    used = set()
    for k, lab in enumerate(labels):
        r0, c0 = proj[k]
        if bs:
            dd = [(np.hypot(b["row"] - r0, b["col"] - c0), j) for j, b in enumerate(bs)]
            dd.sort()
            d, j = dd[0]
            used.add(j)
            print(f"  {lab:14s} ({r0:7.1f},{c0:7.1f})   "
                  f"({bs[j]['row']:7.1f},{bs[j]['col']:7.1f})   "
                  f"{bs[j]['row']-r0:+7.1f} {bs[j]['col']-c0:+7.1f}")
        else:
            print(f"  {lab:14s} ({r0:7.1f},{c0:7.1f})   -- none --")
    # the decisive comparison: sign and size of the image displacement per world mm
    print()
    print("DECISIVE: measured image displacement per 1 mm of world displacement,")
    print("          taken from the nearest blob to each projection (order-independent):")
    if len(bs) >= 4:
        # pair by proximity: the four blobs must map to the four projections
        idx = np.argsort([np.hypot(b["row"] - proj[0][0], b["col"] - proj[0][1])
                          for b in bs])
        # identify the base as the blob nearest the base projection, then sort the rest by
        # their distance from that base blob, matching the analytic ordering
        b0 = bs[int(np.argmin([np.hypot(b["row"] - proj[0][0], b["col"] - proj[0][1])
                               for b in bs]))]
        rest = [b for b in bs if b is not b0]
        for k in (1, 2, 3):
            ar, ac = proj[k]
            dr_a, dc_a = ar - proj[0][0], ac - proj[0][1]
            # the blob whose displacement direction best matches the analytic one
            best = None
            for b in rest:
                dr_m, dc_m = b["row"] - b0["row"], b["col"] - b0["col"]
                if np.hypot(dr_m, dc_m) < 1:
                    continue
                cos = (dr_a * dr_m + dc_a * dc_m) / (np.hypot(dr_a, dc_a)
                                                     * np.hypot(dr_m, dc_m))
                if best is None or cos > best[0]:
                    best = (cos, dr_m, dc_m)
            if best:
                print(f"   {labels[k]:10s} analytic d=({dr_a:+7.1f},{dc_a:+7.1f})  "
                      f"measured d=({best[1]:+7.1f},{best[2]:+7.1f})  cos={best[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
