#!/usr/bin/env python3
"""Measure the camera's actual image axes by rendering markers at known offsets.

WHY MEASURE INSTEAD OF DERIVE: the analytic projection matched the rendered rows to within
a pixel while the COLUMNS were off by tens of pixels -- a pattern that is not a scale error
or an offset, but a sign or axis mix-up.  Rather than reason about which of the four sign
conventions is the wrong one, this renders markers at +x and +y offsets from the optical
axis and reads off where each one appears.  The answer is then a fact about the renderer.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_axis_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools_marker_discriminate import components  # noqa: E402
from run_arena_reconstruct import load_cameras  # noqa: E402

MAT = "marker_emissive"


def render_markers(pts, camera="cam0", radius=0.30, res=(200, 260)):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    cams = [c for c in arena_cameras() if c["name"] == camera]
    geoms = tuple({"name": f"p{k}", "type": "sphere", "size": [radius] * 3,
                   "pos": list(p), "rgba": [1, 1, 1, 1], "material": MAT,
                   "contype": 0, "conaffinity": 0} for k, p in enumerate(pts))
    cfg = BodyConfig(
        scene_preset="meadow_grass", add_tracking_camera=False, add_world_camera=False,
        add_vision=False, extra_cameras=tuple(cams), extra_geoms=geoms,
        extra_materials=({"name": MAT, "rgba": [1, 1, 1, 1], "emission": 1.0,
                          "reflectance": 0.0, "shininess": 0.0, "specular": 0.0},))
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=res, playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(200):
        be.step()
        if k % 100 == 0:
            be.sim.render_as_needed()
    fr = [np.asarray(f) for f in r.frames.get(camera, [])]
    be.close()
    return np.asarray(fr[-1])[..., :3].astype(float)


def main() -> int:
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    cams = load_cameras(geom)
    cam = cams[0]
    print(f"cam0 pos {np.round(cam.pos,3).tolist()}  x_vec {np.round(cam.x_vec,4).tolist()}")
    print(f"     y_vec {np.round(cam.y_vec,4).tolist()}  f {cam.fx:.2f} px  "
          f"principal {(cam.W/2, cam.H/2)}")
    pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0),
           (0.0, 1.0, 1.0), (0.0, -1.0, 1.0)]
    img = render_markers(pts)
    g = img.mean(axis=2)
    comps = [c for c in components(g >= 250.0) if c["area"] >= 4]
    print(f"\n{len(comps)} bright blobs found (expected 5)")
    proj = cam.project(np.array(pts))
    print("\n pt            analytic(row,col)   detected(row,col)   d_row  d_col")
    for k, p in enumerate(pts):
        if k < len(comps):
            # pair by row, which is known to be right
            cand = sorted(comps, key=lambda c: abs(c["row"] - proj[k][0]))
            best = cand[0]
            print(f" {str(p):18s} ({proj[k][0]:6.1f},{proj[k][1]:6.1f})   "
                  f"({best['row']:6.1f},{best['col']:6.1f})   "
                  f"{best['row']-proj[k][0]:+6.1f} {best['col']-proj[k][1]:+6.1f}")
        else:
            print(f" {str(p):18s} ({proj[k][0]:6.1f},{proj[k][1]:6.1f})   -- no blob --")
    # what the data says the axes are: d(col) / d(world x)
    if len(comps) >= 3:
        print("\nimplied column sensitivity (px per mm of world offset):")
        for k, name in ((1, "+x 1 mm"), (2, "-x 1 mm"), (3, "+y 1 mm"), (4, "-y 1 mm")):
            if k < len(comps):
                cand = sorted(comps, key=lambda c: abs(c["row"] - proj[k][0]))
                print(f"   {name}: d_col = {cand[0]['col'] - proj[0][1]:+7.1f} px  "
                      f"d_row = {cand[0]['row'] - proj[0][0]:+7.1f} px")
        print(f"   analytic expectation: +x -> "
              f"{cam.fx * (np.array([1.0,0,1.0])-cam.pos) @ cam.x_vec / 0 - cam.fx*(np.array([0,0,1.0])-cam.pos) @ cam.x_vec / 0:.1f}"
              " (see the projected columns above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
