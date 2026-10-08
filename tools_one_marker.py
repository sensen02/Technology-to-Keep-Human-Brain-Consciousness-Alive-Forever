#!/usr/bin/env python3
"""One marker, one camera: does the analytic projection land on the rendered marker?

THE MINIMAL REPRODUCTION.  The arena results were ambiguous because six markers, the fly
and the grass were all in the frame.  This puts ONE marker in front of one camera, with the
grass scene active and nothing else emissive, and reports:
  * where the camera model says it is,
  * where the brightest blob in the image actually is,
  * the difference.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_one_marker.py
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


def run(pos_mm, radius_mm, scene_preset, camera="cam0", grass_clear=True):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    cams = [c for c in arena_cameras() if c["name"] == camera]
    cfg = BodyConfig(
        scene_preset=scene_preset, add_tracking_camera=False, add_world_camera=False,
        add_vision=False, extra_cameras=tuple(cams),
        extra_geoms=({"name": "solo", "type": "sphere", "size": [radius_mm] * 3,
                      "pos": list(pos_mm), "rgba": [1, 1, 1, 1], "material": MAT,
                      "contype": 0, "conaffinity": 0},),
        extra_materials=({"name": MAT, "rgba": [1, 1, 1, 1], "emission": 1.0,
                          "reflectance": 0.0, "shininess": 0.0, "specular": 0.0},))
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=(200, 260), playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(300):
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
    tests = [
        ("meadow_grass, marker at (0,0,0.5)", "meadow_grass", (0.0, 0.0, 0.5)),
        ("meadow_grass, marker at (0,0,3.0)", "meadow_grass", (0.0, 0.0, 3.0)),
        ("meadow_grass, marker at (2,-2,0.5)", "meadow_grass", (2.0, -2.0, 0.5)),
        ("blank scene, marker at (0,0,0.5)", "blank", (0.0, 0.0, 0.5)),
    ]
    for name, preset, pos in tests:
        try:
            img = run(pos, 0.30, preset)
        except Exception as exc:
            print(f"{name}: RAISED {type(exc).__name__}: {exc}")
            continue
        g = img.mean(axis=2)
        proj = cam.project(np.array([pos]))[0]
        comps = [c for c in components(g >= 250.0) if c["area"] >= 4]
        comps.sort(key=lambda c: -c["area"])
        best = comps[0] if comps else None
        if best:
            d = float(np.hypot(best["row"] - proj[0], best["col"] - proj[1]))
            print(f"{name:36s} proj({proj[0]:6.1f},{proj[1]:6.1f}) "
                  f"blob({best['row']:6.1f},{best['col']:6.1f}) area {best['area']:3d} "
                  f"fill {best['fill']:.2f}  d={d:5.1f} px  n_ge250={len(comps)}")
        else:
            print(f"{name:36s} proj({proj[0]:6.1f},{proj[1]:6.1f}) "
                  f"NO blob >= 250 with area >= 4 (max grey {g.max():.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
