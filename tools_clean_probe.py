#!/usr/bin/env python3
"""A CLEAN axis/principal-point probe: one marker per render, all inside the cleared disc.

WHY ONE AT A TIME.  Every earlier probe put several markers in one frame, and the ones 4 mm
from the centre sat OUTSIDE the 4.2 mm cleared disc, i.e. in grass -- so the "detections"
could be grass and the offsets were meaningless.  Rendering one marker per image removes both
problems: no merging (the discs were overlapping at 1 mm separation, measured area 241 px for
four markers) and no grass near the marker.

WHAT IT MEASURES.  With three markers at known world positions the image displacement per
millimetre of world displacement is read off directly, and the principal point follows from
where a marker on the optical-axis ray lands.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_clean_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POSITIONS = [
    ("origin",       (0.0, 0.0, 1.0)),
    ("+y 0.5mm",     (0.0, 0.5, 1.0)),
    ("-y 0.5mm",     (0.0, -0.5, 1.0)),
    ("+x 0.5mm",     (0.5, 0.0, 1.0)),
    ("-x 0.5mm",     (-0.5, 0.0, 1.0)),
    ("z 2.0mm",      (0.0, 0.0, 2.0)),
]


def one_render(pos, camera="cam0", radius=0.8, res=(200, 260)):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    from run_arena_record import MARKER_MATERIAL_NAME
    cams = [c for c in arena_cameras() if c["name"] == camera]
    cfg = BodyConfig(scene_preset="meadow_grass", add_tracking_camera=False,
                     add_world_camera=False, add_vision=False,
                     extra_cameras=tuple(cams),
                     extra_geoms=({"name": "solo", "type": "sphere",
                                   "size": [radius] * 3, "pos": list(pos),
                                   "rgba": [1, 1, 1, 1], "material": MARKER_MATERIAL_NAME,
                                   "contype": 0, "conaffinity": 0},),
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


def brightest_blob(img, min_area=40):
    from scipy import ndimage
    g = img.mean(axis=2)
    lab, n = ndimage.label(g >= 250.0)
    best = None
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(yy) >= min_area and (best is None or len(yy) > best[2]):
            best = (float(yy.mean()), float(xx.mean()), int(len(yy)))
    return best


def main() -> int:
    from run_arena_reconstruct import load_cameras
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    cam = load_cameras(geom)[0]
    print(f"cam0 pos {np.round(cam.pos, 3).tolist()} f={cam.fx:.2f} px "
          f"principal=({cam.H/2}, {cam.W/2})")
    seen = {}
    for name, pos in POSITIONS:
        img = one_render(pos)
        b = brightest_blob(img)
        p = cam.project(np.array(pos))[0]
        seen[name] = {"world": list(pos), "analytic": [float(p[0]), float(p[1])],
                      "blob": None if b is None else [b[0], b[1]], "area": None if b is None else b[2]}
        if b is None:
            print(f"  {name:10s} analytic ({p[0]:7.1f},{p[1]:7.1f})  NO BLOB >= 40 px")
        else:
            print(f"  {name:10s} analytic ({p[0]:7.1f},{p[1]:7.1f})  "
                  f"blob ({b[0]:7.1f},{b[1]:7.1f}) area {b[2]:4d}  "
                  f"d=({b[0]-p[0]:+6.1f},{b[1]-p[1]:+6.1f})")
    # image displacement per mm, from the symmetric pairs
    def dd(a, b):
        A, B = seen[a], seen[b]
        if A["blob"] is None or B["blob"] is None:
            return None
        w = np.array(B["world"]) - np.array(A["world"])
        m = np.array(B["blob"]) - np.array(A["blob"])
        a_ = np.array(B["analytic"]) - np.array(A["analytic"])
        return w, m, a_
    print()
    for pair, label in ((("+x 0.5mm", "-x 0.5mm"), "world x, 1 mm"),
                        (("+y 0.5mm", "-y 0.5mm"), "world y, 1 mm")):
        r = dd(*pair)
        if r is None:
            print(f"  {label}: missing a blob")
            continue
        w, m, a_ = r
        print(f"  {label:16s} measured image d={np.round(m,1).tolist()}  "
              f"analytic d={np.round(a_,1).tolist()}  "
              f"ratio col {m[1]/a_[1] if a_[1] else float('nan'):.3f}")
    print()
    print(json.dumps(seen, indent=2))
    (HERE / "outputs" / "arena" / "clean_probe.json").write_text(
        json.dumps(seen, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
