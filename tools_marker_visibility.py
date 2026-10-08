#!/usr/bin/env python3
"""Are the fiducial markers actually VISIBLE?  Renders the arena with and without them.

WHY THIS EXISTS: a reconstruction that associates zero fiducials can be wrong in two very
different ways -- the camera model can be wrong, or the markers can be invisible.  Those
need different fixes, and the only way to tell them apart is to render the scene with the
markers present and absent and compare the images.  This tool does exactly that and prints
the pixel statistics that decide it.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_marker_visibility.py \\
        --camera cam0 --seconds 1.2 --out outputs/arena/marker_visibility
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MARKER_RADIUS_MM = 0.15
MARKER_MATERIAL_NAME = "marker_emissive"


def build(with_markers: bool, camera: str, seconds: float, res: tuple[int, int]):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras, marker_report
    extra = {}
    if with_markers:
        extra = dict(
            extra_geoms=tuple({"name": m["name"], "type": "sphere",
                               "size": [MARKER_RADIUS_MM] * 3,
                               "pos": list(m["pos_mm"]), "rgba": [1, 1, 1, 1],
                               "material": MARKER_MATERIAL_NAME,
                               "contype": 0, "conaffinity": 0}
                              for m in marker_report()["markers"]),
            extra_materials=({"name": MARKER_MATERIAL_NAME, "rgba": [1, 1, 1, 1],
                              "emission": 1.0, "reflectance": 0.0,
                              "shininess": 0.0, "specular": 0.0},))
    cams = [c for c in arena_cameras() if c["name"] == camera]
    cfg = BodyConfig(scene_preset="meadow_grass", add_tracking_camera=False,
                     add_world_camera=False, add_vision=False,
                     extra_cameras=tuple(cams), **extra)
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=res, playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(int(round(seconds / dt))):
        be.step()
        if k % 300 == 0:
            be.sim.render_as_needed()
    frames = [np.asarray(f) for f in r.frames.get(camera, [])]
    # the marker geometry, in the model, to prove it was added at all
    import mujoco
    gnames = [str(mujoco.mj_id2name(be.model, mujoco.mjtObj.mjOBJ_GEOM, i))
              for i in range(be.model.ngeom)]
    n_marker_geoms = sum(1 for n in gnames if n in
                         {m["name"] for m in marker_report()["markers"]})
    be.close()
    return frames, n_marker_geoms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--out", default="outputs/arena/marker_visibility")
    args = ap.parse_args()
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    rep = {}
    for tag, with_m in (("without_markers", False), ("with_markers", True)):
        frames, ngeom = build(with_m, args.camera, args.seconds,
                              (args.height, args.width))
        if not frames:
            rep[tag] = {"error": "no frames"}
            continue
        a = np.asarray(frames[-1])[..., :3].astype(float).mean(axis=2)
        Image.fromarray(np.asarray(frames[-1])[..., :3].astype("uint8")).save(
            out / f"{args.camera}_{tag}.png")
        rep[tag] = {"n_frames": len(frames), "marker_geoms_in_model": ngeom,
                    "max": float(a.max()), "p99_9": float(np.percentile(a, 99.9)),
                    "n_at_255": int((a >= 255).sum()),
                    "n_at_250": int((a >= 250).sum())}
    # the decisive comparison
    a = np.asarray([rep[k].get("n_at_250", 0) for k in
                    ("without_markers", "with_markers")])
    rep["verdict"] = {
        "markers_add_250plus_pixels": int(a[1] - a[0]) if len(a) == 2 else None,
        "markers_visible": bool(len(a) == 2 and a[1] > a[0]),
        "note": ("if the marker count does not exceed the no-marker count, the markers are "
                 "NOT visible and no camera-model fix can help")}
    (out / "marker_visibility.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
