#!/usr/bin/env python3
"""Render the arena with UNMISTAKABLE markers and draw every projection hypothesis on it.

THE ONE UNAMBIGUOUS TEST.  Numbers have been giving contradictory answers: single-marker
renders say the columns are a constant ~48 px off, while the six-marker constellation says
the unshifted projection is the closest one.  Those cannot both describe the same renderer,
and choosing between them by more arithmetic is how the last three rounds went wrong.

So: make the markers large enough (0.4 mm radius, 15-30 px discs) that there is no doubt which
bright things they are, render ONE camera, and draw
    YELLOW  the analytic projection
    MAGENTA the analytic projection mirrored about the image centre (a principal-point flip)
    CYAN    the analytic projection shifted +48 px in column
and save the image.  Whichever marker has a circle sitting on it names the right model.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_visual_calib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    from run_arena_reconstruct import load_cameras
    from run_arena_record import MARKER_MATERIAL_NAME

    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    decl = np.array([m["pos_mm"] for m in geom["markers"]["markers"]])
    cam = load_cameras(geom)[0]
    out = HERE / "outputs" / "arena" / "visual_calib"
    out.mkdir(parents=True, exist_ok=True)
    cams = [c for c in arena_cameras() if c["name"] == cam.name]
    for radius in (0.4,):
        cfg = BodyConfig(
            scene_preset="meadow_grass", add_tracking_camera=False,
            add_world_camera=False, add_vision=False, extra_cameras=tuple(cams),
            extra_geoms=tuple({"name": f"m{k}", "type": "sphere",
                               "size": [radius] * 3, "pos": list(p),
                               "rgba": [1, 1, 1, 1], "material": MARKER_MATERIAL_NAME,
                               "contype": 0, "conaffinity": 0}
                              for k, p in enumerate(decl)),
            extra_materials=({"name": MARKER_MATERIAL_NAME, "rgba": [1, 1, 1, 1],
                              "emission": 1.0, "reflectance": 0.0,
                              "shininess": 0.0, "specular": 0.0},))
        be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
        r = be.sim.set_renderer(cam.name, camera_res=(260, 400), playback_speed=1.0,
                                output_fps=1e5, buffer_frames=True)
        dt = cfg.timestep_s
        for k in range(300):
            be.step()
            if k % 120 == 0:
                be.sim.render_as_needed()
        fr = [np.asarray(f) for f in r.frames.get(cam.name, [])]
        be.close()
        img = fr[-1][..., :3].astype("uint8").copy()
        H, W = img.shape[0], img.shape[1]
        # a smaller camera model matching this resolution
        # a camera model at THIS resolution, built from the declared pose so the test is
        # about the model and not about the pose source
        from run_arena_reconstruct import PinholeCamera
        spec = {"name": cam.name, "pos_mm": cam.pos.tolist(),
                "xyaxes": list(cams[0]["xyaxes"]), "fovy_deg": cam.fovy_deg,
                "resolution_px": [H, W]}
        from arena import CAMERA_RING
        small = PinholeCamera(spec)
        proj = small.project(decl)
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for k, (rr, cc) in enumerate(proj):
            dr.ellipse([cc - 9, rr - 9, cc + 9, rr + 9], outline=(255, 255, 0), width=1)
            dr.text((cc + 10, rr - 5), f"Y{k}", fill=(255, 255, 0))
            cc2 = W - cc
            dr.ellipse([cc2 - 9, rr - 9, cc2 + 9, rr + 9], outline=(255, 0, 255), width=1)
            dr.text((cc2 + 10, rr + 5), f"M{k}", fill=(255, 0, 255))
            cc3 = cc + 48.5 * (W / 260.0)
            dr.ellipse([cc3 - 9, rr - 9, cc3 + 9, rr + 9], outline=(0, 255, 255), width=1)
            dr.text((cc3 + 10, rr + 16), f"C{k}", fill=(0, 255, 255))
        dest = out / f"projection_hypotheses_{cam.name}.png"
        im.save(dest)
        print(f"wrote {dest}")
        print("YELLOW = analytic projection, MAGENTA = mirror about image centre, "
              "CYAN = analytic + 48 px in column")
        print("projections:", [(round(float(p[0]), 1), round(float(p[1]), 1)) for p in proj])
        print(f"image {W}x{H}; principal point ({W/2}, {H/2})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
