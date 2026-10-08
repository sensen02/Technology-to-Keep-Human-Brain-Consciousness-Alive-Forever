#!/usr/bin/env python3
"""Can a marker be told apart from a saturated grass highlight?  Measured, not argued.

THE PROBLEM, MEASURED: with the emissive 0.15 mm markers in the meadow, 62 of the pixels at
or above 250 grey levels belong to the GRASS and only 15 to the markers.  An absolute
brightness threshold therefore cannot work, no matter how it is tuned, and the earlier
detector's output was mostly specular grass highlights.

WHAT IS TESTED HERE, for several marker radii and two detection criteria:
  * SIZE + COMPACTNESS.  A marker is a filled DISC: its area nearly equals its bounding-box
    area.  A specular highlight on a grass blade is a thin sliver, so its area is a few
    percent of its bounding box.  If the two populations separate on that ratio, the
    detector can use it.
  * CHROME.  A marker in a colour the scene cannot make -- pure red or pure blue -- is
    separable in CHROMA, where grass highlights are white.  This needs the renderer to keep
    the material's colour at emission, which is checked rather than assumed.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_marker_discriminate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAT = "marker_emissive"


def render(radius_mm: float, rgba, camera: str = "cam0", res=(260, 200), seconds=1.0):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras, marker_report
    cams = [c for c in arena_cameras() if c["name"] == camera]
    cfg = BodyConfig(
        scene_preset="meadow_grass", add_tracking_camera=False, add_world_camera=False,
        add_vision=False, extra_cameras=tuple(cams),
        extra_geoms=tuple({"name": m["name"], "type": "sphere",
                           "size": [radius_mm] * 3, "pos": list(m["pos_mm"]),
                           "rgba": list(rgba), "material": MAT,
                           "contype": 0, "conaffinity": 0}
                          for m in marker_report()["markers"]),
        extra_materials=({"name": MAT, "rgba": list(rgba), "emission": 1.0,
                          "reflectance": 0.0, "shininess": 0.0, "specular": 0.0},))
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=res, playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(int(round(seconds / dt))):
        be.step()
        if k % 250 == 0:
            be.sim.render_as_needed()
    fr = [np.asarray(f) for f in r.frames.get(camera, [])]
    from run_arena_reconstruct import load_cameras, PinholeCamera
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    cams2 = load_cameras(geom)
    cam = next(c for c in cams2 if c.name == camera)
    declared = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    proj = cam.project(declared)
    be.close()
    return np.asarray(fr[-1])[..., :3].astype(float), proj


def components(mask: np.ndarray):
    """4-connected components with area and bounding box, via a stack flood fill."""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    out = []
    for r0 in range(H):
        for c0 in range(W):
            if mask[r0, c0] and lab[r0, c0] == 0:
                n += 1
                st = [(r0, c0)]
                lab[r0, c0] = n
                rmin = rmax = r0
                cmin = cmax = c0
                area = 0
                while st:
                    r, c = st.pop()
                    area += 1
                    rmin, rmax = min(rmin, r), max(rmax, r)
                    cmin, cmax = min(cmin, c), max(cmax, c)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] == 0:
                            lab[rr, cc] = n
                            st.append((rr, cc))
                box = (rmax - rmin + 1) * (cmax - cmin + 1)
                out.append({"row": (rmin + rmax) / 2.0, "col": (cmin + cmax) / 2.0,
                            "area": area, "bbox": box, "fill": area / max(box, 1)})
    return out


def main() -> int:
    rep = {}
    for radius, rgba, tag in ((0.15, (1, 1, 1, 1), "white_0.15"),
                              (0.30, (1, 1, 1, 1), "white_0.30"),
                              (0.15, (1, 0, 0, 1), "red_0.15"),
                              (0.30, (1, 0, 0, 1), "red_0.30")):
        img, proj = render(radius, rgba)
        g = img.mean(axis=2)
        grey = {"p99_9": float(np.percentile(g, 99.9)), "max": float(g.max())}
        # criterion 1: grey level
        comp_grey = components(g >= 250.0)
        # criterion 2: chroma -- red-dominant pixels
        chroma = img[:, :, 0] - 0.5 * (img[:, :, 1] + img[:, :, 2])
        comp_chroma = components(chroma >= 40.0)
        # how many components land near a projection?
        hits_grey = sum(1 for c in comp_grey
                        if min(np.hypot(proj[:, 0] - c["row"], proj[:, 1] - c["col"])) <= 10)
        hits_chroma = sum(1 for c in comp_chroma
                          if min(np.hypot(proj[:, 0] - c["row"],
                                          proj[:, 1] - c["col"])) <= 10)
        rep[tag] = {
            "grey": grey,
            "n_components_at_250": len(comp_grey),
            "n_near_projection_grey": int(hits_grey),
            "fill_ratio_near_projection_grey": [
                round(c["fill"], 3) for c in comp_grey
                if min(np.hypot(proj[:, 0] - c["row"], proj[:, 1] - c["col"])) <= 10][:8],
            "max_chroma": float(chroma.max()),
            "n_components_by_chroma": len(comp_chroma),
            "n_near_projection_chroma": int(hits_chroma),
            "fill_ratio_near_projection_chroma": [
                round(c["fill"], 3) for c in comp_chroma
                if min(np.hypot(proj[:, 0] - c["row"], proj[:, 1] - c["col"])) <= 10][:8],
        }
        print(f"{tag:12s} grey: {len(comp_grey):3d} comps, {hits_grey} near a projection "
              f"(fill {rep[tag]['fill_ratio_near_projection_grey'][:4]}) | "
              f"chroma: {len(comp_chroma):3d} comps, {hits_chroma} near a projection "
              f"(fill {rep[tag]['fill_ratio_near_projection_chroma'][:4]})")
    dest = HERE / "outputs" / "arena" / "marker_discrimination.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
