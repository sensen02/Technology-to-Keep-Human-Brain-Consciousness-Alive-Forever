#!/usr/bin/env python3
"""The decisive test: optimal one-to-one assignment of markers to analytic projections.

Renders the six fiducials large (0.4 mm) on the cleared disc, detects the bright blobs, and
finds the assignment that MINIMISES the total distance.  That removes the guessing: no
nearest-neighbour tie-breaking, no hand-picked pairs.  The mean residual then says whether the
declared camera model is right.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_decisive_assign.py
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def render(radius, res, camera="cam0", preset="meadow_grass"):
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras
    from run_arena_record import MARKER_MATERIAL_NAME
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    decl = [tuple(m["pos_mm"]) for m in geom["markers"]["markers"]]
    cams = [c for c in arena_cameras() if c["name"] == camera]
    cfg = BodyConfig(
        scene_preset=preset, add_tracking_camera=False, add_world_camera=False,
        add_vision=False, extra_cameras=tuple(cams),
        extra_geoms=tuple({"name": f"m{k}", "type": "sphere", "size": [radius] * 3,
                           "pos": list(p), "rgba": [1, 1, 1, 1],
                           "material": MARKER_MATERIAL_NAME,
                           "contype": 0, "conaffinity": 0}
                          for k, p in enumerate(decl)),
        extra_materials=({"name": MARKER_MATERIAL_NAME, "rgba": [1, 1, 1, 1],
                          "emission": 1.0, "reflectance": 0.0,
                          "shininess": 0.0, "specular": 0.0},))
    be = BodyBackend(cfg, gl_backend="egl").attach_cpg_baseline()
    r = be.sim.set_renderer(camera, camera_res=res, playback_speed=1.0,
                            output_fps=1e5, buffer_frames=True)
    dt = cfg.timestep_s
    for k in range(300):
        be.step()
        if k % 120 == 0:
            be.sim.render_as_needed()
    fr = [np.asarray(f) for f in r.frames.get(camera, [])]
    be.close()
    return np.asarray(fr[-1])[..., :3].astype(float), decl


def main() -> int:
    from run_arena_reconstruct import PinholeCamera
    from arena import arena_cameras
    from scipy import ndimage
    rep = {}
    for preset in ("meadow_grass", "blank"):
        for radius in (0.4, 1.0):
            try:
                img, decl = render(radius, (260, 400), preset=preset)
            except Exception as exc:
                rep[f"{preset}_{radius}"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            g = img.mean(axis=2)
            lab, n = ndimage.label(g >= 250.0)
            blobs = []
            for i in range(1, n + 1):
                yy, xx = np.nonzero(lab == i)
                if len(yy) >= 40:
                    blobs.append((float(yy.mean()), float(xx.mean()), int(len(yy))))
            c0 = arena_cameras()[0]
            cam = PinholeCamera({"name": "cam0", "pos_mm": c0["pos"],
                                 "xyaxes": list(c0["xyaxes"]), "fovy_deg": 50.0,
                                 "resolution_px": [260, 400]})
            P = cam.project(np.array(decl))
            entry = {"n_blobs": len(blobs),
                     "blobs": [[round(b[0], 1), round(b[1], 1), b[2]] for b in blobs],
                     "projections": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                                     for p in P]}
            B = np.array([[b[0], b[1]] for b in blobs])
            if len(B) >= 6:
                best = None
                for perm in permutations(range(len(B)), 6):
                    d = sum(float(np.hypot(B[perm[k]][0] - P[k][0],
                                           B[perm[k]][1] - P[k][1])) for k in range(6))
                    if best is None or d < best[0]:
                        best = (d, perm)
                entry["assignment_total_px"] = best[0]
                entry["assignment_mean_px"] = best[0] / 6.0
                entry["pairs"] = [{"marker": k,
                                   "proj": [round(float(P[k][0]), 1), round(float(P[k][1]), 1)],
                                   "blob": [round(B[best[1][k]][0], 1),
                                            round(B[best[1][k]][1], 1)],
                                   "d_row": round(B[best[1][k]][0] - float(P[k][0]), 1),
                                   "d_col": round(B[best[1][k]][1] - float(P[k][1]), 1)}
                                  for k in range(6)]
            rep[f"{preset}_{radius}"] = entry
            if "assignment_mean_px" in entry:
                print(f"{preset:14s} r={radius}: {len(blobs)} blobs, "
                      f"optimal assignment mean {entry['assignment_mean_px']:.2f} px")
                for q in entry["pairs"]:
                    print(f"      mk{q['marker']} proj{q['proj']} blob{q['blob']} "
                          f"d=({q['d_row']:+.1f},{q['d_col']:+.1f})")
            else:
                print(f"{preset:14s} r={radius}: only {len(blobs)} blobs; "
                      f"max grey {g.max():.0f}")
    (HERE / "outputs" / "arena" / "decisive_assignment.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True))
    print("wrote outputs/arena/decisive_assignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
