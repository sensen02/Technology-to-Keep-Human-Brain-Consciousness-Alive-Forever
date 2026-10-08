#!/usr/bin/env python3
"""Recover each camera's ACTUAL world pose from the compiled model.

THE ROOT CAUSE THIS ADDRESSES.  The six cameras are declared with MuJoCo's ``xyaxes``, and
the reconstruction built its projection matrices from those same numbers.  That is the
natural thing to do and it is WRONG: measured, the compiled model's camera rotation has
x = (0.866, -0.3536, 0.3536) where the declaration said (0.866, 0.5, 0), i.e. MuJoCo did not
take the declared axes as the camera's basis.  The analytic projection then matched the
rendered ROWS to within a pixel and was off by tens of pixels in the COLUMNS, which is what
a wrong image-X axis looks like.

So the pose is read back from the compiled model (``cam_pos`` and ``cam_quat``, converted
with ``mju_quat2Mat``) for every camera, and THAT is what the reconstruction uses.  This is
also what a real calibration does: you calibrate the camera that exists, not the one you
specified.

Writes ``camera_poses.json`` next to the episode.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv_body/bin/python tools_cam_pose_dump.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def dump(preset: str = "meadow_grass", camera_names=("cam0",), out=None) -> dict:
    import mujoco
    from engine.embodied import BodyBackend, BodyConfig
    from arena import arena_cameras, CAMERA_RING
    cams = [c for c in arena_cameras() if c["name"] in camera_names]
    cfg = BodyConfig(scene_preset=preset, add_tracking_camera=False,
                     add_world_camera=False, add_vision=False,
                     extra_cameras=tuple(cams))
    be = BodyBackend(cfg, gl_backend=None)
    m = be.model
    H, W = int(CAMERA_RING["resolution_px"][0]), int(CAMERA_RING["resolution_px"][1])
    fovy = float(m.cam_fovy[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA,
                                             camera_names[0])])
    fy = (H / 2.0) / np.tan(np.radians(fovy) / 2.0)
    out_doc = {"image_size_px": [H, W], "fovy_from_model_deg": fovy,
               "fy_px": float(fy), "fx_px": float(fy),
               "source": ("cam_pos and cam_quat read back from the COMPILED model and "
                          "converted with mju_quat2Mat; the declared xyaxes are recorded "
                          "alongside for comparison and are NOT used for projection"),
               "cameras": []}
    for name in camera_names:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        pos = np.asarray(m.cam_pos[cid], dtype=float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(m.cam_quat[cid], dtype=float))
        R = R.reshape(3, 3)
        declared = next(c for c in arena_cameras() if c["name"] == name)
        out_doc["cameras"].append({
            "name": name, "pos_mm": pos.tolist(), "R": R.tolist(),
            "fx_px": float(fy), "fy_px": float(fy),
            "image_size_px": [H, W],
            "declared_xyaxes": list(declared["xyaxes"]),
            "note": ("R's ROWS are the camera's x, y and z axes expressed in world "
                     "coordinates; a world point p maps to camera coordinates "
                     "R @ (p - pos), with the camera looking along -z"),
        })
    be.close()
    if out:
        Path(out).write_text(json.dumps(out_doc, indent=2, sort_keys=True))
    return out_doc


def main() -> int:
    from arena import arena_cameras
    names = tuple(c["name"] for c in arena_cameras())
    doc = dump(camera_names=names,
               out=str(HERE / "outputs" / "arena" / "camera_poses.json"))
    print(json.dumps({"image_size_px": doc["image_size_px"],
                      "fovy_from_model_deg": doc["fovy_from_model_deg"],
                      "fy_px": doc["fy_px"],
                      "n_cameras": len(doc["cameras"])}, indent=2))
    for c in doc["cameras"][:2]:
        print(c["name"], "R rows:")
        for row in c["R"]:
            print("   ", np.round(row, 4).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
