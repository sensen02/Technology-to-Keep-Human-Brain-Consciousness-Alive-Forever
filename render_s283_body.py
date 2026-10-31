#!/usr/bin/env python3
"""Render the complete s283 body from archived qpos only.

This is deliberately a visual-only replay: it loads the verified XML once, assigns
archived qpos samples, and calls ``mj_forward`` exactly once per output panel.  It
never calls ``mj_step`` and does not import or invoke a neural runner.

Usage (the output directory must already contain baseline/ and tau_equal/):
    python render_s283_body.py --output /path/to/s283

The parent runner is responsible for EGL setup before invoking this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

XML = Path("/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml")
ARMS = ("baseline", "tau_equal")
TARGET_MS = (0, 100, 200, 300, 400)
PANELS_PER_ARM = len(TARGET_MS)


def sha256_array(a: np.ndarray) -> str:
    """Hash shape, dtype and contiguous bytes without changing the source array."""
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(repr(tuple(x.shape)).encode("ascii"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def load_qpos_time(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read qpos/time, preferring those exact keys and never mutating the NPZ."""
    with np.load(path, allow_pickle=False) as z:
        if "qpos" not in z.files or "time" not in z.files:
            raise KeyError(f"{path}: required qpos and time fields; found {z.files}")
        qpos = np.array(z["qpos"], copy=True)
        time = np.array(z["time"], copy=True)
        keys = list(z.files)
    if qpos.shape != (4001, 49):
        raise ValueError(f"{path}: expected qpos shape (4001, 49), got {qpos.shape}")
    if time.shape != (4001,):
        raise ValueError(f"{path}: expected time shape (4001,), got {time.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(time).all():
        raise ValueError(f"{path}: qpos/time contain non-finite values")
    return qpos, time, {"keys": keys, "qpos_sha256": sha256_array(qpos), "time_sha256": sha256_array(time)}


def nearest_sample(time: np.ndarray, target_ms: float) -> tuple[int, float]:
    # Trajectory time is normally seconds; tolerate a millisecond time vector too.
    scale = 1000.0 if float(np.nanmax(np.abs(time))) < 10.0 else 1.0
    ms = time * scale
    i = int(np.argmin(np.abs(ms - target_ms)))
    return i, float(ms[i])


def camera_for_full_body(mj: Any, model: Any, data: Any) -> Any:
    """Build a free camera framing all body geometry, including the floor."""
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    # geom_xpos/rbound gives a conservative complete-body bound after forward.
    pts = np.asarray(data.geom_xpos, dtype=float)
    rad = np.asarray(model.geom_rbound, dtype=float)
    # Plane/floor rbound is infinite; bound finite body geoms and add a
    # deliberately finite floor margin so the floor remains in the shot.
    finite = np.isfinite(rad) & (rad > 0.0) & (np.asarray(model.geom_type) != int(mj.mjtGeom.mjGEOM_PLANE))
    if not np.any(finite):
        raise RuntimeError("model has no finite geometry bounds")
    lo = np.min(pts[finite] - rad[finite, None], axis=0)
    hi = np.max(pts[finite] + rad[finite, None], axis=0)
    lo[2] = min(lo[2], 0.0)
    hi[2] = max(hi[2], 0.0)
    center = (lo + hi) / 2.0
    extent = float(np.max(hi - lo))
    cam.lookat[:] = center
    cam.azimuth = 135.0
    cam.elevation = -22.0
    cam.distance = max(2.0, extent * 2.0)
    cam.trackbodyid = -1
    return cam


def annotate(image: np.ndarray, arm: str, target_ms: int, actual_ms: float, index: int) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    out = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    text = f"{arm}   target {target_ms} ms   sample {index} ({actual_ms:.3f} ms)"
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, box[2] + 18, box[3] + 14), fill=(20, 20, 24))
    draw.text((9, 7), text, fill=(255, 255, 255), font=font)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="existing s283 output directory")
    ap.add_argument("--model", type=Path, default=XML, help=argparse.SUPPRESS)
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_dir():
        raise SystemExit(f"--output must be an existing directory: {output}")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"model XML does not exist: {model_path}")

    # Import only MuJoCo and PIL at runtime; no runner, simulator stepping, or neural code.
    import mujoco as mj
    from PIL import Image, ImageDraw, ImageFont

    model = mj.MjModel.from_xml_path(str(model_path))  # exactly one model load
    data = mj.MjData(model)
    mesh_type = int(mj.mjtGeom.mjGEOM_MESH)
    mesh_ids = np.flatnonzero(np.asarray(model.geom_type) == mesh_type)
    rgba_before = np.array(model.geom_rgba, copy=True)
    # Visual-only derived modification on the newly loaded model; XML/source untouched.
    model.geom_rgba[:, 3] = 1.0
    rgba_after = np.array(model.geom_rgba, copy=True)

    qpos_hashes: dict[str, dict[str, str]] = {}
    trajectories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in ARMS:
        q, t, info = load_qpos_time(output / arm / "TRAJECTORY.npz")
        trajectories[arm] = (q, t)
        qpos_hashes[arm] = {"before": info["qpos_sha256"]}

    panels: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    fwd_calls = 0
    for arm in ARMS:
        qpos, time = trajectories[arm]
        for target in TARGET_MS:
            idx, actual = nearest_sample(time, target)
            data.qpos[:] = qpos[idx]
            # Explicitly position-only replay: no integration or controls.
            mj.mj_forward(model, data)
            fwd_calls += 1
            cam = camera_for_full_body(mj, model, data)
            renderer = mj.Renderer(model, height=480, width=640)
            renderer.update_scene(data, camera=cam)
            panels.append(annotate(renderer.render(), arm, target, actual, idx))
            renderer.close()
            rows.append({"arm": arm, "target_ms": target, "actual_ms": actual, "sample_index": idx})

    # A source-array rehash documents that loading/rendering did not mutate qpos.
    for arm, (q, _t) in trajectories.items():
        qpos_hashes[arm]["after"] = sha256_array(q)
    cols, rows_per_arm = PANELS_PER_ARM, len(ARMS)
    w, h = panels[0].size
    montage = Image.new("RGB", (cols * w, rows_per_arm * h), (245, 245, 245))
    for n, panel in enumerate(panels):
        montage.paste(panel, ((n % cols) * w, (n // cols) * h))
    montage.save(output / "FULL_BODY.png")

    mesh_alpha_before = np.asarray(rgba_before[mesh_ids, 3], dtype=float)
    mesh_alpha_after = np.asarray(rgba_after[mesh_ids, 3], dtype=float)
    audit = {
        "script": str(Path(__file__).resolve()),
        "model": str(model_path),
        "output": str(output),
        "arms": list(ARMS),
        "targets_ms": list(TARGET_MS),
        "samples": rows,
        "qpos_hashes": qpos_hashes,
        "visual_change": {
            "geom_count": int(model.ngeom),
            "mesh_count": int(len(mesh_ids)),
            "expected_mesh_count": 71,
            "mesh_alpha_before_unique": sorted(set(mesh_alpha_before.tolist())),
            "mesh_alpha_after_unique": sorted(set(mesh_alpha_after.tolist())),
            "all_71_meshes_visible": bool(len(mesh_ids) == 71 and np.all(mesh_alpha_after == 1.0)),
            "geom_rgba_changed_only_alpha": bool(np.array_equal(rgba_before[:, :3], rgba_after[:, :3]) and np.any(rgba_before[:, 3] != rgba_after[:, 3])),
        },
        "calls": {"model_load": 1, "mj_forward": fwd_calls, "fwd_position": 0, "step": 0, "neural": 0},
        "xml_modified": False,
    }
    (output / "FULL_BODY.audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if fwd_calls != 10:
        raise AssertionError(f"expected 10 position forwards, got {fwd_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
