#!/usr/bin/env python3
"""Visual-only full-body renderer for S284 baseline/friction_x2 trajectories.
Reads saved qpos/time only; never steps physics or invokes neural code.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ARMS = ('baseline', 'friction_x2')
TARGET_MS = (0, 100, 200, 300, 400)

def sha256_array(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(repr(tuple(x.shape)).encode())
    h.update(x.tobytes(order='C'))
    return h.hexdigest()

def load_qpos_time(path: Path):
    with np.load(path, allow_pickle=False) as z:
        if 'qpos' not in z.files or 'time' not in z.files:
            raise KeyError(f'{path}: required qpos/time; keys={z.files}')
        q = np.array(z['qpos'], copy=True)
        t = np.array(z['time'], copy=True)
        keys = list(z.files)
    if q.shape != (4001, 49):
        raise ValueError(f'{path}: qpos shape {q.shape}, expected (4001,49)')
    if t.shape != (4001,):
        raise ValueError(f'{path}: time shape {t.shape}, expected (4001,)')
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError(f'{path}: non-finite qpos/time')
    return q, t, {'keys': keys, 'qpos_sha256': sha256_array(q), 'time_sha256': sha256_array(t)}

def nearest(t: np.ndarray, target_ms: float):
    scale = 1000.0 if float(np.nanmax(np.abs(t))) < 10.0 else 1.0
    ms = t * scale
    i = int(np.argmin(np.abs(ms - target_ms)))
    return i, float(ms[i])

def camera_for_full_body(mj: Any, model: Any, data: Any):
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    pts = np.asarray(data.geom_xpos, dtype=float)
    rad = np.asarray(model.geom_rbound, dtype=float)
    finite = np.isfinite(rad) & (rad > 0.0) & (np.asarray(model.geom_type) != int(mj.mjtGeom.mjGEOM_PLANE))
    if not np.any(finite):
        raise RuntimeError('no finite non-floor geometry bounds')
    lo = np.min(pts[finite] - rad[finite, None], axis=0)
    hi = np.max(pts[finite] + rad[finite, None], axis=0)
    lo[2] = min(lo[2], 0.0); hi[2] = max(hi[2], 0.0)
    cam.lookat[:] = (lo + hi) / 2.0
    cam.azimuth = 135.0; cam.elevation = -22.0
    cam.distance = max(2.0, float(np.max(hi - lo)) * 2.0)
    cam.trackbodyid = -1
    return cam

def annotate(image: np.ndarray, arm: str, target: int, actual: float, index: int):
    from PIL import Image, ImageDraw, ImageFont
    out = Image.fromarray(np.asarray(image, dtype=np.uint8), mode='RGB')
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    except OSError:
        font = ImageFont.load_default()
    text = f'{arm}   target {target} ms   sample {index} ({actual:.3f} ms)'
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, box[2] + 18, box[3] + 14), fill=(20, 20, 24))
    draw.text((9, 7), text, fill=(255, 255, 255), font=font)
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_dir():
        raise SystemExit(f'--output must be an existing directory: {output}')
    if not XML.is_file():
        raise SystemExit(f'frozen XML does not exist: {XML}')
    import mujoco as mj
    from PIL import Image
    model = mj.MjModel.from_xml_path(str(XML))
    data = mj.MjData(model)
    mesh_ids = np.flatnonzero(np.asarray(model.geom_type) == int(mj.mjtGeom.mjGEOM_MESH))
    rgba_before = np.array(model.geom_rgba, copy=True)
    model.geom_rgba[:, 3] = 1.0
    rgba_after = np.array(model.geom_rgba, copy=True)
    trajectories = {}; qhash = {}; samples = []
    for arm in ARMS:
        q, t, info = load_qpos_time(output / arm / 'TRAJECTORY.npz')
        trajectories[arm] = (q, t); qhash[arm] = {'before': info['qpos_sha256'], 'time': info['time_sha256']}
    panels = []; fwd = 0
    for arm in ARMS:
        q, t = trajectories[arm]
        for target in TARGET_MS:
            idx, actual = nearest(t, target)
            data.qpos[:] = q[idx]
            mj.mj_forward(model, data); fwd += 1
            renderer = mj.Renderer(model, height=480, width=640)
            renderer.update_scene(data, camera=camera_for_full_body(mj, model, data))
            panels.append(annotate(renderer.render(), arm, target, actual, idx)); renderer.close()
            samples.append({'arm': arm, 'target_ms': target, 'actual_ms': actual, 'sample_index': idx})
    for arm, (q, _) in trajectories.items(): qhash[arm]['after'] = sha256_array(q)
    w, h = panels[0].size
    montage = Image.new('RGB', (len(TARGET_MS) * w, len(ARMS) * h), (245, 245, 245))
    for i, panel in enumerate(panels): montage.paste(panel, ((i % len(TARGET_MS))*w, (i//len(TARGET_MS))*h))
    montage.save(output / 'FULL_BODY.png')
    audit = {'script': str(Path(__file__).resolve()), 'model': str(XML), 'output': str(output), 'arms': list(ARMS), 'targets_ms': list(TARGET_MS), 'samples': samples, 'qpos_hashes': qhash, 'visual_change': {'geom_count': int(model.ngeom), 'mesh_count': int(len(mesh_ids)), 'expected_mesh_count': 71, 'mesh_alpha_before_unique': sorted(set(rgba_before[mesh_ids,3].tolist())), 'mesh_alpha_after_unique': sorted(set(rgba_after[mesh_ids,3].tolist())), 'all_71_meshes_visible': bool(len(mesh_ids)==71 and np.all(rgba_after[mesh_ids,3]==1.0)), 'geom_rgb_unchanged': bool(np.array_equal(rgba_before[:,:3], rgba_after[:,:3]))}, 'calls': {'model_load': 1, 'mj_forward': fwd, 'fwd_position': 0, 'step': 0, 'neural': 0}, 'xml_modified': False}
    (output / 'FULL_BODY.audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    if fwd != 10: raise AssertionError(f'expected 10 forwards, got {fwd}')
    return 0

if __name__ == '__main__': raise SystemExit(main())
