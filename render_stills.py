"""Plain rendered stills of the walking fly -- images, not charts.

WHAT THIS PRODUCES
------------------
Nothing here is a plot: no axes, no legends, no subplot furniture.  It renders the
physical fly directly from FlyGym/MuJoCo and writes PNGs you can just look at:

  * ``stills/track_*.png``   -- the tracking camera, one file per moment, higher
                                resolution than the video, showing the fly's legs
                                at different gait phases
  * ``stills/world_*.png``   -- a WORLD-FIXED camera above the arena, so the same
                                moments show the fly at different PLACES on the
                                checkered ground (how far it has walked)
  * ``stills/filmstrip_world.png`` -- the world-camera moments laid out in one
                                image with no plot furniture, so the travel is
                                visible left-to-right
  * ``stills/side_by_side_baseline_vs_neural.png`` -- the same time index from the
                                two already-recorded walks (plain CPG baseline vs
                                the neural-modulated arm), extracted from the
                                written mp4 files

UNITS / HONEST NOTES
--------------------
The model is in MILLIMETRES (gravity -9810 mm/s^2).  The gait is produced by
FlyGym's demo tripod CPG, which is an ENGINEERING BASELINE, not the connectome.
These are pictures of a simulation, not of a real fly.

Run: ./venv_body/bin/python render_stills.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs" / "embodied_body"
STILLS = OUT / "stills"

from engine.embodied import BodyBackend, BodyConfig  # noqa: E402


def look_at_xyaxes(camera_pos, target):
    """xyaxes for a camera at camera_pos looking at target (MuJoCo convention).

    MuJoCo camera frame: the first triple is the X axis, the second the Y axis,
    and the camera looks along -Z with Z = X cross Y.  This is computed rather
    than guessed -- the earlier hand-written orientation rendered pure white and
    the cause turned out to be distance/clipping, not handedness.
    """
    C = np.asarray(camera_pos, float)
    T = np.asarray(target, float)
    z = C - T
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("camera and target coincide")
    z = z / n
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    y = up - np.dot(up, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return tuple(np.concatenate([x, y]).tolist())


def add_world_camera(be, pos, fovy=45.0):
    """Add a world-fixed camera looking straight down, if the API allows it."""
    from flygym.utils.math import Rotation3D  # noqa: F401  (kept for parity)
    try:
        # the worldbody is an MjsBody: cameras are added with add_camera(), not
        # the generic add().  An earlier version used add() and failed.
        be.world.mjcf_root.worldbody.add_camera(
            name="worldcam", pos=[float(v) for v in pos],
            xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], fovy=float(fovy))
        return True
    except Exception as exc:                                    # noqa: BLE001
        print("world camera unavailable:", type(exc).__name__, exc)
        return False


def to_uint8(frame):
    f = np.asarray(frame)
    return f if f.dtype == np.uint8 else (255 * np.clip(f, 0, 1)).astype("uint8")


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(to_uint8(arr)).save(path)
    return path


def tile(images, cols, path, pad=6, bg=255):
    """Lay images out in a grid WITHOUT any captions, axes or frames."""
    from PIL import Image
    ims = [Image.fromarray(to_uint8(a)) for a in images]
    w, h = ims[0].size
    rows = int(np.ceil(len(ims) / cols))
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad),
                       (bg, bg, bg))
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    canvas.save(path)
    return path


def main():
    STILLS.mkdir(parents=True, exist_ok=True)
    made = []

    # ------------------------------------------------------------------ walk
    cfg = BodyConfig(seed=0, timestep_s=1e-4, spawn_position_mm=(0.0, 0.0, 0.5),
                     world_half_size_mm=1000.0)
    # The world camera must satisfy MuJoCo's far clip plane.  The model sets
    # zfar = 250 mm, and a top-down camera 190 mm up rendered PURE WHITE (measured:
    # mean 255, std 0.0) while the same orientation at 120 mm and an oblique view at
    # 45 mm both rendered.  So the arena view is oblique and stays well inside zfar.
    target = (45.0, 12.0, 1.0)           # roughly the middle of the walked path
    camino = (45.0 - 24.0, 12.0 + 88.0, 62.0)
    cfg.add_world_camera = True          # must be set before the world is compiled
    cfg.world_camera_pos_mm = camino
    cfg.world_camera_xyaxes = look_at_xyaxes(camino, target)
    cfg.world_camera_fovy = 62.0
    be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
    have_world = be.world_camera_spec is not None
    cams = [be.camera_spec] + ([be.world_camera_spec] if have_world else [])
    # The flat ground is rendered with a REFLECTIVE material, so the fly appears
    # mirrored underneath and the picture looks like two flies.  Measured: the
    # reflection was present in every earlier still.  Reflectance is a model
    # property read at render time, so it can be zeroed after compile.
    _zeroed = []
    for gi in range(be.model.ngeom):
        name = str(be.model.geom(gi).name).lower()
        mid = int(be.model.geom_matid[gi])
        if "ground" in name and mid >= 0 and float(be.model.mat_reflectance[mid]) > 0:
            _zeroed.append((name, mid, float(be.model.mat_reflectance[mid])))
            be.model.mat_reflectance[mid] = 0.0
    print("ground reflections removed:", _zeroed)

    # 5 frames per simulated second per camera keeps memory and wall time modest
    renderer = be.sim.set_renderer(cams, camera_res=(600, 800),
                                   playback_speed=1.0, output_fps=5,
                                   buffer_frames=True)

    seconds = 6.0
    n = int(seconds / cfg.timestep_s)
    thorax = []
    for k in range(n):
        be.step()
        if k % 200 == 0:
            thorax.append(np.asarray(be.observe().thorax_position_mm, float))
        be.sim.render_as_needed()

    frames = {c: [np.asarray(f) for f in renderer.frames.get(c, [])] for c in cams}
    print({c: len(v) for c, v in frames.items()},
          "thorax end (mm):", np.round(thorax[-1], 2).tolist())

    # pick six moments spread across the walk
    track = frames[be.camera_spec]
    idx = np.linspace(0, len(track) - 1, min(6, len(track))).astype(int)
    for j, i in enumerate(idx):
        made.append(save_png(track[i], STILLS / f"track_{j:02d}_t{i*0.2:.1f}s.png"))

    if have_world and frames.get(be.world_camera_spec):
        wf = frames[be.world_camera_spec]
        idxw = np.linspace(0, len(wf) - 1, min(6, len(wf))).astype(int)
        for j, i in enumerate(idxw):
            made.append(save_png(wf[i], STILLS / f"world_{j:02d}_t{i*0.2:.1f}s.png"))
        made.append(tile([wf[i] for i in idxw], cols=3,
                         path=STILLS / "filmstrip_world.png"))

    be.close()

    # ------------------------------------------- baseline vs neural stills
    try:
        import imageio.v3 as iio
        picks = {}
        for arm, name in (("baseline", "body_walk_seed0.mp4"),
                          ("neural", "loop_neural_walk_seed0.mp4")):
            p = OUT / name
            if not p.exists():
                continue
            frames_arm = []
            for i, fr in enumerate(iio.imiter(p)):
                if i % 25 == 0:                       # one per simulated second
                    frames_arm.append(np.asarray(fr))
                if len(frames_arm) >= 5:
                    break
            picks[arm] = frames_arm
        if len(picks) == 2:
            n_use = min(len(picks["baseline"]), len(picks["neural"]))
            rows = []
            for i in range(n_use):
                rows.append(picks["baseline"][i])
            for i in range(n_use):
                rows.append(picks["neural"][i])
            made.append(tile(rows, cols=n_use,
                             path=STILLS / "side_by_side_baseline_vs_neural.png"))
            for i in range(n_use):
                made.append(save_png(picks["baseline"][i],
                                     STILLS / f"baseline_t{i}s.png"))
                made.append(save_png(picks["neural"][i],
                                     STILLS / f"neural_t{i}s.png"))
    except Exception as exc:                                    # noqa: BLE001
        print("mp4 frame extraction unavailable:", type(exc).__name__, exc)

    print("wrote", len(made), "images:")
    for p in made:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
