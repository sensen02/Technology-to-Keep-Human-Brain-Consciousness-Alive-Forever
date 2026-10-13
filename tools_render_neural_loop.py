#!/usr/bin/env python3
"""RENDER THE NEURO-MUSCLE LOOP THROUGH THE ARENA'S CAMERA RING.

WHY THIS IS THE STEP AFTER S1a.  S1a made the muscle-driven body OBSERVABLE, but it rendered it
PASSIVELY -- settled, standing still.  A still body proves the ring is attached; it does not prove
the ring can observe a body that MUSCLES drive.  The platform's whole claim is
brain -> body -> sensing, so the cameras have to see the body while the connectome is driving it.

WHAT DRIVES THE MOTION, and it is the reason this is worth rendering: the loop it wraps contains NO
state-to-action function.  Each muscle's activation IS the measured firing rate of THAT muscle's own
motor neurons (`run_six_leg_neural_loop.run`), so anything that moves in these frames moved because
motor neurons fired.  The frames are therefore evidence about the loop, not decoration.

WHAT IT REUSES.  The loop is NOT reimplemented here: `run_six_leg_neural_loop.run` gained a
`frame_sink` callback and this module supplies one.  A second copy of the sensorimotor chain would
drift from the first, and the drift would be invisible in whichever copy nobody was reading.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_render_neural_loop.py --duration-s 0.6 --stride 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

EPISODE = HERE / "outputs" / "arena" / "stackb_neural_loop"
DETECT_THRESHOLD = 240.0
DETECT_MIN_PX = 3


def render_loop_episode(duration_s: float = 0.6, stride: int = 4, lesion_leg=None,
                        out_dir=None, ring_rotation_deg: float = 0.0) -> dict:
    import mujoco
    from PIL import Image
    from scipy import ndimage
    import run_six_leg_neural_loop as loop
    from arena import CAMERA_RING, FIDUCIAL_MARKERS

    dest = Path(out_dir) if out_dir else EPISODE
    dest.mkdir(parents=True, exist_ok=True)

    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    res = geom.get("resolutions") or {}
    h, w = int(res.get("height", CAMERA_RING["resolution_px"][0])), int(
        res.get("width", CAMERA_RING["resolution_px"][1]))

    # THE RENDERER IS BUILT FROM THE MODEL THE LOOP HANDS THE SINK, NOT FROM A SEPARATE BUILD.
    # MEASURED BUG, and it produced a completely black episode: this script called
    # `loop.build(observable=True)` to read the camera names, and then `loop.run(...)` built the
    # loop's OWN context internally -- a SECOND model and data.  The renderer was therefore bound to
    # model A while the sink received model B's `d`, so `update_scene` drew a scene the renderer did
    # not own (frame max grey 18 against 242 for the passive render, and 0 candidates detected).
    state = {"renderer": None, "cams": None}
    rows, cols, peaks, ncan = {}, {}, {}, {}
    times, n_frames = [], 0

    def sink(mm, d, c, act_trace):
        nonlocal n_frames
        if state["renderer"] is None:
            state["cams"] = sorted(mm.camera(i).name for i in range(mm.ncam)
                                   if mm.camera(i).name.startswith("cam"))
            for cc in state["cams"]:
                (dest / "frames" / cc).mkdir(parents=True, exist_ok=True)
                rows[cc], cols[cc], peaks[cc], ncan[cc] = [], [], [], []
            state["renderer"] = mujoco.Renderer(mm, height=h, width=w)
            print(f"  rendering the neuro-muscle loop: {len(state['cams'])} cameras at {h}x{w} "
                  f"(HxW), stride {stride}")
        renderer = state["renderer"]
        cams = state["cams"]
        t = c * loop.DT_COMMAND_S
        for cam in cams:
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            grey = img.mean(axis=2)
            Image.fromarray(img).save(dest / "frames" / cam / f"f{n_frames:05d}.png")
            lab, n = ndimage.label(grey >= DETECT_THRESHOLD)
            rr, cc, pk = [], [], []
            for i in range(1, n + 1):
                yy, xx = np.nonzero(lab == i)
                if len(yy) >= DETECT_MIN_PX:
                    rr.append(float(yy.mean()))
                    cc.append(float(xx.mean()))
                    pk.append(float(grey[yy, xx].max()))
            rows[cam].extend(rr)
            cols[cam].extend(cc)
            peaks[cam].extend(pk)
            ncan[cam].append(len(rr))
        times.append(float(t))
        n_frames += 1

    summary = loop.run(lesion_leg=lesion_leg, duration_s=duration_s, verbose=False,
                       frame_sink=sink, frame_stride=stride, observable=True,
                       ring_rotation_deg=ring_rotation_deg)
    if state["renderer"] is not None:
        state["renderer"].close()
    cams = state["cams"] or []

    arrays = {}
    for cam in cams:
        arrays[f"px/{cam}/rows"] = np.asarray(rows[cam], dtype=np.float32)
        arrays[f"px/{cam}/cols"] = np.asarray(cols[cam], dtype=np.float32)
        arrays[f"px/{cam}/offsets"] = np.concatenate(
            [[0], np.cumsum(ncan[cam])]).astype(np.int64)
        arrays[f"px/{cam}/n_candidates"] = np.asarray(ncan[cam], dtype=np.int64)
        arrays[f"px/{cam}/peak_grey"] = np.asarray(peaks[cam], dtype=np.float64)
    dec = np.asarray([x["pos_mm"] for x in FIDUCIAL_MARKERS], dtype=np.float64)
    arrays["truth/marker_static_world_mm"] = np.repeat(dec[None, :, :], n_frames, axis=0)
    arrays["truth/time_s"] = np.asarray(times, dtype=np.float64)
    np.savez_compressed(dest / "episode.npz", **arrays)
    (dest / "markers.json").write_text(json.dumps({
        "marker_body_segments": ["c_thorax", "c_rostrum", "c_abdomen3"],
        "static_markers": [{"name": x["name"], "pos_mm": list(x["pos_mm"])}
                           for x in FIDUCIAL_MARKERS],
        "stack": "B (FlyMimic muscles) DRIVEN by the connectome",
        "driver": "run_six_leg_neural_loop.run with a frame_sink; no state-to-action function",
        "lesion_leg": lesion_leg, "frame_stride": stride,
    }, indent=2, sort_keys=True))
    return {"episode": str(dest), "cameras": cams, "n_frames": n_frames,
            "lesion_leg": lesion_leg, "duration_s": duration_s,
            "median_candidates": {c: float(np.median(ncan[c])) for c in cams},
            "excursion": summary.get("excursion", {}) if isinstance(summary, dict) else {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=0.6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lesion-leg", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ring-rotation-deg", type=float, default=0.0)
    args = ap.parse_args()
    info = render_loop_episode(args.duration_s, args.stride, args.lesion_leg, args.out_dir,
                               args.ring_rotation_deg)
    print(json.dumps({"episode": info["episode"], "cameras": info["cameras"],
                      "n_frames": info["n_frames"], "lesion_leg": info["lesion_leg"],
                      "median_candidates": info["median_candidates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
