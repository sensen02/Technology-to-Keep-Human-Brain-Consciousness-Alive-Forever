#!/usr/bin/env python3
"""END-TO-END RECORDING PIPELINE -- step 1 of 3: run the fly and record BOTH
sides of the experiment (the electrode's voltage and an external camera's photos).

WHAT THIS SCRIPT IS FOR
-----------------------
Four things have to be true at once, and this script produces the raw material for
all four:

  R1  the fly walks normally                     -> truth arrays + rendered frames
  R2  the electrode records its signal           -> ``elec/*`` arrays, from a
                                                    modelled artefact + the
                                                    project's OWN 7 um recording
                                                    chain noise budget
  R2b the electrode's own weight changes the walk -> the SAME seed is run with and
                                                    without the payload mass
  R3  the behaviour can be reconstructed from
      camera photos, NOT from the simulator      -> frames are the ONLY thing
                                                    handed to the offline parser
  R4  touch can be inferred from the trajectory  -> ground-truth contact is
                                                    recorded here but is kept
                                                    SEPARATE from what the
                                                    reconstruction may read

DISCIPLINE
----------
Everything the offline parser is allowed to see goes into ``frames/`` plus a
``camera.json`` that contains ONLY the things a real experimenter would have: the
camera pose, its field of view and image size, and a calibration target image with
its known dimensions.  Ground truth (``truth/*``) is written to the same file as
the frames so the comparison can be audited, but the parser is written and run
without reading it -- see the header of ``run_recording_vision.py``.

Units: the FlyGym model is in MM/N/uN*mm; the electrode chain is in V.  Nothing is
converted silently: array names carry their units.

Run (BODY environment):
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      venv_body/bin/python run_recording_pipeline.py --seeds 0,1 --seconds 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs" / "electrode_payload"

# Cameras.  DECLARED configuration -- the numbers a real experimenter would have --
# established by MEASURED trials, not guessed:
#   * from 240 mm straight down the render is a pure-white frame (measured: mean 255,
#     std 0), i.e. that placement sees nothing; 150 mm works;
#   * the fly walks 55.6 mm in 4 s at seed 0 (MEASURED), so a camera that is to keep
#     it in frame for the whole episode must cover roughly 70 x 40 mm.
#
# TWO cameras, because ONE cannot do both jobs and pretending otherwise would make
# the reconstruction look better than it is:
#
#   "worldcam"  fixed overhead, 150 mm up: covers 187 x 140 mm, so it sees the WHOLE
#               trajectory, but the fly is only ~8 x 15 px there.  MEASURED: at that
#               size the measured stride frequency of its blob signal is wrong
#               (20.7 Hz recovered vs 12.0 Hz true), i.e. the gait is NOT recoverable
#               from this camera.
#   "detailcam" CLOSE overhead, 8 mm up: 0.035 mm/px, so a 0.78 mm stride is ~22 px.
#               It covers only 11 x 8 mm, so it is PANNED by a motorised stage to
#               follow the fly, which is what a real close-up rig does.  The pan is
#               commanded from the fly's OWN measured position in the sim -- a
#               deliberate simplification of a real tracker, and it means the detail
#               camera's CENTRING is not something the parser has to solve.
# Both cameras are written into the same frames list.
WORLD_CAM_NAME = "worldcam"
WORLD_CAM_POS_MM = (28.0, 7.0, 150.0)      # centred on the measured 4 s walk path
WORLD_CAM_FOVY_DEG = 50.0
WORLD_CAM_XYAXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)   # straight down, +x right, +y up

DETAIL_CAM_NAME = "detailcam"
DETAIL_CAM_HEIGHT_MM = 14.0
DETAIL_CAM_FOVY_DEG = 50.0
DETAIL_CAM_XYAXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#: The pan follows the fly's SIMULATED position (DECLARED).  A real rig would track the
#: image centroid; using the sim's own position removes tracking error from the
#: reconstruction, and that simplification is reported rather than hidden.
#:
#: LOOKAHEAD = 0 IS A MEASURED CHOICE, not a default.  Leading the fly by a fixed time
#: makes the camera offset proportional to SPEED, so a faster seed pushes the fly
#: further off axis -- measured with a 0.5 s lead: the fly left the 27 x 35 mm field of
#: view for whole episodes and the close-up camera detected only 80-195 of 240 frames.
#: Centring on the current position instead leaves a steady-state lag of about tau times
#: the speed (0.3 s x 20 mm/s = 6 mm) which fits inside the frame, and it removes the
#: noisy velocity term from the command entirely.
DETAIL_PAN_LOOKAHEAD_S = 0.0
#: Pan COMMAND SMOOTHING time constant, seconds.  MEASURED REASON: the simulator's
#: free-joint velocity is dominated by contact impulses -- sampled per frame it swings
#: between 0.5 and 38 mm/s at a 100 us timestep -- so a stage driven by the raw value
#: throws the camera around and the fly leaves an 11 x 8 mm field of view.  A real
#: motorised stage with an encoder loop behaves like a low-pass, so the command is
#: filtered with a first-order lag of this time constant and the COMMANDED position is
#: what gets recorded (and is what the parser is told).
PAN_TAU_S = 0.30

#: The renderer's OWN fps gate QUANTISES: measured, asking for 60 fps while stepping
#: a 16 ms period produced 3 frames in 800 steps instead of ~50, because a render
#: happens on the first sim step at or after ``last + 1/fps - 5e-5`` and the step and
#: render periods beat against each other.  So the gate is opened wide and the cadence
#: is set HERE, by rendering on every ``frame_period_steps``-th step.  The real frame
#: times are then recorded from the sim clock, so the effective fps is MEASURED.
RENDERING_OPEN_FPS = 100000.0

#: Volt per unit of the sim's actuator-force channel.  DECLARED, ILLUSTRATIVE: a
#: real fly EMG amplitude depends on electrode placement and contact area, neither
#: of which is modelled.  With the fitted value below the walking artefact lands a
#: factor of a few above the 7 um chain's own noise floor, which is the situation
#: a real 7 um electrode is in.  It is NOT a measured conversion factor.
ARTEFACT_UV_PER_FORCE_UNIT = 20.0
#: High-pass corner applied to the artefact before sampling, Hz.  DECLARED.
ARTEFACT_HP_HZ = 1.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(0xC0FFEE ^ int(seed))


def build_scene_and_run(seed: int, seconds: float, fps: float, cam_res: tuple[int, int],
                        gl: str | None, load_fraction: float, mass_scale: float,
                        label: str, with_render: bool) -> dict:
    """Run one episode.  Returns arrays in a dict -- no file writes here."""
    from engine.embodied import BodyBackend, BodyConfig
    from electrode_payload import ElectrodePayloadConfig, assert_yardstick, measure_body_mass_kg
    from engine.electrode_recording import RecordingChain

    payload_cfg = None
    if label != "control":
        payload_cfg = ElectrodePayloadConfig(load_fraction=load_fraction,
                                             mass_scale=mass_scale, label=label)
    cfg = BodyConfig(
        seed=seed,
        add_world_camera=True,
        world_camera_name=WORLD_CAM_NAME,
        world_camera_pos_mm=WORLD_CAM_POS_MM,
        world_camera_fovy=WORLD_CAM_FOVY_DEG,
        world_camera_xyaxes=WORLD_CAM_XYAXES,
        extra_cameras=({"name": DETAIL_CAM_NAME, "pos": [0.0, 0.0, DETAIL_CAM_HEIGHT_MM],
                        "xyaxes": list(DETAIL_CAM_XYAXES),
                        "fovy": DETAIL_CAM_FOVY_DEG},),
        electrode_payload=payload_cfg,
    )
    be = BodyBackend(cfg, gl_backend=gl).attach_cpg_baseline()
    dt = cfg.timestep_s
    n_steps = int(round(seconds / dt))
    # Frame cadence: the sim runs at dt = 100 us, so a whole number of steps gives a
    # cadence CLOSE to, but not exactly, the requested fps.  The realised cadence is
    # measured from the sim clock and recorded instead of the request.
    frame_period_steps = max(1, int(round(1.0 / (fps * dt))))
    obs_period = max(1, int(round(1.0 / (200.0 * dt))))    # 200 Hz truth sampling

    renderer = None
    if with_render:
        renderer = be.sim.set_renderer([WORLD_CAM_NAME, DETAIL_CAM_NAME],
                                       camera_res=cam_res,
                                       playback_speed=1.0,
                                       output_fps=RENDERING_OPEN_FPS,
                                       buffer_frames=True)
        # FIND THE CAMERA ID BY NAME, never by a guessed index: the compiled model's
        # camera ORDER is whatever the spec happened to produce, and writing the stage
        # position into the wrong camera would silently pan the FIXED camera instead.
        # MEASURED trap: the first version of this file hard-coded ``cam_pos[1]``.
        import mujoco as _mj
        cam_names = [str(be.model.camera(i).name) for i in range(be.model.ncam)]
        detail_cam_id = _mj.mj_name2id(be.model, _mj.mjtObj.mjOBJ_CAMERA,
                                       DETAIL_CAM_NAME)
        world_cam_id = _mj.mj_name2id(be.model, _mj.mjtObj.mjOBJ_CAMERA,
                                      WORLD_CAM_NAME)
        if detail_cam_id < 0 or world_cam_id < 0:
            raise RuntimeError(f"cameras missing from the compiled model: order was "
                               f"{cam_names}, wanted {WORLD_CAM_NAME!r} and "
                               f"{DETAIL_CAM_NAME!r}")
        pan_cam_id = detail_cam_id
        pan_cam_note = (f"model camera order {cam_names}: world={world_cam_id}, "
                        f"detail={detail_cam_id}")
    else:
        pan_cam_id = None
        pan_cam_note = "no renderer requested"

    T, BODY_P, BODY_Q, THORAX, CF, CP, CFound, AF, CPG, ACTQ = ([] for _ in range(10))
    frames: dict[str, list[np.ndarray]] = {WORLD_CAM_NAME: [], DETAIL_CAM_NAME: []}
    frame_time_s: list[float] = []
    detail_cam_pos: list[tuple[float, float, float]] = []
    detail_cam_offset: list[tuple[float, float, float]] = []
    pan_state = None
    try:
        for k in range(n_steps):
            be.step()
            if k % obs_period == 0:
                obs = be.observe()
                T.append(obs.time_s)
                BODY_P.append(obs.body_positions_mm)
                BODY_Q.append(obs.body_rotations_wxyz)
                THORAX.append(obs.thorax_position_mm)
                CF.append(obs.contact_forces)
                CP.append(obs.contact_positions_mm)
                CFound.append(obs.contact_found_raw.copy())
                AF.append(obs.actuator_forces)
                ACTQ.append(obs.joint_angles_rad)
                try:
                    CPG.append(np.asarray(be.cpg_phase, dtype=float))
                except Exception:
                    CPG.append(np.full(6, np.nan))
            if renderer is not None and k % frame_period_steps == 0:
                # The motorised stage: command the detail camera to hover over where
                # the fly will be in DETAIL_PAN_LOOKAHEAD_S, using the fly's own
                # measured position (a tracker simplification, see the header).
                th = np.asarray(be.data.xpos[1], dtype=float).copy()
                vel = np.asarray(be.data.qvel[0:3], dtype=float).copy()
                desired = th + DETAIL_PAN_LOOKAHEAD_S * vel
                if pan_state is None:
                    pan_state = np.asarray(desired, dtype=float)
                else:
                    alpha = 1.0 - math.exp(-frame_period_steps * dt / PAN_TAU_S)
                    pan_state = pan_state + alpha * (np.asarray(desired, dtype=float)
                                                     - pan_state)
                tgt = pan_state
                be.model.cam_pos[pan_cam_id] = [float(tgt[0]), float(tgt[1]),
                                                DETAIL_CAM_HEIGHT_MM]
                if be.sim.render_as_needed():
                    frame_time_s.append(float(be.data.time))
                    # RECORD THE POSE THAT WAS ACTUALLY RENDERED, read back from the
                    # model, not the value that was commanded.  MEASURED trap: recording
                    # the commanded position instead put the log ~10 frames ahead of the
                    # frame, and every reconstructed world position came out with a
                    # 25 mm bias -- the reconstruction was working and the CALIBRATION LOG
                    # was wrong.  A stage log has to be the pose at capture time.
                    actual = [float(be.model.cam_pos[pan_cam_id][0]),
                              float(be.model.cam_pos[pan_cam_id][1]),
                              float(be.model.cam_pos[pan_cam_id][2])]
                    detail_cam_pos.append(tuple(actual))
                    # how far the fly is from the camera's optical axis, in mm, so the
                    # framing is measurable instead of assumed
                    detail_cam_offset.append((float(th[0] - actual[0]),
                                              float(th[1] - actual[1]), 0.0))
        if renderer is not None:
            frames = {name: [np.asarray(f).copy()
                             for f in renderer.frames.get(name, [])]
                      for name in (WORLD_CAM_NAME, DETAIL_CAM_NAME)}
    finally:
        pass

    truth = {
        "time_s": np.asarray(T, dtype=float),
        "frame_time_s": np.asarray(frame_time_s, dtype=float),
        "body_positions_mm": np.asarray(BODY_P, dtype=np.float32),
        "body_rotations_wxyz": np.asarray(BODY_Q, dtype=np.float32),
        "thorax_mm": np.asarray(THORAX, dtype=np.float32),
        "contact_forces": np.asarray(CF, dtype=np.float32),
        "contact_positions_mm": np.asarray(CP, dtype=np.float32),
        "contact_found_raw": np.asarray(CFound, dtype=np.float32),
        "contact_present": np.asarray(CFound, dtype=np.float32) > 0,
        "actuator_forces": np.asarray(AF, dtype=np.float32),
        "joint_angles_rad": np.asarray(ACTQ, dtype=np.float32),
        "cpg_phase_rad": np.asarray(CPG, dtype=np.float32),
    }

    # ---- electrode: what the 7 um contact actually sees ---------------------
    # SOURCE: the sim's own actuator-force channel, which is the only muscle-drive
    # proxy this model has.  It is a SIMULATED drive, not EMG.  The transfer from
    # that drive to a voltage at the electrode is a DECLARED constant.
    chain = RecordingChain()
    budget = chain.budget()
    noise_rms_uV = float(budget["noise_rms_uV"])
    drive = truth["actuator_forces"].astype(float)
    drive_ac = drive - drive.mean(axis=0, keepdims=True)
    artefact_uV = ARTEFACT_UV_PER_FORCE_UNIT * drive_ac.sum(axis=1)
    # a first-order high-pass at 1 Hz removes the residual drift a real amplifier
    # would also remove; done in the sample domain with a declared corner.
    fs_truth = 1.0 / float(np.median(np.diff(truth["time_s"]))) if len(T) > 2 else 200.0
    a = np.exp(-2.0 * np.pi * ARTEFACT_HP_HZ / fs_truth)
    hp = np.empty_like(artefact_uV)
    acc = 0.0
    prev_in = 0.0
    for i, x in enumerate(artefact_uV):
        acc = a * (acc + x - prev_in)
        prev_in = x
        hp[i] = acc
    rng = _rng(seed)
    noise_uV = rng.normal(0.0, noise_rms_uV, size=hp.shape)
    elec = {
        "time_s": truth["time_s"].copy(),
        "artefact_uV": hp.astype(np.float32),
        "noise_uV": noise_uV.astype(np.float32),
        "recorded_uV": (hp + noise_uV).astype(np.float32),
        "fs_Hz": float(fs_truth),
    }

    model_sig = None
    payload_rep = None
    try:
        from electrode_payload import model_signature
        model_sig = model_signature(be.model)
        if payload_rep is None:
            payload_rep = be.payload_report
    except Exception as exc:      # surfaced, not swallowed
        model_sig = {"error": f"{type(exc).__name__}: {exc}"}
    desc = be.describe()
    if payload_rep is None:
        payload_rep = be.payload_report
    be.close()
    return {"truth": truth, "elec": elec, "frames": frames,
            "detail_cam_pos_mm": np.asarray(detail_cam_pos, dtype=float),
            "detail_cam_offset_mm": np.asarray(detail_cam_offset, dtype=float),
            "pan_cam_note": pan_cam_note,
            "describe": desc,
            "payload": payload_rep, "model_signature": model_sig,
            "recording_chain": {"noise_rms_uV": noise_rms_uV, "band": chain.config.band,
                                "components_uV": {k: float(np.sqrt(v)) * 1e6
                                                  for k, v in budget["components_V2"].items()},
                                "budget": budget},
            "fs_truth_Hz": float(fs_truth)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--gl", default="egl")
    ap.add_argument("--loads", default="0.20,0.05")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    loads = [float(s) for s in args.loads.split(",") if s.strip()]
    conditions = [("control", 0.0, 1.0)] + [
        (f"load{int(round(f * 100)):03d}", f, 1.0) for f in loads]
    if any(abs(f) < 1e-12 for f in loads):
        raise SystemExit("a load condition of 0 duplicates the control; remove it")
    conditions.append(("sham", loads[0], 1e-6))

    index = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seconds": args.seconds, "fps": args.fps,
             "camera_res": [args.width, args.height],
             "conditions": [c[0] for c in conditions], "seeds": seeds,
             "python": sys.version.split()[0], "platform": platform.platform(),
             "episodes": []}
    for seed in seeds:
        for label, frac, scale in conditions:
            t0 = time.perf_counter()
            ep = build_scene_and_run(seed, args.seconds, args.fps,
                                     (args.width, args.height), args.gl,
                                     frac, scale, label,
                                     with_render=not args.no_render)
            wall = time.perf_counter() - t0
            d = OUT / f"episode_{label}_seed{seed}"
            from PIL import Image
            cam_records = {}
            for cam_name, cam_frames in ep["frames"].items():
                cdir = d / ("frames" if cam_name == WORLD_CAM_NAME
                            else f"frames_{cam_name}")
                cdir.mkdir(parents=True, exist_ok=True)
                for f in cdir.glob("*.png"):
                    f.unlink()
                for i, fr in enumerate(cam_frames):
                    img = fr if fr.dtype == np.uint8 else \
                        (255 * np.clip(fr, 0, 1)).astype("uint8")
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    Image.fromarray(img).save(cdir / f"f{i:05d}.png")
                cam_records[cam_name] = {"dir": cdir.name, "n_frames": len(cam_frames)}
            n_fr = cam_records.get(WORLD_CAM_NAME, {}).get("n_frames", 0)
            np.savez_compressed(
                d / "episode.npz",
                **{f"truth/{k}": v for k, v in ep["truth"].items()},
                **{f"elec/{k}": v for k, v in ep["elec"].items()},
                **{"camera/detail_cam_pos_mm": np.asarray(ep["detail_cam_pos_mm"],
                                                          dtype=np.float32),
                   "camera/detail_cam_offset_mm": np.asarray(
                       ep["detail_cam_offset_mm"], dtype=np.float32)},
            )
            ft = np.asarray(ep["truth"]["frame_time_s"], dtype=float)
            if len(ft) != n_fr:
                raise RuntimeError(f"frame/time bookkeeping mismatch: {n_fr} frames "
                                   f"but {len(ft)} frame times")
            dt_fr = np.diff(ft) if len(ft) > 1 else np.asarray([np.nan])
            base_cam = {
                "fps": args.fps,
                "fps_requested": args.fps,
                "fps_achieved": float(1.0 / np.median(dt_fr)) if len(ft) > 1 else None,
                "frame_period_steps": int(round(1.0 / (args.fps * 1e-4))),
                "frame_dt_s_median": float(np.median(dt_fr)) if len(ft) > 1 else None,
                "frame_dt_s_min": float(np.min(dt_fr)) if len(ft) > 1 else None,
                "frame_dt_s_max": float(np.max(dt_fr)) if len(ft) > 1 else None,
                "n_frames": n_fr,
                "frame_time_s": ft.tolist(),
                "pixel_is_square": True,
            }
            world_cam = dict(base_cam, **{
                "name": WORLD_CAM_NAME,
                "frames_dir": cam_records.get(WORLD_CAM_NAME, {}).get("dir", "frames"),
                "pos_mm": list(WORLD_CAM_POS_MM),
                "fovy_deg": WORLD_CAM_FOVY_DEG,
                "xyaxes": list(WORLD_CAM_XYAXES),
                "image_size_px": [int(args.width), int(args.height)],
                "moves": False,
                "provenance": ("camera pose, fovy, image size and frame times are "
                               "DECLARED/MEASURED configuration, exactly what a real "
                               "experimenter would have; no ground truth is needed to "
                               "interpret a frame"),
            })
            detail_pos = np.asarray(ep["detail_cam_pos_mm"], dtype=float)
            detail_cam = dict(base_cam, **{
                "name": DETAIL_CAM_NAME,
                "frames_dir": cam_records.get(DETAIL_CAM_NAME, {}).get("dir",
                                                                        "frames_detailcam"),
                "pos_mm": [float(v) for v in detail_pos[0]] if len(detail_pos)
                          else [0.0, 0.0, DETAIL_CAM_HEIGHT_MM],
                "pos_mm_per_frame": [list(map(float, row)) for row in detail_pos],
                "fovy_deg": DETAIL_CAM_FOVY_DEG,
                "xyaxes": list(DETAIL_CAM_XYAXES),
                "image_size_px": [int(args.width), int(args.height)],
                "moves": True,
                "motion": ("motorised pan stage following the fly; the commanded "
                           "positions are recorded per frame"),
                "provenance": ("pose is DECLARED per frame and recorded; the COMMAND "
                               "that drove the stage came from the simulator's own "
                               "position, which is a TRACKER SIMPLIFICATION and means "
                               "the parser does not have to solve centring.  The "
                               "parser is told only the resulting poses."),
            })
            (d / "camera.json").write_text(json.dumps(world_cam, indent=2, sort_keys=True))
            (d / "camera_detail.json").write_text(json.dumps(detail_cam, indent=2,
                                                             sort_keys=True))
            (d / "payload.json").write_text(json.dumps(ep["payload"], indent=2,
                                                       sort_keys=True, default=str))
            (d / "model_signature.json").write_text(json.dumps(
                {"signature": ep["model_signature"],
                 "recording_chain": ep["recording_chain"]},
                indent=2, sort_keys=True, default=str))
            rec = {"seed": seed, "condition": label,
                   "load_fraction_requested": frac, "mass_scale": scale,
                   "n_frames": n_fr, "wall_seconds": wall,
                   "frames_per_camera": {k: v["n_frames"] for k, v in cam_records.items()},
                   "load_fraction_achieved": (ep["payload"] or {}).get(
                       "load_fraction_achieved"),
                   "payload_total_kg": (ep["payload"] or {}).get("total_payload_kg"),
                   "body_mass_sum_kg": (ep["model_signature"] or {}).get("mass_sum_kg"),
                   "mass_sha1": (ep["model_signature"] or {}).get("mass_sha1"),
                   "dir": str(d)}
            index["episodes"].append(rec)
            print(json.dumps(rec, default=str), flush=True)
    index["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "recording_index.json").write_text(json.dumps(index, indent=2,
                                                         sort_keys=True, default=str))
    print(f"wrote {OUT / 'recording_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
