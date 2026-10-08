#!/usr/bin/env python3
"""SIX-CAMERA HIGH-SPEED RECORDING in the grass meadow.

WHAT THIS RECORDS
-----------------
Eight things, per episode, all in one file plus a camera description:

  * ``frames/cam{k}/NNNNN.png`` -- the photos.  THESE AND THE CAMERA DESCRIPTION ARE THE
    ONLY INPUTS THE RECONSTRUCTION IS ALLOWED.  Everything else in the file is truth,
    for the audit.
  * ``marker/cam{k}``           -- the pixel centroid of every tracked marker in every
                                  frame, MEASURED from the photos by a threshold +
    centroid estimator, so the reconstruction's input is real image measurements and not
    the simulator's own coordinates.
  * ``truth/marker_world_mm``   -- the same markers' true world positions, used ONLY to
    score the reconstruction.
  * ``truth/body_*``            -- the fly's full body state, so absolute angles can be
    computed from truth independently of the pipeline.

WHY MARKERS AT ALL
------------------
A 2.5 mm fly seen by a camera 25 mm away is about 140 px long.  A claw excursion of
0.5 mm is 30 px.  Segmenting a leg out of that -- against grass -- is not a solved
problem, and every method that tries it (silhouette carving, learned pose estimation)
introduces an error the engineer cannot see.  A marker at a known body site turns the
hard part into centroid localisation, whose error is measurable.  The markers here are
0.05 mm spheres: 1 px at the close range used, bright, and on the fly's own joints.

THE WORLD FRAME
---------------
``arena.FIDUCIAL_MARKERS`` places six static markers at DECLARED world positions.  They
are recorded as truth too, so the reconstruction can FIT the world frame from what it
sees and the fit's residual is reported.  Without that step the reconstruction would be
in the cameras' own arbitrary frame and every angle would be meaningless.

Run (BODY environment):
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      venv_body/bin/python run_arena_record.py --seeds 0,1 --seconds 1.5 --fps 400
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
# MUJOCO_GL MUST BE IN THE ENVIRONMENT BEFORE ANYTHING IMPORTS mujoco.  MEASURED: BodyBackend
# does set os.environ["MUJOCO_GL"] from gl_backend, but this module imports mujoco itself inside
# run_episode() BEFORE constructing BodyBackend, and MuJoCo has already fixed its backend by
# then.  The symptom was a hard failure far from the cause:
#   "GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'" followed by
#   "mujoco.FatalError: an OpenGL platform library has not been loaded into this process".
# Setting a default here, at import, means --gl still wins in main() while every earlier import
# path is already safe.
os.environ.setdefault("MUJOCO_GL", "egl")

OUT = HERE / "outputs" / "arena"

from arena import (ARENA_CENTRE_MM, CAMERA_RING, FIDUCIAL_MARKERS,  # noqa: E402
                   FLY_MARKER_RADIUS_MM, arena_camera_report, arena_cameras,
                   marker_report)

#: Fly body segments that carry a marker.  DECLARED, chosen so that the set spans the BODY
#: frame (which is what makes an absolute body orientation recoverable) and gives one marker
#: per leg segment needed for a joint angle.
#:
#: Two on the body axis: thorax and abdomen, a rigid pair that fixes the body frame together
#: with the leg set.  Then, per leg: femur, tibia and tarsus tip, which is the minimum for an
#: inter-segment angle (two points give a DIRECTION, three give an angle between directions).
#: So 2 + 18 = 20 markers.
#:
#: ``c_rostrum`` WAS IN THIS SET AND HAD TO GO.  MEASURED: asking for a marker on it made
#: ``attach_marker_bodies`` delete the ``nmf/c_rostrum`` body from the fly's spec outright --
#: the compiled model came back with 82 bodies where 90 were expected, ``nmf/c_rostrum`` was
#: gone, and its marker geom had been re-parented onto ``nmf/c_thorax`` at the rostrum's
#: neutral offset.  A silently lost BODY is a far worse failure than a missing marker, and the
#: rostrum carries nothing the limb work needs, so it is dropped rather than patched around.
#: The recorder now ASSERTS that no base body disappeared, whichever segments are requested,
#: so this cannot recur unnoticed.
MARKER_BODY_SEGMENTS = (
    "c_thorax", "c_abdomen3",
    "lf_trochanterfemur", "lf_tibia", "lf_tarsus5",
    "lm_trochanterfemur", "lm_tibia", "lm_tarsus5",
    "lh_trochanterfemur", "lh_tibia", "lh_tarsus5",
    "rf_trochanterfemur", "rf_tibia", "rf_tarsus5",
    "rm_trochanterfemur", "rm_tibia", "rm_tarsus5",
    "rh_trochanterfemur", "rh_tibia", "rh_tarsus5",
)

#: Marker appearance.  DECLARED, and it is an EMISSIVE material rather than a white
#: sphere, for a MEASURED reason: in the meadow the grass itself saturates the image
#: (measured: max 255 already present with no marker in the scene), so brightness alone
#: does not separate a marker from a blade.  An emissive material makes the marker the
#: only thing that renders at the top of the range, so a simple threshold works on every
#: camera without per-camera tuning.  This is also what real motion capture does: the
#: markers are RETROREFLECTIVE and the illumination is arranged so they alone come back
#: bright.
#:
#: THE RADIUS HISTORY, all of it MEASURED:
#:   0.15 mm -> the grass swamped it: 62 saturated pixels from grass against 15 from markers,
#:              so the detector reported blade highlights (3-10 per frame, scattered over the
#:              whole image, where there are six markers).
#:   0.80 mm -> detectable but TOO BIG: on the original 2.6 mm constellation the discs merged
#:              into single components (one blob of 798 px stood for several markers), which
#:              destroyed association.  It was also 32 % of the fly's length.
#:   0.30 mm -> what is used now, together with a constellation spread over 12 mm so the discs
#:              no longer touch.  A 0.30 mm disc still spans several pixels at 28 mm range.
MARKER_RADIUS_MM = 0.30
MARKER_MATERIAL_NAME = "marker_emissive"
MARKER_RGB = (1.0, 1.0, 1.0, 1.0)

RENDERING_OPEN_FPS = 100000.0


def marker_specs() -> tuple[dict, ...]:
    """The six static fiducials as MJCF geoms for ``BodyConfig.extra_geoms``."""
    out = []
    for m in FIDUCIAL_MARKERS:
        out.append({"name": m["name"], "type": "sphere",
                    "size": [float(MARKER_RADIUS_MM)] * 3,
                    "pos": [float(v) for v in m["pos_mm"]],
                    "rgba": list(MARKER_RGB),
                    "material": MARKER_MATERIAL_NAME,
                    "contype": 0, "conaffinity": 0})
    return tuple(out)


def _fly_marker_specs(body_seg_names: tuple[str, ...]) -> tuple[dict, ...]:
    """Markers attached to fly segments, as extra bodies injected into the fly's spec."""
    return tuple({"segment": seg, "offset_m": (0.0, 0.0, 0.0),
                  "radius_m": 0.05e-3, "label": f"mk_{seg}"} for seg in body_seg_names)


def detect_markers(rgb: np.ndarray, thresh_sigma: float | None = None,
                   min_px: int = 1, max_px: int = 60,
                   abs_level: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """Marker detector: intensity-weighted connected-component centroids above a level.

    THE ONE DETECTOR, USED ON EVERY CAMERA AND EVERY FRAME.  Nothing about the fly's
    expected position enters it, so the measurements it produces are independent of the
    simulator's state -- which is what makes them a legitimate reconstruction input.

    WHY THE THRESHOLD IS ABSOLUTE AND NOT STATISTICAL, WHICH IS THE OPPOSITE OF WHAT WAS
    TRIED FIRST.  A robust-sigma threshold was the first design and it found NOTHING
    (measured: 0 candidates in 480 frames of all six cameras).  The reason is a property
    of the scene, not of the threshold: the meadow is full of bright blades, so the
    frame's own robust spread is large (median absolute deviation scale about 38 grey
    levels) and a 6-sigma cut lands ABOVE the image maximum -- unreachable.  What
    separates a marker from a blade is not statistics but the marker's EMISSIVE material,
    which makes it the only thing at the top of the range.  So the level is absolute,
    near saturation, and it is honest about the assumption it rests on: this detector
    works BECAUSE the markers are emissive, and it would fail on a scene where anything
    else saturates.  The per-frame count of saturated pixels is recorded with every
    episode so that assumption can be checked rather than trusted.
    """
    a = np.asarray(rgb[..., :3], dtype=float).mean(axis=2)
    if thresh_sigma is not None:
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        level = med + float(thresh_sigma) * max(1.4826 * mad, 1.0)
    else:
        level = float(abs_level)
    mask = a >= level
    if not mask.any():
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    return _centroids(a, mask, min_px, max_px)


def _centroids(a: np.ndarray, mask: np.ndarray, min_px: int, max_px: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of ``mask``, returned as intensity-weighted centroids."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    H, W = mask.shape
    stack: list[tuple[int, int]] = []
    for r0 in range(H):
        row_m, row_l = mask[r0], lab[r0]
        for c0 in range(W):
            if row_m[c0] and row_l[c0] == 0:
                n += 1
                row_l[c0] = n
                stack.append((r0, c0))
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] == 0:
                            lab[rr, cc] = n
                            stack.append((rr, cc))
    if n == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    rows_out, cols_out = [], []
    for cid in range(1, n + 1):
        rr, cc = np.nonzero(lab == cid)
        if len(rr) < min_px or len(rr) > max_px:
            continue
        w = a[rr, cc]
        s = float(w.sum())
        if s <= 0:
            continue
        rows_out.append(float((rr * w).sum() / s))
        cols_out.append(float((cc * w).sum() / s))
    return np.asarray(rows_out), np.asarray(cols_out)


def run_episode(seed: int, seconds: float, fps: float, cam_res: tuple[int, int],
                rig: str | None, gl: str | None, with_render: bool = True,
                marker_radius_mm: float = FLY_MARKER_RADIUS_MM,
                max_px: int | None = None) -> dict:
    from engine.embodied import BodyBackend, BodyConfig
    from electrode_payload import ElectrodePayloadConfig
    import mujoco

    payload_cfg = None
    if rig is not None:
        payload_cfg = ElectrodePayloadConfig(rig=rig, label=rig)
    cams = arena_cameras()
    cfg = BodyConfig(seed=seed, scene_preset="meadow_grass",
                     add_world_camera=False, add_tracking_camera=False,
                     add_vision=False, extra_cameras=cams,
                     extra_geoms=marker_specs(),
                     extra_materials=({"name": MARKER_MATERIAL_NAME,
                                       "rgba": [1.0, 1.0, 1.0, 1.0],
                                       "emission": 1.0, "reflectance": 0.0,
                                       "shininess": 0.0, "specular": 0.0},),
                     # FLY-BORNE MARKERS GO IN THROUGH THE CONFIG, NOT AFTERWARDS.
                     # MEASURED, AND THIS IS WHY THE PREVIOUS EPISODE HAS NONE: calling
                     # ``attach_marker_bodies(be.fly, ...)`` after BodyBackend() reports 21
                     # bodies added and changes NOTHING, because world.add_fly()/compile()
                     # have already run.  BodyConfig.fly_markers applies it before the
                     # compile.  ``marker_radius_mm`` is the identity budget: see
                     # FLY_MARKER_RADIUS_MM.
                     # radius_mm IS ALREADY IN MODEL UNITS (mm); do NOT multiply by 1e-3.
                     fly_markers={"segments": MARKER_BODY_SEGMENTS,
                                  "radius_mm": float(marker_radius_mm),
                                  "material": MARKER_MATERIAL_NAME,
                                  "offset_m": "outward"},
                     electrode_payload=payload_cfg)
    be = BodyBackend(cfg, gl_backend=gl)
    marker_attach = be.marker_report
    # A MARKER MUST NOT COST A BODY, AND THIS IS CHECKED RATHER THAN ASSUMED.  MEASURED: the
    # fly-marker helper, in its first form, silently deleted ``nmf/c_rostrum``, and when the
    # head markers were requested it also deleted ``nmf/c_haustellum`` and BOTH EYES -- the same
    # failure family as the first payload attempt (extra bodies are capacity-limited).
    #
    # THE CHECK IS A LANDMARK LIST, NOT A SECOND COMPILE.  The obvious way to write it -- build
    # the same config with fly_markers=None and diff the body names -- is a trap that is already
    # documented in body_backend: ``NeuroMechFly(name=...)`` hands back a SHARED spec, so
    # compiling a second world inside one episode corrupts the real fly (measured previously as
    # "repeated name 'c_thorax/...' in body"; measured again here as a fly that renders as
    # nothing but grass).  The landmarks below are the parts whose loss is the actual failure
    # mode, checked directly against the compiled model.
    _LANDMARKS = ("nmf/c_thorax", "nmf/c_rostrum", "nmf/c_haustellum", "nmf/l_eye",
                  "nmf/r_eye", "nmf/l_wing", "nmf/r_wing", "nmf/l_haltere", "nmf/r_haltere",
                  "nmf/c_abdomen3", "nmf/lf_tibia", "nmf/rh_tarsus5")
    _present = {str(mujoco.mj_id2name(be.model, mujoco.mjtObj.mjOBJ_BODY, i))
                for i in range(be.model.nbody)}
    _lost = [n for n in _LANDMARKS if n not in _present]
    if _lost:
        raise RuntimeError(
            f"attaching the fly markers REMOVED {len(_lost)} landmark body/bodies from the "
            f"model: {_lost}.  Refusing to record an episode on a model that lost bodies -- "
            "this is exactly how the first marker and payload attempts failed.")
    # The action source comes after the model checks, so a broken model fails before a single
    # simulation step is spent on it.
    be = be.attach_cpg_baseline()
    # BLOB SIZE GATE, SCALED FROM THE IMAGE INSTEAD OF HARD-CODED.  MEASURED: the previous
    # episode was recorded with ``max_px=60``, which was right for a 200-row frame where a
    # 0.30 mm-radius fiducial covers ~17 px, and WRONG at 800 rows where it covers ~210 px.
    # The detector therefore rejected every fiducial and returned only grass specks, which is
    # why the recorded ``px/*`` arrays contain no fiducial at all.  The gate is now derived
    # from the frame height so a resolution change cannot silently do that again.
    # AREA SCALES AS THE SQUARE OF THE LINEAR SIZE, so the gate has to as well.  The historical
    # 60 px was sized for 200 rows, where a 0.30 mm-radius fiducial covers roughly 60 px; at
    # 800 rows the SAME fiducial covers ~214 px and would be clipped by any linear scaling.
    # The gate is therefore
    # 60 px at 200 rows scaled by (rows/200)**2: 60 at 200 rows, 960 at 800 rows.
    _H = int(max(cam_res))
    blob_max_px = int(max_px) if max_px else int(round(60.0 * (_H / 200.0) ** 2))

    cam_names = [c["name"] for c in cams]
    m = be.model
    cam_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cam_names}
    if any(v < 0 for v in cam_ids.values()):
        raise RuntimeError(f"cameras missing: {cam_ids}")

    # body ids for the truth side
    all_names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))
                 for i in range(m.nbody)]
    marker_body_ids = {}
    for seg in MARKER_BODY_SEGMENTS:
        # The body that IS the segment: ``nmf/c_head``, not the marker child which is
        # ``nmf/c_head/mk_c_head``.  Matching ``endswith("/" + seg)`` picks the segment
        # itself and excludes the marker, which is why the exact suffix matters here.
        hit = [i for i, nm in enumerate(all_names) if nm and nm.endswith("/" + seg)]
        if not hit:
            raise RuntimeError(
                f"no body for segment {seg!r}.  Bodies present that mention it: "
                f"{[nm for nm in all_names if nm and seg in nm][:4]}")
        marker_body_ids[seg] = hit[0]
    static_marker_ids = {}
    for mk in FIDUCIAL_MARKERS:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mk["name"])
        if i < 0:
            raise RuntimeError(f"static marker geom {mk['name']!r} missing")
        static_marker_ids[mk["name"]] = i
    # FLY MARKER GEOM IDS, AND A HARD CHECK THAT THEY EXIST.  The truth recorded for a
    # fly marker must be the position of the MARKER ITSELF (a geom), not of the segment
    # origin: they coincide only if the marker really is at a zero offset, and the whole
    # point of the marker is that its rendered blob is what the camera sees.  MEASURED:
    # before this check the episode silently recorded segment origins for markers that had
    # never been compiled into the model at all.
    marker_geom_ids = {}
    _geom_names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i))
                   for i in range(m.ngeom)]
    _all_body_names = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))
                       for i in range(m.nbody)]
    for seg in MARKER_BODY_SEGMENTS:
        want = "/mk_" + seg + "_g"
        hit = [i for i, nm in enumerate(_geom_names) if nm and nm.endswith(want)]
        if not hit:
            raise RuntimeError(
                f"fly marker geom for segment {seg!r} is missing from the compiled model; "
                f"the marker set would be incomplete.  Geoms present that mention it: "
                f"{[nm for nm in _geom_names if nm and seg in nm][:4]}")
        marker_geom_ids[seg] = hit[0]
    # THE SIZE IS CHECKED IN MODEL UNITS, because a radius in the wrong unit is invisible
    # rather than wrong: a 0.00013-unit sphere is a fraction of a pixel and every render test
    # simply shows "no marker", which is indistinguishable from "no geom".
    # MARKERS MUST BE MASSLESS, AND THE CHECK IS ON THE FLY'S TOTAL MASS.  There is no
    # ``geom_mass`` on this MjModel (MEASURED: AttributeError, only geom_aabb and friends), so
    # the check goes through the payload module's own yardstick, which sums ``body_mass`` and
    # compares it against the measured unloaded fly mass.  That is the right test anyway: it
    # fails on ANY added mass, from any source, and it is the same instrument that proved the
    # payload fractions.  MEASURED: without mass=0 on the marker geoms, 20 markers shifted the
    # fly's resting height from z = 1.4207 to 1.800 model units and destabilised the solver,
    # because MuJoCo computes geom inertia from the SIZE NUMBERS (0.13 here) as if they were
    # metres -- ~9 kg of "mass" on a 1e-6 kg fly.
    from electrode_payload import assert_yardstick
    marker_mass_check = assert_yardstick(m)
    marker_mass_check["note"] = ("the fly's total mass must be the unloaded yardstick; markers "
                                 "are visual only")
    _sz = float(m.geom_size[marker_geom_ids[MARKER_BODY_SEGMENTS[0]]][0])
    if abs(_sz - float(marker_radius_mm)) > 1e-9 or _sz < 0.01:
        raise RuntimeError(
            f"fly marker radius compiled to {_sz!r} in MODEL UNITS (mm) but "
            f"{float(marker_radius_mm)!r} was requested; the model is in MILLIMETRES, so a "
            "radius passed in metres becomes an invisible sub-pixel sphere.")
    # EACH MARKER GEOM MUST RIDE ITS OWN SEGMENT -- AT THE OFFSET THAT WAS REQUESTED.
    #
    # This check used to demand a ZERO offset, because a marker hard on the segment origin is
    # what makes its world position the joint centre.  MEASURED, by rendering: a marker at the
    # segment ORIGIN is INSIDE the body's own mesh and is invisible at any usable radius, so the
    # design now pushes each marker outward by a fixed offset in its own segment's frame.  The
    # invariant that still matters, and is still checked, is that the geom is on ITS OWN segment
    # body and sits at exactly the offset recorded for it -- a marker re-parented onto another
    # body (measured previously for c_rostrum, which landed on nmf/c_thorax) would measure the
    # wrong thing, and a marker at an unrecorded offset would silently shift the bone it defines.
    _requested = {}
    for _sp in (marker_attach or {}).get("specs", ()):
        _requested[_sp["segment"]] = tuple(float(v) for v in _sp["offset_m"])
    _seg_body_id = {}
    for seg in MARKER_BODY_SEGMENTS:
        hits = [i for i, nm in enumerate(_all_body_names) if nm and nm.endswith("/" + seg)]
        if hits:
            _seg_body_id[seg] = hits[0]
    _offenders = []
    for seg, gid in marker_geom_ids.items():
        bid = int(m.geom_bodyid[gid])
        own_ok = _seg_body_id.get(seg) == bid
        want = _requested.get(seg)
        got = tuple(float(v) for v in m.geom_pos[gid])
        off_ok = want is None or max(abs(a - b) for a, b in zip(want, got)) < 1e-9
        if not own_ok or not off_ok:
            _offenders.append((seg, _geom_names[gid], _all_body_names[bid], got, want, own_ok))
    if _offenders:
        raise RuntimeError(
            "fly marker geoms are not riding their own segment at the requested offset: "
            + "; ".join(f"{s2}: geom {g2} on {b2} offset {o2} wanted {w2} own_body={ok2}"
                        for s2, g2, b2, o2, w2, ok2 in _offenders)
            + ".  A marker that does not sit on its own segment, at its recorded offset, would "
              "measure the wrong thing.")

    dt = cfg.timestep_s
    n_steps = int(round(seconds / dt))
    frame_period_steps = max(1, int(round(1.0 / (fps * dt))))
    obs_period = max(1, int(round(1.0 / (500.0 * dt))))

    renderer = None
    if with_render:
        renderer = be.sim.set_renderer(cam_names, camera_res=cam_res,
                                       playback_speed=1.0,
                                       output_fps=RENDERING_OPEN_FPS,
                                       buffer_frames=True)

    frame_time_s: list[float] = []
    truth_static: list[np.ndarray] = []
    truth_fly: list[np.ndarray] = []
    T, THORAX, BODY_P, BODY_Q, CFound, CF, AF, ACTQ = ([] for _ in range(8))
    try:
        for k in range(n_steps):
            be.step()
            if k % obs_period == 0:
                obs = be.observe()
                T.append(obs.time_s)
                THORAX.append(obs.thorax_position_mm)
                BODY_P.append(obs.body_positions_mm)
                BODY_Q.append(obs.body_rotations_wxyz)
                CFound.append(obs.contact_found_raw.copy())
                CF.append(obs.contact_forces)
                AF.append(obs.actuator_forces)
                ACTQ.append(obs.joint_angles_rad)
            if renderer is not None and k % frame_period_steps == 0:
                if be.sim.render_as_needed():
                    frame_time_s.append(float(be.data.time))
                    # Truth: marker world positions.  Static markers are geoms, so their
                    # positions come from geom_xpos; fly markers are bodies.
                    sm = np.array([be.data.geom_xpos[static_marker_ids[mk["name"]]]
                                   for mk in FIDUCIAL_MARKERS], dtype=float)
                    truth_static.append(sm)
                    truth_fly.append(np.array([be.data.geom_xpos[marker_geom_ids[s]]
                                               for s in MARKER_BODY_SEGMENTS],
                                              dtype=float))
        frames = {}
        if renderer is not None:
            frames = {n: [np.asarray(f).copy() for f in renderer.frames.get(n, [])]
                      for n in cam_names}
    finally:
        pass

    # ---- PIXEL MEASUREMENTS: the reconstruction's actual input -----------------
    marker_px: dict[str, np.ndarray] = {}
    marker_peak: dict[str, list] = {}
    n_cand: dict[str, list] = {}
    for n in cam_names:
        rows_all, cols_all, npk, ncd = [], [], [], []
        for fr in frames.get(n, []):
            rr, cc = detect_markers(fr, max_px=blob_max_px)
            rows_all.append(rr)
            cols_all.append(cc)
            ncd.append(int(len(rr)))
            npk.append(float(np.asarray(fr[..., :3], dtype=float).mean(axis=2).max()))
        marker_px[n] = {"n_candidates": np.array(ncd, dtype=int),
                        "peak_grey": np.array(npk, dtype=float),
                        "rows": rows_all, "cols": cols_all}
        marker_peak[n] = npk
        n_cand[n] = ncd

    truth = {
        "time_s": np.asarray(T, dtype=float),
        "frame_time_s": np.asarray(frame_time_s, dtype=float),
        "thorax_mm": np.asarray(THORAX, dtype=np.float32),
        "body_positions_mm": np.asarray(BODY_P, dtype=np.float32),
        "body_rotations_wxyz": np.asarray(BODY_Q, dtype=np.float32),
        "contact_found_raw": np.asarray(CFound, dtype=np.float32),
        "contact_present": np.asarray(CFound, dtype=np.float32) > 0,
        "contact_forces": np.asarray(CF, dtype=np.float32),
        "actuator_forces": np.asarray(AF, dtype=np.float32),
        "joint_angles_rad": np.asarray(ACTQ, dtype=np.float32),
        "marker_static_world_mm": np.asarray(truth_static, dtype=np.float32),
        "marker_fly_world_mm": np.asarray(truth_fly, dtype=np.float32),
    }
    return {"truth": truth, "frames": frames, "marker_px": marker_px,
            "n_candidates": n_cand, "cam_names": cam_names,
            "marker_body_ids": marker_body_ids,
            "marker_geom_ids": marker_geom_ids,
            "marker_attach": marker_attach,
            "marker_mass_check": marker_mass_check,
            "blob_max_px": blob_max_px,
            "marker_radius_mm": float(marker_radius_mm),
            "static_marker_ids": static_marker_ids,
            "describe": be.describe(),
            "payload": be.payload_report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--fps", type=float, default=400.0)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--gl", default="egl")
    ap.add_argument("--rigs", default="bare,tether,telemetry")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--marker-radius-mm", type=float, default=FLY_MARKER_RADIUS_MM,
                    help="radius of the FLY-BORNE markers; the identity budget says the "
                         "diameter must stay under the 0.257 mm clearance to the nearest "
                         "confusable marker (tools_identity_budget.py)")
    ap.add_argument("--max-px", type=int, default=None,
                    help="blob size gate; defaults to a value scaled from the frame height")
    args = ap.parse_args()
    # --gl wins, and it has to be set BEFORE run_episode imports mujoco (see the note at the top).
    os.environ["MUJOCO_GL"] = args.gl

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arena_geometry.json").write_text(json.dumps(
        {"camera_ring": CAMERA_RING, "camera_report": arena_camera_report(),
         "markers": marker_report(),
         "marker_body_segments": list(MARKER_BODY_SEGMENTS),
         "column_convention": ("REAL multi-camera work defines the world frame only "
                               "through visible markers; both frames are kept and "
                               "their rigid transform is fitted and reported"),
         "resolutions": {"width": args.width, "height": args.height},
         "fly_marker_radius_mm": args.marker_radius_mm,
         "frame_rate_hz": args.fps, "seconds": args.seconds},
        indent=2, sort_keys=True, default=str))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rigs = [s for s in args.rigs.split(",") if s.strip()]
    index = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seconds": args.seconds, "fps": args.fps,
             "resolution": [args.width, args.height], "seeds": seeds, "rigs": rigs,
             "python": sys.version.split()[0], "platform": platform.platform(),
             "episodes": []}
    for seed in seeds:
        for rig in rigs:
            t0 = time.perf_counter()
            ep = run_episode(seed, args.seconds, args.fps, (args.width, args.height),
                             None if rig == "bare" else rig, args.gl,
                             with_render=not args.no_render,
                             marker_radius_mm=args.marker_radius_mm,
                             max_px=args.max_px)
            wall = time.perf_counter() - t0
            d = OUT / f"arena_{rig}_seed{seed}"
            n_fr = len(ep["truth"]["frame_time_s"])
            from PIL import Image
            for n in ep["cam_names"]:
                cdir = d / "frames" / n
                cdir.mkdir(parents=True, exist_ok=True)
                for f in cdir.glob("*.png"):
                    f.unlink()
                for i, fr in enumerate(ep["frames"].get(n, [])):
                    img = fr if fr.dtype == np.uint8 else \
                        (255 * np.clip(fr, 0, 1)).astype("uint8")
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    Image.fromarray(img).save(cdir / f"f{i:05d}.png")
            np.savez_compressed(
                d / "episode.npz",
                **{f"truth/{k}": v for k, v in ep["truth"].items()},
                # FLAT arrays plus frame offsets, NOT ragged object arrays: object arrays
                # need allow_pickle to load, and a reconstruction input that requires
                # unpickling is both a security smell and a portability trap.
                **{f"px/{n}/rows": np.concatenate(
                    [np.asarray(r, dtype=np.float32) for r in ep["marker_px"][n]["rows"]]
                ) if any(len(r) for r in ep["marker_px"][n]["rows"])
                else np.zeros(0, dtype=np.float32) for n in ep["cam_names"]},
                **{f"px/{n}/cols": np.concatenate(
                    [np.asarray(c, dtype=np.float32) for c in ep["marker_px"][n]["cols"]]
                ) if any(len(c) for c in ep["marker_px"][n]["cols"])
                else np.zeros(0, dtype=np.float32) for n in ep["cam_names"]},
                **{f"px/{n}/offsets": np.concatenate(
                    [[0], np.cumsum([len(r) for r in ep["marker_px"][n]["rows"]])]
                ).astype(np.int32) for n in ep["cam_names"]},
                **{f"px/{n}/n_candidates": ep["marker_px"][n]["n_candidates"]
                   for n in ep["cam_names"]},
                **{f"px/{n}/peak_grey": ep["marker_px"][n]["peak_grey"]
                   for n in ep["cam_names"]},
            )
            (d / "markers.json").write_text(json.dumps(
                {"marker_body_segments": list(MARKER_BODY_SEGMENTS),
                 "static_markers": [dict(m) for m in FIDUCIAL_MARKERS],
                 "marker_radius_mm": 0.05,
                 "attach_report": ep["marker_attach"],
                 "detector": ("absolute high threshold at 6 robust sigmas above the "
                              "frame median; intensity-weighted connected-component "
                              "centroids; no expected position used")},
                indent=2, sort_keys=True, default=str))
            (d / "payload.json").write_text(json.dumps(ep["payload"], indent=2,
                                                       sort_keys=True, default=str))
            rec = {"seed": seed, "rig": rig, "n_frames": n_fr, "wall_seconds": wall,
                   "frames_per_camera": {n: len(ep["frames"].get(n, []))
                                         for n in ep["cam_names"]},
                   "candidates_per_camera_median": {
                       n: float(np.median(ep["n_candidates"][n])) if
                       ep["n_candidates"][n] else 0.0 for n in ep["cam_names"]},
                   "dir": str(d)}
            index["episodes"].append(rec)
            print(json.dumps(rec, default=str), flush=True)
    index["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "arena_index.json").write_text(json.dumps(index, indent=2, sort_keys=True,
                                                     default=str))
    print(f"wrote {OUT / 'arena_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
