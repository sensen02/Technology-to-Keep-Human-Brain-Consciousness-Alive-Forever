#!/usr/bin/env python3
"""S1a: MAKE THE MUSCLE-DRIVEN BODY OBSERVABLE.

THE PROBLEM, MEASURED.  This platform has two embodied stacks that share the neural tier and
nothing else:

  * stack A  `engine/embodied/loop.py` + NeuroMechFly + 42 POSITION actuators.  It owns the camera
             ring, the scene, the tactile chain and the electrode -- and its gait is a
             motion-capture replay driven by two CPG knobs.
  * stack B  `run_six_leg_neural_loop.py` + FlyMimic + 90 Hill muscles.  Its docstring states the
             property stack A lacks: "There is NO state-to-action function in it.  A muscle's
             activation IS the measured firing of that muscle's own motor neurons."

MEASURED isolation: `run_six_leg_neural_loop.py` contains camera 0, scene 0, electrode 0, render 0,
vision 0; `engine/embodied/body_backend.py` and `loop.py` contain muscle 0, Hill 0.  So the body that
neurons actually drive is the body nobody can see.

WHAT THIS SCRIPT DOES.  It does NOT touch the model's dynamics.  It attaches the SAME observation
geometry the arena already declares -- `arena.arena_cameras()` and `arena.FIDUCIAL_MARKERS` are
plain geometry, independent of which stack owns the body -- and renders frames in the recorder's own
on-disk format, so the EXISTING camera -> geometry chain can be pointed at stack B unchanged:

    tools_episode_health.py      the recording-health gate (fiducial hit rate)
    run_arena_reconstruct.py     triangulation and the world-frame fit
    limb_tracking.py             the per-segment 3D tracker

WHAT IT DOES NOT DO.  It does not make stack B walk or stand: `stood = False` in 5 of 5
configurations is a BODY-PHYSICS problem and belongs to a separate workstream.  Observability and
locomotion are different questions, and conflating them is how "the platform is nearly there" gets
said about a body that cannot support itself.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_make_stackb_observable.py --frames 12
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

STACK_B_XML = HERE / "outputs" / "muscles_six_legs" / "fruitfly_six_leg_standing.xml"
EPISODE = HERE / "outputs" / "arena" / "stackb_observable"
DETECT_THRESHOLD = 240.0
DETECT_MIN_PX = 3

#: THE FLY'S OWN MARKERS, ON STACK B.  The keys are STACK A's logical segment names, deliberately:
#: `arena_geometry.json` declares `marker_body_segments` in those names, `limb_tracking.py` reads the
#: segment list from there and indexes `truth/marker_fly_world_mm` by it, so keeping the keys means
#: the whole existing camera -> geometry -> limb chain works on stack B with ZERO changes (which is
#: the point of attaching the arena's own geometry rather than inventing a second convention).
#:
#: The VALUES are stack B's actual MJCF body names, and they are NOT a renaming of stack A's: this
#: model splits the leg differently.  MEASURED from the spec's body tree, stack B has
#:     LFCoxa -> LFTrochanter -> LFFemur -> LFTibia -> LFTarsus1..5
#: but for the mid and hind legs there is NO trochanter at all:
#:     LMCoxa -> LMFemur -> LMTibia -> LMTarsus1..5
#: so stack A's `lf_trochanterfemur` has no single counterpart.  All six legs are therefore bound to
#: their FEMUR, which is the one segment that is anatomically the same limb in all six legs -- rather
#: than to the fore leg's trochanter, which would make the fore leg's bone a different bone from the
#: other five and silently corrupt every joint angle computed across legs.
STACKB_MARKER_BODY: dict[str, str] = {
    "c_thorax": "Thorax",
    "c_abdomen3": "A3",
    "lf_trochanterfemur": "LFFemur", "lf_tibia": "LFTibia", "lf_tarsus5": "LFTarsus5",
    "lm_trochanterfemur": "LMFemur", "lm_tibia": "LMTibia", "lm_tarsus5": "LMTarsus5",
    "lh_trochanterfemur": "LHFemur", "lh_tibia": "LHTibia", "lh_tarsus5": "LHTarsus5",
    "rf_trochanterfemur": "RFFemur", "rf_tibia": "RFTibia", "rf_tarsus5": "RFTarsus5",
    "rm_trochanterfemur": "RMFemur", "rm_tibia": "RMTibia", "rm_tarsus5": "RMTarsus5",
    "rh_trochanterfemur": "RHFemur", "rh_tibia": "RHTibia", "rh_tarsus5": "RHTarsus5",
}

#: The torso markers need a bigger stand-off than a leg marker: a leg is a thin capsule, while the
#: thorax is a thick mesh, so the same stand-off puts a leg marker outside its cuticle and a thorax
#: marker inside it.  MEASURED in stack A, where a single distance put both the thorax and the
#: abdomen markers at 0 % visibility.
STACKB_TORSO_MARGIN_MM = 0.30
STACKB_LEG_MARGIN_MM = 0.06


def _find_body(spec, name: str):
    """Depth-first search of the UNCOMPILED spec tree, because names are only unique by path here."""
    stack = [spec.worldbody]
    while stack:
        b = stack.pop()
        if str(getattr(b, "name", "")) == name:
            return b
        stack.extend(list(b.bodies))
    return None


def attach_fly_markers(spec, radius_mm: float | None = None, margin_mm: dict | None = None):
    """Put one bright sphere on each of stack B's own segments -- the markers `limb_tracking` needs.

    BEFORE THIS, the stack B episodes carried `truth/marker_static_world_mm` ONLY, i.e. the twelve
    ARENA fiducials.  That is enough to recover camera geometry and the body's POSITION, and it is
    why the earlier delivery had to say "limb kinematics are not established": no marker rides the
    fly, so `limb_tracking.py` -- which reads `truth/marker_fly_world_mm` -- could not run on these
    episodes at all.

    THE OFFSET IS MEASURED FROM THIS MODEL, NOT COPIED FROM STACK A.  A marker at a segment's ORIGIN
    is inside the body's own mesh and renders as nothing (MEASURED in stack A, at every radius
    tried), so each marker needs a stand-off.  Stack A's numbers (0.18 mm legs / 0.55 mm torso) were
    tuned for NeuroMechFly's mesh sizes, and stack B is a different model with different meshes, so
    copying them would be an assumption dressed as a measurement.  Instead the stand-off is computed
    here from the segment's OWN geoms: the model is compiled once with the markers at zero offset,
    and each marker is then pushed along the segment's DORSAL direction far enough to clear the
    bounding sphere of every geom on that body, plus a margin.  "Dorsal" means world +Z projected
    perpendicular to the segment's own bone axis -- stack A's rule, and the one that measured well
    there, because a marker pushed RADIALLY from a leg ends up inside the NEXT segment.

    Returns a report; `render_episode` and the recorder both use `geom_name_by_segment` to read the
    marker positions back out of `geom_xpos`, so the truth and the rendered markers are the SAME
    geoms (an earlier stack A attempt recorded segment ORIGINS while claiming they were markers).
    """
    import mujoco

    if radius_mm is None:
        from arena import FLY_MARKER_RADIUS_MM
        radius_mm = float(FLY_MARKER_RADIUS_MM)
    margin_mm = margin_mm or {}
    from electrode_payload import ensure_marker_material
    material = ensure_marker_material(spec)

    geom_name_by_segment: dict[str, str] = {}
    added: list[dict] = []
    for seg, body_name in STACKB_MARKER_BODY.items():
        body = _find_body(spec, body_name)
        if body is None:
            raise KeyError(f"stack B has no body named {body_name!r} for marker segment {seg!r}")
        name = f"mk_{seg}"
        g = body.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [float(radius_mm), 0.0, 0.0]
        g.pos = [0.0, 0.0, 0.0]
        g.material = material
        g.contype = 0
        g.conaffinity = 0
        # MASSLESS.  MEASURED in stack A: MuJoCo computes geom inertia from the SIZE NUMBERS as if
        # they were metres, so a 0.12-unit sphere carries kilograms on a microgram fly, shifting the
        # resting height and destabilising the solver.  A marker must not change the mechanics it
        # measures.
        g.mass = 0.0
        g.density = 0.0
        geom_name_by_segment[seg] = name
        added.append({"segment": seg, "body_name": body_name, "geom_name": name})

    # PASS 1: compile a throwaway model AT THE NEUTRAL POSE and measure each body's geometry.
    probe = spec.compile()
    pd = mujoco.MjData(probe)
    mujoco.mj_resetDataKeyframe(probe, pd, 0)
    mujoco.mj_forward(probe, pd)
    # `MjModel` HAS NO `geom_names` / `body_names` ATTRIBUTE -- MEASURED: AttributeError on both.  The
    # names come from `mj_id2name`, which returns None for anonymous elements, so every lookup here is
    # explicitly None-guarded rather than relying on a list of names existing.
    def _name(objtype, i):
        n = mujoco.mj_id2name(probe, objtype, i)
        return None if n is None else str(n)

    geom_id = {n: i for i in range(probe.ngeom)
               if (n := _name(mujoco.mjtObj.mjOBJ_GEOM, i)) in set(
                   geom_name_by_segment.values())}
    body_id = {n: i for i in range(probe.nbody)
               if (n := _name(mujoco.mjtObj.mjOBJ_BODY, i)) is not None}

    offsets: dict[str, list[float]] = {}
    dorsal_report: dict[str, dict] = {}
    for seg, body_name in STACKB_MARKER_BODY.items():
        bid = body_id.get(body_name)
        if bid is None:
            raise KeyError(f"compiled model lost body {body_name!r}")
        specs = [b for b in STACKB_MARKER_BODY if STACKB_MARKER_BODY[b] == body_name]
        if len(specs) > 1:
            raise RuntimeError(f"body {body_name!r} claimed by segments {specs}")
        wall = np.asarray(pd.xmat[bid], dtype=float).reshape(3, 3)
        # DORSAL, in the body's own frame: world +Z expressed locally, minus its component along the
        # bone axis.  The bone axis is the direction to the segment's first child body, which is what
        # the segment actually extends toward; a leaf segment (tarsus5) has no child, so it falls back
        # to world +Z itself, which for a horizontal tarsus is already nearly dorsal.
        up_local = wall.T @ np.array([0.0, 0.0, 1.0])
        kids = [i for i in range(probe.nbody) if int(probe.body_parentid[i]) == bid]
        if kids:
            axis = np.asarray(probe.body_pos[kids[0]], dtype=float)
            n = float(np.linalg.norm(axis))
            axis = axis / n if n > 1e-12 else up_local
        else:
            axis = up_local
        dorsal = up_local - float(up_local @ axis) * axis
        nd = float(np.linalg.norm(dorsal))
        if nd < 1e-6:
            # The bone axis is parallel to world +Z: pick any perpendicular, deterministically.
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(float(fallback @ axis)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            dorsal = fallback - float(fallback @ axis) * axis
            nd = float(np.linalg.norm(dorsal))
        dorsal = dorsal / nd
        # HOW FAR IS FAR ENOUGH, from the segment's own geoms: for every geom on this body, its
        # furthest extent along `dorsal` measured in the BODY frame.
        reach = 0.0
        for gi in range(probe.ngeom):
            if int(probe.geom_bodyid[gi]) != bid:
                continue
            c = np.asarray(probe.geom_pos[gi], dtype=float)
            r = float(probe.geom_rbound[gi])
            reach = max(reach, float(c @ dorsal) + r)
        is_torso = seg in ("c_thorax", "c_abdomen3")
        margin = float(margin_mm.get(seg, STACKB_TORSO_MARGIN_MM if is_torso
                                     else STACKB_LEG_MARGIN_MM))
        dist = reach + margin
        offsets[seg] = [float(v) for v in (dorsal * dist)]
        dorsal_report[seg] = {"geom_reach_mm": round(reach, 4), "margin_mm": margin,
                              "distance_mm": round(dist, 4),
                              "dorsal_body_frame": [round(float(v), 4) for v in dorsal]}
        # WRITE IT BACK INTO THE SPEC (the throwaway model is already stale and is discarded).
        gid_local = None
        for g in _find_body(spec, body_name).geoms:
            if str(getattr(g, "name", "")) == geom_name_by_segment[seg]:
                gid_local = g
                break
        if gid_local is None:
            raise RuntimeError(f"cannot find marker geom {geom_name_by_segment[seg]!r} to offset")
        gid_local.pos = offsets[seg]

    radii = [float(probe.geom_size[gi][0]) for gi in geom_id.values()]
    if radii and abs(max(radii) - float(radius_mm)) > 1e-9:
        raise RuntimeError(
            f"marker radius compiled to {max(radii)!r} but {float(radius_mm)!r} was requested; this "
            "model is in MILLIMETRES, so a radius passed in metres becomes an invisible sphere")
    return {"n_markers": len(added), "radius_mm": float(radius_mm), "material": material,
            "markers": added, "geom_name_by_segment": geom_name_by_segment,
            "offsets_mm": offsets, "offset_derivation": dorsal_report,
            "units": "MODEL UNITS = mm",
            "note": ("stack A's logical segment names mapped onto stack B's own body names; all six "
                     "legs bind the FEMUR because stack B has no trochanter on the mid and hind legs")}


def attach_observation_geometry(spec, ring_rotation_deg: float = 0.0,
                                offwidth: int = 800, offheight: int = 640):
    """Add the arena's DECLARED fiducials and camera ring to ANY spec of stack B.

    ONE IMPLEMENTATION, TWO CALLERS: this script's own renderer and
    `run_six_leg_neural_loop.build(observable=True)`.  Two copies would drift, and the drift would
    show up as a ring that matches in one path and not the other.
    """
    import mujoco
    from electrode_payload import ensure_marker_material
    from arena import arena_cameras, FIDUCIAL_MARKERS

    # THE OFFSCREEN FRAMEBUFFER MUST BE ENLARGED: `flygym`'s `_load_mjcf` sets it to 640x640, so a
    # 640-row x 800-col render fails with "Image width 800 > framebuffer width 640".
    spec.visual.global_.offwidth = int(offwidth)
    spec.visual.global_.offheight = int(offheight)
    mat = ensure_marker_material(spec)
    for m in FIDUCIAL_MARKERS:
        g = spec.worldbody.add_geom()
        g.name = f"fid_{m['name']}"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE      # the enum; MjsGeom.type rejects a bare 1
        g.size = [float(m["radius_mm"]), 0.0, 0.0]
        g.pos = [float(v) for v in m["pos_mm"]]
        g.material = mat
        g.contype = 0
        g.conaffinity = 0                          # a reference marker must not touch anything
    cam_names = []
    for c in arena_cameras():
        cam = spec.worldbody.add_camera()
        cam.name = c["name"]
        pos = np.asarray(c["pos"], dtype=float)
        if ring_rotation_deg:
            th = np.radians(float(ring_rotation_deg))
            R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0, 0, 1]])
            pos = R @ pos
        cam.pos = [float(v) for v in pos]
        # `MjsCamera` HAS NO `xyaxes` -- the XML attribute becomes `quat` at parse time.  The
        # xyaxes -> quat -> xyaxes ROUND TRIP IS CHECKED, because a wrong layout convention would
        # silently mis-orient the ring.
        mm = np.asarray(c["xyaxes"], dtype=float)
        R3 = np.column_stack([mm[:3], mm[3:], np.cross(mm[:3], mm[3:])])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R3.ravel())
        back = np.zeros(9)
        mujoco.mju_quat2Mat(back, q)
        back = back.reshape(3, 3)
        if not np.allclose(back[:, 0], mm[:3], atol=1e-9) or \
                not np.allclose(back[:, 1], mm[3:], atol=1e-9):
            raise RuntimeError(f"xyaxes -> quat round trip changed {c['name']}'s orientation")
        cam.quat = [float(v) for v in q]
        cam.fovy = float(c["fovy"])
        cam_names.append(c["name"])
    return cam_names


def build_spec(ring_rotation_deg: float = 0.0, fly_markers: bool = False,
               marker_report: dict | None = None):
    """Stack B's model + the arena's declared fiducials and camera ring.  Dynamics untouched.

    `fly_markers=True` additionally puts one sphere on each of the fly's own segments, which is what
    lets `limb_tracking.py` run on stack B at all.  The offsets are MEASURED from this model's own
    geoms (see `attach_fly_markers`), not copied from stack A.
    """
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    spec = _load_mjcf(str(STACK_B_XML))
    cam_names = attach_observation_geometry(spec, ring_rotation_deg)
    report = None
    if fly_markers:
        report = attach_fly_markers(spec)
        if marker_report is not None:
            marker_report.clear()
            marker_report.update(report)
    return spec, cam_names


def fly_marker_truth(model, data, geom_name_by_segment: dict, order: list[str]) -> np.ndarray:
    """Read the fly markers' world positions OUT OF THE SAME GEOMS THAT WERE RENDERED.

    This is the stack A rule, restated because it was violated once there: an earlier attempt recorded
    each segment's ORIGIN while claiming to record its MARKER, so the "truth" was a different point
    from the bright sphere in the image and the tracker's error was measured against the wrong thing.
    Here `geom_xpos` of `mk_<segment>` IS the sphere that the camera saw, offsets included.

    The column order is `order` -- the caller's declared segment list -- so the truth array and
    `marker_body_segments` cannot silently disagree.
    """
    import mujoco

    ids = {str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)): i
           for i in range(model.ngeom)}
    missing = [order.index(s) for s in order if geom_name_by_segment[s] not in ids]
    if missing:
        raise RuntimeError(
            f"fly markers missing from the compiled model at columns {missing}: "
            f"{[geom_name_by_segment[order[i]] for i in missing]}.  Refusing to record a truth "
            "array in which some columns are not marker geoms")
    return np.asarray([np.asarray(data.geom_xpos[ids[geom_name_by_segment[s]]], dtype=np.float64)
                       for s in order], dtype=np.float64)


def render_episode(n_frames: int, ring_rotation_deg: float = 0.0, settle_steps: int = 4000,
                   out_dir=None, fly_markers: bool = True) -> dict:
    import mujoco
    from PIL import Image
    from scipy import ndimage
    from arena import CAMERA_RING, FIDUCIAL_MARKERS, marker_grey

    marker_report: dict = {}
    spec, cam_names = build_spec(ring_rotation_deg, fly_markers=fly_markers,
                                marker_report=marker_report)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    # THE RESOLUTION COMES FROM THE RECORD, NOT FROM THE DECLARATION.  MEASURED DISCREPANCY, and
    # it cost a render: `arena.py`'s CAMERA_RING declares `resolution_px = [200, 260]`, while
    # `arena_geometry.json` -- the record the reconstruction and the health gate actually consume --
    # says height 640 / width 800, and stack A's own frames are 640x800.  The declaration is FOUR
    # TIMES smaller than what was used.  Rendering at the declared value made a 0.30 mm fiducial
    # about three pixels across, so its pixels were nearly all anti-aliased edge: the frame's max
    # grey was 243 and `blobs_from_png`'s mean>=240 threshold found 4 of 12 markers instead of 12.
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    res = geom.get("resolutions") or {}
    h = int(res.get("height") or CAMERA_RING["resolution_px"][0])
    w = int(res.get("width") or CAMERA_RING["resolution_px"][1])
    declared = tuple(int(v) for v in CAMERA_RING["resolution_px"])
    print(f"  rendering {n_frames} frames x {len(cam_names)} cameras at {h}x{w} (HxW)")
    if (h, w) != declared:
        print(f"  NOTE: arena.py DECLARES resolution_px={declared} but the record says {(h, w)}; "
              f"using the record (see the delivery note on this discrepancy)")
    renderer = mujoco.Renderer(model, height=h, width=w)

    # A CALLER-SUPPLIED DESTINATION, because an injection that re-renders must not overwrite the
    # good episode.  MEASURED: the first version of the acceptance test re-rendered in place to make
    # its rotated control, so "restoring" the episode restored the ROTATED one and destroyed the
    # 12-frame artifact -- the suite then judged a 3-frame rotated episode and failed.
    dest = Path(out_dir) if out_dir else EPISODE
    dest.mkdir(parents=True, exist_ok=True)
    for c in cam_names:
        (dest / "frames" / c).mkdir(parents=True, exist_ok=True)

    # px/ group in the recorder's own format: flat rows/cols + offsets + per-frame counts
    rows_by_cam, cols_by_cam, peaks_by_cam, ncan_by_cam = {}, {}, {}, {}
    areas_by_cam: dict[str, list] = {}
    for c in cam_names:
        rows_by_cam[c], cols_by_cam[c], peaks_by_cam[c], ncan_by_cam[c] = [], [], [], []
        areas_by_cam[c] = []

    for f in range(n_frames):
        mujoco.mj_step(model, data)
        for c in cam_names:
            renderer.update_scene(data, camera=c)
            img = renderer.render()
            grey = marker_grey(img)
            Image.fromarray(img).save(dest / "frames" / c / f"f{f:05d}.png")
            mask = grey >= DETECT_THRESHOLD
            lab, n = ndimage.label(mask)
            rr, cc, pk, ar = [], [], [], []
            for i in range(1, n + 1):
                yy, xx = np.nonzero(lab == i)
                if len(yy) >= DETECT_MIN_PX:
                    rr.append(float(yy.mean()))
                    cc.append(float(xx.mean()))
                    pk.append(float(grey[yy, xx].max()))
                    ar.append(int(len(yy)))
            rows_by_cam[c].extend(rr)
            cols_by_cam[c].extend(cc)
            peaks_by_cam[c].extend(pk)
            areas_by_cam[c].extend(ar)
            ncan_by_cam[c].append(len(rr))
    renderer.close()

    arrays = {}
    for c in cam_names:
        arrays[f"px/{c}/rows"] = np.asarray(rows_by_cam[c], dtype=np.float32)
        arrays[f"px/{c}/cols"] = np.asarray(cols_by_cam[c], dtype=np.float32)
        off = np.concatenate([[0], np.cumsum(ncan_by_cam[c])]).astype(np.int64)
        arrays[f"px/{c}/offsets"] = off
        arrays[f"px/{c}/n_candidates"] = np.asarray(ncan_by_cam[c], dtype=np.int64)
        arrays[f"px/{c}/peak_grey"] = np.asarray(peaks_by_cam[c], dtype=np.float64)
        arrays[f"px/{c}/area"] = np.asarray(areas_by_cam[c], dtype=np.int64)
    # the DECLARED fiducial positions, under the `truth/` prefix the chain expects for the static
    # constellation (they are declared constants here, not a simulator reading)
    dec = np.asarray([m["pos_mm"] for m in FIDUCIAL_MARKERS], dtype=np.float64)
    arrays["truth/marker_static_world_mm"] = np.repeat(dec[None, :, :], n_frames, axis=0)
    arrays["truth/time_s"] = np.arange(n_frames, dtype=np.float64) * float(model.opt.timestep)
    segment_order = list(STACKB_MARKER_BODY) if fly_markers else []
    if fly_markers:
        arrays["truth/marker_fly_world_mm"] = np.repeat(
            fly_marker_truth(model, data, marker_report["geom_name_by_segment"],
                             segment_order)[None, :, :], n_frames, axis=0).astype(np.float32)
    np.savez_compressed(dest / "episode.npz", **arrays)

    # markers.json: the chain reads `marker_body_segments` from it
    (dest / "markers.json").write_text(json.dumps({
        "marker_body_segments": segment_order,
        "static_markers": [{"name": m["name"], "pos_mm": list(m["pos_mm"])} for m in FIDUCIAL_MARKERS],
        "stack": "B (FlyMimic muscles)", "source_model": str(STACK_B_XML),
        "fly_markers": marker_report or None,
    }, indent=2, sort_keys=True))
    return {"episode": str(dest), "cameras": cam_names, "n_frames": n_frames,
            "resolution_hxw": [h, w],
            "n_fly_markers": len(segment_order),
            "fly_marker_radius_mm": (marker_report or {}).get("radius_mm"),
            "fly_marker_offsets_mm": (marker_report or {}).get("offsets_mm"),
            "median_candidates_per_camera_frame": {c: float(np.median(ncan_by_cam[c]))
                                                   for c in cam_names}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ring-rotation-deg", type=float, default=0.0)
    ap.add_argument("--settle-steps", type=int, default=4000)
    ap.add_argument("--no-fly-markers", action="store_true",
                    help="render the arena geometry only, without the fly's own markers")
    ap.add_argument("--out-dir", default=None,
                    help="destination episode directory; defaults to outputs/arena/stackb_observable")
    args = ap.parse_args()
    info = render_episode(args.frames, args.ring_rotation_deg, args.settle_steps, args.out_dir,
                          fly_markers=not args.no_fly_markers)
    print(json.dumps(info, indent=2, sort_keys=True))
    print(f"\nwrote {info['episode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
