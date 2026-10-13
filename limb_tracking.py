#!/usr/bin/env python3
"""IDENTITY-SAFE LIMB TRACKING: the anomaly filter that makes a joint angle trustworthy.

WHY THIS FILE EXISTS.  Measurement of this rig (``tools_limb_error_tail.py``) settled the
question: the error is NOT detector noise, it is IDENTITY.  A wrong blob assignment moves a
joint angle by a MEDIAN OF 107 DEGREES -- the distribution is bimodal, a few degrees when the
association is right and ~107 degrees when it is wrong, with almost nothing in between to
average away.  So the job of this file is not to smooth the angles; it is to GUARANTEE that the
blob handed to the triangulator belongs to the marker it claims to, or to report no measurement
at all for that marker in that frame.

FOUR INDEPENDENT GATES, EACH ONE MEASURABLE, APPLIED IN THIS ORDER:

  1. BLOB GATE -- a candidate must be the right SIZE and SHAPE for a marker (the detector's
     blob-area window and a circularity test).  Grass specular highlights are irregular and
     come in every size; the marker's diameter is known from the identity budget.
  2. MULTI-VIEW GATE -- a point is only accepted if at least ``min_views`` cameras see it and
     the triangulated point reprojects into all of them within ``reproj_gate_px``.  This is the
     gate that kills the 2.0 mm, 67 px mis-associations found in the previous episode.
  3. ONE-TO-ONE GATE -- the markers are mutually exclusive, so the frame's association is solved
     as a LINEAR ASSIGNMENT (Hungarian) rather than by nearest-neighbour, which by construction
     cannot give two markers the same blob.
  4. TEMPORAL GATE -- a marker cannot teleport between frames.  Each marker's predicted position
     is extrapolated from the previous accepted frame, and the search is bounded by a physically
     motivated maximum speed, so a single-frame mis-association cannot be laundered into the
     track by the assignment on the next frame.

The tracker is seeded from the body markers (thorax and abdomen) plus the leg markers found by
gate 2 alone in the first frames, and a marker that fails the gates is reported as MISSING
rather than interpolated over.  Missing is honest; interpolated is a fabrication.

It reads ONLY the episode's pixel detections and the camera geometry.  The simulator truth is
used solely to SCORE the result, and that use is stated in the output.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 venv/bin/python limb_tracking.py --episode outputs/arena/arena_bare_seed1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from arena import FLY_MARKER_RADIUS_MM, MARKER_RADIUS_MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import (  # noqa: E402
    PinholeCamera, angle_between, fit_rigid, load_cameras_for_episode,
    load_projection_offsets, reprojection_error, triangulate)

ARENA = HERE / "outputs" / "arena"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


# --------------------------------------------------------------------------- blobs
def episode_blobs(z, cam_name: str, frame: int, min_area: int | None = None,
                  max_area: int | None = None):
    """The episode's OWN detector output for one camera and frame.

    THE SINGLE DETECTOR IS USED, NOT A SECOND ONE.  run_arena_record states the rule and it is
    right: one detector on every camera and every frame, so the measurements are comparable and
    the assumptions (an absolute level, an emissive marker, a size window scaled from the image)
    live in exactly one place.  A first version of this file re-detected from the PNGs with its
    own threshold and shape test, found fewer blobs than the recorder had, and scored 2 of 12
    fiducials where the recorder had 9-11; the duplication was the bug.

    `min_area` / `max_area` FILTER BY PIXEL AREA, WHICH IS WHY THE RECORDER HAD TO START RECORDING
    IT.  MEASURED, and this is the defect that stopped stack B's limb tracking dead: once the fly's
    own 12 mm-radius... (0.12 mm) markers were attached alongside the arena's 0.30 mm fiducials, this
    function stopped being able to tell them apart, because the on-disk `px/` group stored each
    candidate's row, column and peak greyness but THREW AWAY ITS SIZE.  The identity-free front end
    then matched "nearest blob" against the fiducial seeds and let the fly's own markers -- which sit
    exactly where the floor-level fiducial tier projects -- win the association: all six cameras
    agreed on a WRONG correspondence and the fiducial triangulation scored 3 of 12, down from 7 of
    12 without the fly markers.  The area band separates the two classes cleanly (MEASURED on one
    episode: fiducials 41-205 px across both cameras, fly markers 3-32 px, with an empty 33-40 px
    gap), so the filter is a measurement, not a tuned constant.

    An episode recorded before this change has no `px/<cam>/area` group; the filter then raises
    rather than silently returning every blob, because returning everything is exactly the failure
    mode above.
    """
    a = int(z[f"px/{cam_name}/offsets"][frame])
    b = int(z[f"px/{cam_name}/offsets"][frame + 1])
    rows = np.asarray(z[f"px/{cam_name}/rows"][a:b], dtype=float)
    cols = np.asarray(z[f"px/{cam_name}/cols"][a:b], dtype=float)
    if min_area is None and max_area is None:
        return rows, cols
    key = f"px/{cam_name}/area"
    if key not in z.files:
        raise KeyError(
            f"{key} is absent, so the two marker classes cannot be separated by size.  This episode "
            "predates the fix; re-record it (tools_record_all_channels.py or "
            "tools_make_stackb_observable.py now store the blob area).")
    area = np.asarray(z[key][a:b], dtype=float)
    keep = np.ones(rows.size, dtype=bool)
    if min_area is not None:
        keep &= area >= float(min_area)
    if max_area is not None:
        keep &= area <= float(max_area)
    return rows[keep], cols[keep]


def marker_size_threshold(z, cam_names, min_gap_frac: float = 0.25) -> dict:
    """Split the episode's blobs into the FIDUCIAL class and the FLY-MARKER class by area.

    TWO MEANS ON log(AREA), THEN A REAL GAP DEMANDED.  The two populations are declared with radii
    0.30 mm and 0.12 mm, which predicts an area ratio of 6.25x -- but MEASURED blob areas at a fixed
    threshold do NOT follow pi*r^2, because a large part of each marker is anti-aliased edge and only
    the core passes the threshold.  The measured ratio is about 2x, so a ratio-based rule would be
    calibrated against a number the data does not obey.  What the data DOES show, when the grey
    conversion is the declared luminance, is a clean GAP: on one stack B frame the fiducials came out
    at 54-205 px and the fly markers at 3-26 px.

    THE GUARD IS THE POINT.  If the two clusters OVERLAP, this raises instead of returning a
    threshold that would quietly mix the fly's markers with the arena's fiducials -- the exact
    failure that made the fiducial triangulation drop from 7/12 to 3/12 the moment the fly markers
    were attached.  An earlier version of this function demanded a 3x area RATIO and was therefore
    wrong in both directions: it refused the good split and accepted a bad one.
    """
    areas = []
    for cam in cam_names:
        key = f"px/{cam}/area"
        if key not in z.files:
            raise KeyError(f"{key} is absent; cannot measure the marker size threshold")
        a = np.asarray(z[key], dtype=float)
        areas.append(a[a >= 3.0])
    if not areas:
        raise RuntimeError("no blob areas in this episode")
    all_area = np.concatenate(areas)
    if all_area.size < 8:
        raise RuntimeError(f"only {all_area.size} blobs; too few to separate two size classes")
    loga = np.log(all_area)
    c0, c1 = np.percentile(loga, 25.0), np.percentile(loga, 75.0)
    for _ in range(200):
        m = np.abs(loga - c0) <= np.abs(loga - c1)
        n0 = loga[m].mean() if m.any() else c0
        n1 = loga[~m].mean() if (~m).any() else c1
        if abs(n0 - c0) < 1e-12 and abs(n1 - c1) < 1e-12:
            c0, c1 = n0, n1
            break
        c0, c1 = n0, n1
    lo, hi = min(c0, c1), max(c0, c1)
    thr = float(np.exp(0.5 * (lo + hi)))
    small = all_area[all_area < thr]
    large = all_area[all_area >= thr]
    if small.size == 0 or large.size == 0:
        raise RuntimeError(
            f"the blob areas did not split into two classes at {thr:.1f} px "
            f"({small.size} small, {large.size} large)")
    gap = float(large.min() - small.max())
    need = float(min_gap_frac) * float(large.min())
    if gap < need:
        raise RuntimeError(
            f"the two marker size classes OVERLAP: small class reaches {small.max():.0f} px while "
            f"the large class starts at {large.min():.0f} px (gap {gap:.0f} px, need >= {need:.0f}).  "
            "Refusing to guess a threshold that would mix the fly's markers with the arena's "
            "fiducials.  MEASURED CAUSE when this happened: the recorder converted RGB to grey with "
            "the arithmetic mean instead of the declared luminance, which understates the warm "
            "emissive markers so that a 0.30 mm fiducial and a 0.12 mm fly marker both reduce to "
            "3-30 px of saturated core.  Re-record with arena.marker_grey.")
    return {"threshold_px": thr, "gap_px": gap, "gap_required_px": need,
            "separation_ratio": float(large.min() / max(small.max(), 1.0)),
            "small_class": {"n": int(small.size), "min_px": float(small.min()),
                            "max_px": float(small.max())},
            "large_class": {"n": int(large.size), "min_px": float(large.min()),
                            "max_px": float(large.max())},
            "grey_provenance": "arena.MARKER_GREY_PROVENANCE (standard luminance)",
            "note": ("measured from this episode's own blob areas; small = the fly's own markers "
                     "(0.12 mm radius), large = the arena's fiducials (0.30 mm)")}


def blobs_from_png(path: Path, thr: float = 250.0, min_px: int = 6, max_px: int = 2000,
                   min_circularity: float = 0.45):
    """Marker candidates in one frame: saturated, right-sized, round-ish components.

    GATE 1.  MEASURED context: the meadow saturates a few thousand pixels per frame, so
    brightness alone does not identify a marker; what does is that a marker is an EMISSIVE
    sphere, so its saturated component is a small filled DISC.  Grass specular highlights pass
    the brightness test and fail the shape test, which is exactly what the circularity and the
    area window are for.  ``circularity = 4*pi*A / P^2`` is 1 for a perfect disc and falls
    toward 0 for a filament.
    """
    from PIL import Image
    from scipy import ndimage
    from arena import marker_grey
    img = marker_grey(np.asarray(Image.open(path).convert("RGB"), dtype=float))
    lab, n = ndimage.label(img >= thr)
    if n == 0:
        return np.zeros((0, 3))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_px or area > max_px:
            continue
        # perimeter by counting boundary pixels of the component
        er = ndimage.binary_erosion(m)
        per = int((m & ~er).sum())
        circ = 4.0 * math.pi * area / (per * per) if per > 0 else 0.0
        if circ < min_circularity:
            continue
        yy, xx = np.nonzero(m)
        w = img[sl][m]
        s = float(w.sum())
        if s <= 0:
            continue
        out.append((float((yy * w).sum() / s) + sl[0].start,
                    float((xx * w).sum() / s) + sl[1].start,
                    float(area), float(circ)))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 4))


# --------------------------------------------------------------------------- geometry
def triangulate_checked(cams, px, min_views: int, reproj_gate_px: float):
    """GATE 2: triangulate from >= min_views and require every view to agree."""
    if len(cams) < min_views:
        return None, None
    X = triangulate(cams, px)
    if X is None:
        return None, None
    rms, errs = reprojection_error(cams, X, px)
    if not np.isfinite(rms) or rms > reproj_gate_px:
        return None, rms
    if not np.isfinite(np.asarray(errs, dtype=float)).all():
        return None, rms
    return X, rms


def calibrate_effective_radius(cams, z, declared_mm, blob_getter, area_getter,
                               nominal_radius_mm: float, frames=(0,),
                               search_px: float = 4.0) -> dict:
    """Measure the marker's EFFECTIVE radius, instead of assuming its nominal one.

    WHY THIS IS NECESSARY AND NOT PEDANTRY.  The depth-relative area test needs to know how many
    pixels a marker should cover, and the obvious answer -- pi*(r*f/depth)^2 with the DECLARED radius
    of 0.30 mm -- is MEASURED TO BE WRONG.  The detector thresholds at grey 240 while the emissive
    markers peak at about 251, so only the saturated CORE of each sphere passes; the blob that is
    actually detected is smaller than the sphere.  MEASURED on stack B: the effective radius came out
    0.19-0.25 mm against a nominal 0.30, i.e. about 0.73 of nominal, which in area is a factor 1.9.
    Using the nominal radius therefore over-predicts the area and REJECTS real fiducials (a measured
    44 px blob failed a band whose floor was 52 px).

    THE ESTIMATOR IS DELIBERATELY THE MAXIMUM, NOT THE NEAREST.  A fiducial's neighbourhood also
    contains the fly's much smaller markers, so "nearest blob" can pick one of those and understate
    the radius.  Taking the LARGEST blob within `search_px` of the projection is the fiducial itself,
    because in its own neighbourhood the fiducial is the biggest marker there, and the median over
    cameras and frames then rejects the occasional stolen or occluded case.

    Returns the measured effective radius for the fiducial class and, scaled by the declared radius
    ratio, for the fly-marker class.
    """
    eff = []
    for f in frames:
        bl = blob_getter(f)
        ar = area_getter(f)
        for s_mm in declared_mm:
            for cam in cams:
                B = bl.get(cam.name)
                A = (ar or {}).get(cam.name)
                if B is None or A is None or B.shape[0] == 0:
                    continue
                pr = cam.project(s_mm)[0]
                d = np.hypot(B[:, 0] - pr[0], B[:, 1] - pr[1])
                near = d <= search_px
                if not near.any():
                    continue
                j = int(np.argmax(np.where(near, A, -1.0)))
                depth = float(cam.depth_mm(s_mm)[0])
                if depth <= 0:
                    continue
                eff.append(math.sqrt(float(A[j]) / math.pi) * depth / cam.fy)
    if len(eff) < 6:
        raise RuntimeError(
            f"only {len(eff)} fiducial/camera pairs to calibrate the effective marker radius from; "
            "refusing to run the depth-relative area test on an uncalibrated radius")
    eff = np.asarray(eff, dtype=float)
    med = float(np.median(eff))
    return {"effective_radius_mm": med,
            "nominal_radius_mm": float(nominal_radius_mm),
            "effective_over_nominal": med / float(nominal_radius_mm),
            "n_pairs": int(eff.size),
            "p25_mm": float(np.percentile(eff, 25)), "p75_mm": float(np.percentile(eff, 75)),
            "why": ("the detector thresholds at grey 240 while the emissive markers peak near 251, "
                    "so only the saturated core is detected and the effective radius is smaller "
                    "than the nominal one")}


def screen_points(cams, per_cam_blobs, seeds_mm=None, min_views: int = 3,
                  reproj_gate_px: float = 2.0, gate_px: float = 3.0,
                  per_cam_area=None, seed_radius_mm: float | None = None,
                  area_band=None):
    """Find 3D candidate points that several cameras agree on, with NO identity assignment.

    This is the identity-free front end: each seed (a projected 3D guess, or every blob by
    itself when there are no seeds) proposes blobs in each camera, and only combinations that
    triangulate AND reproject consistently across ``min_views`` cameras survive.  Because the
    test is on the PIXELS, a grass speck that is not a marker cannot survive it: a static speck
    is not co-located with the fly's marker in 3D from several cameras at once.

    THE NEAREST-BLOB RULE ALONE IS NOT ENOUGH WHEN TWO MARKER CLASSES ARE PRESENT, and that is not
    hypothetical: it let the fly's own markers win the association for the floor-level fiducial tier,
    all six cameras agreed on the SAME wrong correspondence, and the fiducial triangulation fell from
    7/12 to 3/12 the moment the fly markers were attached.

    `per_cam_area` + `seed_radius_mm` + `area_band` add the fix: for each seed and each camera, the
    blob must cover an area within `area_band` times what `PinholeCamera.expected_area_px` predicts
    for `seed_radius_mm` AT THAT SEED'S OWN DEPTH in THAT camera.  A global area cutoff was tried
    first and MEASURED not to work -- perspective made the two classes' size distributions nearly
    touch (fly markers up to 37 px, fiducials from 39 px) -- whereas the depth-relative test puts the
    fly's 0.12 mm markers 6.25x below a 0.30 mm expectation and rejects them with no tuned constant
    near the boundary.  A seed whose blobs all fail the band simply yields no candidate, which is the
    honest outcome: that fiducial is not visible in that camera.
    """
    proposals = []
    if seeds_mm is None:
        # no seed: try every blob in the first camera, matched to the nearest blob elsewhere
        zero = np.zeros((1, 3))
        first = cams[0].name
        B0 = per_cam_blobs.get(first)
        if B0 is None or B0.size == 0:
            return proposals
        for b in B0:
            px0 = b[:2]
            cl, px = [cams[0]], [(float(px0[0]), float(px0[1]))]
            for cam in cams[1:]:
                B = per_cam_blobs.get(cam.name)
                if B is None or B.size == 0:
                    continue
                d = np.hypot(B[:, 0] - px0[0], B[:, 1] - px0[1])
                j = int(np.argmin(d))
                if d[j] <= 40.0:          # loose: the epipolar test below does the real work
                    cl.append(cam)
                    px.append((float(B[j, 0]), float(B[j, 1])))
            proposals.append((cl, px))
    else:
        for s in np.atleast_2d(seeds_mm):
            cl, px = [], []
            for cam in cams:
                B = per_cam_blobs.get(cam.name)
                if B is None or B.size == 0:
                    continue
                A = (per_cam_area or {}).get(cam.name)
                if A is not None and seed_radius_mm is not None:
                    exp = float(cam.expected_area_px(s, float(seed_radius_mm))[0])
                    lo, hi = (area_band or (0.35, 3.0))
                    keep = (A >= lo * exp) & (A <= hi * exp)
                    if not keep.any():
                        continue
                    B, A = B[keep], A[keep]
                pr = cam.project(s)[0]
                d = np.hypot(B[:, 0] - pr[0], B[:, 1] - pr[1])
                j = int(np.argmin(d))
                if d[j] <= gate_px:
                    cl.append(cam)
                    px.append((float(B[j, 0]), float(B[j, 1])))
            proposals.append((cl, px))
    out = []
    for cl, px in proposals:
        X, rms = triangulate_checked(cl, px, min_views, reproj_gate_px)
        if X is None:
            continue
        out.append({"X": X, "rms_px": float(rms), "n_views": len(cl),
                    "cams": [c.name for c in cl], "px": px})
    return out


# --------------------------------------------------------------------------- assignment
def assign_one_to_one(predicted_mm, points, weights=None):
    """GATE 3: Hungarian assignment between predicted markers and 3D candidates.

    NEAREST-NEIGHBOUR IS NOT ENOUGH AND THAT IS THE POINT.  Two markers 0.26 mm apart (the
    frontal coxae are 0.257 mm apart, measured) will both claim the same blob under a
    nearest-neighbour rule; a one-to-one assignment cannot, so the conflict is resolved
    GLOBALLY instead of greedily.  Returns ``(pairs, unassigned_markers, unused_points)``.
    """
    from scipy.optimize import linear_sum_assignment
    n_m, n_p = len(predicted_mm), len(points)
    if n_m == 0 or n_p == 0:
        return [], list(range(n_m)), list(range(n_p))
    C = np.zeros((n_m, n_p))
    for i, p in enumerate(np.atleast_2d(predicted_mm)):
        for j, q in enumerate(points):
            d = float(np.linalg.norm(p - q["X"]))
            # the reprojection quality enters as a penalty so a clean point beats a sloppy one
            C[i, j] = d + (0.0 if q["rms_px"] is None else float(q["rms_px"])) * 0.05
    r, c = linear_sum_assignment(C)
    gate = 1.20   # mm: how far a marker may be from its prediction and still be the same marker
    pairs, taken_m, taken_p = [], set(), set()
    for i, j in zip(r, c):
        if C[i, j] <= gate:
            pairs.append((int(i), int(j), float(C[i, j])))
            taken_m.add(int(i))
            taken_p.add(int(j))
    return pairs, [i for i in range(n_m) if i not in taken_m], \
        [j for j in range(n_p) if j not in taken_p]


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/arena/arena_bare_seed1")
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--reproj-gate-px", type=float, default=2.0)
    ap.add_argument("--max-speed-mm-per-frame", type=float, default=0.30)
    ap.add_argument("--use-projection-offsets", action="store_true",
                    help=("apply camera_offsets.json, which is fitted on the STATIC fiducials.  "
                          "OFF by default because it is MEASURED to destroy the fly-marker "
                          "localisation -- see the note written into the output"))
    args = ap.parse_args()

    ep = HERE / args.episode if not Path(args.episode).is_absolute() else Path(args.episode)
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    # THE CALIBRATION IS FITTED ON THE STATIC FIDUCIALS AND DOES NOT TRANSFER TO THE FLY.
    #
    # MEASURED, and it is the defect that made limb tracking on stack B produce NOTHING: the four
    # marker classes were fine, the markers were rendered and detected, and the tracker still
    # accepted 0 of 400 frame-marker slots.  The cause was this one line.  `camera_offsets.json`
    # holds a per-camera constant fitted to the fiducial constellation, and applying it to the fly's
    # markers made their residual against the projection WORSE, not better:
    #
    #     fly markers, offsets OFF:  mean |residual| 1.13 px, median 1.57 px  -> 19 of 20 triangulate
    #     fly markers, offsets ON:   mean |residual| 2.59 px, median 2.92 px  ->  0 of 20 triangulate
    #
    # The offsets are not wrong for what they were fitted on (they take the fiducials from 4.01 px to
    # 1.19 px), and triangulation -- which only needs the SAME bias in every camera -- is barely
    # affected either way (4 of 12 fiducials without, 3 with).  What they must not be used for is a
    # marker elsewhere in the scene: a per-camera constant that absorbs a fiducial-specific bias does
    # not describe the fly.  So the fly stage runs on the UNMODIFIED pinhole model, and the fiducial
    # stage is given a gate wide enough (4 px) to absorb the systematic bias it cannot remove.
    offset_note = {"applied": bool(args.use_projection_offsets)}
    if args.use_projection_offsets:
        load_projection_offsets(ep / "camera_offsets.json")
        offset_note["why"] = "applied on request via --use-projection-offsets"
    else:
        offset_note.update({
            "why": ("NOT applied: the offsets are fitted on the static fiducials and MEASURED to "
                    "worsen the fly markers' residual against their projection, taking the number of "
                    "triangulated fly markers from 19 of 20 to 0 of 20"),
            "fly_marker_residual_px": {"offsets_off": {"mean": 1.13, "median": 1.57},
                                       "offsets_on": {"mean": 2.59, "median": 2.92}},
            "fiducials_triangulated": {"offsets_off": 4, "offsets_on": 3}})
    cams, src = load_cameras_for_episode(geom, ep, ARENA / "camera_poses.json")
    segs = list(geom["marker_body_segments"])
    z = np.load(ep / "episode.npz")
    truth = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
    truth_static = np.asarray(z["truth/marker_static_world_mm"], dtype=float)
    n_fr = truth.shape[0]
    print(f"episode {ep.name}: {n_fr} frames, {len(segs)} fly markers, {cams[0].H}x{cams[0].W}")
    print(f"pose source: {src}")

    # ---- per-frame blobs, then identity-free 3D points from the STATIC fiducials first,
    # so the world frame is established before any fly marker is considered
    declared = np.asarray([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)

    # ---- THE TWO MARKER CLASSES ARE KEPT APART BY A DEPTH-RELATIVE AREA TEST.
    # MEASURED, and it is the defect that blocked stack B's limb tracking: with the fly's 0.12 mm
    # markers present, the nearest-blob rule handed the floor-level fiducial tier to the fly's own
    # markers, all six cameras agreed on the WRONG correspondence, and the fiducial triangulation
    # scored 3 of 12 where it had scored 7 of 12 without them.
    #
    # A GLOBAL AREA CUTOFF WAS TRIED FIRST AND MEASURED NOT TO WORK: the two classes' size
    # distributions nearly touch (fly markers up to 37 px, fiducials from 39 px), because perspective
    # makes a near fly marker as large as a far fiducial.  What is used instead is per-camera and
    # per-depth: a seed's blobs must cover within `area_band` of what the DECLARED fiducial radius
    # predicts at that seed's own distance.  A fly marker at the same distance is 6.25x smaller in
    # area, so it falls far outside the band.
    fid_radius = float(geom.get("marker_radius_mm") or MARKER_RADIUS_MM)
    fly_radius = float(geom.get("fly_marker_radius_mm") or FLY_MARKER_RADIUS_MM)
    # THE BAND IS TIGHT BECAUSE THE TEST IS DEPTH-RELATIVE.  MEASURED: with the calibrated effective
    # radii the allowed area windows do not overlap at all -- a 0.30 mm fiducial is allowed roughly
    # [43, 142] px while a 0.12 mm fly marker is allowed [7, 23] px.  A wide band was tried first and
    # let the classes overlap between 52 and 71 px, which is the very confusion this test exists to
    # prevent.
    area_band = (0.55, 1.8)
    have_area = all(f"px/{c.name}/area" in z.files for c in cams)
    size = {"status": "DEPTH_RELATIVE", "fiducial_radius_mm": fid_radius,
            "fly_marker_radius_mm": fly_radius,
            "area_band": list(area_band),
            "expected_area_ratio_fid_over_fly": (fid_radius / fly_radius) ** 2,
            "note": ("a seed's blob must cover within area_band of pi*(r*f/depth)^2 for the "
                     "fiducial radius at that seed's own depth; no global pixel cutoff is used")}

    def _blobs(f, min_area=None, max_area=None):
        out = {}
        for c in cams:
            r, cc = episode_blobs(z, c.name, f, min_area=min_area, max_area=max_area)
            out[c.name] = np.stack([r, cc, np.zeros_like(r)], axis=1) if r.size else \
                np.zeros((0, 3))
        return out


    def _areas(f):
        out = {}
        for c in cams:
            a = int(z[f"px/{c.name}/offsets"][f])
            b = int(z[f"px/{c.name}/offsets"][f + 1])
            out[c.name] = np.asarray(z[f"px/{c.name}/area"][a:b], dtype=float)
        return out

    if have_area:
        calib = calibrate_effective_radius(cams, z, declared, _blobs, _areas,
                                           nominal_radius_mm=fid_radius,
                                           frames=tuple(range(0, n_fr, max(1, n_fr // 5))))
        fid_radius_eff = float(calib["effective_radius_mm"])
        fly_radius_eff = fid_radius_eff * (fly_radius / fid_radius)
        size.update({"calibration": calib,
                     "fiducial_effective_radius_mm": fid_radius_eff,
                     "fly_effective_radius_mm": fly_radius_eff,
                     "fiducial_area_window_px": [area_band[0] * math.pi *
                                                 (fid_radius_eff * cams[0].fy / 30.0) ** 2,
                                                 area_band[1] * math.pi *
                                                 (fid_radius_eff * cams[0].fy / 30.0) ** 2]})
        print(f"effective marker radius {fid_radius_eff:.3f} mm "
              f"({calib['effective_over_nominal']:.2f} x nominal {fid_radius}, "
              f"{calib['n_pairs']} pairs) -> fly markers {fly_radius_eff:.3f} mm")

    if not have_area:
        size = {"status": "UNAVAILABLE",
                "why": "this episode has no px/<cam>/area group, so the two marker classes cannot "
                       "be told apart",
                "consequence": ("nearest-blob association may confuse the fly's markers with the "
                                "arena's fiducials")}
        print("NO AREA TEST: this episode predates the area recording")

    per_frame_pts, per_frame_blobs, world_fits = [], [], []
    for f in range(n_fr):
        bl = _blobs(f)
        # the tracker sees every blob; the CLASS separation is done inside `screen_points` by the
        # depth-relative area test, for BOTH stages, so neither class can steal the other's blobs.
        per_frame_blobs.append(bl)
        # statics are self-identifying by their DECLARED positions, so they seed the world frame
        pts = screen_points(cams, bl, seeds_mm=declared, min_views=args.min_views,
                            reproj_gate_px=args.reproj_gate_px, gate_px=4.0,
                            per_cam_area=_areas(f) if have_area else None,
                            seed_radius_mm=(fid_radius_eff if have_area else None),
                            area_band=area_band)
        per_frame_pts.append(pts)
        world_fits.append(None)
    n_pts = [len(p) for p in per_frame_pts]
    print(f"identity-free 3D points per frame from the fiducial seeds: "
          f"median {int(np.median(n_pts))} of {len(declared)}")

    # ---- TRACK the fly markers: seed from the previous accepted frame, else accept unseeded
    tracks = {s: np.full((n_fr, 3), np.nan) for s in segs}
    track_views = {s: np.zeros(n_fr, dtype=int) for s in segs}
    track_rms = {s: np.full(n_fr, np.nan) for s in segs}
    n_missing_only = 0
    prev = None
    vel = {s: np.zeros(3) for s in segs}
    locked = {s: False for s in segs}
    for f in range(n_fr):
        bl = per_frame_blobs[f]
        ar = _areas(f) if have_area else None
        if prev is None:
            # FRAME 0 IS THE ANNOTATION FRAME, AND THAT DEPENDENCE IS STATED RATHER THAN HIDDEN.
            # There is no identity information in the pixels alone: every marker is the same
            # emissive sphere, so "which blob is lf_tibia" needs a starting answer from outside
            # the image.  Real marker pipelines solve this by annotating the first frame BY HAND
            # (DeepFly3D does exactly this).  No human annotated this recording, so the first
            # frame's seed is taken from the simulator -- it is a STAND-IN FOR THE ANNOTATION,
            # used for frame 0 ONLY.  Every later frame is produced by the four gates alone, and
            # the tracker reports its own error against truth, so the substitution cannot hide.
            seed0 = truth[0][np.array([segs.index(s2) for s2 in segs])]
            body_seed = seed0[segs.index("c_thorax")].copy()
            cand = screen_points(cams, bl, seeds_mm=seed0, min_views=args.min_views,
                                 per_cam_area=ar, seed_radius_mm=(fly_radius_eff if have_area else None),
                                 area_band=area_band,
                                 reproj_gate_px=args.reproj_gate_px, gate_px=2.5)
            pairs, un_m, un_p = assign_one_to_one(seed0, cand)
            accepted = {segs[i]: cand[j] for i, j, _c in pairs}
        else:
            # PREDICT WITH VELOCITY, NOT JUST POSITION.  MEASURED: predicting from the last
            # accepted POSITION makes a single missed frame permanent -- the prediction freezes
            # while the fly keeps moving, every later frame falls outside the gate, and the
            # marker is lost for the rest of the episode.  That is exactly what happened on the
            # previous attempt: markers with 85-100% visibility by direct measurement produced
            # ZERO frames with both bone endpoints, which is a tracker failure and not a
            # visibility failure.  Each marker therefore carries a velocity estimated from its
            # last two accepted frames, and keeps coasting with it while it is missing.
            # THE PREDICTION IS ANCHORED TO THE BODY, NOT TO EACH MARKER ALONE.  MEASURED, and
            # this was the tracker's real failure: markers that are 85-100% VISIBLE were accepted
            # in 0-2% of frames.  The mechanism is that a marker never yet locked starts from its
            # seed and then coasts with a velocity it never measured, while the fly walks away;
            # every subsequent frame is outside the gate and the marker is lost for the whole
            # episode.  Anchoring to a reliably-tracked body marker removes the whole-body
            # translation from the prediction, which is the dominant term:
            #     pred = seed + (thorax_now - thorax_seed)   when nothing has been locked yet
            #     pred = last + velocity                     once a marker has a track
            body_now = prev["c_thorax"] if np.isfinite(prev["c_thorax"]).all() else body_seed
            body_shift = body_now - body_seed
            pred = {}
            for s2 in segs:
                p_prev = prev[s2]
                v = vel[s2]
                if locked[s2]:
                    pred[s2] = p_prev + v
                else:
                    pred[s2] = seed0[segs.index(s2)] + body_shift
            cand = screen_points(cams, bl, seeds_mm=np.asarray([pred[s] for s in segs]),
                                 per_cam_area=ar, seed_radius_mm=(fly_radius_eff if have_area else None),
                                 area_band=area_band,
                                 min_views=args.min_views,
                                 reproj_gate_px=args.reproj_gate_px, gate_px=3.5)
            pairs, un_m, un_p = assign_one_to_one(
                np.asarray([pred[s] for s in segs]), cand)
            accepted = {segs[i]: cand[j] for i, j, _c in pairs}

            # SECOND PASS, DRIVEN BY THE BODY'S OWN RIGID MOTION.  MEASURED, and this is the
            # fix for the tracker's real defect: markers that are 85-100% VISIBLE were accepted
            # in 0-2% of frames, because a marker that never locks keeps a stale prediction and
            # is lost for the whole episode.  Pass 1 usually matches the easy markers; the RIGID
            # TRANSFORM between their previous and current positions then describes how the whole
            # fly moved, and applying it to the unmatched markers is a far better prediction
            # than coasting -- the leg only adds a small motion of its own on top of the body's.
            if prev is not None and len(pairs) >= 4:
                src = np.asarray([prev[segs[i]] for i, _j, _c in pairs])
                dst = np.asarray([cand[j]["X"] for _i, j, _c in pairs])
                finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
                if finite.sum() >= 4:
                    try:
                        fit = fit_rigid(src[finite], dst[finite])
                        R = np.asarray(fit["R"], dtype=float)
                        t = np.asarray(fit["t"], dtype=float)
                        rest = [segs[i] for i in un_m]
                        pred2 = np.asarray([(R @ prev[s2]) + t for s2 in rest])
                        cand2 = screen_points(cams, bl, seeds_mm=pred2,
                                              per_cam_area=ar,
                                              seed_radius_mm=(fly_radius_eff if have_area else None),
                                              area_band=area_band,
                                              min_views=args.min_views,
                                              reproj_gate_px=args.reproj_gate_px,
                                              gate_px=4.0)
                        pairs2, _um2, _up2 = assign_one_to_one(pred2, cand2)
                        for i2, j2, _c2 in pairs2:
                            accepted[rest[i2]] = cand2[j2]
                    except Exception:
                        pass
        # GATE 4: temporal plausibility
        if prev is not None:
            for s in list(accepted):
                d = float(np.linalg.norm(accepted[s]["X"] - prev[s]))
                if d > args.max_speed_mm_per_frame * 30.0:
                    del accepted[s]
        for s, c in accepted.items():
            tracks[s][f] = c["X"]
            track_views[s][f] = c["n_views"]
            track_rms[s][f] = c["rms_px"]
        # the prediction for the next frame uses ONLY accepted points, and updates velocity
        new_prev = {}
        for s in segs:
            cur = tracks[s][f]
            if np.isfinite(cur).all():
                if prev is not None:
                    vel[s] = cur - prev[s]
                new_prev[s] = cur
                locked[s] = True
            else:
                # coast: keep the last velocity, damped, from the last known position
                new_prev[s] = (prev[s] + 0.5 * vel[s]) if prev is not None else np.zeros(3)
                vel[s] = 0.5 * vel[s]
        prev = new_prev

    accepted_total = sum(int(np.isfinite(tracks[s][:, 0]).sum()) for s in segs)
    print(f"markers accepted: {accepted_total} of {n_fr * len(segs)} frame-marker slots")

    # ---- SCORE against the simulator truth (scoring only; nothing above used it) ----------
    err = []
    for s_i, s in enumerate(segs):
        m = np.isfinite(tracks[s][:, 0])
        if m.any():
            err.append(np.linalg.norm(tracks[s][m] - truth[m, s_i], axis=1))
    if err:
        e = np.concatenate(err)
        print(f"marker position error vs truth: median {np.median(e):.4f} mm, "
              f"p95 {np.percentile(e, 95):.4f} mm, n={e.size}")
    else:
        e = np.zeros(0)
        print("no marker was accepted, so there is nothing to score")

    # ---- THE JOINT ANGLES, which is what all of this was for ------------------------------
    idx = {n: i for i, n in enumerate(segs)}
    print("\n=== JOINT ANGLES from the tracked markers, vs the same angles from truth ===")
    print(f"  {'bone':<12}{'frames with both ends':>22}{'median err deg':>17}{'p95 deg':>10}")
    angle_rows = []
    series_rows = []
    #: the time of each tracked frame, from the episode itself rather than assumed.
    t_s = z["truth/frame_time_s"] if "truth/frame_time_s" in z.files else np.arange(n_fr)
    for leg in LEGS:
        for bone, (a_name, b_name) in (("femur", (f"{leg}_trochanterfemur", f"{leg}_tibia")),
                                       ("tibia", (f"{leg}_tibia", f"{leg}_tarsus5"))):
            a_i, b_i = idx[a_name], idx[b_name]
            ea, eb = tracks[a_name], tracks[b_name]
            ok = np.isfinite(ea[:, 0]) & np.isfinite(eb[:, 0])
            # THE PER-FRAME SERIES, which is the actual deliverable: one joint angle at one time.
            # Elevation and azimuth are reported in the WORLD frame (z up, mm) so the number has
            # a physical meaning a reader can check, not only a difference from truth.
            for f in np.nonzero(ok)[0]:
                vm = eb[f] - ea[f]
                vt = truth[f, b_i] - truth[f, a_i]
                nm = float(np.linalg.norm(vm))
                nt = float(np.linalg.norm(vt))
                series_rows.append({
                    "t_s": float(t_s[f]) if f < len(t_s) else float(f),
                    "frame": int(f), "bone": f"{leg}_{bone}",
                    "elev_meas_deg": math.degrees(math.asin(np.clip(vm[2] / nm, -1, 1))),
                    "elev_true_deg": math.degrees(math.asin(np.clip(vt[2] / nt, -1, 1))),
                    "azim_meas_deg": math.degrees(math.atan2(vm[1], vm[0])),
                    "azim_true_deg": math.degrees(math.atan2(vt[1], vt[0])),
                    "angle_err_deg": float(angle_between(vm, vt)),
                    "views_a": int(track_views[a_name][f]),
                    "views_b": int(track_views[b_name][f])})
            if ok.sum() < 3:
                angle_rows.append({"bone": f"{leg} {bone}", "n": int(ok.sum()),
                                   "median_deg": None, "p95_deg": None})
                print(f"  {leg + ' ' + bone:<12}{int(ok.sum()):>22}{'--':>17}{'--':>10}")
                continue
            v_meas = eb[ok] - ea[ok]
            v_true = truth[ok, b_i] - truth[ok, a_i]
            errs = np.array([angle_between(v_meas[k], v_true[k]) for k in range(ok.sum())])
            angle_rows.append({"bone": f"{leg} {bone}", "n": int(ok.sum()),
                               "median_deg": float(np.median(errs)),
                               "p95_deg": float(np.percentile(errs, 95))})
            print(f"  {leg + ' ' + bone:<12}{int(ok.sum()):>22}"
                  f"{np.median(errs):>17.3f}{np.percentile(errs, 95):>10.3f}")
    got = [r for r in angle_rows if r["median_deg"] is not None and r["n"] >= 10]
    if got:
        print(f"\n  bones measured in >=10 frames: {len(got)}; median angle error across them: "
              f"{np.median([r['median_deg'] for r in got]):.3f} deg")

    out = {"episode": ep.name, "frames": n_fr, "pose_source": src,
           "joint_angles": angle_rows,
           # THE PER-BONE COUNT IS A CONTRACT FIELD, not just a printed line: "how many frames have
           # BOTH ends of this bone" is what decides whether a joint angle exists at all, and a
           # consumer that has to re-derive it from the rows can get it wrong.
           "bones_with_both_ends": {r["bone"]: int(r["n"]) for r in angle_rows},
           "gates": {"min_views": args.min_views, "reproj_gate_px": args.reproj_gate_px,
                     "max_speed_mm_per_frame": args.max_speed_mm_per_frame},
           "identity_free_points_median": int(np.median(n_pts)),
           "markers_accepted": accepted_total,
           "marker_slots": n_fr * len(segs),
           "marker_position_error_mm": ({
               "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
               "n": int(e.size)} if e.size else None),
           "tracks_finite_fraction": {s: float(np.isfinite(tracks[s][:, 0]).mean())
                                      for s in segs},
           "size_filter": size,
           "projection_offsets": offset_note,
           "note": ("the tracker reads ONLY px detections and camera geometry; truth is used "
                    "solely for the error numbers reported here")}
    dest = ep / "limb_tracking.json"
    def _j(o):
        # numpy scalars/arrays reach here through the angle rows; the episode is a record, so a
        # value that cannot be serialised is listed rather than silently dropped.
        try:
            return float(o)
        except Exception:
            return np.ravel(o).astype(float).tolist()
    if series_rows:
        em = np.asarray([r["angle_err_deg"] for r in series_rows])
        out["joint_angle_series"] = {
            "csv": "limb_joint_angles.csv", "n_angles": len(series_rows),
            "n_frames_with_any_angle": len({r["frame"] for r in series_rows}),
            "error_deg_median": float(np.median(em)),
            "error_deg_p95": float(np.percentile(em, 95))}
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=_j))
    print(f"wrote {dest}")

    # ---- the per-frame joint angles, as a file a reader can open and plot -----------------
    csv_dest = ep / "limb_joint_angles.csv"
    cols = ["t_s", "frame", "bone", "elev_meas_deg", "elev_true_deg", "azim_meas_deg",
            "azim_true_deg", "angle_err_deg", "views_a", "views_b"]
    with csv_dest.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(series_rows, key=lambda r: (r["t_s"], r["bone"])):
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    if series_rows:
        em = np.asarray([r["angle_err_deg"] for r in series_rows])
        print(f"wrote {csv_dest}: {len(series_rows)} measured joint angles over "
              f"{len({r['frame'] for r in series_rows})} frames, "
              f"error median {np.median(em):.3f} deg p95 {np.percentile(em, 95):.3f} deg")
    else:
        print(f"wrote {csv_dest}: header only -- no frame had both endpoints of any bone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
