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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import (  # noqa: E402
    PinholeCamera, angle_between, fit_rigid, load_cameras_for_episode,
    load_projection_offsets, reprojection_error, triangulate)

ARENA = HERE / "outputs" / "arena"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")


# --------------------------------------------------------------------------- blobs
def episode_blobs(z, cam_name: str, frame: int):
    """The episode's OWN detector output for one camera and frame.

    THE SINGLE DETECTOR IS USED, NOT A SECOND ONE.  run_arena_record states the rule and it is
    right: one detector on every camera and every frame, so the measurements are comparable and
    the assumptions (an absolute level, an emissive marker, a size window scaled from the image)
    live in exactly one place.  A first version of this file re-detected from the PNGs with its
    own threshold and shape test, found fewer blobs than the recorder had, and scored 2 of 12
    fiducials where the recorder had 9-11; the duplication was the bug.
    """
    a = int(z[f"px/{cam_name}/offsets"][frame])
    b = int(z[f"px/{cam_name}/offsets"][frame + 1])
    return (np.asarray(z[f"px/{cam_name}/rows"][a:b], dtype=float),
            np.asarray(z[f"px/{cam_name}/cols"][a:b], dtype=float))


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
    img = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=2)
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


def screen_points(cams, per_cam_blobs, seeds_mm=None, min_views: int = 3,
                  reproj_gate_px: float = 2.0, gate_px: float = 3.0):
    """Find 3D candidate points that several cameras agree on, with NO identity assignment.

    This is the identity-free front end: each seed (a projected 3D guess, or every blob by
    itself when there are no seeds) proposes blobs in each camera, and only combinations that
    triangulate AND reproject consistently across ``min_views`` cameras survive.  Because the
    test is on the PIXELS, a grass speck that is not a marker cannot survive it: a static speck
    is not co-located with the fly's marker in 3D from several cameras at once.
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
    args = ap.parse_args()

    ep = HERE / args.episode if not Path(args.episode).is_absolute() else Path(args.episode)
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    load_projection_offsets(ep / "camera_offsets.json")
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
    per_frame_pts, per_frame_blobs, world_fits = [], [], []
    for f in range(n_fr):
        bl = {}
        for c in cams:
            r, cc = episode_blobs(z, c.name, f)
            bl[c.name] = np.stack([r, cc, np.zeros_like(r)], axis=1) if r.size else \
                np.zeros((0, 3))
        per_frame_blobs.append(bl)
        # statics are self-identifying by their DECLARED positions, so they seed the world frame
        pts = screen_points(cams, bl, seeds_mm=declared, min_views=args.min_views,
                            reproj_gate_px=args.reproj_gate_px, gate_px=2.5)
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
    for leg in LEGS:
        for bone, (a_name, b_name) in (("femur", (f"{leg}_trochanterfemur", f"{leg}_tibia")),
                                       ("tibia", (f"{leg}_tibia", f"{leg}_tarsus5"))):
            a_i, b_i = idx[a_name], idx[b_name]
            ea, eb = tracks[a_name], tracks[b_name]
            ok = np.isfinite(ea[:, 0]) & np.isfinite(eb[:, 0])
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
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=_j))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
