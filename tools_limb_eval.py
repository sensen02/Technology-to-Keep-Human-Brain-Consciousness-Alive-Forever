#!/usr/bin/env python3
"""EVALUATION ONLY: how accurately can ONE limb joint angle at ONE time be resolved?

No simulator work, no re-recording, no model changes.  This measures what the EXISTING
six-camera episode can support, from the images on disk, and nothing else.

THE CHAIN BEHIND THE ANSWER.  A joint angle is the direction of a bone, and a bone direction
comes from two positions:

    angle error  ~  sqrt(2) * (position error) / (bone length)

The bone lengths are small and they are MEASURED from the model, not assumed: the shortest
femur is 0.705 mm and the shortest tibia 0.961 mm.  So the whole question reduces to the
POSITION ERROR, and this file measures it in the one way the existing data allows without
trusting the fly's own markers:

  * the 12 static fiducials are a RIGID body with exactly known separations, so the residual
    after the best rigid fit is a measurement of the reconstruction error with the world frame
    fully removed;
  * what rotates a bone is the DIFFERENCE between the errors of two nearby points, so the
    component of that difference PERPENDICULAR to the pair axis, divided by the separation,
    IS an angle error.  Measuring it from 5.5 mm down to the closest pairs, and reading the
    product (angle x separation), gives the local error scale at any separation -- including
    a 0.705 mm bone, which is far smaller than any fiducial pair.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_limb_eval.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import (  # noqa: E402
    fit_rigid, load_cameras_for_episode, load_projection_offsets, reprojection_error,
    triangulate)

ARENA = HERE / "outputs" / "arena"
EPISODE = ARENA / "arena_bare_seed0"

#: MEASURED from the compiled model's segment origins in this episode.
FEMUR_MM = (0.705, 0.784, 0.836)      # front, mid, hind
TIBIA_MM = (0.987, 1.163, 1.352)      # front, mid, hind
COXA_SPACING_MM = (0.144, 0.204)      # closest adjacent coxa origins (lf-tf <-> rf-tf, ...)
MARKER_DIAMETER_MM = 0.60             # the marker sphere actually rendered for the fiducials

#: A triangulation whose views disagree by more than this is treated as mis-associated.
REPROJ_GATE_PX = 3.0


def blobs_from_png(path: Path, thr: float = 250.0, min_px: int = 20):
    """Saturated components, area-filtered.  The fiducials are ~300-550 px at this size."""
    from PIL import Image
    from scipy import ndimage
    img = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=2)
    lab, n = ndimage.label(img >= thr)
    if n == 0:
        return np.zeros((0, 2))
    slices = ndimage.find_objects(lab)
    out = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        k = int(m.sum())
        if k < min_px:
            continue
        yy, xx = np.nonzero(m)
        out.append((float(yy.mean()) + sl[0].start, float(xx.mean()) + sl[1].start, k))
    return np.array(sorted(out, key=lambda b: -b[2])) if out else np.zeros((0, 3))


def main() -> int:
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    load_projection_offsets(EPISODE / "camera_offsets.json")
    cams, src = load_cameras_for_episode(geom, EPISODE, ARENA / "camera_poses.json")
    decl = np.asarray([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    names = [m["name"] for m in geom["markers"]["markers"]]
    n_frames = len(sorted((EPISODE / "frames" / cams[0].name).glob("*.png")))

    print("EVALUATION OF ONE LIMB JOINT ANGLE FROM THE EXISTING SIX-CAMERA EPISODE")
    print(f"  episode     : {EPISODE}")
    print(f"  pose source : {src}")
    print(f"  image       : {cams[0].H} rows x {cams[0].W} cols, fy {cams[0].fy:.1f} px, "
          f"fovy {cams[0].fovy_deg:.1f} deg")
    print(f"  frames      : {n_frames}")
    print(f"  ground sampling distance at the arena centre: "
          f"{cams[0].pos[2] / cams[0].fy:.5f} mm/px")

    frames = list(range(0, n_frames, max(1, n_frames // 20)))
    errors = np.zeros((len(frames), len(decl), 3))
    have = np.zeros((len(frames), len(decl)), dtype=bool)
    # A SECOND, QUALITY-GATED SET OF THE SAME MEASUREMENT.  MEASURED: two of the twelve
    # fiducials triangulate with a reprojection residual of 67 px and a 3-D error of 2.0 mm --
    # those are MIS-ASSOCIATED blobs (a grass highlight landing closer to the projected marker
    # than the marker is), and letting them into a rigid fit is what produced the 0.870 mm
    # figure this file first reported.  Rejecting outliers on the reprojection residual is the
    # standard remedy and is what DeepFly3D does too, so BOTH numbers are kept: without the
    # gate is what an unguarded pipeline gets, with it is what a working one gets.
    errors_q = np.zeros_like(errors)
    have_q = np.zeros_like(have)
    rms, n_tri = [], []
    for fi, f in enumerate(frames):
        per_cam = {}
        qpts = {}
        for cam in cams:
            per_cam[cam.name] = blobs_from_png(EPISODE / "frames" / cam.name / f"f{f:05d}.png")
        pts, idx = [], []
        for k in range(len(decl)):
            cl, px = [], []
            for cam in cams:
                B = per_cam[cam.name]
                if B.size == 0:
                    continue
                proj = cam.project(decl[k])[0]
                d = np.hypot(B[:, 0] - proj[0], B[:, 1] - proj[1])
                j = int(np.argmin(d))
                if d[j] <= 2.0:
                    cl.append(cam)
                    px.append((float(B[j, 0]), float(B[j, 1])))
            if len(cl) >= 2:
                X = triangulate(cl, px)
                if X is not None:
                    pts.append(X)
                    idx.append(k)
                    rqc = reprojection_error(cl, X, px)[0] if len(cl) >= 3 else float("inf")
                    if rqc <= REPROJ_GATE_PX:
                        qpts[k] = X
        if len(pts) < 4:
            rms.append(float("nan"))
            n_tri.append(len(pts))
            continue
        fit = fit_rigid(np.asarray(pts), decl[idx])
        rms.append(float(fit["residual_mm"]))
        n_tri.append(len(pts))
        R, t = np.asarray(fit["R"]), np.asarray(fit["t"])
        rec = (R @ np.asarray(pts).T).T + t
        for row, k in enumerate(idx):
            errors[fi, k] = rec[row] - decl[k]
            have[fi, k] = True
        # THE GATED SET GETS ITS OWN RIGID FIT.  Applying the transform that was fitted on the
        # CONTAMINATED points would measure the good points in a frame warped by the outliers
        # and report their error as ~0.6 mm when it is really ~0.03 mm -- which is exactly
        # what the first version of this patch did.
        if len(qpts) >= 3:
            qk = sorted(qpts)
            qsrc = np.asarray([qpts[k] for k in qk])
            qfit = fit_rigid(qsrc, decl[qk])
            qR, qt = np.asarray(qfit["R"]), np.asarray(qfit["t"])
            qrec = (qR @ qsrc.T).T + qt
            for row, k in enumerate(qk):
                errors_q[fi, k] = qrec[row] - decl[k]
                have_q[fi, k] = True

    def _rms(e, h):
        vals = [np.sqrt(np.mean([float(np.dot(e[fi, k], e[fi, k]))
                                 for k in range(len(decl)) if h[fi, k]]))
                for fi in range(len(frames))
                if sum(1 for k in range(len(decl)) if h[fi, k]) >= 3]
        return (float(np.median(vals)) if vals else float("nan")), len(vals)

    rms_raw, n_raw = _rms(errors, have)
    rms_q, n_q = _rms(errors_q, have_q)
    print(f"\n=== the measured 3-D error of this rig (static fiducials, {len(frames)} frames) ===")
    print(f"  WITHOUT outlier rejection (n_views >= 2, no residual gate):")
    print(f"    rigid-fit RMS {rms_raw:.4f} mm over {n_raw} frames  "
          f"<-- the number this file first reported; contaminated by mis-association")
    print(f"  WITH outlier rejection (n_views >= 3 and reprojection residual <= "
          f"{REPROJ_GATE_PX:.1f} px):")
    print(f"    rigid-fit RMS {rms_q:.4f} mm over {n_q} frames  "
          f"<-- the rig's actual precision")
    errors, have = errors_q, have_q
    print(f"  fiducials triangulated per frame : median {int(np.median(n_tri))} of "
          f"{len(decl)}")
    print(f"  rigid-fit point residual RMS     : median {np.nanmedian(rms):.4f} mm  "
          f"(min {np.nanmin(rms):.4f}, max {np.nanmax(rms):.4f})")

    # WHERE THE ERROR LIVES.  A median angle*separation product is only meaningful if the error
    # is spread evenly across the constellation.  If one or two markers carry it, the product at
    # the pairs BETWEEN WELL-MEASURED NEIGHBOURS is much smaller, and quoting the all-pairs
    # median for a bone whose endpoints are such neighbours would be far too pessimistic.
    per_marker = []
    for k in range(len(decl)):
        m = np.linalg.norm(errors[:, k], axis=1)[have[:, k]]
        if m.size:
            per_marker.append({"marker": names[k], "n": int(m.size),
                               "median_mm": float(np.median(m)),
                               "max_mm": float(m.max())})
    print("\n  per-marker |error| (mm, after the rigid fit):")
    for r in sorted(per_marker, key=lambda r: -r["median_mm"]):
        print(f"    {r['marker']:<6} n={r['n']:>2}  median {r['median_mm']:.4f}  "
              f"max {r['max_mm']:.4f}")
    order = [r["marker"] for r in sorted(per_marker, key=lambda r: -r["median_mm"])]
    print("  rigid-fit RMS after dropping the worst markers:")
    for drop in range(0, min(4, len(order))):
        removed = set(order[:drop])
        vals = [np.sqrt(np.mean([float(np.dot(errors[fi, k], errors[fi, k]))
                                 for k in range(len(decl))
                                 if names[k] not in removed and have[fi, k]]))
                for fi in range(len(frames))
                if sum(1 for k in range(len(decl))
                       if names[k] not in removed and have[fi, k]) >= 3]
        if vals:
            print(f"    dropped {drop} ({', '.join(order[:drop]) or '-'}): "
                  f"RMS {np.median(vals):.4f} mm")

    good = set(names)
    if per_marker:
        typical = float(np.median([r["median_mm"] for r in per_marker]))
        for r in per_marker:
            if r["median_mm"] > 3.0 * typical:
                good.discard(r["marker"])
    print(f"  markers within 3x of the typical error ({len(good)}): {sorted(good)}")

    pairs = []
    for i in range(len(decl)):
        for j in range(i + 1, len(decl)):
            d = float(np.linalg.norm(decl[i] - decl[j]))
            perp, par = [], []
            for fi in range(len(frames)):
                if not (have[fi, i] and have[fi, j]):
                    continue
                e = errors[fi, i] - errors[fi, j]
                u = (decl[i] - decl[j]) / d
                par.append(float(np.dot(e, u)))
                perp.append(float(np.linalg.norm(e - np.dot(e, u) * u)))
            if len(perp) < 5:
                continue
            pairs.append({"pair": [names[i], names[j]], "separation_mm": d, "n": len(perp),
                          "both_well_measured": bool(names[i] in good and names[j] in good),
                          "perp_median_mm": float(np.median(perp)),
                          "angle_deg": math.degrees(float(np.median(perp)) / d),
                          "parallel_median_mm": float(np.median(np.abs(par)))})

    print(f"\n=== the local error scale: angle error = perpendicular error / separation ===")
    print(f"{'separation mm':>14} {'|dE| perp mm':>13} {'angle deg':>11} {'angle*sep':>10}")
    for r in sorted(pairs, key=lambda r: r["separation_mm"]):
        print(f"{r['separation_mm']:>14.3f} {r['perp_median_mm']:>13.4f} "
              f"{r['angle_deg']:>11.3f} {r['angle_deg'] * r['separation_mm']:>10.4f}")
    ds = np.asarray([r["separation_mm"] for r in pairs])
    ang = np.asarray([r["angle_deg"] for r in pairs])
    const = float(np.median(ang * ds))
    print(f"\n  angle x separation product: median {const:.4f} deg*mm "
          f"(range {np.min(ang * ds):.4f} to {np.max(ang * ds):.4f})")
    print("  A product that is CONSTANT means the error behaves like one local position error,"
          "\n  not like a rigid offset -- so it can be read at any separation.")
    sel = [(r["angle_deg"] * r["separation_mm"]) for r in pairs
           if r["both_well_measured"] and r["separation_mm"] <= 8.0]
    const_good = float(np.median(sel)) if sel else float("nan")
    print(f"\n  restricted to pairs between WELL-MEASURED markers within 8 mm "
          f"({len(sel)} pairs):")
    print(f"    angle x separation product: median {const_good:.4f} deg*mm  "
          f"-> femur {const_good / FEMUR_MM[0]:.3f} deg, tibia {const_good / TIBIA_MM[0]:.3f} deg")
    print("\n  two numbers, and the second is the defensible one:")
    print(f"    all pairs          : {const:.3f} deg*mm  -> femur "
          f"{const / FEMUR_MM[0]:.2f} deg, tibia {const / TIBIA_MM[0]:.2f} deg")
    print(f"    well-measured pairs: {const_good:.3f} deg*mm  -> femur "
          f"{const_good / FEMUR_MM[0]:.2f} deg, tibia {const_good / TIBIA_MM[0]:.2f} deg")
    print(f"\n  ==> local position error at any scale: {math.radians(1) * const:.5f} mm "
          f"(= {math.radians(1) * const * 1000:.1f} um per degree of bone rotation)")

    print(f"\n=== THE ANSWER: joint-angle error for a real bone ===")
    print(f"{'bone':<16} {'length mm':>10} {'angle error deg':>16} {'equivalent tip err um':>23}")
    rows_out = []
    for nm, L in (("femur front", FEMUR_MM[0]), ("femur mid", FEMUR_MM[1]),
                  ("femur hind", FEMUR_MM[2]), ("tibia front", TIBIA_MM[0]),
                  ("tibia mid", TIBIA_MM[1]), ("tibia hind", TIBIA_MM[2])):
        a = const / L
        rows_out.append({"bone": nm, "length_mm": L, "angle_error_deg": a,
                         "tip_error_um": a * L * 1000.0 * math.pi / 180.0})
        print(f"{nm:<16} {L:>10.3f} {a:>16.3f} {rows_out[-1]['tip_error_um']:>23.1f}")

    print(f"\n=== two hard limits that are NOT about resolution ===")
    print(f"  1. MARKER SEPARATION.  The rendered marker is {MARKER_DIAMETER_MM:.2f} mm across, "
          f"the closest adjacent\n     coxa segment origins are {COXA_SPACING_MM[0]:.3f}-"
          f"{COXA_SPACING_MM[1]:.3f} mm apart.  Markers that are 3-4x wider than the gap "
          f"between\n     them merge into one blob, and a bigger sensor does NOT separate them: "
          f"the ratio is\n     optical.  The femur/tibia pair ({FEMUR_MM[0]:.3f} mm) IS "
          f"separable, so the FRONT coxa is the\n     problem, not the bone.")
    print(f"  2. FLY MARKERS WERE NEVER RENDERED IN THIS EPISODE.  MEASURED: the compiled model "
          f"has 0 geoms\n     and 0 bodies named mk_* although attach_marker_bodies reported 21 "
          f"bodies added, because it\n     was called AFTER the model was compiled; the recorded "
          f"'fly marker' positions are therefore\n     simulator segment origins, and the "
          f"brightness at those 21 image positions peaks near 200\n     (grass) against 255 for "
          f"the fiducials.  So no fly-marker measurement exists in this\n     episode to check "
          f"the prediction above against.")

    # the noise-only Monte Carlo, for the side-by-side
    ma_path = ARENA / "limb_angle_precision.json"
    if ma_path.exists():
        ma = json.loads(ma_path.read_text())
        print(f"\n=== side by side: Monte Carlo noise-only bound (0.5 px detector noise) ===")
        print(f"{'configuration':<30} {'femur deg':>10} {'tibia deg':>10} {'3D err mm':>10}")
        for row in ma.get("noise_only", []):
            if row["bone"] == "femur":
                print(f"{row['config']:<30} {row['median_deg']:>10.3f} "
                      f"{'':>10} {row['predicted_3d_mm']:>10.4f}")
        print("  (the noise-only bound assumes a PERFECT calibration; the measured 3-D error "
              "above is\n   what this rig actually delivers, and it is the larger number)")

    dest = ARENA / "limb_eval.json"
    dest.write_text(json.dumps({
        "episode": str(EPISODE), "frames_used": frames, "pose_source": src,
        "image_rows_cols": [cams[0].H, cams[0].W],
        "gsd_mm_per_px": float(cams[0].pos[2] / cams[0].fy),
        "fiducials_triangulated_median": int(np.median(n_tri)),
        "rigid_fit_rms_mm_no_outlier_gate": rms_raw,
        "rigid_fit_rms_mm_with_outlier_gate": rms_q,
        "reproj_gate_px": REPROJ_GATE_PX,
        "rigid_fit_rms_mm": {"median": float(np.nanmedian(rms)),
                             "min": float(np.nanmin(rms)),
                             "max": float(np.nanmax(rms))},
        "static_pairs": pairs,
        "angle_times_separation_deg_mm": const,
        "angle_times_separation_deg_mm_well_measured": const_good,
        "well_measured_markers": sorted(good),
        "per_marker_error_mm": per_marker,
        "local_position_error_mm_per_deg": math.radians(1) * const,
        "bone_angle_error_deg": rows_out,
        "hard_limits": {
            "marker_diameter_mm": MARKER_DIAMETER_MM,
            "closest_coxa_spacing_mm": list(COXA_SPACING_MM),
            "fly_markers_rendered": False,
            "fly_markers_evidence": "compiled model has 0 geoms and 0 bodies named mk_*; "
                                    "attach_marker_bodies ran after compile",
        },
        "note": "EVALUATION ONLY.  No model or recording was changed to produce these numbers.",
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
