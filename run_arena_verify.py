#!/usr/bin/env python3
"""CAN A LEG JOINT ANGLE AT TIME t BE RESOLVED?  Answered on the REAL recording.

This is the experiment the Monte Carlo can only approximate.  Everything the limb answer
depends on -- the true camera poses, the true detector noise, the true blob merging, the true
marker associations and the true calibration residual -- is present here as measured data
instead of as an assumption.

WHAT COMES FROM WHERE, STATED PLAINLY, because this is where a pipeline can fool itself:

  * the MEASUREMENT side reads only the episode's detected blobs (``px/*/rows|cols``) and the
    camera geometry.  No simulator position enters the triangulation.
  * the TRUTH side (``truth/*``) is used for exactly two things: to decide WHICH blob belongs to
    WHICH segment, and to say what the right answer was.  Identity labelling by truth is not a
    measurement shortcut -- tracking identity from a known frame is what DeepFly3D does too --
    but it is a real dependence and it is reported as one, with the number of markers that
    could be found without it (via one-to-one assignment) printed next to it.

THREE QUESTIONS, THREE ANSWERS:

  1. THE BONE ANGLE.  Femur and tibia absolute directions, per leg, per frame, measured vs
     truth, plus the knee angle and the recovered bone length.
  2. THE LOCAL ERROR SCALE.  The static fiducials are a rigid body, so the DIFFERENCE between
     two markers' errors is the quantity that rotates a bone.  Its component perpendicular to
     the pair axis, divided by the pair separation, is an angle error -- and measuring it at
     separations from 5.5 to 15.5 mm and extrapolating to a 0.705 mm bone turns the fiducial
     residual into a limb-angle prediction that is independent of the fly markers.
  3. THE BOTTLENECK.  How many of the 21 fly markers are separately detectable at all.  Markers
     0.60 mm across on a 2.5 mm fly are closer together than they are wide, so the limit here is
     OPTICAL, not resolution: a bigger sensor does not separate them.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python run_arena_verify.py
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
    PinholeCamera, angle_between, fit_rigid, load_cameras_for_episode,
    load_projection_offsets, reprojection_error, triangulate)

ARENA = HERE / "outputs" / "arena"
EPISODE = ARENA / "arena_bare_seed0"
GEOM_PATH = ARENA / "arena_geometry.json"
POSE_PATH = ARENA / "camera_poses.json"
OFFSET_PATH = EPISODE / "camera_offsets.json"

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
GATE_PX = 2.5


def load_cands(z, cam_name: str, frame: int):
    off = z[f"px/{cam_name}/offsets"]
    a, b = int(off[frame]), int(off[frame + 1])
    return (np.asarray(z[f"px/{cam_name}/rows"][a:b], dtype=float),
            np.asarray(z[f"px/{cam_name}/cols"][a:b], dtype=float))


def assign_nearest(points_mm: np.ndarray, cams, cands: dict, gate_px: float):
    """For each 3D point: the nearest detected blob inside the gate, in each camera."""
    views = [[] for _ in range(len(points_mm))]
    for ci, cam in enumerate(cams):
        rows, cols = cands.get(cam.name, (np.array([]), np.array([])))
        if rows.size == 0:
            continue
        pr = cam.project(points_mm)
        d = np.hypot(rows[None, :] - pr[:, 0:1], cols[None, :] - pr[:, 1:2])
        j = np.argmin(d, axis=1)
        dmin = d[np.arange(len(points_mm)), j]
        for k in range(len(points_mm)):
            if dmin[k] <= gate_px:
                views[k].append((ci, float(rows[j[k]]), float(cols[j[k]]), float(dmin[k])))
    return views


def triangulate_views(views, cams):
    """(3D point, reprojection rms px, n_views) from a point's per-camera blob picks."""
    if len(views) < 2:
        return None, float("nan"), len(views)
    cl = [cams[ci] for ci, _r, _c, _d in views]
    px = [(r, c) for _ci, r, c, _d in views]
    X = triangulate(cl, px)
    if X is None:
        return None, float("nan"), len(views)
    e, _ = reprojection_error(cl, X, px)
    return X, float(e), len(views)


def main() -> int:
    geom = json.loads(GEOM_PATH.read_text())
    load_projection_offsets(OFFSET_PATH)
    cams, src = load_cameras_for_episode(geom, EPISODE, POSE_PATH)
    z = np.load(EPISODE / "episode.npz")

    seg_names = list(geom["marker_body_segments"])
    static_names = [m["name"] for m in geom["markers"]["markers"]]
    static_mm = np.asarray([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    fly_truth = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
    static_truth = np.asarray(z["truth/marker_static_world_mm"], dtype=float)
    n_frames = fly_truth.shape[0]

    print(f"episode        : {EPISODE}")
    print(f"camera pose src: {src}")
    print(f"frame size     : {cams[0].H} rows x {cams[0].W} cols "
          f"(fy = {cams[0].fy:.2f} px, fovy {cams[0].fovy_deg:.1f} deg)")
    print(f"frames         : {n_frames}")
    print(f"fly markers    : {len(seg_names)}  statics: {len(static_names)}  gate: {GATE_PX} px")
    print(f"bone lengths (MEASURED from the model): femur 0.705 mm, tibia 0.961 mm\n")

    idx = {n: i for i, n in enumerate(seg_names)}
    leg_bones = {}
    for leg in LEGS:
        leg_bones[leg] = (idx[f"{leg}_trochanterfemur"], idx[f"{leg}_tibia"], idx[f"{leg}_tarsus5"])

    # ---------------- truth joint angles, and what the model says they are -------------
    def bone_angles(p):
        """(femur direction, tibia direction, knee angle) per leg, from 3 points each."""
        out = {}
        for leg, (a, b, c) in leg_bones.items():
            u = p[b] - p[a]
            v = p[c] - p[b]
            out[leg] = {"femur_vec": u, "tibia_vec": v,
                        "femur_len": float(np.linalg.norm(u)),
                        "tibia_len": float(np.linalg.norm(v)),
                        "knee_deg": angle_between(u, v)}
        return out

    truth_ang = [bone_angles(fly_truth[f]) for f in range(n_frames)]

    femur_len_truth = np.array([[truth_ang[f][l]["femur_len"] for l in LEGS] for f in range(n_frames)])
    tibia_len_truth = np.array([[truth_ang[f][l]["tibia_len"] for l in LEGS] for f in range(n_frames)])
    print("truth bone lengths, median over frames and legs (mm):")
    print("  femur", np.round(np.median(femur_len_truth, axis=0), 4),
          " overall", round(float(np.median(femur_len_truth)), 4))
    print("  tibia", np.round(np.median(tibia_len_truth, axis=0), 4),
          " overall", round(float(np.median(tibia_len_truth)), 4))

    # ---------------- per-frame reconstruction ----------------
    rec_femur_err, rec_tibia_err, rec_knee_err = [], [], []
    rec_femur_len, rec_tibia_len = [], []
    n_views_per_marker = []
    n_found_per_frame = []
    n_distinct_blobs_over_fly = []
    static_fit_rms = []
    static_residual = []          # (frame, 12, 3) error vectors in the fitted world frame
    static_residual_idx = []
    per_frame_records = []

    for f in range(n_frames):
        cands = {c.name: load_cands(z, c.name, f) for c in cams}

        # --- statics: nearest blob to the DECLARED position, then rigid fit to the world frame
        sviews = assign_nearest(static_mm, cams, cands, GATE_PX)
        spts, sidx = [], []
        for k, v in enumerate(sviews):
            X, _e, _n = triangulate_views(v, cams)
            if X is not None:
                spts.append(X)
                sidx.append(k)
        frame_world = None
        if len(spts) >= 3:
            fit = fit_rigid(np.asarray(spts), static_mm[sidx])
            static_fit_rms.append(float(fit["point_rms"]))
            R = np.asarray(fit["R"], dtype=float)
            t = np.asarray(fit["t"], dtype=float)
            frame_world = (R, t)
            # error vectors IN THE FITTED WORLD FRAME: reconstruction mapped to declared
            rec_in_world = (R @ np.asarray(spts).T).T + t
            ev = np.zeros((len(static_names), 3))
            have = np.zeros(len(static_names), dtype=bool)
            for row, k in enumerate(sidx):
                ev[k] = rec_in_world[row] - static_mm[k]
                have[k] = True
            static_residual.append(ev)
            static_residual_idx.append(have)
        else:
            static_fit_rms.append(float("nan"))
            static_residual.append(None)
            static_residual_idx.append(None)

        # --- fly markers: nearest blob to the TRUTH projection, then triangulate the pixels
        fviews = assign_nearest(fly_truth[f], cams, cands, GATE_PX)
        fpts = [None] * len(seg_names)
        for k, v in enumerate(fviews):
            if len(v) < 2:
                continue
            X, _e, _n = triangulate_views(v, cams)
            fpts[k] = X
            n_views_per_marker.append(len(v))
        n_found_per_frame.append(sum(1 for p in fpts if p is not None))

        # a fly marker can only be measured up to the world frame the statics established
        if frame_world is not None:
            R, t = frame_world
            fpts_w = [None if p is None else (R @ p + t) for p in fpts]
        else:
            fpts_w = fpts

        # how many SEPARATE blobs exist inside the fly's projected neighbourhood
        used = set()
        for v in fviews:
            for ci, r, c, _d in v:
                used.add((ci, round(r / 8.0), round(c / 8.0)))
        n_distinct_blobs_over_fly.append(len(used))

        fe = {l: np.nan for l in LEGS}
        te = {l: np.nan for l in LEGS}
        ke = {l: np.nan for l in LEGS}
        fl = {l: np.nan for l in LEGS}
        tl = {l: np.nan for l in LEGS}
        for leg, (a, b, c) in leg_bones.items():
            if fpts_w[a] is None or fpts_w[b] is None:
                continue
            u = np.asarray(fpts_w[b]) - np.asarray(fpts_w[a])
            fl[leg] = float(np.linalg.norm(u))
            fe[leg] = angle_between(u, truth_ang[f][leg]["femur_vec"])
            if fpts_w[c] is None:
                continue
            v = np.asarray(fpts_w[c]) - np.asarray(fpts_w[b])
            tl[leg] = float(np.linalg.norm(v))
            te[leg] = angle_between(v, truth_ang[f][leg]["tibia_vec"])
            ke[leg] = abs(angle_between(u, v) - truth_ang[f][leg]["knee_deg"])
        rec_femur_err.append(fe)
        rec_tibia_err.append(te)
        rec_knee_err.append(ke)
        rec_femur_len.append(fl)
        rec_tibia_len.append(tl)

        per_frame_records.append({
            "frame": f, "t_s": float(z["truth/frame_time_s"][f]),
            "markers_found": int(sum(1 for p in fpts if p is not None)),
            "static_fit_rms_mm": static_fit_rms[-1],
            "femur_err_deg": fe, "tibia_err_deg": te, "knee_err_deg": ke,
            "femur_len_mm": fl, "tibia_len_mm": tl,
            "femur_truth_deg": {l: truth_ang[f][l]["femur_vec"].tolist() for l in LEGS},
        })

    def flat(d):
        out = {l: np.asarray([r[l] for r in d], dtype=float) for l in LEGS}
        return out

    F, T, K = flat(rec_femur_err), flat(rec_tibia_err), flat(rec_knee_err)
    FL, TL = flat(rec_femur_len), flat(rec_tibia_len)

    print(f"\n=== 1. MEASURED LIMB ANGLES (pixels only), {n_frames} frames ===")
    print(f"static fiducial rigid-fit RMS : median {np.nanmedian(static_fit_rms):.4f} mm "
          f"over {int(np.sum(np.isfinite(static_fit_rms)))}/{n_frames} frames")
    print(f"fly markers triangulated/frame: median {int(np.median(n_found_per_frame))} "
          f"of {len(seg_names)}   distinct 8px blobs over the fly: "
          f"median {int(np.median(n_distinct_blobs_over_fly))}")
    print(f"\n{'leg':<5} {'femur med':>10} {'femur p95':>10} {'tibia med':>10} {'tibia p95':>10} "
          f"{'knee med':>10} {'n femur':>8}")
    rows_limb = []
    for leg in LEGS:
        fv, tv, kv = F[leg], T[leg], K[leg]
        rows_limb.append({"leg": leg,
                          "femur_median_deg": float(np.nanmedian(fv)),
                          "femur_p95_deg": float(np.nanpercentile(fv[np.isfinite(fv)], 95)) if np.isfinite(fv).any() else None,
                          "tibia_median_deg": float(np.nanmedian(tv)),
                          "tibia_p95_deg": float(np.nanpercentile(tv[np.isfinite(tv)], 95)) if np.isfinite(tv).any() else None,
                          "knee_median_deg": float(np.nanmedian(kv)),
                          "n_femur": int(np.isfinite(fv).sum()),
                          "n_tibia": int(np.isfinite(tv).sum())})
        print(f"{leg:<5} {np.nanmedian(fv):>10.3f} "
              f"{np.nanpercentile(fv[np.isfinite(fv)], 95) if np.isfinite(fv).any() else float('nan'):>10.3f} "
              f"{np.nanmedian(tv):>10.3f} "
              f"{np.nanpercentile(tv[np.isfinite(tv)], 95) if np.isfinite(tv).any() else float('nan'):>10.3f} "
              f"{np.nanmedian(kv):>10.3f} {int(np.isfinite(fv).sum()):>8d}")
    allf = np.concatenate([F[l] for l in LEGS])
    allt = np.concatenate([T[l] for l in LEGS])
    allk = np.concatenate([K[l] for l in LEGS])
    print(f"\nALL LEGS femur median {np.nanmedian(allf):.3f} deg, "
          f"tibia median {np.nanmedian(allt):.3f} deg, knee median {np.nanmedian(allk):.3f} deg")
    print("recovered bone lengths (mm): femur median "
          f"{np.nanmedian(np.concatenate([FL[l] for l in LEGS])):.4f}, tibia median "
          f"{np.nanmedian(np.concatenate([TL[l] for l in LEGS])):.4f}")
    print("  (truth femur ~0.705-0.836, tibia ~0.961-1.368; a length that comes out SHORT means"
          "\n   the two endpoints were resolved less accurately than the bone is long)")

    # ---------------- 2. local error scale from the rigid fiducials ----------------
    print("\n=== 2. THE SAME NUMBER FROM THE STATIC RIGID BODY (no fly markers involved) ===")
    pair_rows = []
    declared_pairs = []
    for i in range(len(static_names)):
        for j in range(i + 1, len(static_names)):
            declared_pairs.append((i, j, float(np.linalg.norm(static_mm[i] - static_mm[j]))))
    for i, j, d in declared_pairs:
        perp, par, ok = [], [], 0
        for ev, have in zip(static_residual, static_residual_idx):
            if ev is None or not (have[i] and have[j]):
                continue
            e = ev[i] - ev[j]
            u = (static_mm[i] - static_mm[j]) / d
            par.append(float(np.dot(e, u)))
            perp.append(float(np.linalg.norm(e - np.dot(e, u) * u)))
            ok += 1
        if ok < 10:
            continue
        perp = np.asarray(perp)
        pair_rows.append({"pair": [static_names[i], static_names[j]], "separation_mm": d,
                          "n_frames": ok,
                          "perp_median_mm": float(np.median(perp)),
                          "angle_median_deg": math.degrees(float(np.median(perp)) / d),
                          "parallel_median_mm": float(np.median(np.abs(par)))})
    ds = np.asarray([r["separation_mm"] for r in pair_rows])
    ang = np.asarray([r["angle_median_deg"] for r in pair_rows])
    # angle error from a point-error difference falls as 1/d: fit angle*d = const
    const = float(np.median(ang * ds))
    print(f"{'separation mm':>13} {'perp err mm':>12} {'angle err deg':>14}")
    for r in sorted(pair_rows, key=lambda r: r["separation_mm"]):
        print(f"{r['separation_mm']:>13.3f} {r['perp_median_mm']:>12.4f} "
              f"{r['angle_median_deg']:>14.3f}")
    print(f"\nproduct angle*separation: median {const:.4f} deg*mm "
          f"(a constant means the error behaves as one local point error, not as a rigid offset)")
    for L, nm in ((0.705, "femur 0.705 mm"), (0.961, "tibia 0.961 mm")):
        print(f"  predicted angle error for a {nm} bone: {const / L:.3f} deg")

    # ---------------- 3. one leg, one time point, written out ----------------
    print("\n=== 3. A SINGLE JOINT ANGLE AT A SINGLE TIME (what was asked) ===")
    best_leg, best_frames = None, -1
    for leg in LEGS:
        n = int(np.isfinite(F[leg]).sum())
        if n > best_frames:
            best_leg, best_frames = leg, n
    fl_ok = np.where(np.isfinite(F[best_leg]) & np.isfinite(T[best_leg]))[0]
    print(f"leg {best_leg}: femur measured in {int(np.isfinite(F[best_leg]).sum())}/{n_frames} "
          f"frames, tibia in {int(np.isfinite(T[best_leg]).sum())}")
    samples = []
    for f in (fl_ok[0], fl_ok[len(fl_ok) // 2], fl_ok[-1]) if fl_ok.size else ():
        a, b, c = leg_bones[best_leg]
        t = float(z["truth/frame_time_s"][f])
        print(f"\n  t = {t:.4f} s  (frame {f})")
        for nm, (p, q) in (("femur", (a, b)), ("tibia", (b, c))):
            tu = truth_ang[f][best_leg][f"{nm}_vec"]
            tlen = float(np.linalg.norm(tu))
            print(f"    {nm}: truth length {tlen:.4f} mm, "
                  f"direction (unit) {np.round(tu / tlen, 4).tolist()}")
        print(f"    femur angle error {F[best_leg][f]:.3f} deg, "
              f"tibia angle error {T[best_leg][f]:.3f} deg, "
              f"knee angle error {K[best_leg][f]:.3f} deg")
        print(f"    measured femur length {FL[best_leg][f]:.4f} mm "
              f"(truth {truth_ang[f][best_leg]['femur_len']:.4f}), "
              f"tibia length {TL[best_leg][f]:.4f} mm "
              f"(truth {truth_ang[f][best_leg]['tibia_len']:.4f})")
        samples.append({"frame": f, "t_s": t,
                        "femur_err_deg": float(F[best_leg][f]),
                        "tibia_err_deg": float(T[best_leg][f]),
                        "knee_err_deg": float(K[best_leg][f]),
                        "femur_len_measured_mm": float(FL[best_leg][f]),
                        "femur_len_truth_mm": float(truth_ang[f][best_leg]["femur_len"]),
                        "tibia_len_measured_mm": float(TL[best_leg][f]),
                        "tibia_len_truth_mm": float(truth_ang[f][best_leg]["tibia_len"])})

    # ---------------- the same question by TWO-TIER ring, from the Monte Carlo ----------------
    rig = {}
    ma_path = ARENA / "limb_angle_precision.json"
    if ma_path.exists():
        ma = json.loads(ma_path.read_text())
        for row in ma.get("noise_only", []):
            rig[(row["config"], row["bone"])] = row["median_deg"]

    dest = ARENA / "limb_verify.json"
    dest.write_text(json.dumps({
        "episode": str(EPISODE),
        "camera_pose_source": src,
        "frame_size_rows_cols": [cams[0].H, cams[0].W],
        "frames": n_frames,
        "gate_px": GATE_PX,
        "identity_labelling": "truth projections were used to label blobs; the 3D position of "
                              "every point comes from the pixels alone",
        "static_fit_rms_mm": {"median": float(np.nanmedian(static_fit_rms)),
                              "n_frames": int(np.sum(np.isfinite(static_fit_rms)))},
        "markers_triangulated_per_frame": {"median": int(np.median(n_found_per_frame)),
                                           "of": len(seg_names)},
        "distinct_blobs_over_fly": {"median": int(np.median(n_distinct_blobs_over_fly))},
        "per_leg": rows_limb,
        "all_legs": {"femur_median_deg": float(np.nanmedian(allf)),
                     "tibia_median_deg": float(np.nanmedian(allt)),
                     "knee_median_deg": float(np.nanmedian(allk)),
                     "n_femur": int(np.isfinite(allf).sum()),
                     "n_tibia": int(np.isfinite(allt).sum())},
        "recovered_lengths_mm": {
            "femur_median": float(np.nanmedian(np.concatenate([FL[l] for l in LEGS]))),
            "tibia_median": float(np.nanmedian(np.concatenate([TL[l] for l in LEGS]))),
            "femur_truth_median": float(np.median(femur_len_truth)),
            "tibia_truth_median": float(np.median(tibia_len_truth))},
        "static_pairs": pair_rows,
        "angle_times_separation_deg_mm": const,
        "predicted_angle_deg": {"femur_0.705mm": const / 0.705, "tibia_0.961mm": const / 0.961},
        "samples": samples,
        "monte_carlo_noise_only": {f"{k[0]}|{k[1]}": v for k, v in rig.items()},
        "per_frame": per_frame_records,
    }, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {dest}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.asarray(z["truth/frame_time_s"], dtype=float)
        fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for leg in LEGS:
            ax[0].plot(tt, np.degrees(np.arccos(np.clip(
                np.array([truth_ang[f][leg]["femur_vec"][2] /
                          max(1e-9, np.linalg.norm(truth_ang[f][leg]["femur_vec"]))
                          for f in range(n_frames)]), -1, 1))) - 90.0,
                label=f"{leg} truth", lw=1.6)
            ax[1].plot(tt, F[leg], label=f"{leg}", lw=1.0, alpha=0.8)
        ax[0].set_ylabel("femur elevation vs\ntruth, deg (truth)")
        ax[0].legend(ncol=3, fontsize=8)
        ax[1].set_ylabel("femur direction\nerror, deg (measured)")
        ax[1].legend(ncol=3, fontsize=8)
        ax[1].axhline(float(np.nanmedian(allf)), color="k", ls="--", lw=1,
                      label="median")
        ax[2].plot(tt, FL[best_leg], label=f"{best_leg} measured femur length")
        ax[2].axhline(float(np.median(femur_len_truth)), color="r", ls="--",
                      label="truth length")
        ax[2].set_ylabel("femur length, mm")
        ax[2].set_xlabel("time, s")
        ax[2].legend(fontsize=8)
        fig.suptitle("Leg joint angle from 6 cameras vs simulator truth "
                     f"({cams[0].H}x{cams[0].W}, {n_frames} frames)")
        fig.tight_layout()
        fig_path = ARENA / "limb_verify.png"
        fig.savefig(fig_path, dpi=110)
        print(f"wrote {fig_path}")
    except Exception as exc:  # pragma: no cover
        print(f"figure skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
