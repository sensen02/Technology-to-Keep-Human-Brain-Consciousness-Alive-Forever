#!/usr/bin/env python3
"""WOULD A BETTER CAMERA FIX THE LIMB ANGLE?  A rig trade study on REAL camera specs.

EVALUATION ONLY -- no model, geometry or recording is changed.

THE QUESTION IS NOT "HOW MANY PIXELS".  A joint angle is a bone direction, and a bone
direction error is ``sqrt(2) * position_error / bone_length``.  So the useful question is
what the POSITION error would be, and it has two parts that behave completely differently:

  * a DETECTOR part, which scales as 1 / (rows x ring tightness) and which a better camera
    DOES buy down;
  * an ASSOCIATION / OCCLUSION part -- which blob in a grass-filled image is fiducial 7 --
    which a better camera does NOT fix at all, and which the previous round measured to be
    the dominant term in this episode (rigid-fit RMS 0.870 mm, against 0.052 mm from the
    noise-only Monte Carlo, i.e. 17x).

So this file measures both, in that order, before quoting any camera.

PART A -- THE HONEST FLOOR OF THE CURRENT RIG, BY SPLIT-HALF CALIBRATION.  Fitting the
per-camera pixel offsets on the same 12 fiducials that are then scored makes the score
circular, so the fiducials are SPLIT: offsets are fitted on half of them and the 3-D error
is measured on the held-out half.  Repeated over many random splits.  That number -- not
the self-fitted one -- is the rig's precision, and it is the baseline a new camera must beat.

PART B -- REAL CAMERAS.  Machine-vision and high-speed cameras that actually exist, with
their datasheet resolution and frame rate, mapped through the same six-camera ring geometry
used elsewhere in the project.  The angle error is read off the Monte Carlo, whose scaling
in rows and ring radius is VERIFIED here against the Monte Carlo itself before being used.

PART C -- THE TWO LIMITS A CAMERA CANNOT BUY.  Marker separation on the fly is an OPTICAL
ratio (marker diameter over segment spacing), and leg kinematics need a FRAME RATE that a
high-resolution sensor usually cannot deliver at full frame.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_camera_trade.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_arena_reconstruct import fit_rigid, load_cameras_for_episode, load_projection_offsets  # noqa: E402
from tools_limb_angle_precision import FEMUR_MM, TIBIA_MM, angle_error, ring  # noqa: E402
from tools_limb_eval import blobs_from_png  # noqa: E402

ARENA = HERE / "outputs" / "arena"
EPISODE = ARENA / "arena_bare_seed0"

MARKER_DIAMETER_MM = 0.60
CLOSEST_SPACING_MM = 0.144


# ---------------------------------------------------------------- principal-point triangulation
class PP:
    """A camera wrapper whose principal point can be MOVED, consistently.

    WHY THIS EXISTS.  ``PinholeCamera.project`` computes ``col = W/2 + fx*dx/depth`` from the
    HARD-CODED half-image, so the measured per-camera offsets that ``build_camera`` folds into
    ``K[0,2]/K[1,2]`` enter only ``P`` (used by the linear DLT) and NOT the Gauss-Newton
    refinement inside ``run_arena_reconstruct.triangulate``, which recomputes the projection
    from ``W/2`` again.  The two halves of one triangulation therefore use different camera
    models, and the refinement pulls the solution back toward the un-offset camera.  Nothing
    in this file depends on that being true or false -- the wrapper below keeps one model
    throughout, and the difference is printed so it is visible rather than hidden.
    """

    def __init__(self, cam, dx=0.0, dy=0.0):
        self.base = cam
        self.name = cam.name
        self.H, self.W = cam.H, cam.W
        self.fx, self.fy = cam.fx, cam.fy
        self.pos, self.R, self.Rwc, self.t = cam.pos, cam.R, cam.Rwc, cam.t
        self.K = np.array([[cam.fx, 0.0, cam.W / 2.0 + dx],
                           [0.0, cam.fy, cam.H / 2.0 + dy],
                           [0.0, 0.0, 1.0]])
        self.P = self.K @ np.hstack([cam.Rwc, cam.t[:, None]])

    def project(self, pts_mm):
        pr = self.base.project(pts_mm)
        return pr + np.array([self.K[1, 2] - self.base.H / 2.0,
                              self.K[0, 2] - self.base.W / 2.0])


def triangulate_pp(cams, px, n_refine: int = 4):
    """DLT + Gauss-Newton, BOTH against the same principal point."""
    if len(cams) < 2:
        return None
    A = []
    for c, (r, col) in zip(cams, px):
        A.append(col * c.P[2] - c.P[0])
        A.append(r * c.P[2] - c.P[1])
    try:
        _u, _s, vt = np.linalg.svd(np.asarray(A, dtype=float))
    except np.linalg.LinAlgError:
        return None
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    for _ in range(int(n_refine)):
        J, res = [], []
        for c, (r, col) in zip(cams, px):
            pc = c.Rwc @ X + c.t
            depth = -pc[2]
            if depth <= 1e-9:
                return None
            u = c.K[0, 2] + c.fx * pc[0] * (1.0 / depth)
            v = c.K[1, 2] - c.fy * pc[1] * (1.0 / depth)
            d = 1.0 / depth
            J.append(np.array([c.fx * d, 0.0, c.fx * pc[0] * d * d]) @ c.Rwc)
            J.append(np.array([0.0, -c.fy * d, c.fy * pc[1] * d * d]) @ c.Rwc)
            res.append(u - col)
            res.append(v - r)
        step, *_ = np.linalg.lstsq(np.asarray(J), -np.asarray(res), rcond=None)
        X = X + step
        if float(np.linalg.norm(step)) < 1e-10:
            break
    return X


def reproj_rms(cams, X, px):
    e = [math.hypot(c.project(X)[0][0] - r, c.project(X)[0][1] - col)
         for c, (r, col) in zip(cams, px)]
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------- part A
def part_a(n_frames=22, seed=0):
    from scipy.optimize import least_squares
    geom = json.loads((ARENA / "arena_geometry.json").read_text())
    load_projection_offsets(EPISODE / "camera_offsets.json")
    base_cams, src = load_cameras_for_episode(geom, EPISODE, ARENA / "camera_poses.json")
    decl = np.asarray([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)
    names = [m["name"] for m in geom["markers"]["markers"]]

    n_avail = len(sorted((EPISODE / "frames" / base_cams[0].name).glob("*.png")))
    frames = list(range(0, n_avail, max(1, n_avail // n_frames)))[:n_frames]

    # observations: blob centroid per (frame, camera, marker), nearest within a gate
    obs = {}
    for f in frames:
        for ci, cam in enumerate(base_cams):
            B = blobs_from_png(EPISODE / "frames" / cam.name / f"f{f:05d}.png")
            for k in range(len(decl)):
                pr = cam.project(decl[k])[0]
                if B.size == 0:
                    continue
                d = np.hypot(B[:, 0] - pr[0], B[:, 1] - pr[1])
                j = int(np.argmin(d))
                if d[j] <= 2.0:
                    obs[(f, ci, k)] = (float(B[j, 0]), float(B[j, 1]), float(B[j, 2]))

    def residual(p, keys):
        out = []
        for (f, ci, k) in keys:
            r, c, _a = obs[(f, ci, k)]
            cam = PP(base_cams[ci], p[2 * ci], p[2 * ci + 1])
            pr = cam.project(decl[k])[0]
            out.append(pr[0] - r)
            out.append(pr[1] - c)
        return np.asarray(out, dtype=float)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(decl))
    held_out_err, self_err, offsets = [], [], []
    n_trusted = []
    for _ in range(12):
        rng.shuffle(idx)
        train = set(idx[: len(idx) // 2].tolist())
        test = [k for k in range(len(decl)) if k not in train]
        tr_keys = [key for key in obs if key[2] in train]
        if len(tr_keys) < 30:
            continue
        sol = least_squares(residual, np.zeros(2 * len(base_cams)), args=(tr_keys,),
                            loss="soft_l1", f_scale=3.0, x_scale="jac")
        offsets.append(sol.x.tolist())

        def measure(keys_markers):
            errs, ntr = [], []
            for f in frames:
                cams = [PP(base_cams[ci], sol.x[2 * ci], sol.x[2 * ci + 1])
                        for ci in range(len(base_cams))]
                pts, tg = [], []
                for k in keys_markers:
                    cl, px = [], []
                    for ci in range(len(base_cams)):
                        key = (f, ci, k)
                        if key in obs:
                            cl.append(cams[ci])
                            px.append((obs[key][0], obs[key][1]))
                    if len(cl) < 3:
                        continue
                    X = triangulate_pp(cl, px)
                    if X is None or reproj_rms(cl, X, px) > 3.0:
                        continue
                    pts.append(X)
                    tg.append(k)
                if len(pts) >= 3:
                    errs.append(np.linalg.norm(np.asarray(pts) - decl[tg], axis=1))
                    ntr.append(len(pts))
            if not errs:
                return float("nan"), 0
            return float(np.median(np.concatenate(errs))), int(np.median(ntr))

        e_held, n_h = measure(test)
        e_self, _ = measure([k for k in range(len(decl))])
        if not math.isnan(e_held):
            held_out_err.append(e_held)
            self_err.append(e_self)
            n_trusted.append(n_h)
    return {"pose_source": src, "frames_used": len(frames),
            "observations": len(obs), "n_splits": len(held_out_err),
            "held_out_error_mm": {"median": float(np.median(held_out_err)),
                                  "min": float(np.min(held_out_err)),
                                  "max": float(np.max(held_out_err))},
            "self_fitted_error_mm_median": float(np.median(self_err)),
            "n_fiducials_scored_median": int(np.median(n_trusted)) if n_trusted else 0,
            "fiducials_per_frame_available": {names[k]: int(sum(1 for key in obs if key[2] == k))
                                              for k in range(len(decl))},
            "offsets_px_first_split": offsets[0] if offsets else None}


# ---------------------------------------------------------------- part B
#: TAKEN from manufacturer datasheets (NOT measured here).  rows, cols, fps at full frame.
#: The frame rate is the FULL-FRAME rate; all of these drop resolution to go faster.
CAMERAS = [
    ("current sim render (baseline)", 800, 640, 306, "not a real camera: the renderer"),
    ("Basler acA1300-200um", 1024, 1280, 200, "USB3, global shutter, mono"),
    ("Basler acA1920-155um", 1200, 1920, 155, "USB3"),
    ("Basler boost boA1936-400cm", 1216, 1936, 400, "CoaXPress-12"),
    ("Basler boost boA2048-380cm", 2048, 2048, 380, "CoaXPress-12"),
    ("Mikrotron EoSens 4CXP", 2048, 2048, 563, "CoaXPress-6"),
    ("Photron FASTCAM Mini WX100", 2048, 2048, 1080, "high-speed, 12-bit"),
    ("Chronos 2.1-HD", 1080, 1920, 1000, "high-speed, 10-bit"),
    ("Photron FASTCAM Mini AX50", 1024, 1024, 6400, "high-speed, 12-bit"),
]

#: FOVY of the ring, and its radius, held at the project's verified values.
FOVY_DEG = 50.0
RING_MM = 22.0


def gsd_mm(rows, radius_mm, fovy_deg=FOVY_DEG):
    """mm per pixel at the arena centre: distance / focal length in pixels."""
    d = radius_mm * math.sqrt(2.0)          # 45 deg elevation
    fy = (rows / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return d / fy


def part_b():
    # VERIFY the scaling law against the Monte Carlo before using it: the Monte Carlo numbers
    # in limb_angle_precision.json are computed, not scaled, and the law must reproduce them.
    ma = json.loads((ARENA / "limb_angle_precision.json").read_text())
    grid = {}
    for row in ma["noise_only"]:
        grid.setdefault(row["bone"], {})[(row["radius_mm"], row["rows"])] = row["median_deg"]
    fem = grid["femur"]
    ref_r, ref_rows = 22.0, 640.0
    ref = fem[(ref_r, ref_rows)]
    print("  scaling law check (angle ~ 1/rows ~ 1/radius), femur:")
    worst = 0.0
    for (r, rows), v in sorted(fem.items()):
        # GROUND SAMPLING DISTANCE IS PROPORTIONAL TO radius/rows, so a CLOSER ring and a
        # BIGGER sensor both shrink it.  The first version of this line had the radius
        # INVERTED (ref_r/r), which made a 3 mm ring look 56x WORSE than a 22 mm one; the
        # check below is what caught it.
        pred = ref * (r / ref_r) * (ref_rows / rows)
        rel = abs(pred - v) / v
        worst = max(worst, rel)
        print(f"    r={r:>4.0f} rows={rows:>4.0f}: monte carlo {v:7.3f} deg, "
              f"law {pred:7.3f} deg, {100 * rel:5.1f}% off")
    print(f"  worst deviation {100 * worst:.1f}%")
    if worst > 0.15:
        raise SystemExit(f"the scaling law is off by {100 * worst:.1f}%; refusing to use it")
    print("  the law holds to within 15%, so it is used below")

    out = []
    for name, rows, cols, fps, note in CAMERAS:
        g = gsd_mm(rows, RING_MM)
        out.append({
            "camera": name, "rows": rows, "cols": cols, "fps_full_frame": fps, "note": note,
            "gsd_mm_per_px_at_r22": g,
            "marker_diameter_px": MARKER_DIAMETER_MM / g,
            "closest_coxa_spacing_px": CLOSEST_SPACING_MM / g,
            "femur_angle_deg_r22": ref * (ref_rows / rows),
            "tibia_angle_deg_r22": grid["tibia"][(ref_r, ref_rows)] * (ref_rows / rows),
        })
    return ref, grid["tibia"][(ref_r, ref_rows)], out


# ---------------------------------------------------------------- report
def figure(ref_femur, ref_tibia, a, table):
    """Two panels, so the trade-off and the measured floor can be SEEN, not just tabulated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    radii = np.array([3.0, 4, 5, 6, 8, 10, 15, 22])
    for rows, style in ((800, "-"), (1024, "--"), (2048, "-."), (4096, ":")):
        fem = ref_femur * (640.0 / rows) * (radii / 22.0)
        ax[0].plot(radii, fem, style, lw=2, label=f"{rows} rows sensor")
    ax[0].scatter([22] * len(table), [r["femur_angle_deg_r22"] for r in table],
                  c="k", zorder=5, s=30, label="real cameras at r=22 mm")
    for r in table:
        ax[0].annotate(r["camera"].split("(")[0].strip()[:22],
                       (22, r["femur_angle_deg_r22"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[0].axhline(a["held_out_error_mm"]["median"] * math.sqrt(2.0) / FEMUR_MM * 57.2958,
                  color="r", ls="--", lw=2,
                  label=f"MEASURED floor, recalibrated "
                        f"({a['held_out_error_mm']['median']:.4f} mm -> "
                        f"{a['held_out_error_mm']['median'] * math.sqrt(2.0) / FEMUR_MM * 57.2958:.1f} deg)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("camera ring radius, mm")
    ax[0].set_ylabel("femur joint-angle error, degrees (noise-only)")
    ax[0].set_title("price of optics: ring tightness and sensor size\n"
                    "(lower is better; ring r=22 -> 8 mm is free and beats every camera upgrade)")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    labels = ["no gate\n(mis-associated\nblobs kept)",
              "outlier gate\n(stored offsets)",
              "outlier gate +\nsplit-half\nrecalibration",
              "noise-only Monte\nCarlo prediction\n(0.5 px detector)"]
    vals = [0.8696, 0.0653, a["held_out_error_mm"]["median"], 0.0417]
    colors = ["#c0392b", "#e67e22", "#27ae60", "#2980b9"]
    bars = ax[1].bar(labels, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax[1].annotate(f"{v:.4f} mm\n{v * math.sqrt(2.0) / FEMUR_MM * 57.2958:.1f} deg femur",
                       (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom",
                       fontsize=9)
    ax[1].set_ylabel("3-D position error, mm (rigid-fit RMS)")
    ax[1].set_title("what the same six cameras actually deliver\n"
                    "the angle error is sqrt(2)*sigma/bone_length, bone = 0.705 mm")
    ax[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p = ARENA / "camera_trade.png"
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


def main() -> int:
    print("=" * 78)
    print("PART A -- WHAT THE CURRENT RIG ACTUALLY DELIVERS (split-half, held-out fiducials)")
    print("=" * 78)
    a = part_a()
    print(f"  pose source            : {a['pose_source']}")
    print(f"  frames used            : {a['frames_used']}   observations: {a['observations']}")
    print(f"  splits                 : {a['n_splits']}")
    print(f"  3-D error, HELD OUT    : {a['held_out_error_mm']['median']:.4f} mm "
          f"(min {a['held_out_error_mm']['min']:.4f}, max {a['held_out_error_mm']['max']:.4f})")
    print(f"  3-D error, self-fitted : {a['self_fitted_error_mm_median']:.4f} mm "
          f"<-- circular, quoted only to show how much the circularity flatters it")
    print(f"  fiducials scored/frame : median {a['n_fiducials_scored_median']} of 12")
    print("  per-fiducial observations over the whole run (12 = seen in every frame/camera):")
    for k, v in sorted(a["fiducials_per_frame_available"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<6} {v:>5}")

    e = a["held_out_error_mm"]["median"]
    print(f"\n  ==> implied joint-angle error at this rig's precision {e:.4f} mm:")
    for nm, L in (("femur front", FEMUR_MM), ("tibia front", TIBIA_MM)):
        print(f"      {nm:<12} {math.degrees(math.sqrt(2.0) * e / L):>7.2f} deg")

    print()
    print("=" * 78)
    print("PART B -- REAL CAMERAS, SAME SIX-CAMERA RING (r = 22 mm, fovy 50 deg)")
    print("=" * 78)
    ref_femur, ref_tibia, table = part_b()
    print(f"\n  {'camera':<30}{'Mpx':>6}{'fps':>7}{'GSD um':>8}{'marker px':>10}"
          f"{'coxa gap px':>12}{'femur deg':>11}{'tibia deg':>11}")
    for r in table:
        mp = r["rows"] * r["cols"] / 1e6
        print(f"  {r['camera']:<30}{mp:>6.1f}{r['fps_full_frame']:>7}"
              f"{r['gsd_mm_per_px_at_r22'] * 1000:>8.2f}{r['marker_diameter_px']:>10.1f}"
              f"{r['closest_coxa_spacing_px']:>12.2f}"
              f"{r['femur_angle_deg_r22']:>11.3f}{r['tibia_angle_deg_r22']:>11.3f}")
    print("  (angle columns are the NOISE-ONLY bound: a perfect calibration and a perfect"
          "\n   association.  Part A is what this rig does with neither.)")

    print(f"\n  the same cameras with the ring tightened from r=22 mm to r=8 mm:")
    print(f"  {'camera':<30}{'GSD um':>8}{'marker px':>10}{'femur deg':>11}{'tibia deg':>11}")
    tight = []
    for name, rows, cols, fps, note in CAMERAS:
        g = gsd_mm(rows, 8.0)
        fem = ref_femur * (8.0 / 22.0) * (640.0 / rows)
        tib = fem * 0.735
        tight.append({"camera": name, "gsd_mm_per_px": g, "femur_deg": fem, "tibia_deg": tib})
        print(f"  {name:<30}{g * 1000:>8.2f}{MARKER_DIAMETER_MM / g:>10.1f}"
              f"{fem:>11.3f}{tib:>11.3f}")

    print()
    print("=" * 78
          )
    print("PART C -- THE TWO THINGS NO CAMERA BUYS")
    print("=" * 78)
    print(f"  1. MARKER SEPARATION IS AN OPTICAL RATIO, NOT A PIXEL COUNT.")
    print(f"     marker diameter {MARKER_DIAMETER_MM:.2f} mm vs closest adjacent coxa spacing "
          f"{CLOSEST_SPACING_MM:.3f} mm")
    print(f"     -> the markers are {MARKER_DIAMETER_MM / CLOSEST_SPACING_MM:.1f}x wider than "
          f"the gap, so they merge at EVERY resolution:")
    for r in table:
        print(f"        {r['camera']:<30} coxa gap = "
              f"{r['closest_coxa_spacing_px']:>6.2f} px, marker = "
              f"{r['marker_diameter_px']:>6.1f} px")
    print(f"     what DOES fix it: shrink the marker (radius <~0.05 mm), or stop trying to")
    print(f"     separate adjacent coxae and measure the bone from markers that ARE far apart")
    print(f"     (femur ends {FEMUR_MM:.3f} mm, tibia ends {TIBIA_MM:.3f} mm).")
    print(f"\n  2. FRAME RATE vs RESOLUTION IS A HARD TRADE, AND LEG KINEMATICS NEEDS RATE.")
    print(f"     a walking fly's step is ~5-10 Hz; resolving a joint angle needs >=20 samples")
    print(f"     per cycle => >=200 fps, and resolving a tarsal CONTACT transient (1-5 ms)")
    print(f"     needs 1-5 kHz.  The cameras that reach 1 kHz+ at full frame are the high-speed")
    print(f"     ones, whose sensors are noisier and need far more light per frame -- and light")
    print(f"     budget / exposure is the instrument side, deliberately out of scope here.")
    print(f"     The current episode is 107 frames over 0.3499 s = ~306 fps at 800x640.")

    print(f"\n  DECISION.  Resolution is NOT the binding constraint.  In order:")
    print(f"    (1) fix the fly markers (they were never rendered -- measured, 0 mk_* geoms);")
    print(f"    (2) fix association in grass clutter (that is where 0.870 mm comes from);")
    print(f"    (3) then tighten the ring: r=22 -> r=8 mm is 2.75x better for FREE, no new")
    print(f"        camera at all, and beats every camera upgrade in the table above;")
    print(f"    (4) only then buy a camera, and pick RATE first (1 kHz class) since the HD")
    print(f"        machine-vision options cannot exceed 563 fps at full frame.")

    dest = ARENA / "camera_trade.json"
    dest.write_text(json.dumps({
        "kind": "EVALUATION ONLY: no model, geometry or recording changed",
        "part_a_current_rig": a,
        "part_b_cameras_at_r22": table,
        "part_b_cameras_at_r8": tight,
        "monte_carlo_reference": {"config": "r=22 rows=640", "femur_deg": ref_femur},
        "fovy_deg": FOVY_DEG, "marker_diameter_mm": MARKER_DIAMETER_MM,
        "closest_coxa_spacing_mm": CLOSEST_SPACING_MM,
        "camera_specs_source": "TAKEN from manufacturer datasheets, NOT measured here",
    }, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")
    figure(ref_femur, ref_tibia, a, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
