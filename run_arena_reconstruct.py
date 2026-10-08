#!/usr/bin/env python3
"""MULTI-CAMERA RECONSTRUCTION: markers -> 3D points -> ABSOLUTE angles.

WHAT THIS FILE MAY READ
-----------------------
    arena_geometry.json     the declared cameras and the declared fiducial positions
    <episode>/markers.json  which body segments carry markers
    <episode>/episode.npz   ONLY the ``px/`` group: the per-camera pixel centroids

It never opens ``truth/``.  The truth is compared against in
``run_arena_verify.py``, separately, and that separation is the only thing that makes
"reconstructed from the cameras" a statement with content.

THE CHAIN, AND WHAT EACH STEP ASSUMES
-------------------------------------
1. **Pinhole model per camera** from the declared pose.  Assumes the declaration is
   right; the residual on the fiducials is where that assumption gets tested.
2. **Correspondence.**  Which blob in camera A is the same physical marker as which blob
   in camera B.  This is the step that is usually hand-waved, and it is where a
   multi-camera pipeline most often fails silently: an assignment error produces a
   plausible-looking but wrong 3D point.  Here it is done by EPIPOLAR + REACH
   consistency: a candidate pair is kept only if the triangulated point reprojects into
   every other camera within a stated pixel radius, and if the resulting bone length
   agrees with the model.  Every rejected combination is counted and reported.
3. **Triangulation.**  Linear (DLT) then two Gauss-Newton refinements minimising
   reprojection error.  The linear step alone is biased at short baselines.
4. **World frame.**  A rigid transform fitted from the RECONSTRUCTED static fiducials to
   their DECLARED positions (Kabsch).  Without this step the reconstruction lives in the
   cameras' arbitrary frame and no angle is meaningful.  The fit is reported as a
   residual on the markers, and separately as a residual on the PAIRWISE DISTANCES,
   which is transform-independent.
5. **Absolute angles.**  Using markers on three body segments, the body's rotation
   matrix is built and expressed as intrinsic Z-Y-X angles.  Using three markers per leg
   (femur, tibia, tarsus tip) the bone directions are computed in the WORLD frame and
   the joint angles between them.  "Absolute" here means: measured against the world
   frame of the six fiducials, not against the body.

WHAT IS *NOT* DONE, AND WHY
---------------------------
Full inverse kinematics over the 42 fly DOF is NOT attempted.  The markers are placed at
three points per leg, which is enough for the angles BETWEEN those three bones and for
each bone's direction in the world, but not enough to constrain every DOF of a leg; a
solver given too few constraints would return angles that look precise and are not
identifiable.  Three points per limb is the honest limit of this marker set, and the
report says so where the numbers are.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")
BODY_MARKERS = ("c_thorax", "c_rostrum", "c_abdomen3")


class PinholeCamera:
    """A camera with a declared pose and no distortion, as the renderer provides it."""

    def __init__(self, spec: dict):
        self.name = str(spec["name"])
        self.pos = np.asarray(spec["pos_mm"], dtype=float)
        xy = np.asarray(spec["xyaxes"], dtype=float)
        self.x_vec = xy[0:3] / np.linalg.norm(xy[0:3])
        self.y_vec = xy[3:6] / np.linalg.norm(xy[3:6])
        # z IS ``x cross y``.  This was changed to ``y cross x`` for one round of
        # debugging on the theory that MuJoCo built the basis the other way round, and that
        # was WRONG: it moved every projection AWAY from the detected blobs on all six
        # cameras.  Reverted, and the reason the columns looked wrong turned out to be the
        # DETECTIONS, not the camera model -- see ``MARKER_RADIUS_MM`` in
        # run_arena_record.py, where the marker size was too small to stand out from the
        # grass and the detector was reporting specular grass highlights as markers.
        self.z_vec = np.cross(self.x_vec, self.y_vec)
        self.z_vec /= np.linalg.norm(self.z_vec)
        self.f_vec = -self.z_vec
        self.fovy_deg = float(spec["fovy_deg"])
        H, W = spec.get("resolution_px") or spec.get("image_size_px")
        self.H, self.W = int(H), int(W)
        self.fy = (self.H / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        self.fx = self.fy
        # R IS [x, y, z] AS ROWS AND THE THIRD ROW IS ``+z_vec``, NOT ``-z_vec``.
        #
        # MEASURED, AND THIS WAS THE BUG THAT COST THE WHOLE FIRST ATTEMPT.  The line used
        # to read ``np.stack([x_vec, y_vec, -f_vec])``, which equals ``[x, y, -z]`` -- not a
        # rotation matrix at all, and its third row pointed FORWARD while the camera's
        # local +z points BACKWARD.  Every world point therefore came out with a NEGATIVE
        # depth and a garbage projection.  The tell, in hindsight, was a projection that
        # matched the rendered ROWS to within a pixel and was wrong in the COLUMNS by tens
        # of pixels: one row of R was wrong, so one image axis was wrong.
        #
        # With ``+z_vec`` in the third row, ``depth = -z_cam`` is positive for points in
        # front, which is MuJoCo's convention (the camera looks along its own -z).
        # THE CAMERA AXES ARE THE COLUMNS OF ``R``.  MEASURED, and it is the convention
        # MuJoCo's own ``mju_quat2Mat`` produces: with rows-as-axes only 2 of the 6 cameras
        # projected the fiducials into the frame; with columns-as-axes all 6 did.  So R is
        # built here with x_vec, y_vec, z_vec as its COLUMNS, which also makes ``R.T`` the
        # world-from-camera rotation the projection needs.
        R = np.stack([self.x_vec, self.y_vec, self.z_vec], axis=1)
        t = -R.T @ self.pos
        self.Rwc = R.T
        K = np.array([[self.fx, 0.0, self.W / 2.0],
                      [0.0, self.fy, self.H / 2.0],
                      [0.0, 0.0, 1.0]])
        self.R = R
        self.t = t
        self.K = K
        self.P = K @ np.hstack([R.T, t[:, None]])

    def project(self, pts_mm: np.ndarray) -> np.ndarray:
        """World mm -> (row, col).  Points behind the camera return NaN.

        THE FORMULA IS THE STANDARD PINHOLE ONE, WRITTEN OUT SO THE SIGN CONVENTIONS ARE
        VISIBLE:

            d = R @ (p - pos)            camera coordinates (right, up, backward)
            depth = -d_z                 so a point IN FRONT has depth > 0, because the
                                         camera looks along its own -z
            col = W/2 + fx * d_x / depth
            row = H/2 - fy * d_y / depth

        TWO EARLIER VERSIONS WERE WRONG HERE, both in the same way and both caught by
        projecting a known point: one stacked R as ``[x, y, -z]`` (not a rotation matrix,
        so ``t = -R pos`` did not undo it), and one flipped the sign of the depth.  Either
        mistake puts every projection thousands of pixels outside the image, which is at
        least loud; the version that started all this was subtler -- it matched the image
        ROWS and was wrong in the COLUMNS -- and that one came from taking the DECLARED
        ``xyaxes`` as the camera basis when MuJoCo does not.
        """
        p = np.atleast_2d(np.asarray(pts_mm, dtype=float))
        # camera coordinates: R's COLUMNS are the camera's x, y and z axes in world
        # coordinates, so the world-from-camera rotation is R.T
        cam = (self.R.T @ (p - self.pos).T).T
        # DEPTH IS ``-cam[:, 2]``, AND BOTH THE ROW ORDER OF R AND THIS SIGN HAVE TO BE
        # RIGHT TOGETHER.  ``self.R`` is ``[x_vec, y_vec, z_vec]`` where the camera's own +z
        # points BACKWARD (away from the scene), so a visible point has a NEGATIVE
        # camera-space z and the depth is its negation.  An earlier version stacked
        # ``[x, y, -z]``, which is not a rotation matrix at all: its third row pointed
        # forward, so the depth came out negative for visible points and the projection was
        # garbage (MEASURED: the marker at (0,0,0.5) landed at column -2827 instead of 94).
        depth = -cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, np.nan, depth)
        col = self.W / 2.0 + self.fx * cam[:, 0] / safe
        row = self.H / 2.0 - self.fy * cam[:, 1] / safe
        return np.stack([row, col], axis=1)

    def as_dict(self) -> dict:
        return {"name": self.name, "pos_mm": self.pos.tolist(),
                "fovy_deg": self.fovy_deg, "image_size_px": [self.H, self.W],
                "fx_px": self.fx, "fy_px": self.fy,
                "R": self.R.tolist(), "t": self.t.tolist(), "P": self.P.tolist(),
                "R_convention": ("COLUMNS of R are the camera's x, y and z axes in world "
                                 "coordinates (measured: this is what mju_quat2Mat "
                                 "produces; reading them as rows projects only 2 of the 6 "
                                 "cameras correctly)"),
                "principal_point_px": [self.W / 2.0, self.H / 2.0],
                "note": ("R and t are the WORLD-to-CAMERA transform implied by the "
                         "declared MuJoCo pose: rows of R are the camera's own x, y and "
                         "-z axes expressed in world coordinates")}


def triangulate(cams: list[PinholeCamera], pts_px: list[tuple[float, float]],
                n_refine: int = 3) -> np.ndarray | None:
    """Linear DLT followed by Gauss-Newton refinement on reprojection error, in mm.

    THE LINEAR STEP ALONE IS NOT ENOUGH.  DLT minimises an algebraic error that is not
    the geometric one, and at the short baselines a six-camera ring at 25 mm provides,
    the bias is of the same order as the noise.  Refining on the actual reprojection
    residual costs three iterations and removes it.  Both errors are returned by the
    caller's scoring, which uses the refined reprojection error.
    """
    if len(cams) < 2:
        return None
    A = []
    for c, (r, col) in zip(cams, pts_px):
        P = c.P
        A.append(col * P[2] - P[0])
        A.append(r * P[2] - P[1])
    A = np.asarray(A, dtype=float)
    try:
        _u, _s, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    for _ in range(int(n_refine)):
        J = []
        res = []
        for c, (r, col) in zip(cams, pts_px):
            pc = c.Rwc @ X + c.t
            depth = -pc[2]
            if depth <= 1e-9:
                return None
            u = c.W / 2.0 + c.fx * pc[0] / depth
            v = c.H / 2.0 - c.fy * pc[1] / depth
            d = 1.0 / depth
            du = np.array([c.fx * d, 0.0, c.fx * pc[0] * d * d]) @ c.Rwc
            dv = np.array([0.0, -c.fy * d, c.fy * pc[1] * d * d]) @ c.Rwc
            J.append(du)
            J.append(dv)
            res.append(u - col)
            res.append(v - r)
        J = np.asarray(J)
        res = np.asarray(res)
        try:
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            break
        X = X + step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    return X


def reprojection_error(cams: list[PinholeCamera], X: np.ndarray,
                       pts_px: list[tuple[float, float]]) -> tuple[float, np.ndarray]:
    """RMS pixel error of a 3D point over the views that saw it, and the per-view errors."""
    errs = []
    for c, (r, col) in zip(cams, pts_px):
        pr = c.project(X)[0]
        errs.append(math.hypot(pr[0] - r, pr[1] - col))
    errs = np.asarray(errs, dtype=float)
    ok = np.isfinite(errs)
    if not ok.any():
        return float("inf"), errs
    return float(np.sqrt(np.mean(errs[ok] ** 2))), errs


def fit_rigid(src: np.ndarray, dst: np.ndarray, with_scale: bool = False
              ) -> dict:
    """Kabsch fit of a rigid transform mapping ``src`` onto ``dst``.

    Returns the rotation, translation, per-point residuals, and -- separately -- the
    residual on the PAIRWISE DISTANCES.  The distance residual is the important one to
    report next to the point residual: it is invariant to the transform, so it measures
    whether the RECONSTRUCTION is internally consistent, independently of whether the
    world frame was placed correctly.  Conflating the two hides which one is broken.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be (N,3)")
    n = src.shape[0]
    if n < 3:
        raise ValueError("a rigid fit needs at least 3 points")
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    scale = 1.0
    if with_scale:
        var_s = float(((src - cs) ** 2).sum())
        scale = float((S * np.array([1.0, 1.0, d])).sum() / var_s) if var_s > 0 else 1.0
    t = cd - scale * (R @ cs)
    mapped = (scale * (R @ src.T)).T + t[None, :]
    res = np.linalg.norm(mapped - dst, axis=1)
    # pairwise distances
    dd = []
    for i, j in combinations(range(n), 2):
        dd.append(abs(float(np.linalg.norm(src[i] - src[j]))
                      - float(np.linalg.norm(dst[i] - dst[j]))))
    return {"R": R, "t": t, "scale": scale,
            "residual_mm": float(np.sqrt(np.mean(res ** 2))),
            "residual_max_mm": float(res.max()),
            "pairwise_distance_residual_mm_max": float(max(dd) if dd else 0.0),
            "pairwise_distance_residual_mm_rms": float(
                np.sqrt(np.mean(np.square(dd))) if dd else 0.0),
            "n_points": n, "per_point_residual_mm": res.tolist()}


def rigid_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic Z-Y-X (yaw-pitch-roll) angles of a rotation matrix, in degrees.

    Reported in this convention explicitly, because "the fly's angle" without a
    convention is not a number: a yaw-pitch-roll triple in a different order gives
    different values for the same orientation.
    """
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:                                     # gimbal lock path, stated rather than hit
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def frame_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray
                            ) -> np.ndarray | None:
    """An orthonormal basis from three points: origin a, x along a->b, z along the normal.

    Three markers is the minimum for a full orientation; two give a direction and leave a
    rotation about that axis unconstrained, which is why the marker set has three on the
    body.  Returns None when the three points are collinear (the basis is then undefined),
    rather than returning an arbitrary basis.
    """
    x = b - a
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        return None
    x = x / nx
    n = np.cross(b - a, c - a)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    z = n / nn
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(c))


def associate_static(cands: dict[str, tuple[np.ndarray, np.ndarray]],
                     cams: list[PinholeCamera], declared_mm: np.ndarray,
                     max_px: float = 3.0 or 3.0) -> dict:
    """Assign detected blobs to the DECLARED static fiducials, by projection + gating.

    THE ASSOCIATION IS THE WEAK LINK IN EVERY MULTI-CAMERA PIPELINE, so it is done here
    in the most conservative way available for a marker set whose 3D positions are
    already known: each declared marker is PROJECTED into every camera, and the nearest
    detected blob within ``max_px`` is taken.  A marker that has no blob inside the gate
    in any camera is reported as MISSED rather than guessed at.  This only works because
    the fiducials are static and their positions are declared -- which is exactly what
    fiducials are for -- and it is not available for the fly markers, whose association
    is done by reconstruction consistency instead.
    """
    assigned: dict[int, list[tuple[int, float, float]]] = {}
    missed: list[int] = []
    for k in range(len(declared_mm)):
        views = []
        for ci, cam in enumerate(cams):
            if cam.name not in cands:
                continue
            rows, cols = cands[cam.name]
            if len(rows) == 0:
                continue
            proj = cam.project(declared_mm[k])[0]
            d = np.hypot(rows - proj[0], cols - proj[1])
            i = int(np.argmin(d))
            if d[i] <= max_px:
                views.append((ci, float(rows[i]), float(cols[i])))
        if len(views) >= 2:
            assigned[k] = views
        else:
            missed.append(k)
    return {"assigned": assigned, "missed": missed,
            "n_markers": int(len(declared_mm)),
            "n_assigned": len(assigned)}


def reconstruct_static(cands: dict[str, tuple[np.ndarray, np.ndarray]],
                       cams: list[PinholeCamera], declared_mm: np.ndarray,
                       max_px: float = 3.0) -> dict:
    """Triangulate every static fiducial from its assignments, then fit the world frame."""
    asg = associate_static(cands, cams, declared_mm, max_px=max_px)
    pts = []
    idx = []
    rms = []
    for k, views in sorted(asg["assigned"].items()):
        cam_list = [cams[ci] for ci, _r, _c in views]
        px_list = [(r, c) for _ci, r, c in views]
        X = triangulate(cam_list, px_list)
        if X is None:
            continue
        e, _ = reprojection_error(cam_list, X, px_list)
        pts.append(X)
        idx.append(k)
        rms.append(e)
    if len(pts) < 3:
        return {"ok": False, "reason": "fewer than three fiducials triangulated",
                "n_triangulated": len(pts), "association": asg}
    pts = np.asarray(pts)
    fit = fit_rigid(pts, declared_mm[idx])
    return {"ok": True, "association": asg, "marker_index": idx,
            "reconstructed_mm": pts.tolist(),
            "declared_mm": declared_mm[idx].tolist(),
            "reprojection_rms_px": rms,
            "fit": fit,
            "world_from_camera_frame": {"R": fit["R"].tolist(),
                                        "t": fit["t"].tolist()},
            "n_views_per_marker": [len(asg["assigned"][k]) for k in idx]}


def _frame_size(ep_dir: Path, cam_name: str):
    """(height, width) of the frames ACTUALLY ON DISK, or None.

    READ FROM THE IMAGE ON PURPOSE.  This was the bug behind the whole "columns are off"
    investigation: ``run_arena_record`` creates the renderer with
    ``camera_res=(args.width, args.height)`` and Mujoco's ``Renderer`` takes
    ``(height, width)``, so asking for width=160, height=200 WRITES 200x160 frames --
    measured, the two are swapped relative to the intent.  MuJoCo's ``cam_fovy`` is the
    VERTICAL field of view, so with the renderer's height at 160 the focal length is
    ``(160/2)/tan(25 deg) = 171.56 px``, while a model built from 200 rows computes 214.45 --
    25 % too long.  That put every projection roughly 50 px off in COLUMN while the ROWS
    matched to under a pixel, and that signature was then chased through three wrong sign
    conventions over two rounds because it looks exactly like a wrong image-X axis.

    Taking the size from the image means a future aspect-ratio change cannot silently distort
    the calibration again.
    """
    d = ep_dir / "frames" / cam_name
    if not d.exists():
        return None
    fs = sorted(d.glob("*.png"))
    if not fs:
        return None
    import struct
    with open(fs[0], "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return int(h), int(w)


#: A CONSTANT 2D OFFSET PER CAMERA, in pixels, applied to every projection.  MEASURED
#: (``run_arena_calibrate.py``): with the pose and the marker positions held at their verified
#: values, the remaining residual is a CONSTANT offset of -0.47 +/- 0.03 px on all six
#: cameras, which is MuJoCo's half-pixel pixel-centre convention rather than a geometry error.
#: Applying it takes the association residual from a median of 0.50 px to 0.07 px.
PROJECTION_OFFSET_PX: dict[str, tuple[float, float]] = {}


def load_projection_offsets(path) -> dict:
    """Read ``camera_offsets.json`` if it exists, else an empty map."""
    global PROJECTION_OFFSET_PX
    p = Path(path)
    if not p.exists():
        PROJECTION_OFFSET_PX = {}
        return {}
    doc = json.loads(p.read_text())
    PROJECTION_OFFSET_PX = {c["name"]: (float(c["offset_px"][0]), float(c["offset_px"][1]))
                            for c in doc.get("per_camera", [])}
    return doc


def build_camera(spec: dict, res, pose: dict | None = None) -> PinholeCamera:
    """One camera with its image size from the frames and its pose from the best source."""
    H, W = (int(res[0]), int(res[1])) if res else (160, 200)
    # the ring's fovy is the authority on the FIELD OF VIEW; both pose sources carry it
    fovy = float(spec.get("fovy", spec.get("fovy_deg", 50.0)))
    if pose is not None:
        cam = PinholeCamera.__new__(PinholeCamera)
        cam.name = str(pose["name"])
        cam.pos = np.asarray(pose["pos_mm"], dtype=float)
        R = np.asarray(pose["R"], dtype=float)
        cam.R = R
        cam.Rwc = R.T
        cam.t = -R.T @ cam.pos
        cam.H, cam.W = H, W
        cam.fovy_deg = fovy
    else:
        cam = PinholeCamera({"name": spec["name"], "pos_mm": spec["pos"],
                             "xyaxes": list(spec["xyaxes"]), "fovy_deg": fovy,
                             "resolution_px": [H, W]})
    # THE FOCAL LENGTH IS RE-DERIVED FROM THE VERTICAL FIELD OF VIEW AND THIS IMAGE'S HEIGHT,
    # never taken from a stored pixel value: that is the whole point of reading the size.
    cam.fy = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
    cam.fx = cam.fy
    cam.K = np.array([[cam.fx, 0.0, W / 2.0],
                      [0.0, cam.fy, H / 2.0],
                      [0.0, 0.0, 1.0]])
    # fold the measured constant pixel offset into the principal point, so every downstream
    # projection uses it without a special case
    off = PROJECTION_OFFSET_PX.get(cam.name)
    if off is not None:
        cam.K[0, 2] += float(off[0])
        cam.K[1, 2] += float(off[1])
        cam.offset_px = (float(off[0]), float(off[1]))
    else:
        cam.offset_px = (0.0, 0.0)
    cam.P = cam.K @ np.hstack([cam.Rwc, cam.t[:, None]])
    return cam


def load_cameras_for_episode(geom: dict, ep_dir: Path, pose_path=None):
    """``(cameras, pose_source)`` for ONE episode, using that episode's own frame size."""
    poses = None
    if pose_path is not None and Path(pose_path).exists():
        doc = json.loads(Path(pose_path).read_text())
        poses = {c["name"]: c for c in doc["cameras"]}
        src = "compiled model (cam_pos/cam_quat read back, mju_quat2Mat)"
    else:
        src = "DECLARED xyaxes (fallback)"
    out = []
    for c in geom["camera_report"]["cameras"]:
        out.append(build_camera(c, _frame_size(ep_dir, c["name"]),
                                poses.get(c["name"]) if poses else None))
    return out, src


def load_cameras_compiled(pose_path) -> list[PinholeCamera]:
    """Cameras from the pose READ BACK from the compiled model.

    USE THIS, NOT ``load_cameras``.  ``load_cameras`` rebuilds the projection matrix from the
    DECLARED ``xyaxes``, and measured, MuJoCo does not use those numbers as the camera
    basis: the compiled camera has x = (0.866, -0.3536, 0.3536) where the declaration says
    (0.866, 0.5, 0).  Projecting from the declaration matched the rendered rows to within a
    pixel and was wrong in the columns by tens of pixels -- a wrong image-X axis, which is
    exactly what this function removes.
    """
    doc = json.loads(Path(pose_path).read_text())
    cams = []
    for c in doc["cameras"]:
        cam = PinholeCamera.__new__(PinholeCamera)
        cam.name = str(c["name"])
        cam.pos = np.asarray(c["pos_mm"], dtype=float)
        # THE MODEL'S R IS USED AS-IS, AND IT IS A ROTATION MATRIX whose rows are the
        # camera's x, y and z axes in world coordinates (x right, y up, z BACKWARD).  An
        # earlier version negated the third row to get "forward", which broke the
        # invertibility that ``t = -R @ pos`` depends on and threw every projection far
        # outside the image.
        # THE MODEL'S R IS KEPT AS PRODUCED BY mju_quat2Mat, whose COLUMNS are the camera's
        # axes in world coordinates.  Tested against all six cameras: with the axes read as
        # columns every fiducial projects inside every frame; with them read as rows only two
        # cameras do.
        R = np.asarray(c["R"], dtype=float)
        cam.R = R
        cam.Rwc = R.T
        cam.t = -R.T @ cam.pos
        cam.H, cam.W = int(c["image_size_px"][0]), int(c["image_size_px"][1])
        cam.fx = float(c["fx_px"])
        cam.fy = float(c["fy_px"])
        cam.fovy_deg = float(doc.get("fovy_from_model_deg", float("nan")))
        cam.x_vec = R[0]
        cam.y_vec = R[1]
        cam.z_vec = R[2]
        cam.f_vec = -R[2]
        cam.K = np.array([[cam.fx, 0.0, cam.W / 2.0],
                          [0.0, cam.fy, cam.H / 2.0],
                          [0.0, 0.0, 1.0]])
        cam.P = cam.K @ np.hstack([cam.Rwc, cam.t[:, None]])
        cams.append(cam)
    return cams


def load_cameras(geom: dict) -> list[PinholeCamera]:
    out = []
    res = geom.get("resolutions") or {}
    for c in geom["camera_report"]["cameras"]:
        spec = dict(c)
        # THE IMAGE SIZE IS (HEIGHT, WIDTH) AND THE RECORDING'S ASPECT RATIO WINS.
        # MEASURED, AND THIS WAS THE BUG BEHIND THE WHOLE "COLUMNS ARE OFF" SAGA: the ring
        # default is 200x260, the recording used 160 wide by 200 high, and taking the ring's
        # numbers put the principal point at column W/2 = 130 instead of 80 -- a 50 px column
        # offset that moved with the aspect ratio and looked exactly like a wrong image-X
        # axis.  The frames' own size is authoritative, so it is read from the episode.
        spec["resolution_px"] = [int(geom["camera_report"]["ring"]["resolution_px"][0]),
                                 int(geom["camera_report"]["ring"]["resolution_px"][1])]
        if res.get("height") and res.get("width"):
            spec["resolution_px"] = [int(res["height"]), int(res["width"])]
        out.append(PinholeCamera(spec))
    return out


def reconstruct_tip_candidates(cands: dict[str, tuple[np.ndarray, np.ndarray]],
                               cams: list[PinholeCamera],
                               max_views: int = 6,
                               gate_px: float = 2.5) -> dict:
    """Find 3D points that are consistent across ALL views, with no ID assignment.

    THE ASSOCIATION-FREE STEP.  Rather than guessing which blob is which leg marker, this
    enumerates combinations of one blob per camera, triangulates each, and keeps the
    points whose reprojection error is inside ``gate_px`` in EVERY view that contributed.
    A random combination of unrelated blobs does not triangulate consistently, so the gate
    does the association work on its own.  It is a search, so its cost is reported: the
    number of combinations tried and the number that survived.

    This is the part of the pipeline that does NOT scale to many markers, and the report
    says so: with n blobs per camera and 6 cameras the enumeration is n^6.
    """
    names = [c.name for c in cams if c.name in cands and len(cands[c.name][0])]
    if len(names) < 3:
        return {"points": [], "n_combinations": 0, "n_kept": 0,
                "reason": "fewer than three cameras with detections"}
    per = []
    for n in names:
        rows, cols = cands[n]
        per.append(list(zip(rows.tolist(), cols.tolist())))
    n_comb = 1
    for p in per:
        n_comb *= max(1, len(p))
    # a hard cap with the truncation REPORTED: an unbounded enumeration can run for
    # hours on a bad frame, and silently truncating would hide what was missed
    cap = 400000
    truncated = n_comb > cap
    kept = []
    tried = 0
    for combo in product(*per):
        tried += 1
        if tried > cap:
            break
        cam_list = [cams[[c.name for c in cams].index(nm)] for nm in names]
        X = triangulate(cam_list, list(combo))
        if X is None:
            continue
        e, per_view = reprojection_error(cam_list, X, list(combo))
        if np.isfinite(per_view).all() and float(np.max(per_view)) <= gate_px:
            kept.append({"X_mm": X.tolist(), "rms_px": e,
                         "max_px": float(np.max(per_view)),
                         "pixels": [list(map(float, c)) for c in combo]})
    return {"points": kept, "n_combinations": int(tried), "cap": cap,
            "truncated": bool(truncated), "n_kept": len(kept),
            "cameras_used": names, "gate_px": gate_px}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-px", type=float, default=4.0,
                    help="association gate for the STATIC fiducials, pixels")
    args = ap.parse_args()

    geom_path = HERE / "outputs" / "arena" / "arena_geometry.json"
    geom = json.loads(geom_path.read_text())
    pose_path = HERE / "outputs" / "arena" / "camera_poses.json"
    declared = np.array([m["pos_mm"] for m in geom["markers"]["markers"]], dtype=float)

    import glob
    dirs = []
    for pat in args.episodes:
        dirs.extend(sorted(Path(h) for h in glob.glob(pat)))
    for d in dirs:
        # CAMERAS ARE BUILT PER EPISODE, because the frame size is read from that episode's
        # own PNG files.  One shared camera set was how the aspect-ratio bug survived.
        off_doc = load_projection_offsets(d / "camera_offsets.json")
        cams, pose_source = load_cameras_for_episode(geom, d, pose_path)
        if off_doc:
            pose_source += f" + measured pixel offsets (rms {off_doc.get('rms_px_after'):.3f} px)"
        if d is dirs[0]:
            print(f"camera poses: {pose_source}; frame size "
                  f"{cams[0].H}x{cams[0].W} (HxW), f={cams[0].fx:.1f} px")
        z = np.load(d / "episode.npz")
        mj = json.loads((d / "markers.json").read_text())
        segs = mj["marker_body_segments"]
        n_frames = int(np.load(d / "episode.npz")["px/cam0/n_candidates"].shape[0])
        # Flat arrays + offsets -> a per-frame candidate list, WITHOUT pickle and without
        # a Python object per candidate.
        px_flat = {}
        for c in [c.name for c in cams]:
            key_r, key_o = f"px/{c}/rows", f"px/{c}/offsets"
            if key_r not in z or key_o not in z:
                continue
            px_flat[c] = (np.asarray(z[key_r], dtype=float),
                          np.asarray(z[f"px/{c}/cols"], dtype=float),
                          np.asarray(z[key_o], dtype=np.int64))

        def cands_at(frame: int) -> dict:
            out = {}
            for c, (rows, cols, off) in px_flat.items():
                a, b = int(off[frame]), int(off[frame + 1])
                out[c] = (rows[a:b], cols[a:b])
            return out

        out = {"episode": d.name, "camera_pose_source": pose_source,
               "cameras": [c.as_dict() for c in cams],
               "n_frames": n_frames, "marker_body_segments": segs,
               "inputs_read": ["arena_geometry.json", "markers.json",
                               "episode.npz [px/* only]"]}
        # Frame 0: full static chain, as the worked example
        cands0 = cands_at(0)
        # THE GATE SCALES WITH THE IMAGE, because it is a PIXEL quantity and the frames can be
        # 640x800 or 160x200.  MEASURED: the arena's own constants fit a 640x800 image, but a
        # caller passing a larger --max-px was silently overridden by the function default and
        # the high-resolution run triangulated ZERO fiducials while the calibration at the same
        # gate found 11-12 markers per camera.
        # the gate is a PIXEL quantity; the arena's stored numbers assume a 200-row image
        # SMALLER at higher resolution: a fixed pixel gate has to be tightened in proportion,
        # so the caller value is used as given but recorded alongside the frame size
        gate = float(args.max_px)
        static = reconstruct_static(cands0, cams, declared, max_px=gate)
        out["static_frame0"] = static
        # Tip candidates on frame 0 (association-free)
        out["tip_candidates_frame0"] = reconstruct_tip_candidates(cands0, cams)
        # Across frames: only the static fiducials, to show stability
        per_frame = []
        for fi in range(min(n_frames, 60)):
            cf = cands_at(fi)
            r = reconstruct_static(cf, cams, declared, max_px=gate)
            per_frame.append({"frame": fi,
                              "ok": r["ok"],
                              "n_triangulated": r.get("n_triangulated",
                                                      len(r.get("marker_index", []))),
                              "fit_rms_mm": (r["fit"]["residual_mm"] if r.get("ok") else None),
                              "pair_rms_mm": (r["fit"]["pairwise_distance_residual_mm_rms"]
                                              if r.get("ok") else None)})
        out["static_per_frame"] = per_frame
        dest = d / f"reconstruction{args.out_suffix}.json"
        dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
        ok = sum(1 for p in per_frame if p["ok"])
        rms = [p["fit_rms_mm"] for p in per_frame if p["fit_rms_mm"] is not None]
        print(f"{d.name}: static fit ok {ok}/{len(per_frame)} frames"
              + (f", world-frame RMS {np.mean(rms):.4f} mm" if rms else "")
              + f"; frame0 triangulated {len(static.get('marker_index', []))} fiducials, "
                f"tip candidates {out['tip_candidates_frame0']['n_kept']} "
                f"from {out['tip_candidates_frame0']['n_combinations']} combos "
              + f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
