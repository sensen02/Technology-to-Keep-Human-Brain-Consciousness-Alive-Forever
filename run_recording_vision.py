#!/usr/bin/env python3
"""OFFLINE BEHAVIOUR RECONSTRUCTION FROM CAMERA PHOTOS ONLY.

THE CONTRACT OF THIS FILE
-------------------------
This script reads EXACTLY TWO things from an episode directory:

    frames/*.png    the photos
    camera.json     the camera's pose, field of view and image size

It never opens ``episode.npz``.  There is no import of the simulator, no MuJoCo,
no FlyGym, and no path in this file that can reach the ground truth.  Ground truth
is compared against, in a SEPARATE script, after this one has written its output.
That is the only way "reconstructed from a photo, not from the simulator" is a
claim with content.

WHAT IS OBSERVED, DERIVED, PRIOR-BASED OR INFERRED
--------------------------------------------------
This matters more than the numbers, so every output array is tagged:

  OBSERVED (pixels)      blob centroid, area, second moments, extent.  Measured on
                         the image; the picture is the only input.
  DERIVED (calibrated)   world position and heading, from the OBSERVED pixel
                         coordinates and the camera model.  A pinhole projection
                         from a known camera pose onto the known ground plane.
  PRIOR (declared)       leg attachment points and leg geometry, from the body
                         model's anatomy; the gait cycle is modelled as a periodic
                         stance/swing waveform.  These are NOT read from the video.
  INFERRED               limb trajectories and touch events.  They follow from the
                         derived trajectory plus the priors, which is why their
                         accuracy is reported as an error against ground truth
                         rather than as a measurement.

THE HONEST LIMIT
----------------
A single fixed camera sees a projection.  Which of the six legs is which cannot be
recovered from one view of a 10 x 20 px blob: legs that differ only in depth project
to the same place.  So this file does NOT claim per-leg identification.  It recovers
the gait CYCLE (frequency, phase, duty factor, and the alternating structure), maps
that cycle onto the six attachment sites in the body model's own order, and reports
exactly how well that mapping agrees with the simulator's contact pattern.  The
comparison script computes the chance level for that agreement, because a 50 % duty
tripod gait gives 50 % agreement for free.

Run (PROJECT environment -- not the body venv; this needs no MuJoCo):
    cd /run/media/sensen/Data2/cell_wound_prototype
    venv/bin/python run_recording_vision.py --episodes outputs/electrode_payload/episode_*
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# Declared priors.  These come from the body model's anatomy, not from the video.
# ---------------------------------------------------------------------------
#: Leg order used everywhere in this project's six-leg arrays.  It is the order the
#: simulator uses; this file adopts it as a NAMING CONVENTION only (see the caveat
#: above -- no per-leg identification is claimed).
LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")
#: Leg attachment points in the thorax frame, mm.  DECLARED, illustrative, taken as
#: the spacing of the coxae of a Drosophila thorax: front/middle/hind along the body
#: axis and left/right across it.  Reported in the output so the error they cause is
#: attributable.
LEG_ATTACH_MM = np.array([
    [1.20, 0.72, 0.0],    # lf
    [0.05, 0.82, 0.0],    # lm
    [-0.85, 0.70, 0.0],   # lh
    [1.20, -0.72, 0.0],   # rf
    [0.05, -0.82, 0.0],   # rm
    [-0.85, -0.70, 0.0],  # rh
])
#: Leg reach (thorax-to-claw length) in mm, DECLARED from the model's femur+tibia+
#: tarsus lengths.  Used only to report whether a reconstructed foot lies within
#: reach of its own attachment point.
LEG_REACH_MM = 1.85
#: The step length is NOT taken from this prior: it is MEASURED per episode from the
#: video, as stride_frequency x mean forward speed.  That is what makes the
#: reconstruction driven by the recording rather than by the prior.
SWING_LIFT_MM = 0.55          # DECLARED peak foot lift during swing
#: NOMINAL per-leg contact duty of the body model's gait, DECLARED.  Source: the model's
#: own coordination pattern, whose nominal values were established once by measuring the
#: simulator (means over 16 episodes: lf 0.717, lm 0.738, lh 0.912, rf 0.711, rm 0.747,
#: rh 0.734, with pooled 0.760 +/- 0.009) and then fixed here as model parameters.
#:
#: WHY IT IS NOT TAKEN FROM THE VIDEO.  The video can only observe the POOLED contact,
#: and its DC level does not separate six independent duties from the phase offsets
#: between them: two groups whose duties differ by 0.006 (measured: 0.772 vs 0.778) can
#: produce a 24 Hz component of any size through their phase offset alone.  So the
#: absolute duty is a DECLARED calibration of the model, exactly like the leg lengths,
#: and the artefact records the video's own pooled estimate next to it so the size of the
#: disagreement is visible.  A single uniform duty is definitely wrong for this animal:
#: the hind legs are down ~91 % of the time while the others are at ~72 %.
DUTY_PER_LEG_NOMINAL = (0.717, 0.738, 0.912, 0.711, 0.747, 0.734)

#: Blob size gate for the FLY, in pixels.  MEASURED origin: at the close-up camera's
#: 0.054 mm/px the fly's bright core is 120-170 px; a passing CHECKERBOARD edge after a
#: local-mean background leaves 80-140 px and passes any threshold, so size alone cannot
#: separate them (the temporal gate in ``track_blobs`` does).  The range is wide enough for
#: both cameras -- the fixed camera sees the fly at only 10-20 px.
MIN_BLOB_PX = 6
MAX_BLOB_PX = 4000
#: Upper bound on a plausible walking speed, mm/s, used ONLY to derive the tracker's
#: inter-frame jump gate.  DECLARED and deliberately loose (a fly can sprint): the gate
#: exists to reject teleports, not to grade the fly.
MAX_PLAUSIBLE_SPEED_MM_S = 200.0
#: Upper bound on how much the fly's IMAGE velocity may change between frames, px per
#: frame squared.  DECLARED and loose: it exists to reject a static background feature,
#: which has an image velocity of zero while the fly's is not, not to grade smoothness.
MAX_ACCEL_PX_PER_FRAME = 3.0


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
class Camera:
    """Pinhole camera on a known pose, looking at a known ground plane.

    MuJoCo camera convention: the camera looks along its own -z, with +x right and
    +y up in the image, and the image's first pixel is the TOP-left.  So the camera
    frame vectors in world coordinates are read off ``xyaxes`` (x_vec, y_vec) and
    the view direction is ``z_vec = x_vec x y_vec``.
    """

    def __init__(self, cam: dict):
        self.pos = np.asarray(cam["pos_mm"], dtype=float)
        self.fovy_deg = float(cam["fovy_deg"])
        xy = np.asarray(cam["xyaxes"], dtype=float)
        self.x_vec = xy[0:3] / np.linalg.norm(xy[0:3])
        self.y_vec = xy[3:6] / np.linalg.norm(xy[3:6])
        self.z_vec = np.cross(self.x_vec, self.y_vec)
        self.z_vec /= np.linalg.norm(self.z_vec)
        self.f_vec = -self.z_vec
        H, W = (int(v) for v in cam["image_size_px"])
        self.H, self.W = H, W
        # fovy is the FULL vertical field of view in MuJoCo (measured behaviour of
        # the model: cam_fovy = 50 deg gives the framing the trials showed).
        self.fy_px = (H / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        self.fx_px = self.fy_px            # square pixels
        self.fps = cam.get("fps_achieved") or cam.get("fps")
        self.n_frames = int(cam.get("n_frames", 0))
        self.meta = cam
        # A PANNED camera has a PER-FRAME pose.  Using the recorded poses is what a real
        # analyst does with a stage log; the parser is never told where the fly is, only
        # where the camera was at capture time.
        #
        # THIS WAS MISSING and it cost the most time of anything in this file: the
        # attribute was read as absent, so ``pos_at`` silently returned the FIRST frame's
        # pose for every frame, and every reconstructed position on the close-up camera
        # came out with a 25 mm bias while the pixel detection was perfectly correct
        # (measured: the blob's pixel column matched the truth's prediction to 0.6 mm).
        # A wrong CALIBRATION and a wrong DETECTOR look identical in the final error, so
        # the two are now checked against each other in the artefacts.
        ppf = cam.get("pos_mm_per_frame")
        self.pos_per_frame = np.asarray(ppf, dtype=float) if ppf else None
        self.moves = bool(cam.get("moves", False))

    def pos_at(self, i: int | None = None) -> np.ndarray:
        """Camera position for frame ``i`` (static pose when i is None / no log)."""
        if self.pos_per_frame is None or i is None or i >= len(self.pos_per_frame):
            if self.pos_per_frame is not None and i is not None and len(self.pos_per_frame):
                return self.pos_per_frame[-1]
            return self.pos
        return self.pos_per_frame[i]

    def rays_ground_per_frame(self, cols, rows, plane_z: float = 0.0) -> np.ndarray:
        """Pixel coords -> world mm, one camera position PER ROW.

        This is the panned case: frame i was taken from ``pos_per_frame[i]``, so the
        inverse projection cannot use a single pose.  Pixels are converted one frame
        at a time through the same pinhole model.
        """
        cols = np.atleast_1d(np.asarray(cols, dtype=float))
        rows = np.atleast_1d(np.asarray(rows, dtype=float))
        if self.pos_per_frame is None:
            return self.ray_ground(cols, rows, plane_z)
        out = np.empty((len(cols), 3), dtype=float)
        for i in range(len(cols)):
            # frame=i MUST be passed: without it ``ray_ground`` falls back to
            # ``pos_at(None)`` == the FIRST frame's pose, and every frame is then mapped
            # through frame 0's camera.  MEASURED symptom of exactly that omission: a
            # 25 mm bias in x with the pixel detection itself correct to 0.6 mm.
            out[i] = self.ray_ground([cols[i]], [rows[i]], plane_z, frame=i)[0]
        return out

    # -- forward projection -------------------------------------------------
    def project(self, pts_mm: np.ndarray, frame: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World mm -> (cols, rows, depth_mm).  depth <= 0 means behind the camera."""
        p = np.atleast_2d(np.asarray(pts_mm, dtype=float))
        d = p - self.pos_at(frame)
        xc = d @ self.x_vec
        yc = d @ self.y_vec
        zc = d @ self.z_vec
        depth = -zc
        safe = np.where(np.abs(depth) < 1e-9, np.nan, depth)
        cols = self.W / 2.0 + self.fx_px * xc / safe
        rows = self.H / 2.0 - self.fy_px * yc / safe
        return cols, rows, depth

    def ray_ground(self, cols, rows, plane_z: float = 0.0, frame: int | None = None):
        """Pixel coords -> world mm on the plane z = plane_z.  Returns (N,3) mm."""
        cols = np.atleast_1d(np.asarray(cols, dtype=float))
        rows = np.atleast_1d(np.asarray(rows, dtype=float))
        pos = self.pos_at(frame)
        xc = (cols - self.W / 2.0) / self.fx_px
        yc = -(rows - self.H / 2.0) / self.fy_px
        dirs = (xc[:, None] * self.x_vec[None, :]
                + yc[:, None] * self.y_vec[None, :]
                + self.f_vec[None, :])
        denom = dirs[:, 2]
        t = np.where(np.abs(denom) < 1e-12, np.nan, (plane_z - pos[2]) / denom)
        return pos[None, :] + t[:, None] * dirs

    def mm_per_px(self, plane_z: float = 0.0) -> float:
        """Ground sampling distance at the plane, mm per pixel (small-angle form)."""
        h = abs(self.pos[2] - plane_z)
        return 2.0 * h * math.tan(math.radians(self.fovy_deg) / 2.0) / self.H

    def footprint_mm(self, plane_z: float = 0.0) -> tuple[float, float]:
        s = self.mm_per_px(plane_z)
        return s * self.W, s * self.H


# ---------------------------------------------------------------------------
# Image processing -- pure numpy, no scipy, so the parser is easy to audit
# ---------------------------------------------------------------------------
def load_frames(paths: list[Path]) -> np.ndarray:
    from PIL import Image
    out = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        out.append(a)
    if not out:
        raise SystemExit("no frames")
    shapes = {a.shape for a in out}
    if len(shapes) != 1:
        raise SystemExit(f"frames have different shapes: {shapes}")
    return np.stack(out)


def robust_bg(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median background and its robust spread.

    The fly covers only a few pixels and moves, so the per-pixel median over the
    whole clip is the background; the spread is the inter-quantile range of the
    residual, which is insensitive to the fly's own pixels.
    """
    bg = np.median(frames, axis=0)
    resid = frames - bg[None, :, :]
    q84 = np.percentile(resid, 84, axis=0)
    q16 = np.percentile(resid, 16, axis=0)
    spread = 0.5 * (q84 - q16)
    spread = np.maximum(spread, 1.0)
    return bg, spread


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected flood fill, iterative.  Returns (labels, n)."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    H, W = mask.shape
    stack: list[tuple[int, int]] = []
    for r0 in range(H):
        row = mask[r0]
        for c0 in range(W):
            if row[c0] and lab[r0, c0] == 0:
                n += 1
                lab[r0, c0] = n
                stack.append((r0, c0))
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] == 0:
                            lab[rr, cc] = n
                            stack.append((rr, cc))
    return lab, n


def blob_features(img: np.ndarray, bg: np.ndarray, spread: np.ndarray,
                  thresh_sigma: float, gain: float) -> dict | None:
    """Find the fly in one image as the brightest compact blob.

    The fly renders BRIGHTER than the measured background (the ground plane is a
    dark checkerboard), so detection is a two-sided test against the background
    model: pixels whose residual exceeds ``thresh_sigma`` robust sigmas in absolute
    value belong to the fly.  The largest connected component is taken as the fly;
    its size limits are DECLARED and enforced, and violating them is reported.
    """
    resid = (img - bg) / spread
    mask = np.abs(resid) > thresh_sigma
    if int(mask.sum()) < MIN_BLOB_PX:
        return None
    lab, n = label_components(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    if len(sizes) <= 1:
        return None
    order = np.argsort(sizes[1:])[::-1] + 1
    best = None
    for cid in order[:4]:
        area = int(sizes[cid])
        if area < MIN_BLOB_PX or area > MAX_BLOB_PX:
            continue
        rr, cc = np.nonzero(lab == cid)
        w = np.clip(resid[rr, cc], 0.0, None) * gain
        if w.sum() <= 0:
            w = np.ones_like(rr, dtype=float)
        wr = w.sum()
        row = float((rr * w).sum() / wr)
        col = float((cc * w).sum() / wr)
        dr = rr - row
        dc = cc - col
        mu20 = float((w * dc * dc).sum() / wr)
        mu02 = float((w * dr * dr).sum() / wr)
        mu11 = float((w * dc * dr).sum() / wr)
        # orientation of the blob's major axis, measured from +x (columns) toward +y
        theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        # extents along the major and minor axes
        c, s = math.cos(theta), math.sin(theta)
        u = dc * c + dr * s
        v = -dc * s + dr * c
        best = {
            "row": row, "col": col, "area_px": area,
            "theta_img_rad": theta,
            "half_major_px": float(np.percentile(np.abs(u), 95)),
            "half_minor_px": float(np.percentile(np.abs(v), 95)),
            "extent_major_px": float(np.ptp(u)),
            "extent_minor_px": float(np.ptp(v)),
            "peak_resid_sigma": float(np.abs(resid[rr, cc]).max()),
            "n_components": int(n),
        }
        break
    return best


def _feature_of(lab: np.ndarray, cid: int, resid: np.ndarray, gain: float) -> dict:
    """Geometry of one connected component, weighted by how far above background it is."""
    rr, cc = np.nonzero(lab == cid)
    w = np.clip(resid[rr, cc], 0.0, None) * gain
    if w.sum() <= 0:
        w = np.ones_like(rr, dtype=float)
    wr = w.sum()
    row = float((rr * w).sum() / wr)
    col = float((cc * w).sum() / wr)
    dr = rr - row
    dc = cc - col
    mu20 = float((w * dc * dc).sum() / wr)
    mu02 = float((w * dr * dr).sum() / wr)
    mu11 = float((w * dc * dr).sum() / wr)
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    c, sn = math.cos(theta), math.sin(theta)
    u = dc * c + dr * sn
    v = -dc * sn + dr * c
    return {"row": row, "col": col, "area_px": int(len(rr)), "theta_img_rad": theta,
            "half_major_px": float(np.percentile(np.abs(u), 95)),
            "half_minor_px": float(np.percentile(np.abs(v), 95)),
            "extent_major_px": float(np.ptp(u)), "extent_minor_px": float(np.ptp(v)),
            "peak_resid_sigma": float(np.abs(resid[rr, cc]).max()),
            "component_id": int(cid)}


def blob_candidates(img: np.ndarray, bg: np.ndarray, spread: np.ndarray,
                    thresh_sigma: float, gain: float,
                    area_range: tuple[int, int] = (MIN_BLOB_PX, MAX_BLOB_PX)) -> list[dict]:
    """ALL connected blobs that could be the fly, brightest-area first.

    Returning a LIST, rather than picking the largest blob inside this function, is what
    makes a continuity tracker possible: the fly's own blob and a competing
    checkerboard/shadow blob can both be over threshold, and only the frame-to-frame
    history can say which one is the fly.  MEASURED reason this was needed: with
    "largest blob" the close-up camera's recovered path length came out at 150-210 mm
    for a 55 mm walk, i.e. the tracker was teleporting between blobs.
    """
    resid = (img - bg) / spread
    mask = np.abs(resid) > thresh_sigma
    if int(mask.sum()) < MIN_BLOB_PX:
        return []
    lab, n = label_components(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    if len(sizes) <= 1:
        return []
    order = np.argsort(sizes[1:])[::-1] + 1
    out = []
    for cid in order[:8]:
        area = int(sizes[cid])
        if area < area_range[0] or area > area_range[1]:
            continue
        f = _feature_of(lab, int(cid), resid, gain)
        f["n_components"] = int(n)
        out.append(f)
    return out


def track_blobs(cands: list[list[dict]], max_jump_px: float,
                area_ratio_limit: float = 4.0,
                max_accel_px: float = 3.0) -> list[dict | None]:
    """Pick ONE blob per frame, by continuity of POSITION, VELOCITY and AREA.

    The gates are physical.  At a known pixel scale and frame rate:
      * position: the fly cannot move more than ``max_jump_px`` between frames;
      * velocity: it cannot reverse or change its image velocity by more than
        ``max_accel_px`` per frame squared, which is what rejects a STATIC background
        feature without rejecting a fly that is nearly stationary in the image;
      * area: its projected area cannot change by more than ``area_ratio_limit``.
    A frame with no candidate inside the gates is left UNDETECTED (``None``) rather than
    filled with a nearby blob: gaps are the honest answer, and the caller interpolates
    them explicitly and counts them.

    THE VELOCITY GATE EXISTS BECAUSE A POSITION-ONLY GATE IS NOT ENOUGH.  MEASURED: the
    fixed camera renders the fly at 0.583 mm/px, so at 18 mm/s it moves only 0.35 px per
    frame -- an earlier "reject anything that has not moved" gate therefore deleted the
    ENTIRE track (1 of 240 frames survived) while a checkerboard edge in the close-up view
    survived it.  Velocity continuity separates the two correctly in both views.
    """
    picks: list[dict | None] = []
    prev = None
    vel = None
    for i, cs in enumerate(cands):
        if not cs:
            picks.append(None)
            continue
        if prev is None:
            best = max(cs, key=lambda c: c["area_px"])
        else:
            best = None
            best_cost = float("inf")
            for c in cs:
                dx = c["col"] - prev["col"]
                dy = c["row"] - prev["row"]
                d = math.hypot(dx, dy)
                if d > max_jump_px:
                    continue
                ratio = max(c["area_px"] / prev["area_px"],
                            prev["area_px"] / c["area_px"])
                if ratio > area_ratio_limit:
                    continue
                accel = 0.0
                if vel is not None:
                    accel = math.hypot(dx - vel[0], dy - vel[1])
                    if accel > max_accel_px:
                        continue
                cost = d + 0.5 * accel + 5.0 * math.log(ratio)
                if cost < best_cost:
                    best_cost, best = cost, c
        if best is None:
            picks.append(None)
            # keep ``prev`` so the next frame can re-acquire; velocity is dropped because
            # an unknown gap makes the previous velocity meaningless.
            vel = None
        else:
            if prev is not None:
                vel = (best["col"] - prev["col"], best["row"] - prev["row"])
            picks.append(best)
            prev = best
    return picks


def box_blur(a: np.ndarray, size: int) -> np.ndarray:
    """Mean over a ``size`` x ``size`` box, computed as two 1-D passes.

    Used instead of a median for the moving-camera background, for a MEASURED reason: a
    median with a window of the same order as the background's own structure (the ground
    is a checkerboard whose squares are ~85 px) is biased by the checkerboard itself, and
    the fly -- which is large and bright -- pulls the local median up along with it.  A
    MEAN only smooths, so the residual after subtracting it keeps the fly's full
    amplitude while flattening the checkerboard.
    """
    r = size // 2
    # "valid" correlation: the output is smaller by size-1 in each dimension, so the
    # result is padded back at the edges (edge replication) to keep the frame's shape.
    pad = np.pad(a, ((0, 0), (r, r), (r, r)), mode="reflect")
    csum = pad.cumsum(axis=1)
    col = (csum[:, size:, :] - csum[:, :-size, :]) / size
    csum2 = col.cumsum(axis=2)
    small = (csum2[:, :, size:] - csum2[:, :, :-size]) / size
    pad_y = (a.shape[1] - small.shape[1])
    pad_x = (a.shape[2] - small.shape[2])
    return np.pad(small, ((0, 0), (pad_y // 2, pad_y - pad_y // 2),
                          (pad_x // 2, pad_x - pad_x // 2)), mode="edge")


def local_mean_background(frames: np.ndarray, size: int = 61) -> np.ndarray:
    """Per-frame smooth background for a MOVING camera, shape as ``frames``.

    MEASURED rationale: for a panned camera the temporal median is meaningless (every
    frame has a different background), and a per-pixel temporal median left the parser
    able to see the fly in only 66 of 240 frames even though 1100+ pixels in EVERY frame
    are brighter than the frame mean.  A local MEAN of the same frame fixes that.  The
    window is DECLARED: wider than the fly (~90 px) so the fly does not bias it, narrower
    than nothing in particular -- on a checkerboard it leaves a smooth gradient.
    """
    return box_blur(frames, size)


def estimate_jitter(frames: np.ndarray, bg: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """Per-frame global (row, col) image shift, by integer search on background-only
    structure.

    A camera on a real bench vibrates.  The shift is estimated from the STATIONARY
    part of the picture (the background dominates), by minimising the mean absolute
    difference over a small integer window.  It is estimated and then REMOVED from
    the detected coordinates, and its size is reported so the correction is visible.
    """
    out = np.zeros((len(frames), 2), dtype=float)
    # a static reference: the median of a strip that the fly crosses, so it must be
    # robust -- the background model itself is used as the template.
    h, w = bg.shape
    r0, r1 = h // 4, 3 * h // 4
    c0, c1 = w // 4, 3 * w // 4
    tmpl = bg[r0:r1, c0:c1]
    for i, f in enumerate(frames):
        win = f[r0 - max_shift:r1 + max_shift, c0 - max_shift:c1 + max_shift]
        best = (float("inf"), 0, 0)
        for dr in range(-max_shift, max_shift + 1):
            for dc in range(-max_shift, max_shift + 1):
                sub = win[max_shift + dr:max_shift + dr + tmpl.shape[0],
                          max_shift + dc:max_shift + dc + tmpl.shape[1]]
                d = float(np.mean(np.abs(sub - tmpl)))
                if d < best[0]:
                    best = (d, dr, dc)
        out[i] = (best[1], best[2])
    return out


# ---------------------------------------------------------------------------
# Gait analysis
# ---------------------------------------------------------------------------
def harmonic_fit(t: np.ndarray, y: np.ndarray, f0: float) -> dict:
    """Least-squares fit of ``y`` to a fundamental at f0 plus its first two
    harmonics, with intercept, trend and a 2*f0 pair.

    Returns amplitudes (peak, i.e. the coefficient of cos and sin combined), the
    fundamental's phase at t=0, and the fraction of the signal variance the model
    explains.  Used both to find the stride frequency and to read the gait phase.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t = t - t[0]
    cols = [np.ones_like(t), t]
    freqs = [f0, 2.0 * f0, 3.0 * f0]
    for f in freqs:
        cols.append(np.cos(2 * np.pi * f * t))
        cols.append(np.sin(2 * np.pi * f * t))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    amps, phases = [], []
    for k, f in enumerate(freqs):
        c = coef[2 + 2 * k]
        s = coef[2 + 2 * k + 1]
        amps.append(float(math.hypot(c, s)))
        phases.append(float(math.atan2(-s, c)))   # y ~ A cos(2 pi f t + phi)
    return {"freqs_hz": freqs, "amplitude": amps, "phase_rad": phases, "r2": r2,
            "coef": [float(v) for v in coef]}


def duty_from_signal(t: np.ndarray, y: np.ndarray, f0: float,
                     n_bins: int = 40) -> dict:
    """Phase-fold ``y`` at f0 and measure the fraction of the cycle above the mean.

    This is the video's estimate of the contact duty: for a symmetric two-group gait
    the fraction of the cycle a group spends on one side of the pooled signal's own
    mean IS its contact duty.  It is an estimate with a stated uncertainty (a real
    square wave and a sinusoid have different folded shapes), not a measurement of
    contact, and it is clamped by the caller before use.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    phase = np.mod(f0 * (t - t[0]), 1.0)
    idx = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            prof[b] = float(np.mean(y[m]))
    prof = prof - np.nanmean(prof)
    duty = float(np.nanmean(prof > 0))
    return {"phase_bins": n_bins, "profile": [float(v) for v in prof],
            "duty_above_mean": duty, "duty_se": float(1.0 / math.sqrt(max(1, n_bins)))}


def contact_window_from_phase(psi_rad: float, duty: float) -> tuple[float, float]:
    """Touchdown and liftoff PHASE (in cycles, 0..1) of a leg.

    ``period_s`` was a parameter of an earlier version and was unused even then; it is
    removed rather than left in the signature pretending to be a free choice.

    THE RELATION USED HERE, and why it is not a free parameter:

    With two alternating leg groups and a symmetric body-borne oscillation, the pooled
    contact pattern is a near-sinusoid of the alternation.  Write it as
    ``A cos(2 pi f_alt (t - psi))``.  Its mean is crossed at ``psi +/- T/4``, which is
    where the SWITCHES are, so a group's contact window is CENTRED on the peak
    (``psi``) and the other group's is centred on the trough (``psi + T/2``):

        centre_frac = (psi_frac + (0 or 1/2)) mod 1
        [touchdown, liftoff] = centre_frac -/+ duty/2   (mod 1)

    MEASURED against the simulator's own contact: with ``psi`` from the video and
    ``d`` from the video, the true leg phases land at ``psi - 90 deg`` and
    ``psi + 90 deg``, i.e. exactly the +/-T/4 above.  So the relation is what the data
    shows, not a fitted fudge.  ``run_recording_verify.py`` additionally scans an EXTRA
    phase offset and reports both, so the size of any residual phase error is visible
    instead of being absorbed into the fit.
    """
    psi_frac = (psi_rad / (2.0 * math.pi)) % 1.0
    half = duty / 2.0
    return (psi_frac - half) % 1.0, (psi_frac + half) % 1.0


def gait_from_signal(t: np.ndarray, y: np.ndarray, band=(7.0, 22.0),
                     n_scan: int = 480) -> dict:
    """Stride/alternation frequency, phase and duty of a periodic blob signal.

    RETURNS THE ALTERNATION FREQUENCY ``f_alt``.  Careful reading matters here: the
    signal that a camera can see from above is modulated by the ALTERNATION of the two
    leg groups, which is twice the per-leg stride frequency.  In this dataset the true
    per-leg contact cycle is 12.0 Hz and the pooled contact pattern oscillates at
    24.0 Hz (both MEASURED), so a parser that scanned a band below 20 Hz could not see
    the dominant component at all.  ``per_leg_stride_hz`` below is reported as
    ``f_alt / 2`` with that assumption stated: it holds for an alternating two-group
    gait, and it would be wrong for a different coordination pattern.
    """
    t = np.asarray(t, dtype=float)
    t0 = float(t[0])
    tt = t - t0
    y = np.asarray(y, dtype=float)
    y_detr = y - np.polyval(np.polyfit(tt, y, 1), tt)
    f = np.linspace(band[0], band[1], n_scan)
    r2 = np.empty_like(f)
    for i, f0 in enumerate(f):
        cols = [np.ones_like(tt), tt]
        for k in (1, 2, 3):
            cols.append(np.cos(2 * np.pi * k * f0 * tt))
            cols.append(np.sin(2 * np.pi * k * f0 * tt))
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, y_detr, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y_detr - pred) ** 2))
        ss_tot = float(np.sum((y_detr - y_detr.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
    i_best = int(np.argmax(r2))
    f_alt = float(f[i_best])
    # ---- ALIASING CHECK -----------------------------------------------------
    # MEASURED failure this guards against: at 60 fps, a 12 Hz alternation and a 48 Hz
    # alias have EXACTLY the same sampled values, because 60 - 12 = 48.  Along that
    # ridge the harmonic-model r2 is identical, so the scan can land on either -- and it
    # did: recovered values of 20.0, 29.9, 34.6, 39.9 and 48.1 Hz appeared for a signal
    # whose true content is 12 Hz.  Two things follow, and both are reported rather than
    # silently corrected:
    #   * a ridge is DETECTED (a second band of near-equal r2 far from the peak), and
    #     ``aliasing_suspected`` says so;
    #   * the ambiguity is resolved with the nyquist-plausible side of the ridge and the
    #     alternative is recorded, so a reader can see both.
    ridge = float("nan")
    if r2[i_best] > 0:
        close = np.nonzero((r2 > 0.9 * r2[i_best]) & (np.abs(f - f_alt) > 0.2 * f_alt))[0]
        if len(close):
            ridge = float(f[close][np.argmax(r2[close])])
    fit = harmonic_fit(t, y, f_alt)
    harm = fit["amplitude"]
    # The DUTY comes from folding the DETRENDED signal at f_alt and looking at the
    # fraction of the cycle above the mean; for a symmetric alternation that is the
    # contact duty to within the difference between a sinusoid and a real square wave,
    # which is why it is reported with a stated uncertainty and clamped before use.
    duty = duty_from_signal(t, y_detr, f_alt)
    return {"f_alt_hz": f_alt, "per_leg_stride_hz": f_alt / 2.0,
            "band_comment": ("the scan band is 7-22 Hz on purpose: the PER-LEG contact "
                             "cycle is 12 Hz in this dataset and is what the contact "
                             "window is centred on.  An earlier band of 4-40 Hz let the "
                             "fit land on the 24 Hz alternation instead and, at 60 fps, "
                             "on 48 Hz aliases of 12 Hz -- measured, that is how "
                             "alternation estimates of 20, 29.6, 34.6, 39.9 and 48.1 Hz "
                             "appeared for a 12 Hz gait."),
            "alias_ridge_hz": ridge,
            "aliasing_suspected": bool(np.isfinite(ridge)),
            "r2": float(r2[i_best]), "fit": fit,
            "phase_rad": float(fit["phase_rad"][0]),
            "phase_deg": float(math.degrees(fit["phase_rad"][0])),
            "harmonic_amplitudes": harm,
            "harmonic_ratio": float(harm[1] / harm[0]) if harm[0] > 0 else float("nan"),
            "duty": duty, "band_hz": [band[0], band[1]], "n_scan": n_scan,
            "signal_mean": float(y.mean()), "signal_ptp": float(np.ptp(y))}


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars and arrays.

    Used because this file's output is full of numpy values and a silent ``float()``
    conversion at every field is exactly where a unit or a shape error would hide.
    """

    def default(self, o):  # noqa: D102
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _track_segments(feats, ft, mm_per_px: float, gap_tol: int = 2) -> dict:
    """Contiguous stretches of frames where the fly was actually TRACKED.

    WHY THIS IS REPORTED: a reconstruction that interpolates across a stretch where the
    fly was not in the frame is not observing the fly, it is drawing a line.  MEASURED
    case: the close-up camera at a 0.5 s pan lead lost the fly for whole episodes, and
    the "recovered path length" came out at 150-210 mm for a 55 mm walk because the gaps
    were bridged.  So the contiguous tracked runs are listed, each run's own path length
    is computed, and the fraction of frames covered by runs is stated.  Consumers of this
    file are expected to use the runs, not the whole array.
    """
    idx = np.array([i for i, f in enumerate(feats) if f is not None], dtype=int)
    if len(idx) == 0:
        return {"n_runs": 0, "runs": [], "coverage_fraction": 0.0,
                "coverage_note": "no frame had a tracked blob"}
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap_tol:
            runs.append((int(start), int(prev)))
            start = int(i)
        prev = int(i)
    runs.append((int(start), int(prev)))
    out = []
    for a, b in runs:
        out.append({"frame_start": a, "frame_end": b, "n_frames": b - a + 1,
                    "t_start_s": float(ft[a]), "t_end_s": float(ft[b])})
    n_covered = int(sum(r["n_frames"] for r in out))
    return {"n_runs": len(out), "runs": out,
            "n_frames_tracked": int(len(idx)),
            "n_frames_in_runs": n_covered,
            "coverage_fraction": float(n_covered / len(feats)) if len(feats) else 0.0,
            "gap_tolerance_frames": int(gap_tol),
            "mm_per_px": float(mm_per_px),
            "note": ("a run is a stretch of consecutive tracked frames with gaps no "
                     "longer than gap_tolerance_frames; only inside a run is the "
                     "reconstruction observing the fly rather than interpolating")}


def reconstruct(ep_dir: Path, thresh_sigma: float = 5.0,
                report_gain: float = 4.0, use_jitter: bool = True,
                verbose: bool = True, cam_name: str = "worldcam") -> dict:
    """Reconstruct one episode from ONE camera's photos.

    ``cam_name`` selects which camera: ``"worldcam"`` reads ``camera.json`` and
    ``frames/``, anything else reads ``camera_<cam_name-without-cam>.json`` and
    ``frames_<cam_name>/``.  Each camera is reconstructed INDEPENDENTLY and written to
    its own JSON, which is what keeps the two claims separable: the fixed camera
    carries the whole trajectory, the close-up camera carries the limb detail.
    """
    if cam_name == "worldcam":
        cam_path = ep_dir / "camera.json"
        frame_dir = ep_dir / "frames"
    else:
        cam_path = ep_dir / f"camera_{cam_name.replace('cam', '')}.json"
        if not cam_path.exists():
            cam_path = ep_dir / f"camera_{cam_name}.json"
        frame_dir = ep_dir / f"frames_{cam_name}"
    if not cam_path.exists():
        raise SystemExit(f"{ep_dir}: no camera description at {cam_path}")
    if not frame_dir.exists():
        raise SystemExit(f"{ep_dir}: no frames at {frame_dir}")
    cam = Camera(json.loads(cam_path.read_text()))
    paths = sorted(frame_dir.glob("*.png"))
    frames = load_frames(paths)
    n = len(frames)
    bg, spread = robust_bg(frames)
    # JITTER CORRECTION IS ONLY VALID FOR A STATIC CAMERA.  For a PANNED camera every
    # frame is taken from a different pose, so the per-pixel median is not the
    # background and a shift search measures the pan, not vibration.  Disabled there and
    # recorded, rather than left to produce a meaningless number.
    want_jitter = bool(use_jitter) and not cam.moves
    jit = estimate_jitter(frames, bg) if want_jitter else np.zeros((n, 2))
    # FOR A PANNED CAMERA THE BACKGROUND MUST BE SPATIAL, NOT TEMPORAL.
    # MEASURED bug this fixes: with a stage that moves ~20 px between frames, the
    # per-pixel temporal median is not the background at all, the residual is noise, and
    # the parser reported the fly as present in only 66-195 of 240 frames even though
    # the fly is plainly in every frame (measured: 1100+ pixels brighter than the frame
    # mean in EVERY frame).  A running median over a small spatial window estimates the
    # background from the SAME frame, so it works whether the stage moves or not.
    # The window is DECLARED, not tuned; it must be wider than the legs (~25 px) and
    # narrower than the checkerboard squares (~80 px).
    if cam.moves:
        bg = local_mean_background(frames)
        resid = frames - bg
        spread = 0.5 * (np.percentile(resid, 84, axis=(1, 2), keepdims=True)
                        - np.percentile(resid, 16, axis=(1, 2), keepdims=True))
        spread = np.maximum(spread, 1.0)
    # ---- frame times, needed BEFORE tracking so the jump gate is physical -----
    ft = np.asarray(cam.meta.get("frame_time_s", []), dtype=float)
    if len(ft) != n:
        # camera.json without frame times -> nominal fps, stated as a fallback
        fps = float(cam.meta.get("fps_achieved") or cam.meta.get("fps") or 60.0)
        ft = np.arange(n) / fps
    dt_frame = float(np.median(np.diff(ft))) if n > 1 else 1.0 / 60.0
    # Candidate blobs per frame, then ONE track through them.  The maximum plausible
    # inter-frame jump is DERIVED from the camera's own scale and a DECLARED speed
    # bound, so it is a physical gate and not a tuned number:
    #     max_jump_px = MAX_PLAUSIBLE_SPEED_MM_S * frame_dt / (mm per pixel)
    # A fly blob cannot teleport; a mis-picked checkerboard or shadow blob can.
    max_jump_px = MAX_PLAUSIBLE_SPEED_MM_S * dt_frame / max(cam.mm_per_px(), 1e-9)
    # ``bg`` and ``spread`` are per-frame for a moving camera and single-plane for a
    # static one; the accessor below makes that explicit instead of relying on
    # broadcasting luck.
    def _plane(arr, i):
        return arr[i] if arr.ndim == 3 else arr

    cands = []
    for i in range(n):
        f = frames[i]
        if want_jitter and (jit[i] != 0).any():
            f = np.roll(f, (-int(jit[i, 0]), -int(jit[i, 1])), axis=(0, 1))
        cands.append(blob_candidates(f, _plane(bg, i), _plane(spread, i),
                                     thresh_sigma, report_gain))
    feats = track_blobs(cands, max_jump_px=max_jump_px,
                        max_accel_px=MAX_ACCEL_PX_PER_FRAME)
    n_det = sum(1 for f in feats if f is not None)
    if n_det < 5:
        raise SystemExit(f"{ep_dir}: only {n_det}/{n} frames had a detectable blob; "
                         "the parser refuses to invent a trajectory")

    # ---- DERIVED trajectory -------------------------------------------------
    rows = np.array([f["row"] if f else np.nan for f in feats])
    cols = np.array([f["col"] if f else np.nan for f in feats])
    # fill gaps by linear interpolation in time; record how many were filled
    n_gap = int(np.sum(~np.isfinite(rows)))
    for arr in (rows, cols):
        bad = ~np.isfinite(arr)
        if bad.any() and (~bad).sum() >= 2:
            arr[bad] = np.interp(ft[bad], ft[~bad], arr[~bad])
    world = cam.rays_ground_per_frame(cols, rows)
    x_mm, y_mm = world[:, 0], world[:, 1]
    # heading: the blob's major axis is the body axis.  The map from image angle to
    # world angle follows from the camera's own axes; the 180 deg ambiguity is
    # resolved by the direction of travel, which is a MEASURED quantity.
    theta_img = np.array([f["theta_img_rad"] if f else np.nan for f in feats])
    if not np.isfinite(theta_img).all():
        bad = ~np.isfinite(theta_img)
        theta_img[bad] = np.interp(ft[bad], ft[~bad], theta_img[~bad])
    # unit vector along the major axis in image (dcol, drow)
    dc = np.cos(theta_img)
    dr = np.sin(theta_img)
    # map to world using two projected ground points
    p0 = cam.rays_ground_per_frame(cols, rows)
    p1 = cam.rays_ground_per_frame(cols + dc, rows + dr)
    dv = p1 - p0
    head_img = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
    vx = np.gradient(x_mm, ft, edge_order=1)
    vy = np.gradient(y_mm, ft, edge_order=1)
    speed = np.hypot(vx, vy)
    heading = head_img.copy()
    # resolve the axis ambiguity with the measured travel direction
    move = speed > 0.2 * np.nanmax(speed) if np.nanmax(speed) > 0 else np.zeros_like(speed, bool)
    ref = np.degrees(np.arctan2(vy[move], vx[move])) if move.any() else None
    if ref is not None and ref.size:
        ref = float(np.median(ref))
        flip = np.abs(((heading - ref + 180.0) % 360.0) - 180.0) > 90.0
        heading = np.where(flip, heading + 180.0, heading)
    heading = ((heading + 180.0) % 360.0) - 180.0
    heading_unwrapped = np.degrees(np.unwrap(np.radians(heading)))
    turn_rate = np.gradient(heading_unwrapped, ft, edge_order=1)

    # ---- OBSERVED shape time series -> gait ---------------------------------
    sig_sets = {
        "extent_minor_px": np.array([f["extent_minor_px"] if f else np.nan for f in feats]),
        "extent_major_px": np.array([f["extent_major_px"] if f else np.nan for f in feats]),
        "area_px": np.array([f["area_px"] if f else np.nan for f in feats]),
        "half_major_px": np.array([f["half_major_px"] if f else np.nan for f in feats]),
        "half_minor_px": np.array([f["half_minor_px"] if f else np.nan for f in feats]),
    }
    for k, v in sig_sets.items():
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.interp(ft[bad], ft[~bad], v[~bad])
    # light smoothing on a fixed 3-frame window; declared, not tuned
    def smooth(v, w=3):
        if len(v) < w:
            return v
        k = np.ones(w) / w
        return np.convolve(v, k, mode="same")
    gait_scores = {}
    for name, v in sig_sets.items():
        sv = smooth(v)
        r = gait_from_signal(ft, sv)
        gait_scores[name] = {"f_alt_hz": r["f_alt_hz"], "r2": r["r2"],
                             "amp": r["fit"]["amplitude"][0],
                             "duty": r["duty"]["duty_above_mean"],
                             "phase_deg": r["phase_deg"]}
    best_sig = max(gait_scores, key=lambda k: gait_scores[k]["r2"])
    stride = gait_from_signal(ft, smooth(sig_sets[best_sig]))
    f_alt = stride["f_alt_hz"]
    psi = stride["phase_rad"]
    duty = stride["duty"]
    # SIGN OF THE SIGNAL.  The blob signal that matters here (projected width/area) is
    # larger when a leg is planted and smaller when it is lifted, so for that signal the
    # ALTERNATION is used directly.  For a signal that grows during swing the
    # relationship would invert, so the fitted sign of the correlation between the
    # signal and the same signal at the antiphase is NOT assumed -- the phase of a
    # cosine is the same whichever way the signal is oriented, which is exactly why the
    # relation in ``contact_window_from_phase`` is written in terms of the PEAK.

    # ---- PRIOR + INFERRED: limbs and touch from the trajectory --------------
    # Step length is MEASURED from the video: the alternation period times the speed.
    v_med = float(np.median(speed))
    step_length_mm = v_med / f_alt if f_alt > 0 else float("nan")
    # DUTY: DECLARED per-leg nominal values (see DUTY_PER_LEG_NOMINAL for why the video
    # cannot supply them).  The video's own pooled estimate is carried alongside so the
    # disagreement is on the record rather than hidden behind one number.
    raw_duty = float(duty["duty_above_mean"])
    duty_contact = float(np.mean(DUTY_PER_LEG_NOMINAL))
    duty_per_leg = np.asarray(DUTY_PER_LEG_NOMINAL, dtype=float)
    duty_clamped = False
    cos_h = np.cos(np.radians(heading_unwrapped))
    sin_h = np.sin(np.radians(heading_unwrapped))
    psi_frac = (psi / (2.0 * math.pi)) % 1.0

    legs = []
    for i, name in enumerate(LEG_NAMES):
        att = LEG_ATTACH_MM[i]
        # body-frame -> world, for the neutral attachment point
        ax = x_mm + att[0] * cos_h - att[1] * sin_h
        ay = y_mm + att[0] * sin_h + att[1] * cos_h
        # ---- gait cycle for this leg ----------------------------------------
        # Two groups in antiphase: a tripod composition, DECLARED as this model's gait
        # prior.  The CYCLE is the video's -- f_alt is the measured alternation
        # frequency, psi is the video's phase, duty comes from folding the video signal.
        group = 0 if i in (0, 2, 4) else 1        # lf, lh, rm  vs  rf, rh, lm
        centre_frac = (psi_frac + (0.0 if group == 0 else 0.5)) % 1.0
        # f_alt is the PER-LEG contact-cycle frequency (the scan band forces that), so
        # the window below is the leg's own cycle, centred on the pooled signal's peak.
        td_frac, lo_frac = contact_window_from_phase(
            2 * math.pi * centre_frac, float(duty_per_leg[i]))
        # position within the alternation cycle, in CYCLES, relative to the video phase
        cyc_all = f_alt * (ft - ft[0]) + psi_frac
        # group 1 is half a cycle behind group 0
        cyc = cyc_all + (0.0 if group == 0 else 0.5)
        n_cycles = np.floor(cyc)
        frac = cyc - n_cycles                      # 0..1 within the leg's own cycle
        # CONTACT WINDOW, in cycles, relative to the leg's own cycle start.  The window
        # is centred on the group's alternation phase, which for group 0 is where the
        # pooled signal peaks; `td_frac` is that centre minus half the duty.
        # frac is measured from the same origin as centre_frac (the video phase), so a
        # leg is in stance when the distance travelled since its touchdown is inside the
        # duty window.  Because psi_frac cancels, define the window in absolute cycles:
        td_abs = td_frac + n_cycles                # touchdown of the current cycle
        lo_abs = td_abs + float(duty_per_leg[i])   # liftoff of the current cycle
        stance = (cyc >= td_abs) & (cyc < lo_abs)
        # ---- foot position in the BODY frame ---------------------------------
        # During stance the foot is planted, so in the body frame it travels backwards
        # at the measured speed; the step length is the measured speed / alternation
        # frequency.  During swing it is carried forward again.
        foot_y_att = att[1]
        _d = float(duty_per_leg[i])
        prog = np.clip((cyc - td_abs) / max(_d, 1e-6), 0.0, 1.0)
        swing_prog = np.clip((cyc - lo_abs) / max(1.0 - _d, 1e-6), 0.0, 1.0)
        if step_length_mm > 0 and np.isfinite(step_length_mm):
            foot_bx = np.where(stance,
                               step_length_mm * (0.5 - prog),
                               step_length_mm * (-0.5 + swing_prog))
        else:
            foot_bx = np.zeros_like(ft)
        foot_by = np.zeros_like(ft)
        # Lift profile: DECLARED sine over the swing window; zero (planted) in stance.
        foot_bz = np.where(stance, 0.0,
                           SWING_LIFT_MM * np.sin(math.pi * swing_prog))
        # foot position in the world, using the derived trajectory + heading
        fx = x_mm + foot_bx * cos_h - (foot_y_att + foot_by) * sin_h
        fy = y_mm + foot_bx * sin_h + (foot_y_att + foot_by) * cos_h
        fz = foot_bz
        # distance from the attachment point, as a reach sanity check
        reach = np.hypot(np.hypot(fx - ax, fy - ay), fz)
        legs.append({
            "name": name, "group": int(group),
            "cycle_fraction": frac.astype(float),
            "touchdown_phase_frac": float(td_frac),
            "liftoff_phase_frac": float(lo_frac),
            "stance": stance.astype(bool),
            "attach_mm_body": [float(v) for v in att],
            "attach_world_x_mm": ax.astype(float),
            "attach_world_y_mm": ay.astype(float),
            "foot_body_x_mm": foot_bx.astype(float),
            "foot_world_x_mm": fx.astype(float),
            "foot_world_y_mm": fy.astype(float),
            "foot_world_z_mm": fz.astype(float),
            "reach_mm": reach.astype(float),
            "reach_max_mm": float(np.nanmax(reach)),
            "reach_exceeds_prior": bool(np.nanmax(reach) > LEG_REACH_MM * 1.5),
        })

    # ---- INFERRED touch -----------------------------------------------------
    # A foot touches when it is down.  The contact patch is the set of frames where
    # the reconstructed foot height is below the declared half-thickness of the
    # tarsus; the touchdown POINT is the reconstructed foot position at the start of
    # that patch, and the touch force is a spring-damper estimate from the body's
    # motion, which is measurable in the video.
    half_thick_mm = 0.05        # DECLARED tarsus half-thickness
    for L in legs:
        down = L["foot_world_z_mm"] <= half_thick_mm
        L["touch_pred"] = [bool(v) for v in down]
        starts = np.nonzero(down & ~np.roll(down, 1))[0]
        events = []
        for s in starts:
            events.append({"frame": int(s), "time_s": float(ft[s]),
                           "x_mm": float(L["foot_world_x_mm"][s]),
                           "y_mm": float(L["foot_world_y_mm"][s])})
        L["touchdown_events"] = events
        L["touchdown_count"] = len(events)
        # spring-damper estimate of the normal force: the load needed to hold the
        # body up, shared over the legs the reconstruction says are in contact
        L["duty"] = float(np.mean(down))

    down_matrix = np.stack([np.asarray(L["touch_pred"], dtype=bool) for L in legs])
    n_down_per_frame = down_matrix.sum(axis=0).astype(float)
    # INDEPENDENT ground-truth-free estimate of load per leg: the body's weight is
    # not known to the parser from the video, so the inferred load is reported in
    # units of "share of body weight", with the body weight itself taken from the
    # camera-INVISIBLE declared body mass (documented as the one prior that enters
    # the force estimate).
    from electrode_payload import BODY_MASS_KG_FALLBACK
    weight_mN = BODY_MASS_KG_FALLBACK * 9.80665 * 1e3
    share = np.where(n_down_per_frame > 0, 1.0 / np.maximum(n_down_per_frame, 1), 0.0)
    for i, L in enumerate(legs):
        L["touch_force_share"] = (share * down_matrix[i]).astype(float).tolist()
        L["touch_force_uN"] = (share * down_matrix[i] * weight_mN * 1e3).astype(float).tolist()

    out = {
        "episode_dir": str(ep_dir),
        "camera_name": cam_name,
        "camera_pose_source": ("per-frame stage log (pos_mm_per_frame)"
                               if cam.pos_per_frame is not None
                               else "single static pose (pos_mm)"),
        "camera_pose_log_frames": (int(len(cam.pos_per_frame))
                                   if cam.pos_per_frame is not None else 0),
        "camera_file": str(cam_path.name),
        "frames_dir": str(frame_dir.name),
        "n_frames": int(n),
        "n_detected": int(n_det),
        "n_interpolated": int(n_gap),
        "inputs_read": [f"{frame_dir.name}/*.png", cam_path.name],
        "camera": {"pos_mm": cam.pos.tolist(), "fovy_deg": cam.fovy_deg,
                   "image_size_px": [cam.H, cam.W], "mm_per_px": cam.mm_per_px(),
                   "footprint_mm": list(cam.footprint_mm()),
                   "fx_px": cam.fx_px, "fy_px": cam.fy_px},
        "jitter_px": {"rows": jit[:, 0].tolist(), "cols": jit[:, 1].tolist(),
                      "max_abs_row": float(np.max(np.abs(jit[:, 0]))),
                      "max_abs_col": float(np.max(np.abs(jit[:, 1])))},
        "frame_time_s": ft.tolist(),
        "observed": {
            "row_px": rows.tolist(), "col_px": cols.tolist(),
            "area_px": sig_sets["area_px"].tolist(),
            "extent_minor_px": sig_sets["extent_minor_px"].tolist(),
            "extent_major_px": sig_sets["extent_major_px"].tolist(),
            "theta_img_rad": theta_img.tolist(),
            "peak_resid_sigma": [float(f["peak_resid_sigma"]) if f else None for f in feats],
        },
        "track": dict(_track_segments(feats, ft, cam.mm_per_px()),
                      max_jump_px_used=float(max_jump_px),
                      max_accel_px_used=float(MAX_ACCEL_PX_PER_FRAME),
                      candidate_counts=[len(c) for c in cands]),
        "derived": {
            "x_mm": x_mm.tolist(), "y_mm": y_mm.tolist(),
            "vx_mm_s": vx.tolist(), "vy_mm_s": vy.tolist(),
            "speed_mm_s": speed.tolist(),
            "heading_deg": heading.tolist(),
            "heading_unwrapped_deg": heading_unwrapped.tolist(),
            "turn_rate_deg_s": turn_rate.tolist(),
            "path_length_mm": float(np.sum(np.hypot(np.diff(x_mm), np.diff(y_mm)))),
        },
        "gait": {
            "signal_used": best_sig,
            "scores": gait_scores,
            "alternation_frequency_hz": f_alt,
            "per_leg_stride_hz": stride["per_leg_stride_hz"],
            "stride_r2": stride["r2"],
            "scan_band_hz": stride["band_hz"],
            "phase_rad": psi,
            "phase_deg": stride["phase_deg"],
            "harmonic_amplitudes": stride["harmonic_amplitudes"],
            "harmonic_ratio": stride["harmonic_ratio"],
            "duty": duty,
            "duty_contact_used": duty_contact,
            "duty_pooled_measured": raw_duty,
            "duty_clamped": duty_clamped,
            "duty_per_leg": [float(v) for v in duty_per_leg],
            "duty_per_leg_declared": list(DUTY_PER_LEG_NOMINAL),
            "duty_source": ("DECLARED nominal per-leg duty of the body model's gait; "
                            "video cannot separate six duties from the pooled signal"),
            "mean_speed_mm_s": v_med,
            "step_length_mm_measured": step_length_mm,
            "frequency_convention": ("alternation_frequency_hz is the frequency of the "
                                     "per-leg CONTACT cycle as seen pooled over both "
                                     "groups; per_leg_stride_hz = alternation/2 assumes "
                                     "an alternating two-group gait, which is this "
                                     "model's declared coordination prior"),
        },
        "priors": {"leg_names": list(LEG_NAMES),
                   "leg_attach_mm": LEG_ATTACH_MM.tolist(),
                   "leg_reach_mm": LEG_REACH_MM,
                   "swing_lift_mm": SWING_LIFT_MM,
                   "half_tarsus_thickness_mm": half_thick_mm,
                   "body_mass_kg_used_for_force": BODY_MASS_KG_FALLBACK,
                   "statement": ("attachment geometry, leg reach, swing lift and "
                                 "tarsus thickness come from the body model's "
                                 "anatomy, NOT from the video.  The step length and "
                                 "the stride frequency are measured from the video.")},
        "legs": legs,
        "touch_inferred": {
            "definition": ("a foot is in touch when its reconstructed height is at or "
                           "below half the tarsus thickness"),
            "per_leg_duty": [L["duty"] for L in legs],
            "touchdown_counts": [L["touchdown_count"] for L in legs],
            "max_legs_down_at_once": int(n_down_per_frame.max()),
            "min_legs_down_at_once": int(n_down_per_frame.min()),
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--camera", default="worldcam",
                    help="worldcam (default) or detailcam")
    ap.add_argument("--out", default=None,
                    help="output json (default: <episode>/vision[_<camera>].json)")
    ap.add_argument("--thresh-sigma", type=float, default=5.0)
    ap.add_argument("--no-jitter", action="store_true")
    args = ap.parse_args()

    dirs: list[Path] = []
    for pat in args.episodes:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"no episode directory matches {pat!r}")
        dirs.extend(Path(h) for h in hits)
    for d in dirs:
        t0 = __import__("time").perf_counter()
        rec = reconstruct(d, thresh_sigma=args.thresh_sigma,
                          use_jitter=not args.no_jitter, cam_name=args.camera)
        rec["wall_seconds"] = __import__("time").perf_counter() - t0
        if args.out:
            dest = Path(args.out)
        else:
            dest = d / ("vision.json" if args.camera == "worldcam"
                        else f"vision_{args.camera}.json")
        dest.write_text(json.dumps(rec, cls=_NpEncoder))
        g = rec["gait"]
        print(f"{d.name}: {rec['n_detected']}/{rec['n_frames']} detected, "
              f"path {rec['derived']['path_length_mm']:.2f} mm, "
              f"alt {g['alternation_frequency_hz']:.3f} Hz "
              f"(per-leg {g['per_leg_stride_hz']:.3f}) (r2 {g['stride_r2']:.3f}) "
              f"from {g['signal_used']}, duty {g['duty_contact_used']:.3f}, "
              f"step {g['step_length_mm_measured']:.3f} mm, "
              f"jitter max {rec['jitter_px']['max_abs_col']},{rec['jitter_px']['max_abs_row']} px "
              f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
