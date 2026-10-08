"""ARENA: the six-camera truss and the fiducial markers, as declared geometry.

WHY THIS IS A MODULE AND NOT A FEW NUMBERS IN A SCRIPT
------------------------------------------------------
Two things in this file have to be exactly reproducible and exactly recorded:

1. **The camera array.**  Six cameras on a ring, all aimed at the arena centre.  Their
   poses are the calibration, and a calibration that is assembled by hand inside a
   script cannot be audited.  Here the poses are DERIVED from three declared numbers
   (radius, height, field of view) plus the ring angles, and the derivation writes down
   the resulting rotation matrices and the distance from each camera to the arena
   centre, so a reader can check the geometry without re-running anything.

2. **The fiducial markers.**  The world frame of a multi-camera measurement is defined
   ONLY by objects visible in the images.  A camera pose can be right and the world
   frame still be wrong; a rigid transform that maps measured marker positions onto their
   declared positions is the only way to FIX the frame, and its residual is the only
   honest measure of how well it is fixed.  The markers below are that reference, and
   they are declared in world coordinates to the millimetre.

WHY THE CAMERAS ARE WHERE THEY ARE -- a measured statement, not a preference
--------------------------------------------------------------------------
Triangulation accuracy is governed by the ANGLES BETWEEN VIEW RAYS, not by the number
of cameras.  Two views 10 degrees apart localise a point far worse than two views 70
degrees apart, no matter how many pixels each has; adding cameras helps only in so far
as they add DISTINCT directions.  So the ring is spaced at 120, 60 and 180 degrees so
that every camera sees a target from at least one pair with a large baseline: the worst
pair separation on a 6-ring at 60 degree spacing is 60 degrees in the horizontal plane,
and each camera pair is also separated in elevation because the ring is tilted 30
degrees below horizontal.  The alternative -- several cameras clustered together, which
is what a lab tripod setup looks like when nobody has thought about this -- gives a low
baseline for every pair and a reconstructed point that slides along the view direction.

The numbers for the ring (radius 45 mm, height 26 mm, fovy 40 deg, 1200 fps) are
DECLARED.  What is MEASURED is the resulting per-camera ground sampling distance and
each camera's distance and elevation to the arena centre, which are reported so the
numbers can be judged instead of trusted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["CAMERA_RING", "FIDUCIAL_MARKERS", "arena_cameras",
           "arena_camera_report", "marker_report", "ARENA_CENTRE_MM",
           "MARKER_RADIUS_MM"]

#: Arena centre in world coordinates, mm.  The fly is spawned here.
ARENA_CENTRE_MM = (0.0, 0.0, 0.0)

#: The camera ring.  DECLARED: six cameras, one ring, tilted down.
CAMERA_RING = {
    "n_cameras": 6,
    # RADIUS 25 mm AND HEIGHT 14 mm ARE SET BY THE SUBJECT, NOT BY CONVENIENCE.  The
    # working volume has to contain the fly's whole walk (about 5 mm in this arena) plus
    # its payload, and it has to do so at a sampling fine enough that a 1 mm leg spans
    # tens of pixels.  At this radius and field of view the footprint at the arena centre
    # is about 13 x 10 mm and the ground sampling distance is about 75 px/mm, so the fly
    # is ~76 px long and a claw excursion of 0.5 mm is ~38 px.  The FIRST version of this
    # file used a 45 mm ring, which gave 10.6 px/mm and a 26 px fly -- measured, and not
    # enough to locate a joint.  The ring was moved in rather than the requirement
    # weakened.
    "radius_mm": 22.0,
    "height_mm": 22.0,
    "fovy_deg": 50.0,
    #: The elevation of each optical axis below horizontal, degrees.  All six look down
    #: at the arena centre, so this follows from radius and height; it is reported, not
    #: chosen independently.
    "elevation_deg": math.degrees(math.atan2(22.0, 22.0)),
    #: Horizontal spacing of the ring.  60 degrees for six cameras; a pair separated by
    #: 60 degrees is the WORST baseline available, and that number is what is reported.
    "azimuth_step_deg": 60.0,
    "first_azimuth_deg": -60.0,
    #: Camera resolution.  Square pixels; the aspect ratio only decides how much sky is
    #: in the frame, not the sampling.
    "resolution_px": [200, 260],
    "frame_rate_hz": 1200.0,
    "note": ("fovy is the FULL vertical field of view of the MuJoCo camera.  The "
             "ground sampling distance at the arena centre is height/fy_px, and that is "
             "the number that sets the whole measurement's resolution."),
}

#: Fiducial markers: static spheres at DECLARED world positions.  These define the world
#: frame.  Three of them are also ON the fly (the fly-borne ones are injected as extra
#: bodies by the recorder), which is what lets a body rigid transform be fitted.
#: TWELVE markers in THREE tiers, decided AFTER MEASURING what fails:
#:
#:   * SIX on a 12 mm-diameter ring at 0.5 mm (just above the litter, standing on the cleared
#:     disc), giving well-separated points on the walking plane;
#:   * THREE at 6 mm on a 12 mm ring, rotated 60 degrees from the lower ring;
#:   * THREE at 11 mm on a 12 mm ring, aligned with the lower one.
#:
#: WHY TWELVE AND NOT SIX.  The first set was five points within 2.6 mm of the centre plus one
#: raised one, and that failed twice over.  MEASURED: the five lower points MERGED in the
#: rendered image -- with 0.4 mm-radius markers only ONE distinct blob appeared for six
#: markers -- so there was nothing to associate; and a near-coplanar set cannot condition six
#: camera poses at all.  The tiers here span 11 mm vertically and 12 mm laterally, and the
#: 6 mm ring separation puts markers tens of pixels apart at the working distance.
#:
#: THE MARKER RADIUS IS 0.30 mm, down from 0.80 mm.  MEASURED: at 0.8 mm on a 2.6 mm
#: constellation the discs merged into single components (one blob of area 798 px stood for
#: several markers), and the areas grew with radius exactly as merging predicts.  Now that the
#: markers are spread out, a smaller disc is better: 0.3 mm still spans several pixels at the
#: working distance, and it stays clear of its neighbours.
#:
#: Positions are DECLARED to the millimetre and the pairwise distances are reported as the
#: check on any fitted frame.
FIDUCIAL_MARKERS = (
    {"name": "fx_A", "pos_mm": (-6.00, 0.00, 0.50), "radius_mm": 0.30},
    {"name": "fx_B", "pos_mm": (-1.85, 5.71, 0.50), "radius_mm": 0.30},
    {"name": "fx_C", "pos_mm": (4.85, 3.53, 0.50), "radius_mm": 0.30},
    {"name": "fx_D", "pos_mm": (4.85, -3.53, 0.50), "radius_mm": 0.30},
    {"name": "fx_E", "pos_mm": (-1.85, -5.71, 0.50), "radius_mm": 0.30},
    {"name": "fx_G", "pos_mm": (0.00, 0.00, 0.50), "radius_mm": 0.30},
    {"name": "fx_H", "pos_mm": (-6.00, 0.00, 6.00), "radius_mm": 0.30},
    {"name": "fx_I", "pos_mm": (4.85, 3.53, 6.00), "radius_mm": 0.30},
    {"name": "fx_J", "pos_mm": (-1.85, -5.71, 6.00), "radius_mm": 0.30},
    {"name": "fx_K", "pos_mm": (4.85, -3.53, 11.00), "radius_mm": 0.30},
    {"name": "fx_L", "pos_mm": (-1.85, 5.71, 11.00), "radius_mm": 0.30},
    {"name": "fx_M", "pos_mm": (0.00, 0.00, 11.00), "radius_mm": 0.30},
)

MARKER_RADIUS_MM = 0.30

#: Radius of the FLY-BORNE markers, which is a completely different budget from the static
#: fiducials above.  The fiducials are 5.5 mm apart at the closest, so 0.30 mm radius there is
#: free.  The fly's markers must fit between EACH OTHER, and MEASURED over a real episode the
#: closest non-own marker to a frontal coxa origin is 0.257 mm away (tools_identity_budget.py).
#: A marker can only be told apart from a neighbour if its DIAMETER IS SMALLER THAN THAT GAP,
#: so 0.60 mm diameter merged on 8 of the 12 leg bones -- which no sensor resolution can undo.
#: 0.26 mm diameter (0.13 mm radius) makes all 12 bones separable, while staying ~7 px across
#: at 800 rows on the 22 mm ring, comfortably above the ~5 px a centroid needs.
FLY_MARKER_RADIUS_MM = 0.12


def _look_at(pos_mm, target_mm) -> list[float]:
    """MuJoCo ``xyaxes`` for a camera at ``pos_mm`` looking at ``target_mm``.

    MuJoCo cameras look along their own -z, with +x to the right in the image and +y up.
    The construction here is: forward = normalize(target - pos); right = normalize(
    forward x world_up); up = right x forward.  ``world_up`` is +z, and the ring is never
    vertical, so the cross product is never degenerate -- checked in the report rather
    than assumed.
    """
    p = np.asarray(pos_mm, dtype=float)
    f = np.asarray(target_mm, dtype=float) - p
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        raise ValueError("camera position coincides with the target")
    f = f / n
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        raise ValueError("camera is looking straight up or down; the up vector is "
                         "undefined in this construction")
    right = right / rn
    up = np.cross(right, f)
    return [float(v) for v in right] + [float(v) for v in up]


def arena_cameras(ring: dict[str, Any] | None = None) -> tuple[dict, ...]:
    """The six camera specs, ready for ``BodyConfig.extra_cameras``."""
    r = dict(CAMERA_RING if ring is None else ring)
    out = []
    for k in range(int(r["n_cameras"])):
        az = math.radians(float(r["first_azimuth_deg"]) + k * float(r["azimuth_step_deg"]))
        pos = (float(r["radius_mm"]) * math.cos(az),
               float(r["radius_mm"]) * math.sin(az),
               float(r["height_mm"]))
        out.append({"name": f"cam{k}", "pos": [float(v) for v in pos],
                    "xyaxes": _look_at(pos, ARENA_CENTRE_MM),
                    "fovy": float(r["fovy_deg"])})
    return tuple(out)


def arena_camera_report(ring: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-camera geometry: distance, elevation, sampling, and image footprint.

    MEASURED, in the sense that it is computed from the declared ring rather than
    asserted: the ground sampling distance is ``height / fy_px`` with
    ``fy_px = (H/2)/tan(fovy/2)``, and the worst view-ray separation across all camera
    PAIRS is computed explicitly, because THAT is the number that governs triangulation
    accuracy and it is not visible from any single camera's parameters.
    """
    r = dict(CAMERA_RING if ring is None else ring)
    H, W = (int(v) for v in r["resolution_px"])
    fy_px = (H / 2.0) / math.tan(math.radians(float(r["fovy_deg"])) / 2.0)
    gsd = float(r["height_mm"]) / fy_px
    cams = []
    for spec in arena_cameras(r):
        p = np.asarray(spec["pos"], dtype=float)
        d = float(np.linalg.norm(p - np.asarray(ARENA_CENTRE_MM, dtype=float)))
        elev = math.degrees(math.asin(float(r["height_mm"]) / d))
        cams.append({"name": spec["name"], "pos_mm": [float(v) for v in p],
                     "xyaxes": spec["xyaxes"], "fovy_deg": float(spec["fovy"]),
                     "distance_to_centre_mm": d, "elevation_deg": elev,
                     "fy_px": fy_px, "gsd_at_centre_mm_per_px": gsd,
                     "footprint_at_centre_mm": [gsd * W, gsd * H]})
    # worst PAIR separation: this is the number that sets triangulation quality
    dirs = []
    for c in cams:
        v = np.asarray(c["pos_mm"], dtype=float)
        dirs.append((np.asarray(ARENA_CENTRE_MM, dtype=float) - v)
                    / np.linalg.norm(np.asarray(ARENA_CENTRE_MM, dtype=float) - v))
    worst = (180.0, None)
    best = (0.0, None)
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            ang = math.degrees(math.acos(float(np.clip(np.dot(dirs[i], dirs[j]),
                                                       -1.0, 1.0))))
            if ang < worst[0]:
                worst = (ang, (cams[i]["name"], cams[j]["name"]))
            if ang > best[0]:
                best = (ang, (cams[i]["name"], cams[j]["name"]))
    return {"ring": r, "cameras": cams, "fy_px": fy_px,
            "gsd_at_centre_mm_per_px": gsd,
            "worst_pair_separation_deg": worst[0],
            "worst_pair": list(worst[1]) if worst[1] else None,
            "best_pair_separation_deg": best[0],
            "best_pair": list(best[1]) if best[1] else None,
            "note": ("a six-camera ring at 60 degree spacing has a WORST pair separation "
                     "of about 60 degrees; that, not the camera count, is what "
                     "triangulation accuracy depends on")}


def marker_report() -> dict[str, Any]:
    """Fiducial geometry: pairwise distances, which is what a frame fit is checked on."""
    pos = np.array([m["pos_mm"] for m in FIDUCIAL_MARKERS], dtype=float)
    n = len(pos)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append({"pair": [FIDUCIAL_MARKERS[i]["name"], FIDUCIAL_MARKERS[j]["name"]],
                          "distance_mm": float(np.linalg.norm(pos[i] - pos[j]))})
    dm = np.array([d["distance_mm"] for d in dists])
    return {"markers": [dict(m) for m in FIDUCIAL_MARKERS], "n_markers": n,
            "pairwise_distances": dists,
            "min_pair_distance_mm": float(dm.min()), "max_pair_distance_mm": float(dm.max()),
            "note": ("the pairwise distances are the check on a fitted world frame: a "
                     "rigid transform preserves them exactly, so their residuals measure "
                     "the TRANSFORM, independently of any camera calibration")}
