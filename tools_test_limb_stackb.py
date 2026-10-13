#!/usr/bin/env python3
"""CAN STACK B'S OWN LIMBS BE MEASURED?  Acceptance for the fly's markers on the muscle-driven body.

The question this answers is the one the camera ring exists for: from the recordings alone, what is
one joint's angle at one instant?  Before this round stack B's episodes carried only the twelve ARENA
fiducials, so `limb_tracking.py` -- which reads `truth/marker_fly_world_mm` -- could not run on them
at all, and the delivery note had to say limb kinematics were not established.

Each check can FAIL, and the injections are the point: an assertion that cannot fail is decoration.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_limb_stackb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

EPISODE = HERE / "outputs" / "arena" / "stackb_all_channels"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:                                                     # noqa: C901
    import mujoco
    import tools_make_stackb_observable as T

    print("1. the fly's own markers are ON stack B's body, and change nothing mechanical")
    rep: dict = {}
    spec, _ = T.build_spec(fly_markers=True, marker_report=rep)
    m_with = spec.compile()
    mass_with = float(sum(m_with.body_mass))
    m_without = T.build_spec(fly_markers=False)[0].compile()
    mass_without = float(sum(m_without.body_mass))
    names = {str(mujoco.mj_id2name(m_with, mujoco.mjtObj.mjOBJ_GEOM, i)) for i in range(m_with.ngeom)}
    present = [g for g in rep["geom_name_by_segment"].values() if g in names]
    check("B1a: all 20 markers are in the COMPILED model", len(present) == 20,
          f"{len(present)} of 20")
    # INJECTION: if the markers were not massless the compiled mass would move.  This is the check
    # that caught the stack A version of this bug, where 20 spheres carried ~9 kg on a 1e-6 kg fly.
    check("B1b: the markers add NO mass (injection: mass must be identical)", 
          abs(mass_with - mass_without) < 1e-12,
          f"with {mass_with:.7f} vs without {mass_without:.7f}")
    gid = {str(mujoco.mj_id2name(m_with, mujoco.mjtObj.mjOBJ_GEOM, i)): i
           for i in range(m_with.ngeom)}
    radii = [float(m_with.geom_size[gid[g]][0]) for g in present]
    from arena import FLY_MARKER_RADIUS_MM
    check("B1c: the radius is in MODEL UNITS (mm), not metres",
          abs(max(radii) - float(FLY_MARKER_RADIUS_MM)) < 1e-9,
          f"{max(radii)} mm requested {FLY_MARKER_RADIUS_MM}")

    print("\n2. the recorded truth is the SAME geoms the camera saw")
    if not (EPISODE / "episode.npz").is_file():
        check("B0: the episode exists", False, f"run tools_record_all_channels.py; {EPISODE}")
        return _finish({})
    z = np.load(EPISODE / "episode.npz")
    mj = json.loads((EPISODE / "markers.json").read_text())
    segs = list(mj["marker_body_segments"])
    truth = np.asarray(z["truth/marker_fly_world_mm"], dtype=float)
    check("B2a: the truth array is (frames, markers, 3)",
          truth.ndim == 3 and truth.shape[1] == len(segs) == 20,
          f"{truth.shape} for {len(segs)} segments")
    geom_rec = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    check("B2b: the column order matches arena_geometry's declared segment list",
          segs == list(geom_rec["marker_body_segments"]),
          "limb_tracking indexes the truth columns by that list, so a mismatch would silently "
          "score the wrong bone")
    # INJECTION: swap two columns and require the live cross-check against geom_xpos to notice.
    order = rep["geom_name_by_segment"]
    d = mujoco.MjData(m_with)
    mujoco.mj_resetDataKeyframe(m_with, d, 0)
    mujoco.mj_forward(m_with, d)
    live = T.fly_marker_truth(m_with, d, order, segs)
    swapped = list(segs)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    try:
        T.fly_marker_truth(m_with, d, order, swapped)
        swapped_ok = True
    except Exception:                                                  # noqa: BLE001
        swapped_ok = False
    check("B2c: a swapped column order is REFUSED, not scored (injection)",
          swapped_ok is False or not np.allclose(
              T.fly_marker_truth(m_with, d, order, swapped), live),
          "swapping two segments must not produce a silently reordered truth array")
    check("B2d: markers are distinct, not all at one point",
          float(np.median(np.linalg.norm(np.diff(live, axis=0), axis=1))) > 0.05,
          f"median spacing {np.median(np.linalg.norm(np.diff(live, axis=0), axis=1)):.3f} mm")

    print("\n3. the two marker classes can be told apart, because the AREA is recorded")
    check("B3a: the recorder now stores each blob's pixel area",
          all(f"px/cam{i}/area" in z.files for i in range(6)), "px/<cam>/area present")
    areas = np.concatenate([np.asarray(z[f"px/cam{i}/area"], dtype=float) for i in range(6)])
    check("B3b: the areas are bimodal, i.e. two marker classes really are present",
          float((areas < 40).mean()) > 0.2 and float((areas > 60).mean()) > 0.2,
          f"{float((areas < 40).mean())*100:.0f} % small, {float((areas > 60).mean())*100:.0f} % large")

    print("\n4. the projection offsets are NOT applied to the fly -- MEASURED, and it decides everything")
    import limb_tracking as LT
    import run_arena_reconstruct as R
    geom = geom_rec
    res = {}
    for tag, use_off in (("off", False), ("on", True)):
        if use_off:
            LT.load_projection_offsets(EPISODE / "camera_offsets.json")
        else:
            R.PROJECTION_OFFSET_PX = {}
        cams, _ = LT.load_cameras_for_episode(geom, EPISODE, LT.ARENA / "camera_poses.json")
        if not use_off:
            for c in cams:
                c.offset_px = (0.0, 0.0)
        f = 0
        bl = {}
        for c in cams:
            a = int(z[f"px/{c.name}/offsets"][f])
            b = int(z[f"px/{c.name}/offsets"][f + 1])
            bl[c.name] = np.stack([z[f"px/{c.name}/rows"][a:b], z[f"px/{c.name}/cols"][a:b],
                                   np.zeros(b - a)], axis=1)
        pts = LT.screen_points(cams, bl, seeds_mm=truth[0], min_views=3, reproj_gate_px=2.0,
                               gate_px=3.5)
        res[tag] = len(pts)
    print(f"   fly markers triangulated from their own truth seeds: offsets off {res['off']}/20, "
          f"offsets on {res['on']}/20")
    check("B4: WITH the fiducial-fitted offsets the fly markers collapse -- so they stay off "
          "(injection)",
          res["off"] >= 15 and res["on"] <= res["off"] - 10,
          f"off {res['off']}/20 vs on {res['on']}/20")
    lt = json.loads((EPISODE / "limb_tracking.json").read_text())
    check("B4b: the tracker's own output declares the offsets were NOT applied",
          lt["projection_offsets"]["applied"] is False, str(lt["projection_offsets"]["applied"]))

    print("\n5. and the point of all of it: A JOINT ANGLE COMES OUT")
    check("B5a: the tracker ran and recorded its size filter",
          "size_filter" in lt, str(lt["size_filter"].get("status")))
    acc = lt.get("markers_accepted", 0)
    check("B5b: fly markers are ACCEPTED, where the earlier chain accepted none", acc > 0,
          f"{acc} of {lt.get('marker_slots')} frame-marker slots")
    e = (lt.get("marker_position_error_mm") or {}).get("median")
    check("B5c: the marker positions are millimetre-accurate", e is not None and e < 0.2,
          f"median {e} mm")
    bones = lt.get("bones_with_both_ends") or {r["bone"]: r["n"] for r in lt.get("joint_angles", [])}
    good = {k: v for k, v in bones.items() if v >= 10}
    check("B5d: at least one bone is measured in >=10 frames", bool(good),
          f"{sorted(good.items())} of {len(bones)} bones")
    ang = lt.get("joint_angle_series") or {}
    check("B5e: the joint-angle error is finite and declared",
          ang.get("error_deg_median") is not None and np.isfinite(ang["error_deg_median"]),
          f"median {ang.get('error_deg_median')} deg over {ang.get('n_angles')} angles, "
          f"p95 {ang.get('error_deg_p95')} deg")
    csv = EPISODE / "limb_joint_angles.csv"
    if csv.is_file():
        rows = [r for r in csv.read_text().splitlines() if r and not r.startswith("t_s")]
        ok = bool(rows)
    else:
        ok = False
    check("B5f: the per-frame joint angles are written to a file a reader can open", ok,
          f"{csv.name}")

    return _finish({"fly_markers_triangulated": res, "markers_accepted": acc,
                    "joint_angle_median_deg": ang.get("error_deg_median"),
                    "bones_measured_10plus_frames": sorted(good)})


def _finish(extra: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS], **extra,
           "units": "mm for lengths, px for residuals and areas, degrees for angles",
           "provenance": ("the markers, the recorded truth and the tracker all come from "
                          "tools_make_stackb_observable.attach_fly_markers / fly_marker_truth and "
                          "limb_tracking.py on outputs/arena/stackb_all_channels"),
           "what_this_does_not_establish": (
               "that the fly is walking, or that all six legs are measured.  119 of 400 "
               "frame-marker slots are accepted and only ONE bone clears 10 frames, because a "
               "six-camera ring around a self-occluding body leaves most markers visible from one "
               "or two cameras per frame.  The angles are also scored against the SIMULATOR's own "
               "marker positions: there is no real-fly recording anywhere in this project, and "
               "frame 0's identity seed is a stand-in for a human annotation.  The markers "
               "themselves sit on the FEMUR of all six legs, because stack B has no trochanter on "
               "the mid and hind legs, so 'femur' here is marker-to-marker, not joint-centre-to-"
               "joint-centre.")}
    dest = HERE / "outputs" / "limb_stackb_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
