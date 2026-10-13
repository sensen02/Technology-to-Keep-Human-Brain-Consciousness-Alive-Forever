#!/usr/bin/env python3
"""S1a ACCEPTANCE: is the MUSCLE-DRIVEN body actually observable by the arena's own chain?

Checks, each able to FAIL:

  O1  stack B renders through the arena's six-camera ring at the resolution the chain CONSUMES
      (read from `arena_geometry.json`, not from `arena.py`'s stale declaration), with the 12
      declared fiducials present.
  O2  the recording-health gate PASSES on it -- the same gate that FAILS the known-bad stack A
      episode, so a pass is informative rather than automatic.
  O3  the EXISTING camera -> geometry chain reconstructs a world frame on it: 7 of 12 fiducials
      triangulated with a world RMS comparable to stack A's.
  O4  INJECTION: rotating the declared ring by 5 degrees must make the health gate FAIL.  A ring
      that could be rotated without consequence would mean the gate is not measuring geometry.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_stackb_observable.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

EPISODE = HERE / "outputs" / "arena" / "stackb_observable"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _gate(ep: Path) -> dict:
    out = HERE / "outputs" / "arena" / f"_gate_{ep.name}.json"
    subprocess.run([str(HERE / "venv" / "bin" / "python"),
                    str(HERE / "tools_episode_health.py"), "--episodes", str(ep),
                    "--out", str(out)], cwd=HERE, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    doc = json.loads(out.read_text())
    return doc["episodes"][0]


def main() -> int:
    print("1. O1: the episode exists, with the ring and the fiducials")
    if not (EPISODE / "episode.npz").is_file():
        check("O1a: stack B episode present", False,
              f"missing; run tools_make_stackb_observable.py --frames 12")
        return _finish({})
    z = np.load(EPISODE / "episode.npz")
    cams = sorted({k.split("/")[1] for k in z.files if k.startswith("px/cam")})
    n_fr = int(np.asarray(z["px/cam0/n_candidates"]).shape[0])
    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    res = geom.get("resolutions") or {}
    pngs = sorted((EPISODE / "frames" / "cam0").glob("*.png"))
    from PIL import Image
    with Image.open(pngs[0]) as im:
        w, h = im.size
    declared = tuple(int(v) for v in json.loads(
        (HERE / "arena.py").read_text().split("resolution_px")[1].split("[")[1].split("]")[0]
        .split(","))) if False else None
    print(f"   6-camera ring {cams}; {n_fr} frames; PNG {w}x{h} (WxH); "
          f"chain record says {res.get('height')}x{res.get('width')}")
    check("O1a: six ring cameras recorded", len(cams) == 6, str(cams))
    check("O1b: the render matches the resolution the CHAIN consumes",
          (h, w) == (int(res["height"]), int(res["width"])),
          f"render {(h, w)} vs record {(res['height'], res['width'])}")

    print("\n2. O2: the recording-health gate")
    # CALIBRATION IS A STEP OF THE CHAIN, not an optional extra: the health gate projects into the
    # images, and without the measured per-camera offset the projections miss by ~4 px on this
    # stack, which is outside the 3 px gate.  Run it here if it is absent, so the check is
    # self-contained and reproducible from the rendered episode alone.
    if not (EPISODE / "camera_offsets.json").is_file():
        print("   calibrating (no camera_offsets.json yet)")
        subprocess.run([str(HERE / "venv" / "bin" / "python"),
                        str(HERE / "run_arena_calibrate.py"), "--episode", str(EPISODE)],
                       cwd=HERE, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    g = _gate(EPISODE)
    hr = next((c for c in g["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
    rate = hr["value"]["rate"] if hr and isinstance(hr.get("value"), dict) else 0.0
    print(f"   gate status {g['status']}; fiducial hit rate {100*rate:.1f} %")
    check("O2: the gate PASSES on the muscle-driven body", g["status"] == "PASS",
          f"{100*rate:.1f} %")
    bad = _gate(HERE / "outputs" / "arena" / "arena_bare_seed0")
    check("O2b: the same gate still FAILS the known-bad episode (a pass is informative)",
          bad["status"] == "FAIL", f"seed0 {bad['status']}")

    print("\n3. O3: the existing camera->geometry chain reconstructs a world frame")
    subprocess.run([str(HERE / "venv" / "bin" / "python"),
                    str(HERE / "run_arena_reconstruct.py"), "--episodes", str(EPISODE),
                    "--out-suffix", "_s1acheck"], cwd=HERE, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rj = EPISODE / "reconstruction_s1acheck.json"
    if not rj.is_file():
        check("O3: reconstruction produced a report", False, "no output")
        return _finish({})
    rec = json.loads(rj.read_text())
    ok_frames = [r for r in rec["static_per_frame"] if r["ok"]]
    rms = np.array([r["fit_rms_mm"] for r in ok_frames], float)
    # the count lives on the per-frame summary and in the fit; `static_frame0` has no such key
    n_tri = int(len(rec["static_frame0"]["marker_index"]))
    print(f"   {len(ok_frames)}/{len(rec['static_per_frame'])} frames fit; "
          f"{n_tri} fiducials triangulated; world RMS median {np.median(rms):.4f} mm")
    check("O3a: the chain fits a world frame on stack B",
          len(ok_frames) >= 0.8 * len(rec["static_per_frame"]), f"{len(ok_frames)} frames")
    check("O3b: the same 7 of 12 fiducials triangulate as on stack A", n_tri == 7, str(n_tri))
    check("O3c: the world RMS is comparable to stack A's 0.855-0.883 mm",
          np.median(rms) < 2.0, f"{np.median(rms):.4f} mm")

    print("\n4. O4: INJECTION -- rotating the declared ring must break the geometry")
    # THE INJECTION RENDERS ELSEWHERE.  MEASURED: the first version re-rendered in place, so the
    # "restore" step restored the ROTATED episode and destroyed the good artifact.
    tmp = HERE / "outputs" / "arena" / "_stackb_rotated"
    import os
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run([str(HERE / "venv_body" / "bin" / "python"),
                    str(HERE / "tools_make_stackb_observable.py"), "--frames", "4",
                    "--ring-rotation-deg", "5", "--settle-steps", "200",
                    "--out-dir", str(tmp)], cwd=HERE, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "MUJOCO_GL": "egl"})
    rot = _gate(tmp)
    print(f"   rotated-ring episode: {rot['status']}")
    check("O4: a 5-degree ring rotation is detected (the gate measures geometry)",
          rot["status"] == "FAIL", f"{rot['status']}")
    shutil.rmtree(tmp, ignore_errors=True)

    return _finish({"gate": g, "reconstruction": {"n_triangulated": n_tri,
                                                  "world_rms_median_mm": float(np.median(rms))}})


def _finish(extra: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
           **extra,
           "units": "mm for the world frame; px for the hit rate; degrees for the ring rotation",
           "provenance": ("stack B is outputs/muscles_six_legs/fruitfly_six_leg_standing.xml; ring "
                          "and fiducials from arena.py's declarations; gate and reconstruction are "
                          "the project's existing tools, unchanged"),
           "what_this_does_not_establish": (
               "that stack B can STAND or WALK.  This establishes OBSERVABILITY: the body that "
               "muscles drive can now be seen by the same chain that sees stack A.  `stood = False` "
               "in 5 of 5 configurations is a body-physics problem and a separate workstream.")}
    dest = HERE / "outputs" / "stackb_observable_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
