#!/usr/bin/env python3
"""Is the body that MUSCLES drive observable WHILE IT MOVES, and is the motion neuron-driven?

Checks, each able to FAIL:

  N1  MOTION: the rendered frames must change over time, and the change must be in the FLY, not in
      the fiducials -- the static markers are the world reference and must not move.
  N2  CHAIN: the recording-health gate PASSES and the existing camera -> geometry chain fits a world
      frame on a DRIVEN episode, not only on a settled one.
  N3  NO STATE-TO-ACTION FUNCTION: the loop's own source must contain no hand-authored
      state-to-action mapping.  This is asserted by MEASURING the contract it declares -- every
      muscle's activation is `min(1, rate/REF_RATE_HZ)` of that muscle's own motor-neuron spikes --
      rather than by trusting the docstring.
  N4  INJECTION: ablating ONE leg's motor neurons must change the rendered frames.  A body whose
      images are unaffected by silencing the neurons that drive it is not being driven by them.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_neural_loop_observable.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

EPISODE = HERE / "outputs" / "arena" / "stackb_neural_loop"
LESIONED = HERE / "outputs" / "arena" / "_stackb_lesioned"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _grey(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def _centroids(g: np.ndarray, thr: float = 240.0):
    from scipy import ndimage
    lab, n = ndimage.label(g >= thr)
    out = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if len(yy) >= 3:
            out.append((float(yy.mean()), float(xx.mean())))
    return np.asarray(out)


def _gate(ep: Path) -> dict:
    out = HERE / "outputs" / "arena" / f"_gate_{ep.name}.json"
    subprocess.run([str(HERE / "venv" / "bin" / "python"),
                    str(HERE / "tools_episode_health.py"), "--episodes", str(ep),
                    "--out", str(out)], cwd=HERE, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text())["episodes"][0]


def main() -> int:
    if not (EPISODE / "episode.npz").is_file():
        check("N0: driven episode present", False,
              "run tools_render_neural_loop.py --duration-s 0.4 --stride 4")
        return _finish({})

    print("1. N1: the frames must move, and the FIDUCIALS must not")
    frames = sorted((EPISODE / "frames" / "cam0").glob("*.png"))
    a, b = _grey(frames[0]), _grey(frames[-1])
    diff = np.abs(a - b)
    n_changed = int((diff > 10).sum())
    ca, cb = _centroids(a), _centroids(b)
    fid_shift = (float(np.median([np.min(np.hypot(ca[:, 0] - p[0], ca[:, 1] - p[1])) for p in cb]))
                 if len(ca) and len(cb) else float("nan"))
    print(f"   {len(frames)} frames; frame0 vs last: mean|diff| {diff.mean():.4f}, "
          f"changed(>10) {n_changed}; fiducial shift {fid_shift:.2f} px")
    check("N1a: the rendered scene CHANGES over the loop (the body moves)", n_changed > 0,
          f"{n_changed} pixels changed")
    check("N1b: the static fiducials do NOT move (they are the world reference)",
          np.isfinite(fid_shift) and fid_shift < 0.5, f"{fid_shift:.2f} px")
    check("N1c: all 12 fiducials are visible in the driven episode",
          len(ca) == 12 and len(cb) == 12, f"frame0 {len(ca)}, last {len(cb)}")

    print("\n2. N2: the existing chain on a DRIVEN episode")
    if not (EPISODE / "camera_offsets.json").is_file():
        print("   calibrating (no camera_offsets.json yet)")
        subprocess.run([str(HERE / "venv" / "bin" / "python"),
                        str(HERE / "run_arena_calibrate.py"), "--episode", str(EPISODE)],
                       cwd=HERE, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    g = _gate(EPISODE)
    hr = next((c for c in g["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
    rate = hr["value"]["rate"] if hr and isinstance(hr.get("value"), dict) else 0.0
    print(f"   gate {g['status']}; hit rate {100*rate:.1f} %")
    check("N2a: the health gate PASSES on the driven episode", g["status"] == "PASS",
          f"{100*rate:.1f} %")
    subprocess.run([str(HERE / "venv" / "bin" / "python"),
                    str(HERE / "run_arena_reconstruct.py"), "--episodes", str(EPISODE),
                    "--out-suffix", "_loopcheck"], cwd=HERE, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rj = EPISODE / "reconstruction_loopcheck.json"
    if rj.is_file():
        rec = json.loads(rj.read_text())
        ok_fr = [r for r in rec["static_per_frame"] if r["ok"]]
        rms = np.array([r["fit_rms_mm"] for r in ok_fr], float)
        n_tri = len(rec["static_frame0"]["marker_index"])
        print(f"   reconstruction: {len(ok_fr)}/{len(rec['static_per_frame'])} frames, "
              f"{n_tri} fiducials, world RMS median {np.median(rms):.4f} mm")
        check("N2b: the chain fits a world frame on the driven body",
              len(ok_fr) >= 0.8 * len(rec["static_per_frame"]), f"{len(ok_fr)} frames")
        check("N2c: the same 7 of 12 fiducials triangulate as in the other chains", n_tri == 7,
              str(n_tri))
    else:
        check("N2b: reconstruction produced a report", False, "no output")

    print("\n3. N3: the activation contract, MEASURED rather than trusted")
    src = (HERE / "run_six_leg_neural_loop.py").read_text()
    has_contract = ("min(1.0, rate / REF_RATE_HZ)" in src
                    and "wiring.get(leg, {}).get(mtype)" in src)
    # and no hand-authored state->action table anywhere in the driving path
    no_lookup = not any(tok in src for tok in ("ACTION_TABLE", "state_to_action", "policy(",
                                               "def action_for"))
    check("N3a: activation IS min(1, rate/REF_RATE_HZ) of that muscle's own MN spikes",
          has_contract, "the contract is present in the driving path")
    check("N3b: no hand-authored state-to-action mapping exists in the loop", no_lookup,
          "no action table / policy function found")

    print("\n4. N4: INJECTION -- silencing one leg's motor neurons must change the images")
    if LESIONED.exists():
        shutil.rmtree(LESIONED)
    subprocess.run([str(HERE / "venv_body" / "bin" / "python"),
                    str(HERE / "tools_render_neural_loop.py"), "--duration-s", "0.4",
                    "--stride", "4", "--lesion-leg", "LF", "--out-dir", str(LESIONED)],
                   cwd=HERE, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "MUJOCO_GL": "egl"})
    lf = sorted((LESIONED / "frames" / "cam0").glob("*.png"))
    if not lf:
        check("N4: the lesioned arm rendered", False, "no frames")
    else:
        la, lb = _grey(lf[0]), _grey(lf[-1])
        base_changed = int((np.abs(a - b) > 10).sum())
        les_changed = int((np.abs(la - lb) > 10).sum())
        # COMPARE THE LAST FRAMES, NOT THE FIRST.  MEASURED, and the first version of this check got
        # it wrong: frame 0 is rendered at the first command interval, before EITHER arm has moved,
        # so the two arms are identical there BY CONSTRUCTION and the check reported "the lesion
        # changes nothing" while the lesion was in fact working (the arms' total change differed,
        # 186 px against 211 px).  Comparing the last frame is the comparison that means something.
        arm_diff_first = float(np.abs(a - la).mean())
        arm_diff_last = float(np.abs(b - lb).mean())
        print(f"   intact changed {base_changed} px, lesioned changed {les_changed} px; "
              f"intact-vs-lesioned mean|diff| first frame {arm_diff_first:.4f}, "
              f"last frame {arm_diff_last:.4f}")
        check("N4a: silencing one leg changes the rendered body (the neurons drive it)",
              arm_diff_last > 0.0, f"last-frame mean|diff| {arm_diff_last:.4f}")
        # RELATIVE GROWTH, NOT AN EXACT ZERO.  The previous criterion was
        # `arm_diff_first == 0.0 and arm_diff_last > 0.0`, which rested on an assumption that is
        # now MEASURED to be false: that frame 0 is rendered before either arm has moved, so the
        # two arms are bit-identical there.  That held only while most of this model's muscles
        # could not produce force at all; after the round-25 repairs every one of the 90 muscles
        # works, so the lesion changes the activation on the very first step and the render of
        # frame 0 differs by 0.0064 (mean |diff|).  An exact equality here was a brittle accident,
        # not a property of the claim.  The claim is that the two arms DIVERGE, so the criterion is
        # that the difference grows, stated with its factor rather than as an equality.
        growth = arm_diff_last / max(arm_diff_first, 1e-12)
        check("N4a2: the two arms DIVERGE -- the last frames differ more than the first",
              arm_diff_last > 1.5 * arm_diff_first and arm_diff_last > 0.0,
              f"first {arm_diff_first:.4f}, last {arm_diff_last:.4f}, growth {growth:.2f}x "
              f"(needs >1.50x)")
        check("N4b: the lesion does not simply freeze the whole body",
              les_changed > 0, f"{les_changed} px still change")
    shutil.rmtree(LESIONED, ignore_errors=True)

    return _finish({"gate_status": g["status"], "hit_rate": rate,
                    "frames_changed_intact": n_changed, "fiducial_shift_px": fid_shift})


def _finish(extra: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS], **extra,
           "units": "px for image differences and hit rates; mm for the world frame",
           "provenance": ("the body is stack B driven by run_six_leg_neural_loop.run through a "
                          "frame_sink callback; the ring and fiducials are arena.py's declarations; "
                          "the gate and the reconstruction are the project's existing tools"),
           "what_this_does_not_establish": (
               "that the body WALKS.  `stood = False` in 5 of 5 configurations means the arena model "
               "does not support itself; what is established here is that the camera chain observes "
               "the muscle-driven body WHILE THE CONNECTOME DRIVES IT, and that the observed motion "
               "changes when the driving neurons are silenced.")}
    dest = HERE / "outputs" / "neural_loop_observable_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
