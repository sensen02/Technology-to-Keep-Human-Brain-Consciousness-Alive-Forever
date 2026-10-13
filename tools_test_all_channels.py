#!/usr/bin/env python3
"""Are the camera ring AND the electrode recorded in ONE run of ONE body?

Checks, each able to FAIL:

  A1  BOTH channels exist in the SAME episode directory and share ONE interval clock.  Two channels
      recorded in two runs are two demonstrations, not an observation layer.
  A2  the camera channel is usable: the existing health gate PASSES on it.
  A3  the electrode channel is coupled: the guard reports PASS on it.
  A4  INJECTION, and it is the point of running them together: silencing one leg's motor neurons must
      change BOTH channels of the SAME run -- the frames and the recorded trace.
  A5  the VISUAL channel's absence is MEASURED, not assumed: binding the front end to this body must
      fail for the named structural reason (it needs a FlyGym Simulation; stack B is raw MjData).

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_all_channels.py
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

EPISODE = HERE / "outputs" / "arena" / "stackb_all_channels"
LESIONED = HERE / "outputs" / "arena" / "_all_channels_lesioned"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _grey(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("L"), dtype=float)


def main() -> int:
    if not (EPISODE / "electrode.npz").is_file():
        check("A0: the two-channel episode exists", False,
              "run tools_record_all_channels.py --duration-s 0.4 --stride 4")
        return _finish({})

    print("1. A1: both channels, one run, one clock")
    z = np.load(EPISODE / "episode.npz")
    e = np.load(EPISODE / "electrode.npz")
    cams = sorted({k.split("/")[1] for k in z.files if k.startswith("px/cam")})
    n_fr = int(np.asarray(z["px/cam0/n_candidates"]).shape[0])
    t_frame = np.asarray(z["truth/time_s"], dtype=float)
    t_elec = np.asarray(e["interval_time_s"], dtype=float)
    print(f"   {len(cams)} cameras, {n_fr} frames; electrode {e['trace_mV'].shape}; "
          f"interval clock {t_elec[:3]} s -> {t_elec[-1]:.3f} s; frame clock {t_frame[:3]} s -> "
          f"{t_frame[-1]:.3f} s")
    check("A1a: both channels are in ONE episode", len(cams) == 6 and e["trace_mV"].size > 0,
          f"{len(cams)} cameras + electrode {e['trace_mV'].shape}")
    # THE SHARED CLOCK, as a fact about the artifact rather than a claim in a docstring: the frames
    # must sit ON the electrode's interval clock, at the recorded indices.  This check FAILS on the
    # earlier version of the recorder, which wrote interval_times[:n_frames] and so understated the
    # frame clock by `stride` (declared 0..0.095 s against a 0..0.38 s span of real renders).
    idx = np.asarray(z["truth/frame_interval_index"], dtype=np.int64)
    stride = int(np.asarray(z["truth/stride"]).ravel()[0])
    same = bool(np.allclose(t_frame, t_elec[idx]))
    dt_ok = (t_frame.size < 2
             or np.allclose(np.diff(t_frame), stride * np.diff(t_elec)[0], atol=1e-9))
    check("A1b: the frames sit ON the electrode's interval clock", same and dt_ok,
          f"indices {idx[:4].tolist()}..., frame dt {np.diff(t_frame)[0]:.4g} s = "
          f"stride {stride} x interval dt {np.diff(t_elec)[0]:.4g} s")
    check("A1c: the electrode covers the whole run the frames span",
          t_elec[-1] >= t_frame[-1] and t_frame[-1] > 0,
          f"electrode to {t_elec[-1]:.3f} s, frames to {t_frame[-1]:.3f} s")
    geo = {k: int(e[k][0]) for k in ("n_neurons_with_soma", "n_tier_neurons")}
    check("A1d: the electrode coverage is carried in the same artifact",
          geo["n_neurons_with_soma"] < geo["n_tier_neurons"],
          f"{geo['n_neurons_with_soma']}/{geo['n_tier_neurons']}")

    print("\n2. A2/A3: both channels are usable")
    if not (EPISODE / "camera_offsets.json").is_file():
        print("   calibrating")
        subprocess.run([str(HERE / "venv" / "bin" / "python"),
                        str(HERE / "run_arena_calibrate.py"), "--episode", str(EPISODE)],
                       cwd=HERE, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    out = HERE / "outputs" / "arena" / "_gate_all_channels.json"
    subprocess.run([str(HERE / "venv" / "bin" / "python"),
                    str(HERE / "tools_episode_health.py"), "--episodes", str(EPISODE),
                    "--out", str(out)], cwd=HERE, check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    g = json.loads(out.read_text())["episodes"][0]
    hr = next((c for c in g["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
    rate = hr["value"]["rate"] if hr and isinstance(hr.get("value"), dict) else 0.0
    print(f"   camera channel: gate {g['status']}, hit rate {100*rate:.1f} %")
    check("A2: the CAMERA channel passes the health gate", g["status"] == "PASS",
          f"{100*rate:.1f} %")

    from engine.embodied.adapters import TierConfig, build_neural_tier
    from engine.embodied.electrode_in_loop import InLoopElectrode, SomaPositions
    import run_six_leg_neural_loop as loop
    built = build_neural_tier(TierConfig())
    gi = np.asarray(built["global_index"], dtype=np.int64)
    roots = np.asarray(built["data"]["root_ids"], dtype=np.int64)[gi]
    elec = InLoopElectrode(SomaPositions.from_banc(roots), n_tier_neurons=int(built["n"]),
                           n_channels=4, dt_neural_ms=float(loop.TIER_DT_MS))
    spikes = []
    loop.run(lesion_leg=None, duration_s=0.3, verbose=False,
             spike_sink=lambda sp, c: spikes.append(sp.copy()))
    sp = np.concatenate(spikes, axis=0)
    dep = elec.dependence(sp)
    print(f"   electrode channel: guard {dep['status']}, n_spikes {dep['n_spikes']}")
    check("A3: the ELECTRODE channel is coupled to the network", dep["status"] == "PASS",
          dep["why"][:80])

    print("\n3. A4: INJECTION -- ONE lesion must change BOTH channels of the SAME run")
    if LESIONED.exists():
        shutil.rmtree(LESIONED)
    subprocess.run([str(HERE / "venv_body" / "bin" / "python"),
                    str(HERE / "tools_record_all_channels.py"), "--duration-s", "0.4",
                    "--stride", "4", "--lesion-leg", "LF", "--no-vision-probe",
                    "--out-dir", str(LESIONED)],
                   cwd=HERE, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "MUJOCO_GL": "egl"})
    if not (LESIONED / "electrode.npz").is_file():
        check("A4: the lesioned two-channel run completed", False, "no output")
    else:
        zl = np.load(LESIONED / "episode.npz")
        el = np.load(LESIONED / "electrode.npz")
        fa = sorted((EPISODE / "frames" / "cam0").glob("*.png"))
        fl = sorted((LESIONED / "frames" / "cam0").glob("*.png"))
        frame_diff = (float(np.abs(_grey(fa[-1]) - _grey(fl[-1])).mean())
                      if fa and fl else 0.0)
        tr, tl = e["trace_mV"], el["trace_mV"]
        trace_diff = float(np.abs(tl - tr).max()) if tl.shape == tr.shape else float("nan")
        print(f"   lesion LF -> frame mean|diff| {frame_diff:.4f}; "
              f"electrode max|Δ| {trace_diff:.4g} mV")
        check("A4a: the lesion changes the CAMERA channel of the same run", frame_diff > 0,
              f"{frame_diff:.4f}")
        check("A4b: the lesion changes the ELECTRODE channel of the same run", trace_diff > 0,
              f"{trace_diff:.4g} mV")
    shutil.rmtree(LESIONED, ignore_errors=True)

    print("\n4. A5: the VISUAL channel's absence, MEASURED")
    from tools_record_all_channels import vision_binding_probe
    probe = vision_binding_probe()
    print(f"   can_bind {probe['can_bind']}; {probe.get('error')}: {probe.get('why', '')[:80]}")
    check("A5: vision CANNOT bind to this body, for the measured structural reason",
          probe["can_bind"] is False and probe.get("error") == "AttributeError",
          probe.get("reason", "")[:100])
    mj = json.loads((EPISODE / "markers.json").read_text())
    check("A5b: the absent channel is RECORDED in the episode's own manifest",
          "channels_absent_from_this_body" in mj
          and mj["channels_absent_from_this_body"]["vision"]["can_bind"] is False,
          str(list(mj.get("channels_recorded_in_this_run", []))))

    return _finish({"gate_status": g["status"], "hit_rate": rate,
                    "vision": probe, "electrode_geometry": elec.geometry()})


def _finish(extra: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS], **extra,
           "units": "px for hit rates and image differences; mV for the electrode; um for geometry",
           "provenance": ("ONE call to run_six_leg_neural_loop.run with both a frame_sink and a "
                          "spike_sink; the electrode is fed the POST-lesion spikes, i.e. the ones "
                          "that set the muscle activations; ring and fiducials from arena.py"),
           "what_this_does_not_establish": (
               "that THREE channels run together.  The visual front end cannot bind to this body at "
               "all -- it needs a FlyGym Simulation and stack B is raw MuJoCo -- so this platform "
               "observes TWO channels on the muscle-driven body, and vision remains a stack A "
               "channel.  Nor does it establish that the body walks: `stood = False` in 5 of 5 "
               "configurations is a body-physics problem.")}
    dest = HERE / "outputs" / "all_channels_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
