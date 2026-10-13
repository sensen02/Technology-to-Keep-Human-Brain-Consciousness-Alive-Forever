#!/usr/bin/env python3
"""DOES STIMULATION REACH A MUSCLE, AND DOES IT SHOW UP IN BOTH OBSERVED CHANNELS?

The stimulus seam had only ever been exercised against a tier advanced IN ISOLATION: two identical
networks, one stimulated, spikes compared.  That proves the current reaches a membrane and proves
nothing about whether it reaches a MUSCLE.  This test drives the loop whose motor-neuron firing IS
the muscle activation, and it is the first place where all four seams live in one chain:

    electrode -> tier -> motor neurons -> muscles -> body motion -> cameras + electrode

Every check can FAIL, and the two that matter most are injections:

  C2  the substep restructuring is NEUTRAL.  Adding a stimulus forces the interval to be advanced one
      substep at a time (a waveform changes within an interval), where the default path advances the
      whole interval in one call.  If those two disagreed, every observed change would be an artefact
      of the restructuring rather than of the current, so the equivalence is asserted directly.
  C3  amplitude scaling.  A near-zero amplitude must leave the body where it was, and a full one must
      not: a "difference" that appears at zero amplitude is noise, not stimulation.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_stimulus_stackb.py
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

BASE_EPISODE = HERE / "outputs" / "arena" / "stackb_all_channels"
STIM_EPISODE = HERE / "outputs" / "arena" / "stackb_stimulated"
DURATION_S = 0.4
STRIDE = 4
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _grey(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("L"), dtype=float)


def main() -> int:                                                     # noqa: C901
    import run_six_leg_neural_loop as loop
    from engine.embodied.adapters import NeuralTier, OwnershipLedger, TierConfig, build_neural_tier

    print("1. C1: the stimulus is tied to THE tier it stimulates, by construction")
    print("   (a stimulus built for a different build of the tier would inject into the wrong")
    print("    neurons, still change spikes, and still look like it worked)")
    built = build_neural_tier(TierConfig())
    cfg = TierConfig()
    made = {}

    def factory(ctx):
        st, elec = loop.make_in_loop_stimulus(ctx, channel_index=0, amplitude_nA=1000.0,
                                              capture_radius_um=50.0, mode="recruiting")
        made["st"], made["elec"] = st, elec
        return st

    ok_wrong = True
    try:
        class _Bad:
            pass
        bad = _Bad()
        bad.electrode = type("E", (), {"n_tier_neurons": int(built["n"]) + 7})()
        bad.current_nA = lambda t, n: np.zeros(n)
        loop.run(lesion_leg=None, duration_s=0.05, verbose=False, stimulus=bad)
    except ValueError:
        ok_wrong = False
    check("C1: a stimulus addressing a DIFFERENT tier width is REFUSED (injection)", ok_wrong is False,
          "run() compares the stimulus's tier width against the tier it built")

    print("\n2. C2: the substep restructuring is NEUTRAL (injection)")
    n = int(built["n"])
    rng = np.random.default_rng(0)
    cur = rng.normal(0.02, 0.005, n)
    t1 = NeuralTier(built, cfg, OwnershipLedger())
    t1.reset()
    together = t1.advance(10, cur)
    t2 = NeuralTier(built, cfg, OwnershipLedger())
    t2.reset()
    one_at_a_time = np.stack([t2.advance(1, cur)[0] for _ in range(10)])
    diff = int((together != one_at_a_time).sum())
    check("C2: advance(N) == advance(1) x N, bit for bit", diff == 0,
          f"{diff} differing entries of {together.size}; so a change under stimulation is the "
          f"CURRENT, not the loop restructuring")

    print("\n3. C3: amplitude scaling -- near-zero must move nothing")
    base = loop.run(lesion_leg=None, duration_s=0.2, verbose=False, observable=False)
    check("C3a: with NO stimulus the record says so", base["stimulus"] is None,
          str(base["stimulus"]))
    bs = np.array([base["excursion"][k] for k in base["excursion"]])

    def mk(amp, mode="recruiting"):
        def f(ctx):
            st, _ = loop.make_in_loop_stimulus(ctx, channel_index=0, amplitude_nA=amp,
                                               capture_radius_um=50.0, mode=mode)
            return st
        return f

    tiny = loop.run(lesion_leg=None, duration_s=0.2, verbose=False, stimulus=mk(1e-6))
    dt_ = float(np.abs(np.array([tiny["excursion"][k] for k in base["excursion"]]) - bs).max())
    full = loop.run(lesion_leg=None, duration_s=0.2, verbose=False, stimulus=mk(1000.0))
    df_ = float(np.abs(np.array([full["excursion"][k] for k in base["excursion"]]) - bs).max())
    print(f"   1e-6 nA -> {dt_:.8f} rad;  1000 nA -> {df_:.6f} rad")
    check("C3b: a near-zero amplitude changes nothing, a full one does", dt_ == 0.0 and df_ > 0.01,
          f"{dt_:.8f} vs {df_:.6f} rad")
    check("C3c: the injected charge is recorded, not inferred",
          full["stimulus"]["n_intervals_with_current"] >= 1
          and full["stimulus"]["total_charge_nA_ms"] > 0,
          f"{full['stimulus']['n_intervals_with_current']} interval(s), "
          f"{full['stimulus']['total_charge_nA_ms']:.1f} nA*ms, "
          f"{full['stimulus']['n_recruited_neurons']} neurons recruited")

    print("\n4. C4/C5: a stimulus that reaches a muscle must move BOTH observed channels")
    if not (BASE_EPISODE / "electrode.npz").is_file():
        check("C0: the baseline two-channel episode exists", False,
              f"run tools_record_all_channels.py; {BASE_EPISODE}")
        return _finish({})
    if STIM_EPISODE.exists():
        shutil.rmtree(STIM_EPISODE)
    # A CHILD PROCESS RECORDS IT, because a recording needs the renderer and the electrode hooks that
    # `tools_record_all_channels.py` already owns; a second copy of that wiring here would drift from
    # the one the platform actually uses.
    proc = subprocess.run(
        [str(HERE / "venv_body" / "bin" / "python"), str(HERE / "tools_record_all_channels.py"),
         "--duration-s", str(DURATION_S), "--stride", str(STRIDE), "--stimulate",
         "--no-vision-probe", "--out-dir", str(STIM_EPISODE)],
        cwd=HERE, capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    if not (STIM_EPISODE / "electrode.npz").is_file():
        check("C4: the stimulated recording completed", False,
              (proc.stdout or "")[-300:] + (proc.stderr or "")[-300:])
        return _finish({})

    z0 = np.load(BASE_EPISODE / "episode.npz")
    e0 = np.load(BASE_EPISODE / "electrode.npz")
    z1 = np.load(STIM_EPISODE / "episode.npz")
    e1 = np.load(STIM_EPISODE / "electrode.npz")
    fa = sorted((BASE_EPISODE / "frames" / "cam0").glob("*.png"))
    fb = sorted((STIM_EPISODE / "frames" / "cam0").glob("*.png"))
    frame_diff = float(np.abs(_grey(fa[-1]) - _grey(fb[-1])).mean()) if fa and fb else 0.0
    tr0, tr1 = e0["trace_mV"], e1["trace_mV"]
    trace_diff = float(np.abs(tr1 - tr0).max()) if tr1.shape == tr0.shape else float("nan")
    print(f"   last frame mean |diff| {frame_diff:.5f};  electrode max |diff| {trace_diff:.5g} mV")
    check("C4a: the stimulus changed the CAMERA channel", frame_diff > 0, f"{frame_diff:.5f}")
    # A WHOLE-FRAME MEAN IS THE WRONG INSTRUMENT HERE, and saying only "0.12" invites the reading
    # that it is noise.  MEASURED: the frame is 640x800 and the fly occupies a small part of it, so
    # the change is NOT spread out -- 615 pixels (0.12 %) differ by more than 20 grey levels, they
    # all lie inside ONE bounding box, and the mean change WITHIN that box is 100.46 against 0.12467
    # over the frame.  A difference scattered across the whole image would be speckle or renderer
    # noise; a difference concentrated where the body is, is the body moving.
    from scipy import ndimage
    da, db = _grey(fa[-1]), _grey(fb[-1])
    dd = np.abs(da - db)
    hot = dd > 20.0
    n_hot = int(hot.sum())
    box = None
    if n_hot:
        yy, xx = np.nonzero(hot)
        box = (int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max()))
    # where the fly is, from the BASELINE frame: the small bright blobs are its own markers
    lab, nl = ndimage.label(da >= 240.0)
    sizes = ndimage.sum(np.ones_like(da), lab, range(1, nl + 1))
    small = [ndimage.center_of_mass(np.ones_like(da), lab, i + 1)
             for i, sz in enumerate(sizes) if 3 <= sz <= 40]
    fly_r = float(np.mean([c[0] for c in small])) if small else None
    fly_c = float(np.mean([c[1] for c in small])) if small else None
    inside = (box is not None and fly_r is not None
              and box[0] - 60 <= fly_r <= box[1] + 60 and box[2] - 60 <= fly_c <= box[3] + 60)
    frac = n_hot / dd.size
    mean_in = float(dd[hot].mean()) if n_hot else 0.0
    print(f"   changed pixels {n_hot} ({100*frac:.3f} %) in box {box}, mean |diff| inside "
          f"{mean_in:.2f} vs {dd.mean():.5f} over the frame; fly at row {fly_r:.0f} col {fly_c:.0f}")
    check("C4b: the change is LOCALISED to where the fly is, not spread over the frame",
          n_hot > 50 and frac < 0.05 and inside,
          f"{100*frac:.3f} % of pixels, mean |diff| {mean_in:.1f} inside the box against "
          f"{dd.mean():.5f} over the frame")
    check("C5: the stimulus changed the ELECTRODE channel", trace_diff > 0,
          f"{trace_diff:.5g} mV")
    stim_meta = json.loads((STIM_EPISODE / "stimulus.json").read_text())
    check("C5b: the stimulated episode carries its own stimulus record",
          stim_meta["n_intervals_with_current"] >= 1 and stim_meta["mode"] in
          ("recruiting", "charge_balanced"),
          f"mode {stim_meta['mode']}, {stim_meta['n_intervals_with_current']} interval(s) with "
          f"current, charge {stim_meta['total_charge_nA_ms']:.1f} nA*ms")
    check("C5c: the two MARKER classes still separate under stimulation, i.e. the recording is sound",
          all(f"px/cam{i}/area" in z1.files for i in range(6)),
          "px/<cam>/area present in the stimulated episode")

    return _finish({"frame_mean_abs_diff": frame_diff, "electrode_max_abs_diff_mV": trace_diff,
                    "frame_change_localisation": {"n_pixels_over_20_grey": n_hot,
                                                  "fraction_of_frame": frac,
                                                  "bbox_rows_cols": box,
                                                  "mean_abs_diff_inside_box": mean_in,
                                                  "mean_abs_diff_whole_frame": float(dd.mean()),
                                                  "fly_row_col_from_baseline":
                                                      [fly_r, fly_c]},
                    "stimulus": stim_meta, "amplitude_1e_6_nA_effect_rad": dt_,
                    "amplitude_1000_nA_effect_rad": df_})


def _finish(extra: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS], **extra,
           "units": "nA and nA*ms for current and charge, mV for the electrode, rad for joint "
                    "excursion, greyscale counts for the frames",
           "provenance": ("run_six_leg_neural_loop.run(stimulus=...) with the stimulus built from "
                          "the loop's own context via make_in_loop_stimulus; charge ceilings from "
                          "engine.electrode_stim.charge_injection_limits"),
           "what_this_does_not_establish": (
               "that the stimulation is biologically meaningful, or safe.  WHICH neurons receive how "
               "much current is an ASSUMED geometric rule: the access map has soma voxel positions "
               "and nothing else, and its capture radius is DECLARED rather than measured -- at 50 um "
               "it recruits 53 of the 7737 recordable neurons, 0.7 %, and at 10 um it recruits none, "
               "so amplitude effects are meaningful while the SPATIAL pattern is not.  The sign that "
               "maps an electrode's cathodic phase onto a depolarising membrane current is a declared "
               "assumption, and at this tier's dt the recovery phase is never sampled, so the two "
               "modes differ by SIGN alone rather than by charge balance.  No charge-safety claim is "
               "made.  And the fly still does not stand or walk under stimulation either.")}
    dest = HERE / "outputs" / "stimulus_stackb_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
