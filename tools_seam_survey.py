#!/usr/bin/env python3
"""Does the DECLARED registration of what can be observed together match the RECORDED artifacts?

The four seams answer "is the interface implemented".  A different question is "which of them can be
observed at the same time, on one body" -- and all four seams can be implemented while only two
channels can be recorded together.  `seams.OBSERVATION_CHANNELS` is the declaration; the recorded
episode is the evidence.  This check binds them, so a registry that drifts away from the artifacts
fails the suite instead of being quoted.

  C1  `OBSERVATION_CHANNELS["together"]["recorded_in_one_run"]` equals the channels the episode's own
      manifest says were recorded, in the same order.
  C2  every channel declared `together: True` has evidence on disk, and every one declared
      `together: False` is recorded as absent in the manifest -- the absent one must be there too, so
      "we could not observe it" is written down rather than being invisible.
  C3  the measured absence is the STRUCTURAL one (AttributeError on binding), not a missing file.
  C4  `survey()["observation_channels"]` agrees with the registry, and does NOT report three
      simultaneous channels.
  C5  the episode's shared clock is real: the frames sit on the interval clock at the recorded
      indices.  This fails on the first version of the recorder, which understated the frame clock by
      exactly `stride`.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_seam_survey.py --check
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(HERE / "outputs" / "seam_survey.json"))
    args = ap.parse_args()

    from engine.embodied import seams

    reg = seams.OBSERVATION_CHANNELS
    mj_path, npz_path, el_path = (EPISODE / "markers.json", EPISODE / "episode.npz",
                                  EPISODE / "electrode.npz")
    if not all(p.is_file() for p in (mj_path, npz_path, el_path)):
        check("C0: the recorded two-channel episode exists", False,
              f"missing under {EPISODE} -- run tools_record_all_channels.py")
        return _finish(args.out, reg)
    mj = json.loads(mj_path.read_text())

    print("1. C1: the declaration matches the manifest")
    declared = list(reg["together"]["recorded_in_one_run"])
    recorded = list(mj.get("channels_recorded_in_this_run", []))
    print(f"   declared {declared} vs recorded {recorded}")
    check("C1: declared simultaneously-recorded channels == the manifest's", declared == recorded,
          str(recorded))

    print("\n2. C2: every declared channel has evidence, present or absent")
    for name in declared:
        spec = reg.get(name) or {}
        ev = HERE / str(spec.get("evidence", "")).split(",")[0].strip()
        ok = bool(spec.get("together")) and ev.exists()
        check(f"C2: channel '{name}' is declared together=True and its evidence exists", ok,
              str(ev.relative_to(HERE)))
    absent = list(reg["together"]["absent_by_construction"])
    man_absent = mj.get("channels_absent_from_this_body", {})
    check("C2b: the absent channel is written down in the manifest, not merely missing",
          set(absent) <= set(man_absent), f"absent {absent}, manifest keys {sorted(man_absent)}")
    for name in absent:
        check(f"C2c: channel '{name}' is declared together=False", reg[name]["together"] is False,
              f"in_stack {reg[name]['in_stack']}")

    print("\n3. C3: the absence is STRUCTURAL, not a missing file")
    for name in absent:
        m = man_absent.get(name, {})
        # `NOT_PROBED` means the probe was skipped for speed; that is not a measured absence, so it
        # must not be accepted here -- otherwise a skipped probe would masquerade as an established
        # structural fact.
        ok = m.get("can_bind") is False and m.get("error") == "AttributeError"
        check(f"C3: '{name}' failed binding with a measured AttributeError", ok,
              f"{m.get('error')}: {str(m.get('why', ''))[:70]}")

    print("\n4. C4: survey() agrees and does not overclaim")
    full = seams.survey()
    surv = full["observation_channels"]
    seam_status = {k: v["status"] for k, v in full.items() if k != "observation_channels"}
    print(f"   seams {seam_status}")
    print(f"   survey says {surv['n_simultaneous']} simultaneous "
          f"{surv['recorded_in_one_run_on_one_body']}, absent {surv['absent_by_construction']}")
    check("C4a: survey reports EXACTLY the declared channels",
          surv["recorded_in_one_run_on_one_body"] == declared
          and surv["absent_by_construction"] == absent, str(surv["n_simultaneous"]))
    check("C4b: survey does NOT claim three simultaneous channels", surv["n_simultaneous"] != 3,
          f"n_simultaneous = {surv['n_simultaneous']}")
    # The POINT of checking this: all four seams being IMPLEMENTED says nothing about how many of
    # them can be OBSERVED AT ONCE, and the two numbers must be reported separately so that
    # "4 IMPLEMENTED" cannot be read as "4 observed together".
    all_impl = all(v == "IMPLEMENTED" for v in seam_status.values()) and len(seam_status) == 4
    check("C4c: all four seams IMPLEMENTED while only TWO channels are observable together",
          all_impl and surv["n_simultaneous"] == 2,
          f"{sum(v == 'IMPLEMENTED' for v in seam_status.values())}/4 seams, "
          f"{surv['n_simultaneous']} simultaneous channels")

    print("\n5. C5: the shared clock is real in the artifact")
    z = np.load(npz_path)
    e = np.load(el_path)
    idx = np.asarray(z["truth/frame_interval_index"], dtype=np.int64)
    stride = int(np.asarray(z["truth/stride"]).ravel()[0])
    t_f = np.asarray(z["truth/time_s"], dtype=float)
    t_e = np.asarray(e["interval_time_s"], dtype=float)
    on_clock = bool(np.allclose(t_f, t_e[idx]))
    dt_ok = t_f.size < 2 or np.allclose(np.diff(t_f), stride * float(np.diff(t_e)[0]), atol=1e-9)
    check("C5: frame times ARE electrode interval times at the recorded indices", on_clock and dt_ok,
          f"frame dt {np.diff(t_f)[0]:.4g} s, stride {stride}, interval dt "
          f"{np.diff(t_e)[0]:.4g} s")
    check("C5b: the frame clock is NOT the naive interval_times[:n_frames]",
          not np.allclose(t_f, t_e[:t_f.size]) or stride == 1,
          f"stride {stride} -- a stride-1 recording cannot tell the two apart, which is exactly why "
          f"the recorded indices are stored")
    check("C5c: the electrode spans the whole interval range the frames were cut from",
          t_e.size >= t_f.size, f"{t_e.size} intervals, {t_f.size} frames")
    # The electrode samples FINER than the frames (one column per neural substep), so it cannot be
    # indexed by the interval clock.  Its own axis must therefore be in the artifact; without it the
    # electrode's time base is undeclared and 800 columns share 80 timestamps.
    et = np.asarray(e["electrode_time_s"], dtype=float) if "electrode_time_s" in e.files else None
    nspi = (int(np.asarray(e["samples_per_interval"]).ravel()[0])
            if "samples_per_interval" in e.files else 0)
    tr = e["trace_mV"]
    check("C5d: the electrode declares its OWN sample axis of the right length",
          et is not None and et.size == tr.shape[1],
          f"trace {tr.shape} vs axis {None if et is None else et.size} samples")
    if et is not None and et.size > 1:
        dt_sub = float(np.diff(et).mean())
        check("C5e: the electrode axis is finer than the frame clock and evenly spaced",
              bool(np.allclose(np.diff(et), dt_sub, atol=1e-12)) and dt_sub < np.diff(t_e)[0],
              f"electrode dt {dt_sub*1e3:.3f} ms vs interval dt {np.diff(t_e)[0]*1e3:.3f} ms, "
              f"{nspi} samples/interval")
        check("C5f: the electrode axis starts on the same time origin as the frames",
              abs(et[0] - t_e[0]) < 1e-12 and abs(et[0] - t_f[0]) < 1e-12,
              f"electrode {et[0]}, interval {t_e[0]}, frame {t_f[0]}")

    return _finish(args.out, reg)


def _finish(out_path: str, reg: dict) -> int:
    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
           "declared_together": list(reg["together"]["recorded_in_one_run"]),
           "declared_absent": list(reg["together"]["absent_by_construction"]),
           "units": "s for clocks, counts elsewhere",
           "provenance": ("seams.OBSERVATION_CHANNELS against the episode manifest and npz files "
                          "written by tools_record_all_channels.py"),
           "what_this_does_not_establish": (
               "that the observed channels are TRUTHFUL about a fly.  Every number here is "
               "simulator output; the electrode amplitude rests on an ILLUSTRATIVE spike->current "
               "kernel, and the camera channel sees arena fiducials rather than the fly's own "
               "markers, so limb kinematics are not established.")}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {out_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
