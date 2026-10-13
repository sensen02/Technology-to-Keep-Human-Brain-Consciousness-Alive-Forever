#!/usr/bin/env python3
"""S2 ACCEPTANCE: is the visual channel wired, and does its bandwidth say what it must?

Checks, each able to FAIL:

  V1  BANDWIDTH is reported, not assumed: the tier's visual neurons (11 of 12,000 measured), the
      dataset's visual population (79,538), and the resulting capture fraction.
  V2  COUPLING: a live scene must inject a different current than a blinded one.  Identical means
      the channel is not driven by the scene -> NOT_RUN.
  V3  THE REFUSAL IS INTACT: `visual_drive_nA` must still refuse without an explicit
      `drive_scale_nA_per_unit`, because this project has no measured transduction gain.  A wiring
      that bypassed that guard would be a regression against the project's own honesty machinery.
  V4  THE CONFLICT is represented rather than hidden: with the tier's visual route enabled, leg
      `lh` has no descending readout, is marked UNAVAILABLE, and its command channel is left
      NEUTRAL -- never filled with a guessed value.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_visual_drive.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from engine.embodied.adapters import TierConfig, build_neural_tier
    from engine.embodied.visual_drive import VisualDrive

    print("1. the tier's visual population, MEASURED")
    cfg = TierConfig()
    built = build_neural_tier(cfg)
    n = int(built["n"])
    a = built["data"]
    gi = np.asarray(built["global_index"], dtype=np.int64)
    vis_all = np.isin(np.asarray(a["ann_super_class"]), list(cfg.visual_classes))
    n_vis = int(vis_all[gi].sum())
    print(f"   tier {n} neurons; visual-class {n_vis}; dataset {int(vis_all.sum())}")
    check("V1a: the tier's visual count is the measured 11, not an assumption", n_vis == 11,
          f"{n_vis} of {n} = {100*n_vis/n:.3f} %")

    print("\n2. build a real scene with vision, and wire the channel")
    from engine.embodied import BodyBackend, BodyConfig
    be = BodyBackend(BodyConfig(seed=0, scene_preset="meadow_grass", add_vision=True,
                                add_tracking_camera=False, add_world_camera=False),
                     gl_backend=None)
    obs = be.observe()
    vd = VisualDrive(be.sim, be.fly.name, built).bind_sim(be.sim)
    bw = vd.bandwidth()
    print(f"   visual neurons {bw['visual_neurons_in_tier']} "
          f"({100*bw['visual_fraction_of_tier']:.3f} % of tier); "
          f"left {bw['left_eye_mapped']}, right {bw['right_eye_mapped']}, "
          f"side-unassigned {bw['eye_side_unassigned']}")
    print(f"   tier captures {100*bw['tier_capture_fraction_of_visual_population']:.4f} % of the "
          f"{bw['visual_neurons_in_dataset']} visual neurons in the dataset")
    check("V1b: the bandwidth block carries the dataset population and the capture fraction",
          bw["visual_neurons_in_dataset"] == 79538
          and 0 < bw["tier_capture_fraction_of_visual_population"] < 1e-3,
          f"{100*bw['tier_capture_fraction_of_visual_population']:.4f} %")
    check("V1c2: the channel uses 4 channels, so a still PATTERNED scene drives it",
          vd.cfg.n_channels == 4,
          f"n_channels={vd.cfg.n_channels} (2 would use temporal contrast only, which is exactly "
          f"zero for a still scene -- measured)")
    check("V1c: the drive scale is declared ASSUMED with a sweep, not asserted",
          "ASSUMED" in bw["drive_scale_provenance"] and len(bw["drive_scale_sweep"]) >= 3,
          f"scale {bw['drive_scale_nA_per_unit']} nA/unit, sweep {bw['drive_scale_sweep']}")

    cur, rep = vd.transduce(obs)
    print(f"   injected {rep['total_current_nA']:.4g} nA into "
          f"{rep['neurons_receiving_current']} neurons")
    check("V1d: the visual current reaches only the visual neurons",
          rep["neurons_receiving_current"] <= max(n_vis, 1) and rep["neurons_receiving_current"] > 0,
          f"{rep['neurons_receiving_current']} <= {n_vis}")
    check("V1e: a non-zero current was actually injected",
          rep["total_current_nA"] != 0, f"{rep['total_current_nA']:.4g} nA")

    print("\n3. V2: the GUARD -- blinding the scene must change the current")
    dep = vd.dependence(obs)
    print(f"   real peak {dep['peak_real_nA']:.4g} nA vs blind peak {dep['peak_blind_nA']:.4g} nA")
    check("V2: the visual channel is COUPLED to the scene (blind changes the current)",
          dep["status"] == "PASS", dep["why"])

    print("\n4. V3: the project's own refusal must still refuse")
    from engine.embodied.vision import UnitConversionRefused, VisionConfig, visual_drive_nA
    feats = vd.front.read(be.sim, be.fly.name, float(obs.time_s))
    refused = False
    try:
        visual_drive_nA(feats, VisionConfig())          # no drive_scale_nA_per_unit
    except UnitConversionRefused:
        refused = True
    except Exception as exc:                            # noqa: BLE001
        refused = False
        print(f"   (unexpected {type(exc).__name__}: {exc})")
    check("V3: visual_drive_nA still refuses without an explicit conversion factor", refused,
          "UnitConversionRefused raised")

    print("\n5. V4: the conflict -- the visual route costs leg lh its command channel")
    built_v = build_neural_tier(TierConfig(include_visual_route=True))
    cov_v = built_v["leg_readout_coverage"]
    print(f"   with the visual route: per-leg {cov_v['per_leg_n_neurons']}, "
          f"unavailable {cov_v['unavailable_legs']}")
    check("V4a: the unavailable leg is REPORTED, not silently dropped",
          cov_v["unavailable_legs"] == ["lh"], f"{cov_v['unavailable_legs']}")
    check("V4b: unavailability is documented with its meaning",
          "unavailable_leg_meaning" in cov_v
          and "never filled with a guessed value" in cov_v["unavailable_leg_meaning"],
          "")
    # and the decoder leaves that leg NEUTRAL rather than guessing
    from engine.embodied.adapters import (CommandDecoder, DecodeConfig, NeuralTier, OwnershipLedger)
    tier_v = NeuralTier(built_v, TierConfig(include_visual_route=True), OwnershipLedger())
    tier_v.reset()
    dec = CommandDecoder(DecodeConfig(), dt_ms_command=5.0, dt_ms_neural=0.5,
                         ledger=OwnershipLedger(), tier=tier_v)
    sp = np.zeros((20, int(built_v["n"])), dtype=bool)
    for j in range(20):
        sp[j] = tier_v.advance(1, np.full(int(built_v["n"]), 0.15))
    dec.calibrate(tier_v, sp, n_intervals=4, settled_from_step=0)
    cmd = dec.decode(sp, source_interval=0)
    print(f"   per-leg scales {tuple(round(v,4) for v in cmd.freq_scale_per_leg)}; "
          f"unavailable {cmd.per_leg_unavailable}")
    idx = list(cmd.per_leg_order).index("lh") if "lh" in cmd.per_leg_order else None
    lh_scale = cmd.freq_scale_per_leg[idx] if idx is not None else None
    check("V4c: the unavailable leg's channel is NEUTRAL (1.0), not guessed",
          lh_scale == 1.0 and tuple(cmd.per_leg_unavailable) == ("lh",),
          f"lh scale = {lh_scale}")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
           "bandwidth": bw, "dependence": dep,
           "visual_route_conflict": {"per_leg_n_neurons": cov_v["per_leg_n_neurons"],
                                     "unavailable_legs": cov_v["unavailable_legs"]},
           "units": "nA for injected current; the drive scale is nA per readout unit (dimensionless)",
           "provenance": ("tier via build_neural_tier; scene via the project's meadow_grass preset; "
                          "features via engine.embodied.vision.OmmatidiaFrontEnd; conversion via "
                          "engine.receptors.to_nA with a declared provenance string"),
           "what_this_does_not_establish": (
               "that the fly sees, or that the visual current is biologically scaled.  The channel "
               "reaches 11 of 12,000 tier neurons -- 0.014 % of the dataset's visual population -- "
               "and its conversion factor is ASSUMED and swept, not measured.")}
    dest = HERE / "outputs" / "visual_drive_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
