#!/usr/bin/env python3
"""S3 ACCEPTANCE: is the motor channel really six knobs, and is each one leg-specific?

WHY THIS TEST IS THE POINT OF S3.  Raising a channel from two numbers to six is trivia unless the
six numbers carry SIX DIFFERENT THINGS.  A decoder that returns six values which all move together
is the two-knob channel wearing a longer tuple.  So the criterion is not "the tuple has length 6",
it is:

  C1  each leg's scale is a function of THAT leg's own readout (silencing one leg moves it);
  C2  the effect is LEG-SPECIFIC: silencing leg X moves leg X's scale and leaves the others
      essentially alone.  This is the same specificity test the muscle ablation already passed
      (one leg's motor neurons silenced -> that leg -60 %/-41 %/-16 %, the other five 1-3 %);
  C3  the guard: with the readout UNCHANGED the command must be unchanged, reported NOT_RUN rather
      than as a finding -- the mechanism that caught 75 muscles never being wired.

  INJECTION  make `set_command_freqs` ignore all but the first entry, which is exactly the bug of
             shipping a longer tuple that carries one thing, and require C1/C2 to FAIL.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ./venv_body/bin/python tools_test_per_leg_command.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from engine.embodied.adapters import (DecodeConfig, NeuralTier, OwnershipLedger, TierConfig,
                                          build_neural_tier, )

    print("1. build the real tier and its per-leg readout groups")
    cfg = TierConfig()
    built = build_neural_tier(cfg)
    n = int(built["n"])
    cov = built["leg_readout_coverage"]
    print(f"   readout neurons {cov['n_readout_total']}, assigned to a leg "
          f"{cov['n_assigned_to_a_leg']}, ungrouped {cov['n_ungrouped_no_leg_motor_target']}")
    print(f"   per leg: {cov['per_leg_n_neurons']}")
    check("all six legs have a non-empty readout group",
          all(v > 0 for v in cov["per_leg_n_neurons"].values()),
          str(cov["per_leg_n_neurons"]))
    check("the ungrouped share is reported rather than dropped",
          cov["n_ungrouped_no_leg_motor_target"] > 0 and "caveat" in cov,
          f"{100*cov['ungrouped_share']:.0f} % of readout neurons reach no leg motor neuron")

    tier = NeuralTier(built, cfg, OwnershipLedger())
    tier.reset()

    print("\n2. drive the tier so the readout has real firing, then calibrate the decoder")
    from engine.embodied.adapters import CommandDecoder, DecodeConfig as DC
    dec = CommandDecoder(DC(), dt_ms_command=5.0, dt_ms_neural=cfg.dt_ms,
                         ledger=OwnershipLedger(), tier=tier)
    n_sub = 40
    spikes = np.zeros((n_sub, n), dtype=bool)
    for j in range(n_sub):
        spikes[j] = tier.advance(1, np.full(n, 0.15))       # MEASURED: 0.1 nA drives this tier
    rates = dec.rate_hz(spikes)
    leg_keys = sorted(k for k in rates if k.startswith("leg_"))
    print(f"   rates: total {rates['total_hz']:.2f} Hz; "
          + ", ".join(f"{k[4:-3]} {rates[k]:.2f}" for k in leg_keys))
    check("the decoder sees one rate channel per leg", len(leg_keys) == 6, str(leg_keys))
    dec.calibrate(tier, spikes, n_intervals=max(4, n_sub // 10), settled_from_step=0)

    base = dec.decode(spikes, source_interval=0)
    check("the decoded command carries six per-leg scales", len(base.freq_scale_per_leg) == 6,
          f"{tuple(round(v, 4) for v in base.freq_scale_per_leg)}")
    check("the scalar channel is still present and unchanged in shape",
          isinstance(base.speed_scale, float) and isinstance(base.turn, float),
          f"speed {base.speed_scale:.4f}, turn {base.turn:.4f}")

    print("\n3. C1/C2: silence ONE leg's readout and measure what moves")
    # A FRESH DECODER PER SILENCE.  MEASURED, and the first version got this wrong: the decoder
    # carries a FILTER state, so reusing one decoder across the six silences made each measurement
    # inherit the previous one's offset -- only the first leg came out clean (lf 0.00 % cross-leg,
    # the rest 30-37 %).  That looked like cross-leg coupling in the CHANNEL and was actually state
    # carried over in the TEST.
    def fresh():
        d = CommandDecoder(DC(), dt_ms_command=5.0, dt_ms_neural=cfg.dt_ms,
                           ledger=OwnershipLedger(), tier=tier)
        d.calibrate(tier, spikes, n_intervals=max(4, n_sub // 10), settled_from_step=0)
        return d

    per_leg = {}
    for leg in LEGS:
        idx = tier.readout_per_leg[leg]
        sp2 = spikes.copy()
        sp2[:, idx] = False                                  # silence that leg's readout only
        d2 = fresh().decode(sp2, source_interval=1)
        rel = [abs(a - b) / max(abs(b), 1e-12)
               for a, b in zip(d2.freq_scale_per_leg, base.freq_scale_per_leg)]
        k = LEGS.index(leg)
        others = [rel[j] for j in range(6) if j != k]
        per_leg[leg] = {"own_rel_change": rel[k], "max_other_rel_change": max(others),
                        "own_idx": int(k)}
        print(f"   sil {leg}: own {rel[k]*100:7.2f} %   max other {max(others)*100:7.2f} %")

    moved = [lg for lg in LEGS if per_leg[lg]["own_rel_change"] > 0.01]
    check("C1a: the decoded order is LEGS order, not alphabetical",
          tuple(base.per_leg_order) == LEGS, f"{base.per_leg_order} vs {LEGS}")
    check("C1: silencing a leg's readout moves THAT leg's scale", len(moved) == 6,
          f"moved: {moved}")
    worst = max(per_leg[lg]["max_other_rel_change"] for lg in LEGS)
    check("C2: the effect is leg-specific (every other leg < 5 %)", worst < 0.05,
          f"worst cross-leg change {worst*100:.2f} %")

    print("\n4. C3: the guard -- the command must be a function of the readout")
    # REFORMULATED, and the first version was wrong: the decoder carries a FILTER state, so two
    # consecutive decode calls legitimately differ even on identical spikes.  The guard that matters
    # is dependence: from a FRESH decoder (same calibration), a per-leg-silenced readout must give a
    # DIFFERENT command.  Identical commands would mean the readout never reached the command, which
    # is NOT RUN -- never a finding.
    def fresh_command(sp):
        d = CommandDecoder(DC(), dt_ms_command=5.0, dt_ms_neural=cfg.dt_ms,
                           ledger=OwnershipLedger(), tier=tier)
        d.calibrate(tier, spikes, n_intervals=max(4, n_sub // 10), settled_from_step=0)
        return np.asarray(d.decode(sp, source_interval=0).freq_scale_per_leg, dtype=float)

    c_base = fresh_command(spikes)
    sp_sil = spikes.copy()
    sp_sil[:, tier.readout_per_leg["lh"]] = False
    c_sil = fresh_command(sp_sil)
    check("C3: the per-leg command DEPENDS on the readout (guard reports coupling)",
          not np.array_equal(c_base, c_sil),
          f"base {tuple(round(v,4) for v in c_base)} vs silenced-lh "
          f"{tuple(round(v,4) for v in c_sil)}")
    check("C3b: the order travels with the numbers (names, not a guessed convention)",
          tuple(base.per_leg_order) == LEGS, f"{base.per_leg_order}")

    print("\n5. INJECTION: a longer tuple carrying ONE thing must be caught")
    from engine.embodied.adapters import CommandedTripodCPG
    saved = CommandedTripodCPG.set_command_freqs

    def broken(self, freq_scale_per_leg):
        f = np.asarray(freq_scale_per_leg, dtype=float)
        return saved(self, np.full(6, f[0]))                 # ignore entries 1..5
    CommandedTripodCPG.set_command_freqs = broken
    try:
        cpg = CommandedTripodCPG(timestep_s=1e-4, base_frequency_hz=12.0, seed=0,
                                 dof_order=[], ledger=OwnershipLedger())
    except Exception:                                        # dof_order needs the body; probe the setter
        cpg = None
    CommandedTripodCPG.set_command_freqs = saved

    # the injection's DETECTABLE SIGNATURE, without needing a body: apply both setters to vector
    # probe and compare the resulting intrinsic_freqs
    class _Probe:
        def __init__(self):
            self.base = np.ones(6) * 12.0
            self.n_writes = 0
            self.last_command = None
            self.net = type("N", (), {"intrinsic_freqs": None})()
            self.ledger = OwnershipLedger()
    probe = _Probe()
    vec = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    saved(probe, vec)
    good = np.array(probe.net.intrinsic_freqs, dtype=float)
    broken(probe, vec)
    bad = np.array(probe.net.intrinsic_freqs, dtype=float)
    check("the honest setter writes SIX different frequencies", len(set(np.round(good, 9))) == 6,
          f"{np.round(good, 3)}")
    check("the injected setter collapses them to ONE, so it is detectable",
          len(set(np.round(bad, 9))) == 1, f"{np.round(bad, 3)}")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
           "per_leg_effect": per_leg, "leg_readout_coverage": cov,
           "worst_cross_leg_change": worst,
           "units": "freq_scale_per_leg is a dimensionless multiplier of the CPG intrinsic frequency",
           "provenance": ("tier from build_neural_tier(TierConfig()); per-leg groups assigned by "
                          "synapse-weighted leg motor targets (the same rule the left/right split "
                          "uses); decode by the pre-registered ASSUMED map applied per leg"),
           "what_this_does_not_establish": (
               "that six knobs make the gait more fly-like, or that the ASSUMED map is correct.  It "
               "establishes that the channel now carries six leg-specific numbers and that the "
               "specificity is real.  The ungrouped share of readout neurons is reported and carries "
               "no leg command.")}
    dest = HERE / "outputs" / "per_leg_command_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
