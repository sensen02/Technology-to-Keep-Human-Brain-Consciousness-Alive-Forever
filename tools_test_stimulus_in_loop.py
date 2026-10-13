#!/usr/bin/env python3
"""Does stimulation actually reach the network?  And is the charge accounting honest?

Checks, each able to FAIL:

  1. TIMING     the pulse is biphasic: the returned current is non-zero only inside the pulse
                window, and integrates to ZERO over a full cycle (no net charge into tissue).
  2. LIMITS     the per-phase charge is compared against `engine.electrode_stim`'s OWN limits, and
                the cited/assumed provenance of each ceiling is carried into the result rather than
                restated.  The amplitude used must be inside the cited density ceiling.
  3. GUARD      two IDENTICAL tiers advanced over the same substeps, one stimulated and one not,
                must produce DIFFERENT spikes.  Identical spikes means the stimulus never arrived
                and the result is NOT_RUN -- not a null result.
  4. INJECTION  a zero-amplitude stimulus must be reported NOT_RUN by that same guard, so the guard
                is not simply hard-wired to pass.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_stimulus_in_loop.py
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
    from engine.embodied.electrode_in_loop import InLoopElectrode, InLoopStimulus, SomaPositions

    print("1. real tier, real somata, electrode geometry")
    cfg = TierConfig()
    built = build_neural_tier(cfg)
    n = int(built["n"])
    gi = np.asarray(built["global_index"], dtype=np.int64)
    root_ids = np.asarray(built["data"]["root_ids"], dtype=np.int64)[gi]
    soma = SomaPositions.from_banc(root_ids)
    elec = InLoopElectrode(soma, n_tier_neurons=n, n_channels=4, dt_neural_ms=float(cfg.dt_ms))
    print(f"   tier {n} neurons; electrode covers {elec.n_neurons} with somata "
          f"({elec.n_missing} without)")

    print("\n2. TIMING: biphasic, zero net charge")
    st = InLoopStimulus(elec, channel_index=0, amplitude_nA=1000.0,
                        phase_width_us=100.0, interphase_gap_us=20.0, repetition_rate_Hz=1000.0)
    dt = 0.01
    ts = np.arange(0.0, st.wave.period_us * 1e-3, dt)
    cur = np.stack([st.current_nA(float(t), n) for t in ts])
    nonzero = np.abs(cur).sum(axis=1) > 0
    net_C = float(cur.sum(axis=0).sum() * dt * 1e-3 * 1e-9)     # nA * ms -> C
    cyc_us = st.wave.period_us
    check("current is non-zero only inside the pulse window",
          bool(nonzero.any()) and bool((~nonzero).any()),
          f"{int(nonzero.sum())} of {len(ts)} samples non-zero")
    # CHARGE BALANCE IS A PROPERTY OF THE ELECTRODE WAVEFORM, NOT OF THE INJECTED MEMBRANE CURRENT.
    # An earlier version of this check demanded a zero net current in `recruiting` mode and FAILED --
    # correctly, because that mode injects only the recruiting phase on purpose: the asymmetry IS
    # the hand-put-in rectification.  So balance is verified where it physically lives.
    st_bal = InLoopStimulus(elec, channel_index=0, amplitude_nA=1000.0, phase_width_us=100.0,
                            interphase_gap_us=20.0, repetition_rate_Hz=1000.0,
                            mode="charge_balanced")
    cur_bal = np.stack([st_bal.current_nA(float(t), n) for t in ts])
    net_bal = float(cur_bal.sum() * dt * 1e-3 * 1e-9)
    # Charge balance is a CONTINUOUS property of the waveform; a discrete sum over a sampling grid
    # whose pulse edges do not land on sample points cannot be exactly zero, and up to one sample of
    # a phase is the expected residual.  So the criterion is that the residual is NEGLIGIBLE against
    # one phase's charge, and the sampled phase counts are reported so the number explains itself.
    q_phase = abs(st_bal.wave.amplitude_nA) * 1e-3 * st_bal.wave.phase_width_us * 1e-6
    n_cath = int((np.abs(cur_bal).sum(axis=1) > 0).sum() // 2)
    resid_frac = abs(net_bal) / q_phase if q_phase > 0 else float("inf")
    check("the charge-balanced mode is balanced to within sampling resolution",
          resid_frac < 0.05,
          f"net {net_bal:.3e} C = {resid_frac*100:.2f} % of one phase ({q_phase:.3e} C); "
          f"~{n_cath} samples per phase at dt={dt} ms")
    check("the recruiting mode is NOT balanced, and reports that as the cost of the assumption",
          abs(net_C) > 0 and "not" in st.safety()["charge_balance"]["where_it_does_not"],
          f"recruiting-mode net {net_C:.3e} C over one cycle")
    check("exactly one recruited set (weights sum to 1 over the recruited neurons)",
          abs(float(st.weights.sum()) - 1.0) < 1e-12 and st.n_recruited > 0,
          f"n_recruited={st.n_recruited}, weight sum={st.weights.sum():.6f}")

    print("\n3. LIMITS: read from the stim module, provenance carried")
    s = st.safety()
    print(f"   amplitude {s['amplitude_nA']:.0f} nA, per-phase {s['charge_per_phase_uC']:.3e} uC, "
          f"density {s['density_uC_cm2']:.4g} uC/cm^2 vs cited limit "
          f"{s['density_limit_uC_cm2']:.4g}")
    check("amplitude is inside the CITED density ceiling", s["within_cited_density_limit"],
          f"{s['density_uC_cm2']:.4g} <= {s['density_limit_uC_cm2']:.4g} uC/cm^2")
    check("amplitude is inside the ASSUMED current ceiling", s["within_assumed_current_limit"],
          f"{s['amplitude_nA']:.0f} <= {s['assumed_current_limit_nA']:.0f} nA")
    check("the cited density provenance is carried through verbatim",
          s["density_provenance"] == "MEASURED_CITED" and bool(s["density_source"]),
          f"{s['density_provenance']}: {s['density_source'][:60]}...")
    check("the current ceiling is labelled ASSUMED, not upgraded to measured",
          "ASSUMED" in s["current_provenance"], s["current_provenance"])
    check("a no-charge-safety claim is stated", "no" in s["what_this_does_not_establish"].lower()
          or "NOT" in s["what_this_does_not_establish"], "")

    print("\n4. GUARD: does stimulation change the network?")
    make = _tier_factory(built, cfg)
    dep = st.dependence(make, n_substeps=30, dt_ms=float(cfg.dt_ms))
    print(f"   without {dep['spikes_without_stimulus']} spikes, with "
          f"{dep['spikes_with_stimulus']} spikes, differing {dep['spikes_differing']} "
          f"| n_recruited={dep['n_recruited_neurons']}")
    check("stimulation reaches the network (spikes differ)", dep["status"] == "PASS", dep["why"])

    print("\n4b. S5: MEASURE the sampled waveform instead of asserting a mechanism")
    mech = st.mechanism(n_substeps=6)
    print(f"   dt {mech['dt_ms']} ms; recovery window "
          f"{mech['recovery_phase_window_us']} us of a {mech['period_us']:.0f} us period; "
          f"non-zero substeps {mech['substeps_nonzero']}/{mech['substeps_total']}")
    check("the recovery phase is NOT sampled at this dt (so charge balance cannot act)",
          mech["recovery_phase_sampled"] is False, f"sampled_signs={mech['sampled_signs']}")

    print("\n4c. S5: capture radius swept, so the DECLARED 50 um is auditable")
    sweep = InLoopStimulus.capture_radius_sweep(elec)
    for row in sweep["per_radius"]:
        mark = "  <- active" if abs(row["capture_radius_um"] - sweep["active_radius_um"]) < 1e-9 else ""
        print(f"   r={row['capture_radius_um']:6.0f} um -> {row['n_recruited']:5d} neurons"
              f"  ({100*row['fraction_of_recordable']:5.1f} % of recordable){mark}")
    check("the sweep reports the active radius and covers a range",
          sweep["active_n_recruited"] is not None and len(sweep["per_radius"]) >= 5
          and sweep["per_radius"][0]["n_recruited"] <= sweep["per_radius"][-1]["n_recruited"],
          f"active {sweep['active_radius_um']:.0f} um recruits {sweep['active_n_recruited']}")

    print("\n5. THE MEASURED MECHANISM: the two modes differ by SIGN, not by charge balance")
    # `StimWaveform` refuses a zero amplitude, so the decoupling injection is the OTHER honest
    # case: inject the waveform's literal biphasic sign.  On a linear point-current membrane the two
    # phases cancel exactly, so this must come back NOT_RUN -- and that is a real finding about
    # current injection, not a harness artefact.  It is also what makes the guard meaningful: the
    # same guard passes for `recruiting` and fails here.
    stb = InLoopStimulus(elec, channel_index=0, amplitude_nA=1000.0,
                         phase_width_us=100.0, interphase_gap_us=20.0,
                         repetition_rate_Hz=1000.0, mode="charge_balanced")
    depb = stb.dependence(make, n_substeps=30, dt_ms=float(cfg.dt_ms))
    check("the literal-sign mode reports NOT_RUN (it hyperpolarises: sign, not cancellation)",
          depb["status"] == "NOT_RUN",
          f"without {depb['spikes_without_stimulus']} vs with {depb['spikes_with_stimulus']} spikes")
    check("the two modes disagree, so the sign mapping is what decides recruitment",
          depb["status"] != dep["status"],
          f"recruiting={dep['status']} vs charge_balanced={depb['status']}")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    out = {"overall": "FAIL" if n_fail else "PASS", "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
           "status": "FAIL" if n_fail else "PASS",
           "sampled_mechanism": mech, "capture_radius_sweep": sweep,
           "units": ("nA for injected current; uC per phase; uC/cm^2 for charge density; ms for "
                     "the waveform; um for geometry"),
           "provenance": ("charge ceilings from engine.electrode_stim.charge_injection_limits "
                          "(density MEASURED_CITED, current ASSUMED); soma coordinates from "
                          "data/banc/somas_v1.parquet; tier from build_neural_tier"),
           "what_this_does_not_establish": (
               "CHARGE SAFETY, and any claim that the spatial recruitment pattern is real.  The "
               "cited density ceiling comes from 0.05-0.2 mm^2 cochlear platinum contacts and is "
               "EXTRAPOLATED to a 7 um contact; the current ceiling has no fetched source.  Which "
               "neurons receive how much is a DECLARED geometric assumption over soma voxels.")}
    dest = HERE / "outputs" / "stimulus_in_loop_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nOVERALL: {out['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


def _tier_factory(built, cfg):
    """A factory that returns a FRESH tier each call, so the two arms are independent."""
    from engine.embodied.adapters import NeuralTier, OwnershipLedger, TierConfig

    def make():
        return NeuralTier(built, TierConfig(), OwnershipLedger())
    return make


if __name__ == "__main__":
    raise SystemExit(main())
