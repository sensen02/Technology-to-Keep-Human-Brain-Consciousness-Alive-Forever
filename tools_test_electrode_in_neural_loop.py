#!/usr/bin/env python3
"""Is the electrode recording THE SPIKES THAT DRIVE THE BODY?

Checks, each able to FAIL:

  E1  the electrode records a non-trivial trace while the connectome drives the muscles;
  E2  the coverage is CARRIED, not implied: 7737 of 12000 tier neurons have a soma, so 64 % of the
      network is recordable and the rest is counted rather than silently dropped;
  E3  GUARD: zeroing the spikes must change the trace.  Identical would mean the electrode is not
      coupled to the network and the record is NOT RUN;
  E4  INJECTION, and it is the causal one: silencing one leg's motor neurons must change the
      recorded trace.  The electrode sits on the branch that sets the muscle activations, so if
      silencing those neurons left the record untouched, the electrode would be on a parallel
      branch and its "recording" would not be of the driving activity.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_electrode_in_neural_loop.py
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


def _run(duration_s: float, lesion_leg=None):
    """Run the muscle-driving loop, collecting BOTH the electrode trace and the raw spikes."""
    from tools_record_electrode_in_loop import record
    import run_six_leg_neural_loop as loop

    out = record(duration_s, lesion_leg)
    spikes = []
    loop.run(lesion_leg=lesion_leg, duration_s=duration_s, verbose=False,
             spike_sink=lambda sp, c: spikes.append(sp.copy()))
    return out, np.concatenate(spikes, axis=0)


def main() -> int:
    print("1. E1/E2: record while the connectome drives the muscles")
    out, spikes = _run(0.3)
    elec, trace = out["electrode"], out["trace_mV"]
    geo = elec.geometry()
    print(f"   {geo['n_channels']} channels; {geo['n_neurons_with_soma']}/"
          f"{geo['n_tier_neurons']} neurons with a soma; trace {trace.shape}; "
          f"peak {np.abs(trace).max():.4g} mV, non-zero {int((trace != 0).sum())}")
    check("E1: the electrode records a non-trivial trace", np.abs(trace).max() > 0,
          f"peak {np.abs(trace).max():.4g} mV over {trace.shape[1]} substeps")
    check("E2: the soma coverage is CARRIED in the geometry report",
          geo["n_neurons_with_soma"] + geo["n_neurons_without_soma"] == geo["n_tier_neurons"]
          and geo["n_neurons_without_soma"] > 0,
          f"{geo['n_neurons_with_soma']} of {geo['n_tier_neurons']} "
          f"({100*geo['n_neurons_with_soma']/geo['n_tier_neurons']:.0f} %)")

    print("\n2. E3: the guard -- zeroing the spikes must change the trace")
    dep = elec.dependence(spikes)
    print(f"   {dep['status']}: n_spikes {dep['n_spikes']}, peak {dep['trace_peak_mV']:.4g} mV")
    check("E3: the electrode is COUPLED to the network", dep["status"] == "PASS", dep["why"])

    print("\n3. E4: INJECTION -- silencing one leg's motor neurons must change the record")
    out_l, spikes_l = _run(0.3, lesion_leg="LF")
    trace_l = out_l["trace_mV"]
    # the two runs must have the same shape, or the comparison is meaningless
    same_shape = trace_l.shape == trace.shape
    delta = float(np.abs(trace_l - trace).max()) if same_shape else float("nan")
    n_diff = int((trace_l != trace).sum()) if same_shape else -1
    spike_delta = int((spikes_l != spikes).sum()) if spikes_l.shape == spikes.shape else -1
    print(f"   intact vs LF-lesioned: differing spike entries {spike_delta}; "
          f"trace max|Δ| {delta:.4g} mV over {n_diff} samples")
    check("E4a: the two arms produced comparable runs",
          same_shape and spikes_l.shape == spikes.shape, f"{trace.shape} vs {trace_l.shape}")
    check("E4b: silencing LF's motor neurons CHANGES the raw spikes", spike_delta > 0,
          f"{spike_delta} differing entries")
    check("E4c: and it CHANGES the recorded trace (the electrode is on the driving branch)",
          n_diff > 0 and delta > 0, f"{n_diff} samples differ, max |Δ| {delta:.4g} mV")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {"overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
           "n_checks": len(RESULTS), "n_failed": n_fail,
           "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
           "electrode_geometry": geo,
           "trace": {"shape": list(trace.shape), "peak_mV": float(np.abs(trace).max()),
                     "non_zero_samples": int((trace != 0).sum())},
           "lesion": {"leg": "LF", "spike_entries_differing": spike_delta,
                      "trace_samples_differing": n_diff},
           "units": "mV for the trace; um for geometry; the lesion is a spike-mask",
           "provenance": ("spikes from run_six_leg_neural_loop's connectome tier AFTER the acute "
                          "lesion -- the same spikes that set the muscle activations; electrode "
                          "from engine/embodied/electrode_in_loop.InLoopElectrode; soma positions "
                          "from data/banc/somas_v1.parquet (BANC CC BY 4.0)"),
           "what_this_does_not_establish": (
               "that the recorded AMPLITUDE is biology: the spike -> current kernel is ILLUSTRATIVE "
               "and 4263 of 12000 tier neurons have no soma position, so 64 % of the network is "
               "recordable.  What is established is that the electrode sits on the branch that "
               "drives the body, that it is coupled to the network, and that silencing the driving "
               "motor neurons changes what it records.")}
    dest = HERE / "outputs" / "electrode_in_neural_loop_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
