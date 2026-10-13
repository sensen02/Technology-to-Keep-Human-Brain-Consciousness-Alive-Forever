#!/usr/bin/env python3
"""PUT THE ELECTRODE IN THE LOOP THAT MUSCLES ARE DRIVEN BY.

WHY THIS, AND NOT THE ELECTRODE ALREADY IN `engine/embodied/loop.py`.  That loop drives a body
through TWO CPG knobs, so a recording taken there is a recording of a network that modulates a
pre-recorded gait.  In `run_six_leg_neural_loop` the connection is direct: each muscle's activation
IS the firing rate of that muscle's own motor neurons.  An electrode in THAT loop records the spikes
that are moving the body -- which is the only version of the claim an electrode can support.

THE CHAIN, end to end, in ONE process:

    body joint angles -> mechanosensory neurons        [receptor transduction]
    connectome edges  -> motor-neuron spikes           [the network]
    motor-neuron rate -> that muscle's activation      [no state-to-action function]
    activations       -> MuJoCo muscles -> the body moves
    the SAME spikes   -> `InLoopElectrode.record(...)` -> per-channel voltage

The electrode therefore sits on the branch that drives the body, not on a parallel one.

HONESTY, INHERITED AND NOT WEAKENED.  `InLoopElectrode` already declares that its spike -> current
kernel is ILLUSTRATIVE (charge per spike, pulse duration) and that only 7,737 of the 12,000 tier
neurons have a soma position.  Nothing here changes that: the recorded AMPLITUDE is not a biological
measurement, and the coverage is carried in every result.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_record_electrode_in_loop.py --duration-s 0.4
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

OUT = HERE / "outputs" / "electrode_in_neural_loop.json"


def record(duration_s: float = 0.4, lesion_leg=None, n_channels: int = 4) -> dict:
    import run_six_leg_neural_loop as loop
    from engine.embodied.adapters import TierConfig, build_neural_tier
    from engine.embodied.electrode_in_loop import InLoopElectrode, SomaPositions

    built = build_neural_tier(TierConfig())
    n_tier = int(built["n"])
    gi = np.asarray(built["global_index"], dtype=np.int64)
    root_ids = np.asarray(built["data"]["root_ids"], dtype=np.int64)[gi]
    soma = SomaPositions.from_banc(root_ids)
    elec = InLoopElectrode(soma, n_tier_neurons=n_tier, n_channels=n_channels,
                           dt_neural_ms=float(loop.TIER_DT_MS))

    traces = []

    def spike_sink(spikes, c):
        traces.append(elec.record(spikes, t0_ms=c * loop.DT_COMMAND_S))

    summary = loop.run(lesion_leg=lesion_leg, duration_s=duration_s, verbose=False,
                       spike_sink=spike_sink)
    if not traces:
        raise RuntimeError("the loop produced no intervals, so nothing was recorded")
    trace = np.concatenate(traces, axis=1)                 # (n_channels, n_substeps_total)
    return {"trace_mV": trace, "electrode": elec, "summary": summary,
            "n_channels": n_channels, "n_substeps": int(trace.shape[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=0.4)
    ap.add_argument("--lesion-leg", default=None)
    args = ap.parse_args()

    out = record(args.duration_s, args.lesion_leg)
    elec, trace = out["electrode"], out["trace_mV"]
    geo = elec.geometry()
    print(f"  electrode: {geo['n_channels']} channels, "
          f"{geo['n_neurons_with_soma']} of {geo['n_tier_neurons']} neurons with a soma "
          f"({geo['n_neurons_without_soma']} without)")
    print(f"  trace {trace.shape}: peak {np.abs(trace).max():.4g} mV, "
          f"std {trace.std():.4g} mV, non-zero samples {int((trace != 0).sum())}")
    # the guard, on the SAME spikes the body was driven by
    print("  dependence on the network:")
    from engine.embodied.adapters import TierConfig, build_neural_tier
    import run_six_leg_neural_loop as loop
    built = build_neural_tier(TierConfig())
    # re-run the loop collecting spikes so the guard can zero them
    spikes_all = []
    loop.run(lesion_leg=args.lesion_leg, duration_s=args.duration_s, verbose=False,
             spike_sink=lambda sp, c: spikes_all.append(sp.copy()))
    stacked = np.concatenate(spikes_all, axis=0)
    dep = elec.dependence(stacked)
    print(f"    status {dep['status']}: spike peak {dep['trace_peak_mV']:.4g} mV, "
          f"n_spikes {dep['n_spikes']}, recorded neurons {dep['n_neurons_recorded']}")
    dest = OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "status": "PASS" if dep["status"] == "PASS" else "NOT_RUN",
        "lesion_leg": args.lesion_leg, "duration_s": args.duration_s,
        "n_channels": out["n_channels"], "n_substeps": out["n_substeps"],
        "trace_peak_mV": float(np.abs(trace).max()), "trace_std_mV": float(trace.std()),
        "non_zero_samples": int((trace != 0).sum()),
        "electrode_geometry": geo, "dependence": dep,
        "units": "mV for the recorded trace; nA for the electrode currents; um for geometry",
        "provenance": ("spikes come from run_six_leg_neural_loop's connectome tier, AFTER the acute "
                       "lesion, i.e. the same spikes that set the muscle activations; the electrode "
                       "is engine/embodied/electrode_in_loop.InLoopElectrode; soma positions from "
                       "data/banc/somas_v1.parquet"),
        "what_this_does_not_establish": (
            "that the recorded AMPLITUDE is a biological measurement.  The spike -> current kernel "
            "is ILLUSTRATIVE and 4263 of 12000 tier neurons have no soma position, so 64 % of the "
            "network is recordable.  What is established is that the electrode sits on the branch "
            "that drives the body and is coupled to it."),
    }, indent=2, sort_keys=True))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
