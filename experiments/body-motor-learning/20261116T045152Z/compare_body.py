"""Describe matched physical gaits without changing the established behavior gates."""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261116T000536Z"))
from walking_bench import LEGS, W, cycles, read, stand


def summarise(path):
    result = read(path / "RESULT.json")
    file = path / "TRACE.npz"
    if stand.sha(file) != result["trace_sha256"]:
        raise ValueError(f"Trajectory fingerprint mismatch in {path}")
    with np.load(file, allow_pickle=False) as f:
        data = {name: f[name] for name in ("time", "loads", "padz", "foot", "com", "speed")}
    walk = result.get("walk_ms", 9600)
    start = 500
    end = start + walk
    if end >= len(data["time"]):
        return dict(run=str(path), completed_walking_window=False, saved_milliseconds=len(data["time"])-1)
    counts, events = cycles(data, start, end, arm_before=20)
    active = slice(start+1, end+1)
    forces = data["loads"][active]
    total_load = forces.sum(axis=0)
    numer = (forces * data["speed"][active]).sum(axis=0)
    legs = {}
    length = stand.LENGTH
    time0 = float(data["time"][start])
    for i, name in enumerate(LEGS):
        complete = [x for x in events if x["type"] == "complete" and x["leg"] == name]
        onsets = [float(x["start_time"]) - time0 for x in complete]
        cadence = float(1./np.mean(np.diff(onsets))) if len(onsets) >= 2 else None
        legs[name] = dict(
            qualified_cycles=counts[name], qualified_swing_onsets_s=onsets,
            qualified_cadence_hz=cadence,
            loaded_fraction_at_point_one_W=float(np.mean(forces[:,i] >= .10*W)),
            avg_load_W=float(forces[:,i].mean() / W),
            avg_loaded_slip_body_lengths_per_s=float(numer[i] / total_load[i] / length) if total_load[i]>0 else None,
            advances_body_lengths=[float(event["advance"]/length) for event in complete],
            peak_clearances_body_lengths=[float(event["lift"]/length) for event in complete],
        )
    delta = data["com"][end] - data["com"][start]
    return dict(run=str(path), completed_walking_window=True, walk_ms=walk,
                recorded_walk_gates=result.get("walk"), recorded_stand=result.get("terminal_stand"),
                forward_body_lengths_per_s=float(delta[0]/(walk*.001)/length),
                lateral_drift_body_lengths=float(delta[1]/length),
                per_leg=legs, sampling_ms=1, notes="Descriptive estimates; threshold load is not a full stance phase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    stand.put(args.output, [summarise(path.resolve()) for path in args.runs])
