"""Measure actual short turn and withdrawal on the frozen physical body."""
import argparse
from pathlib import Path

import numpy as np

from bench_runtime import arrays, check_forces, prepare, sample, stand_window, turn


def trial(path, start, yaw, pulse):
    body, policy = prepare(path, start)
    begin = 500
    release = begin + pulse
    duration = release + 1200
    measured = body.measure()
    rows = [sample(body, measured)]
    for tick in range(duration):
        if tick % 5 == 0:
            observation = turn.features(body, measured)
            action = policy(observation, 0, yaw if begin <= tick < release else 0)
        measured = body.tick(action)
        check_forces(body)
        rows.append(sample(body, measured))
    trace = arrays(rows)
    trace["yaw"] = np.unwrap(trace["yaw"])
    angle = np.rad2deg(trace["yaw"] - trace["yaw"][begin])
    onset = np.flatnonzero(yaw*angle[begin:] >= 1.)
    settled = stand_window(trace, duration)
    stable = next((i-release for i in range(release+200, duration+1, 5)
                   if stand_window(trace, i)["passed"]), None)
    compact = path / "TRACE.npz"
    np.savez_compressed(compact, **trace)
    result = dict(start=start, yaw=yaw, pulse_ms=pulse, signed_angle_deg=float(angle[-1]),
                  angle_at_release_deg=float(angle[release]), extra_angle_deg=float(angle[-1]-angle[release]),
                  onset_ms=int(onset[0]) if len(onset) else None, earliest_stand_ms=stable,
                  stop=settled, initial_stand=stand_window(trace, begin), nonfoot_peak=float(trace["nonfoot"].max()),
                  drift_bl=float(np.linalg.norm(trace["com"][begin:,:2]-trace["com"][begin,:2],axis=1).max()/turn.stand.LENGTH),
                  trial_native_steps=80*duration, setup_native_steps=80000, trace_sha256=turn.stand.sha(compact))
    turn.stand.put(path / "RESULT.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("yaw", type=int, choices=(-1, 0, 1))
    parser.add_argument("pulse_ms", type=int)
    parser.add_argument("--start", default="nominal")
    args = parser.parse_args()
    trial(args.output, args.start, args.yaw, args.pulse_ms)
