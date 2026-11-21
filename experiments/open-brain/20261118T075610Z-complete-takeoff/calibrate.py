"""Calibrate launch timing on seed 17 and preserve every trial summary."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from takeoff import BASELINE, COLUMNS, Settings, load_neck, load_policy, make_env, save, simulate


def score(result):
    trace = result[0]
    late = trace[trace[:, 0] >= .18]
    if not len(late):
        return -100.
    impacts = np.count_nonzero(trace[:, 11])
    relanded = np.count_nonzero(late[:, 10])
    height = np.percentile(late[:, 3], 10)
    roll = np.percentile(abs(np.rad2deg(late[:, 8])), 95)
    yaw = np.percentile(abs(np.rad2deg(late[:, 7])), 95)
    return float(2 * height + late[-1, 1] - .02 * roll - .018 * yaw
                 - .005 * impacts - .0017 * relanded)


def main(output, rng_seed, count):
    stock = make_env()
    observation = stock.reset().observation
    policy = load_policy(BASELINE / 'flight_reference/flight')
    receipt, levels = load_neck()
    rng = np.random.default_rng(rng_seed)
    output.mkdir()
    highest = -np.inf
    with (output / 'SWEEP.jsonl').open('x') as lines:
        for index in range(count):
            config = replace(Settings(), duration=.28,
                             wing_delay=rng.uniform(.006, .021),
                             wing_ramp=rng.uniform(.007, .020),
                             teacher_delay=rng.uniform(.008, .028),
                             teacher_ramp=rng.uniform(.005, .027),
                             teacher_gain=rng.uniform(.55, 1.),
                             symmetry_time=rng.choice([0., 0., .015, .025]),
                             flight_pitch_deg=rng.uniform(29, 64),
                             pitch_ramp=rng.uniform(.007, .040),
                             speed_ramp=rng.uniform(.018, .090),
                             climb_time=rng.uniform(.074, .150),
                             climb_cm=rng.uniform(.48, 1.16),
                             tuck_tau=rng.uniform(.0035, .0140),
                             kick_rad=rng.uniform(.92, 1.48),
                             kick_hold=rng.uniform(.017, .043))
            result = simulate(policy, stock, observation, levels['takeoff_probe'], receipt['dt_ms'], config)
            value = score(result)
            entry = dict(index=index, score=value, **result[-1]['settings'])
            lines.write(json.dumps(entry, allow_nan=False) + '\n')
            lines.flush()
            if value > highest:
                highest = value
                save(output / f'best_{index:02}', result)
                print('new best', entry, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    main(args.output, args.seed, args.count)
