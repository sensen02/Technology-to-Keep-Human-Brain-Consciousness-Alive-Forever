"""Calibrate the landing phase using seed 17 while preserving every score."""
import argparse
import json
from pathlib import Path

import numpy as np

from land import Approach, commands_from, launch_settings, save, setup_policy, simulate


def measure(result):
    x = result[0]['TRACE.npy']
    approach = x[x[:, 0] > .4214]
    hit = np.flatnonzero(approach[:, 10:12].sum(axis=1) > 0)
    settled = x[x[:, 0] > x[-1, 0] - .1]
    if not len(hit):
        return -25., {}
    first = approach[hit[0]]
    if hit[0] > 0:
        velocity = approach[hit[0] - 1, 4:7] * 10
    else:
        velocity = first[4:7] * 10
    impacts = int(np.count_nonzero(approach[:, 11]))
    feet = float(settled[:, 10].min())
    max_speed = float((np.linalg.norm(settled[:, 4:7], axis=1)*10).max())
    speed = float(np.linalg.norm(velocity[:2]))
    descent = float(abs(velocity[2]))
    angles = np.rad2deg(first[8:10])
    score = (2. * feet - min(40, impacts) * .12 - impacts * .004
             - .024 * max(0., descent - 65.) - .012 * max(0., speed - 70.)
             - .008 * (abs(angles[0]) + abs(angles[1])) - .02 * max_speed
             - .6 * (first[11] > 0.) - .9 * abs(10. * settled[-1, 3] - .905))
    return float(score), {'nonfoot': impacts, 'feet_last_min': feet, 'contact_s': float(first[0]),
                          'horizontal_before_mm_s': speed, 'vertical_before_mm_s': descent,
                          'abs_max_speed_last_mm_s': max_speed, 'contact_roll_and_pitch': angles.tolist()}


def search(seed, count, neck, output):
    info, commands = commands_from(neck)
    policy, stock, observation = setup_policy()
    launch = launch_settings(17)
    rng = np.random.default_rng(seed)
    output.mkdir()
    best = -np.inf
    with (output / 'SWEEP.jsonl').open('x') as stream:
        for index in range(count):
            params = Approach(brake_s=rng.uniform(.140, .265), descent_delay_s=rng.uniform(.015, .080),
                              descent_s=rng.uniform(.255, .385), final_height_cm=rng.uniform(.061, .105),
                              flare_delay_s=rng.uniform(.045, .114), flare_s=rng.uniform(.13, .23),
                              flare_pitch_deg=rng.uniform(5, 32), deploy_cm=rng.uniform(.41, .67),
                              deploy_tau_s=rng.uniform(.008, .021), foot_level=rng.uniform(.79, 1.),
                              reach_cm=rng.uniform(.035, .105), fold_s=rng.uniform(.003, .015),
                              fold_precontact_cm=rng.uniform(.113, .257),
                              stance_blend_s=rng.uniform(.020, .064),
                              neutral_after_contact_s=rng.uniform(.0002, .004),
                              sweep_rad=rng.uniform(-.13, .20), near_amplitude=rng.uniform(.76, 1.))
            result = simulate(policy, stock, observation, commands['intact'], info['dt_ms'], launch, params)
            score, metrics = measure(result)
            item = {'index': index, 'score': score, 'metrics': metrics, 'params': result[1]['approach']}
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            stream.flush()
            if score > best:
                best = score
                save(output / f'best_{index:02}', result)
                print(item, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rng', type=int, required=True)
    parser.add_argument('--count', type=int, default=14)
    args = parser.parse_args()
    search(args.rng, args.count, args.neck, args.output)
