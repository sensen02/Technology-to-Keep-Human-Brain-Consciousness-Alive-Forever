"""Test nearby landing settings on previously seen phases to find one common controller."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from land import Approach, commands_from, launch_settings, save, setup_policy, simulate
from verify_landing import gates, measure


def tune(neck, base, output, rng_seed, count, first):
    facts, commands = commands_from(neck)
    policy, stock, observation = setup_policy()
    rng = np.random.default_rng(rng_seed)
    start = Approach(**json.loads(base.read_text()))
    phases = (first, *(phase for phase in (17, 29, 53) if phase != first))
    output.mkdir()
    with (output / 'QUALIFICATION.jsonl').open('x') as scores:
        for i in range(count):
            def around(value, span):
                return value + rng.uniform(-span, span)
            p = replace(start, brake_s=around(start.brake_s, .012),
                        descent_s=around(start.descent_s, .029),
                        descent_delay_s=around(start.descent_delay_s, .012),
                        flare_pitch_deg=around(start.flare_pitch_deg, 6.),
                        reach_cm=around(start.reach_cm, .015),
                        foot_level=rng.uniform(.81, 1.),
                        deploy_cm=around(start.deploy_cm, .050),
                        fold_precontact_cm=rng.uniform(.147, .202),
                        fold_s=rng.uniform(.006, .016),
                        neutral_after_contact_s=rng.uniform(.00035, .004),
                        stance_blend_s=rng.uniform(.035, .070),
                        synchronized_fold=int(rng.choice([0, 1])),
                        sweep_rad=around(start.sweep_rad, .042))
            outcome = {}
            run = None
            for phase in phases:
                run = simulate(policy, stock, observation, commands['intact'],
                               facts['dt_ms'], launch_settings(phase), p)
                metrics = measure(run[0]['TRACE.npy'], run[1], .4214)
                check = gates(metrics)
                outcome[str(phase)] = dict(check, first_contact_ms=metrics['first_contact_ms'],
                                           impacts=metrics['nonfoot_contact_native_steps'],
                                           six=metrics['last_100ms_min_feet'],
                                           final_height=metrics['last_100ms_height_mm'],
                                           pass_all=all(check.values()))
                if not all(check.values()):
                    break
            scores.write(json.dumps({'index': i, 'params': run[1]['approach'], 'phases': outcome},
                                    allow_nan=False) + '\n')
            scores.flush()
            if sum(row['pass_all'] for row in outcome.values()) >= 2:
                save(output / f'candidate_{i:02}', run)
                print(i, outcome, flush=True)
            if len(outcome) == 3 and all(row['pass_all'] for row in outcome.values()):
                (output / 'PASS_CONFIG.json').write_text(json.dumps(run[1]['approach'], indent=2) + '\n')
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--count', type=int, default=20)
    a = parser.parse_args()
    tune(a.neck, a.base, a.output, a.seed, a.count, a.first)
