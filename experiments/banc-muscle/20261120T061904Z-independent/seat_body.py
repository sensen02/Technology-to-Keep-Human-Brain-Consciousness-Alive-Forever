"""Bounded initial height diagnostic on the unmodified muscle body."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from body_observations import BASE, BodyObservations, MODEL, load_pose, run, summary


def main(out):
    model = mj.MjModel.from_xml_path(str(MODEL))
    observations = BodyObservations(model)
    with np.load(BASE / 'body/archived_pose.npz', allow_pickle=False) as a:
        pose = a['qpos']
    attempts = []
    chosen = None
    for offset in np.linspace(0., -.010, 21):
        q = pose.copy()
        q[2] += offset
        data = load_pose(model, q)
        loads = observations.measure(data)['ground_load_w']
        feet, nonfoot = loads[:-1], loads[-1]
        floor_depths = [float(c.dist) for c in data.contact if observations.floor in c.geom]
        min_gap = min(floor_depths, default=0.)
        passed = feet.min() > .01 and .98 <= feet.sum() <= 1.02 and nonfoot == 0. and min_gap >= -.012
        attempts.append(dict(root_delta_mm=float(offset), initial_foot_weight=float(feet.sum()),
            loaded_legs=int((feet > .01).sum()), min_floor_distance_mm=min_gap, accepted=bool(passed)))
        if passed and chosen is None:
            chosen = q
    result = dict(candidates=attempts, accepted_pose=chosen.tolist() if chosen is not None else None,
                  native_steps=0, trials={})
    if chosen is not None:
        assert np.array_equal(chosen[3:], pose[3:]) and np.array_equal(chosen[:2], pose[:2])
        with np.load(BASE / 'results/static_archived_six_feet_all_muscles.npz', allow_pickle=False) as a:
            prior_command = a['solution'][:model.nu]
        for label, constant in (('passive', None), ('constant_LP', prior_command)):
            trace = run(model, observations, chosen, constant)
            result['trials'][label] = summary(trace)
            result['native_steps'] += len(trace['time']) - 1
            np.savez_compressed(out / f'BODY_seated_{label}.npz', **trace)
    (out / 'SEATING.json').write_text(json.dumps(result, indent=2) + '\n')
    print([row for row in attempts if row['accepted']], result['trials'])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
