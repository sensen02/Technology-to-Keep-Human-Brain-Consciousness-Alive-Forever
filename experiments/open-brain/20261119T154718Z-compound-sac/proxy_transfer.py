"""Replay physical test seeds in the training proxy to explain transfer gaps."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_timed import HERE, HORIZON, new_env
from policy_runtime import actor_mean


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def run(root, prior):
    plan = json.loads((HERE/'physical_plan.json').read_text())
    policies = {arm:load(path) for arm,path in
                (('before',prior), ('after', root/'POLICY.npz'))}
    rows = {}
    for case in plan['cases']:
        name = case['name']
        rows[name] = {}
        for label, policy in policies.items():
            env = new_env()
            sense = env.reset(case['seed'], case['challenge'], case['side'])
            path, samples = [], []
            info = None
            for _ in range(HORIZON):
                action = actor_mean(policy, sense)
                before = env.position.copy()
                x = sense.copy()
                sense, _, done, cut, info = env.advance_actor(sense,action)
                path.append(before.tolist())
                samples.append(x.tolist())
                if done or cut:
                    break
            rows[name][label] = info | dict(
                step_count=len(path), initial_goal_detected=bool(samples[0][0]),
                goal_detected_fraction=float(np.mean([x[0] for x in samples])),
                path_cm=path, observation=samples)
    (HERE/'PROXY_PROBES.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    parser.add_argument('root',type=Path,default=HERE/'BUDGET_1800',nargs='?')
    args = parser.parse_args()
    run(args.root,args.prior)
