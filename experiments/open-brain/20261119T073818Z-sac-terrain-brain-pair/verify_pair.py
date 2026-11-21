"""Independently reload both SAC branches and cross each actor with each brain."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from balanced import BalancedMotor
from forage import Course, HORIZON, MEAL
from policy_runtime import actor_mean


def evaluate(policy, motor, *, food=True, level=2):
    env=Course(motor)
    cases=[]
    for seed in range(92360,92424):
        side=1 if seed%2 else -1
        obs=env.reset(seed,level,side=side,food=food)
        reward=0.
        trail=[env.position.copy().tolist()]
        for _ in range(HORIZON):
            decision=actor_mean(policy,obs)
            obs,r,done,truncated,info=env.step(decision)
            reward+=r
            trail.append(env.position.copy().tolist())
            if done or truncated:
                break
        consumed=.9-env.state.patch.volume if food else 0.
        assert abs(info['ingested']-consumed)<1e-12
        assert info['success']==(info['ingested']>=MEAL)
        cases.append(info | dict(seed=seed,side=side,total=reward,
                                 trail=trail,terrain=asdict(env.terrain)))
    return cases


def summary(cases):
    result=dict(n=len(cases),success=sum(c['success'] for c in cases),
                falls=sum(c['fall'] for c in cases),right_side=sum(c['crossed_side']==c['side'] for c in cases),
                food_avg=float(np.mean([c['ingested'] for c in cases])),
                mean_reward=float(np.mean([c['total'] for c in cases])))
    return result | {'by_open_side':{str(i):{'success':sum(c['success'] for c in cases if c['side']==i),
                   'fall':sum(c['fall'] for c in cases if c['side']==i)} for i in (-1,1)}}


def main(root):
    cal=json.loads((root/'NEURAL_CAL/CALIBRATION.json').read_text())
    dirs={a:root/(a.upper()+'_BALANCED') for a in ('frozen','tuned')}
    calibration={'frozen':cal['frozen_gain'],'tuned':cal['tuned_gain']}
    motors={name:BalancedMotor(cal['trials'][str(gain)]['rows'])
            for name,gain in calibration.items()}
    initial={}
    policies={}
    for name,path in dirs.items():
        with np.load(path/'policy.npz',allow_pickle=False) as f:
            policies[name]={a:f[a] for a in f.files}
        s0=torch.load(path/'00000.pt',weights_only=True,map_location='cpu')
        s1=torch.load(path/'24000.pt',weights_only=True,map_location='cpu')
        identity=json.loads(policies[name]['identity'].item())
        assert s1['identity']==identity and identity['arm']==name
        assert s1['updates']==11701
        assert all(not torch.equal(v1,s0['actor'][k]) for k,v1 in s1['actor'].items())
        initial[name]=s0['actor']
    assert all(torch.equal(t,initial['tuned'][k]) for k,t in initial['frozen'].items())
    assert cal['original_weights_sha256']!=cal['tuned_weights_sha256']
    matching={}
    matrix={}
    for policy,net in policies.items():
        for brain,motor in motors.items():
            rows=evaluate(net,motor)
            matrix[f'{policy}_actor_{brain}_brain']=summary(rows)
            if policy==brain:
                matching[brain]=rows
    controls={}
    for arm,net in policies.items():
        controls[arm]={'on_flat':summary(evaluate(net,motors[arm],level=0)),
                       'without_food':summary(evaluate(net,motors[arm],food=False))}
    report=dict(heldout_seed_range=[92360,92423],arena='navigation surrogate',
                matched_initial_actor=True,table_comparison=matrix,controls=controls,
                paired_cases=matching)
    (root/'PAIRED_HELDOUT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(matrix,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    main(args.output)
