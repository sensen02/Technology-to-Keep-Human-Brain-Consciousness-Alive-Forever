"""Identify continuous P1 neck response from closed-loop, frozen-Eon recordings."""
import argparse
import json
from pathlib import Path

import numpy as np

from controller import HERE, PAIR, decode_levels
from forage import MotorTable
from policy_runtime import decode_policy


def raw_episode(folder,old=False):
    if old:
        with np.load(folder/'LIVE.npz',allow_pickle=False) as body:
            observations=body['observation'].copy()
        identity=json.loads((folder/'RESULT.json').read_text())
        with np.load(identity['checkpoint'],allow_pickle=False) as data:
            policy={k:data[k] for k in data.files}
        steps=[]
        for row in observations:
            _,command=decode_policy(policy,row[1:15].astype(np.float32))
            steps.append((row[0],*command))
        steps=np.asarray(steps)
        with np.load(folder/'NECK.npz',allow_pickle=False) as data:
            drive=data['drive'].copy()
            asc=data['ascending'].copy()
    else:
        with np.load(folder/'BODY.npz',allow_pickle=False) as body:
            steps=body['commands'][:,:3].copy()
        with np.load(folder/'NECK.npz',allow_pickle=False) as data:
            drive=data['drive'].copy()
            asc=data['ascending'].copy()
    return steps,drive,asc


def prepare(source,motor):
    steps,drive,asc=source
    # Prediction target is mean next 80 ms. Sources to this predictor include
    # the scheduled intention and only earlier neck readout and arrivals.
    group=800
    inputs,goal,baseline=[],[],[]
    for q in range(2*group,len(drive)-group+1,group):
        i=np.searchsorted(steps[:,0],q*.0001+1e-9,side='right')-1
        a,t=steps[i,1:]
        l,r=drive[q-group:q].mean(axis=0)
        vl,vr=drive[q-2*group:q-group].mean(axis=0)
        t1,t2=asc[q-group:q].mean(axis=0)*60.
        features=[1.,a,t,a*a,t*t,a*t,l,r,vl,vr,t1,t2,t*(l-r),a*(l+r)]
        inputs.append(features)
        goal.append(decode_levels(*drive[q:q+group].mean(axis=0)))
        # The v1 static-response model inferred left/right through its idealized drive.
        pairs=np.array([[np.interp(t,motor.turns,motor.table[j,:,k])
                         for j in range(len(motor.paces))] for k in range(2)])
        predicted=[np.interp(a,np.r_[0.,motor.paces],np.r_[0.,each]) for each in pairs]
        baseline.append(decode_levels(*predicted))
    return np.asarray(inputs),np.asarray(goal),np.asarray(baseline)


def fitted(rows):
    a,y,_=map(np.vstack,zip(*rows,strict=True))
    w=np.linalg.solve(a.T@a+np.eye(a.shape[1])*.20,a.T@y)
    return w


def run(output):
    cal=json.loads((PAIR/'NEURAL_CAL/CALIBRATION.json').read_text())
    m=MotorTable(cal['trials']['1.0']['rows'])
    episodes={name:prepare(raw_episode(folder,old),m) for name,folder,old in (
              ('previous_frozen_flat',PAIR/'LIVE_FROZEN_FLAT',True),
              ('previous_frozen_gap',PAIR/'LIVE_FROZEN_GAP',True),
              ('new_flat_development',HERE/'FLAT_DEV',False))}
    checks={}
    for name,(x,y,baseline) in episodes.items():
        w=fitted([r for k,r in episodes.items() if k!=name])
        new=x@w
        checks[name]=dict(windows=len(y),static_mae=np.abs(y-baseline).mean(axis=0).tolist(),
                          context_mae=np.abs(y-new).mean(axis=0).tolist())
    w=fitted(list(episodes.values()))
    np.savez_compressed(output.with_suffix('.npz'),weights=w)
    output.write_text(json.dumps(dict(columns=['cm/s','rad/s'],leave_whole_episode_out=checks,
        status='sensitivity probe only, separate from saved SAC actor'),indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
