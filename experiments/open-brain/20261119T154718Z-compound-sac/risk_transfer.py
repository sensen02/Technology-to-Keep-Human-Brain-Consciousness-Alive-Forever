"""Compare the live optical risk stream to the recorded pose's course geometry."""
import argparse
import json
from pathlib import Path

import numpy as np

from terrain_spec import HERE, make_terrain
from common import desired_hazard
from controller import infer


def read(path):
    return json.loads(path.read_text())


def yaw(q):
    w,x,y,z=q
    return float(np.arctan2(2.*(w*z+x*y),1.-2.*(y*y+z*z)))


def first(time, mask):
    indices=np.flatnonzero(mask)
    return float(time[indices[0]]) if len(indices) else None


def inspect(path):
    r=read(path/'AUDIT.json')
    with np.load(r['policy'],allow_pickle=False) as file:
        policy={key:file[key] for key in file.files}
    with np.load(path/'BODY.npz',allow_pickle=False) as body:
        poses,trace,observed=(body[k] for k in ('poses','trace','observation'))
    cfg=make_terrain(r['seed'],r['challenge'],r['side'])
    times=observed[:,0]
    idx=np.maximum(0,np.searchsorted(trace[:,0],times,side='right')-1)
    gate=cfg.gates[0]
    before_lip=trace[idx,1]<gate.near
    want=[]
    revised=[]
    for row,q in zip(observed,poses[idx],strict=True):
        ideal=desired_hazard(cfg,q[:2],yaw(q[3:7]))
        want.append(ideal)
        sense=row[1:15].astype(np.float32)
        sense[4:10]=ideal
        revised.append(infer(policy,sense,profile='strong')[1])
    actual=observed[:,5:11]
    target=np.array(want)
    threshold=.25
    geometry_gate=(target.max(axis=1)>=threshold)&before_lip
    camera_gate=(actual.max(axis=1)>=threshold)&before_lip
    return dict(
        challenge=r['challenge'], seed=r['seed'], side=r['side'], first_fall_s=r['first_fall_s'],
        lip=r['gates'][0], threshold=threshold,
        evaluated_pre_lip_frames=int(before_lip.sum()),
        expected_discrimination_frames=int(geometry_gate.sum()),
        camera_detected_frames=int(camera_gate.sum()),
        camera_hit_frames=int(np.count_nonzero(geometry_gate&camera_gate)),
        first_geometry_risk_s=first(times,geometry_gate),first_camera_risk_s=first(times,camera_gate),
        observed=actual.tolist(), geometric_reference=target.tolist(), t=times.tolist(),
        exact_xy_cm=trace[idx,1:3].tolist(),
        live_commands=observed[:,17:19].tolist(),offline_geometry_commands=np.array(revised).tolist(),
        note='Geometric projection is an offline reference at replayed poses; it is not an input to an evaluated policy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms',choices=('BASELINE','SELECTED'),nargs='+')
    args=parser.parse_args()
    rows={arm:{name:inspect(HERE/f'VISIBLE_{arm}'/name) for name in ('SLOPE_L','COMPOUND_L')}
          for arm in args.arms}
    prefix='BOTH' if len(args.arms)>1 else args.arms[0]
    (HERE/f'RISK_TRANSFER_{prefix}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    for arm,cases in rows.items():
        for name,r in cases.items():
            print(arm,name,r['first_geometry_risk_s'],r['first_camera_risk_s'],
                  r['camera_hit_frames'],r['expected_discrimination_frames'],flush=True)
