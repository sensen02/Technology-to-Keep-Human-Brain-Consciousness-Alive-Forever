"""Independent check of food, stable completion and the actual BANC transport."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np


CASES=('FINAL_FLAT_A','FINAL_FLAT_B','WIDE_L','WIDE_R','BRIDGE_L','BRIDGE_R','LONG_L','LONG_R','NO_FOOD')
THRESHOLD=.18
BRAIN_SHA='71904e7a9765e5b0fb58afbe9c0ce27efa591f84997b51f0996f58a6eafbeaf3'


def inspect(folder):
    r=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'NECK.npz',allow_pickle=False) as f:
        emission,arrival,delays=(f[key] for key in ('emitted','arrived','delays'))
        assert f['weights'].item()==r['model_sha']==BRAIN_SHA
        assert len(emission)==round(r['duration_s']/.0001)
        assert len(delays)==17
        assert all(not arrival[:d,i].any() and np.array_equal(emission[:-d,i],arrival[d:,i])
                   for i,d in enumerate(delays))
        asc=int(f['ascending'].sum())
    with np.load(folder/'BODY.npz',allow_pickle=False) as f:
        pose,trace,meal,obs=(f[key] for key in ('poses','trace','meal','observation'))
        assert np.isfinite(pose).all() and np.isfinite(trace).all() and np.isfinite(meal).all()
        assert abs(meal[:,4].sum()-r['stock_loss'])<1e-10
        assert abs(meal[-1,1]-r['intake'])<1e-10
        assert np.array_equal(r['intake_target_pass'],meal[:,4].sum()>=THRESHOLD)
    tail=trace[trace[:,0]>=r['duration_s']-.14]
    speed=np.linalg.norm(tail[:,7:9],axis=1)
    assert len(tail)>=27
    steady=bool(np.all((tail[:,4]>=4)&(tail[:,6]>.94)&(speed<=.08)))
    assert r['ended_standing']==steady
    assert r['meal_pass']==(r['intake_target_pass'] and steady and r['first_fall_s'] is None)
    assert r['native_extra_force']==0.
    cfg=r['geometry']
    attained=False
    margin=None
    first=None
    max_h=None
    if cfg['side']:
        approached=np.flatnonzero(trace[:,1]>=cfg['near']-.045)
        if len(approached):
            row=trace[approached[0]]
            margin=float(10*(cfg['side']*row[2]-cfg['edge']))
            mask=obs[:,0]<=row[0]
            if mask.any():
                max_h=float(obs[mask,5:11].max())
        for row in trace[trace[:,1]>=cfg['far']]:
            if row[3]>.07 and row[6]>.85 and cfg['side']*row[2]>=cfg['edge'] and row[4]>=3:
                attained=True
                first=float(row[0])
                break
    return r | dict(neural_steps=len(emission),neck_events=int(arrival.sum()),front_an_events=asc,
                    independent_stock=True,independent_stand=True,banc_delays_ms=(delays*.1).tolist(),
                    supported_side_cross=attained,side_cross_time_s=first,lip_margin_mm=margin,
                    pre_lip_visual_risk=max_h,last_140ms_peak_speed_mm_s=float(10*speed.max()),
                    last_140ms_min_feet=int(tail[:,4].min()))


def run(root,partial):
    result={}
    groups=defaultdict(list)
    for name in CASES:
        f=root/name
        if not (f/'RESULT.json').is_file():
            if partial:
                continue
            raise FileNotFoundError(f)
        row=inspect(f)
        result[name]=row
        groups[row['challenge'] if row['food'] else 'no_food'].append(row)
    assert result
    rates={}
    for group,rows in groups.items():
        rates[group]=dict(n=len(rows),sustained_meal=sum(r['meal_pass'] for r in rows),
                          ingested_threshold=sum(r['intake_target_pass'] for r in rows),
                          correct_physical_cross=sum(r['supported_side_cross'] for r in rows),
                          falls=sum(r['first_fall_s'] is not None for r in rows),
                          total_stock_drop=sum(r['stock_loss'] for r in rows))
    (root/'FULL_LOOP.json').write_text(json.dumps(dict(cases=result,by_difficulty=rates,
        success='≥0.18 intake with no fall and stable final 140 ms',complete=len(result)==len(CASES)),
        indent=2,allow_nan=False)+'\n')
    print(json.dumps(rates,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.root,args.partial)
