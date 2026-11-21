"""Rescore the unsafe approach windows from six archived physical bridge failures."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

from bridge_reward import CFG, Snapshot, clearance, cost, smoke

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261119T084744Z-closed-loop-forage'
sys.path.insert(0,str(PREV))
from controller import Terrain


def evaluate(output):
    cases=json.loads((PREV/'FULL_LOOP.json').read_text())['cases']
    rows=[]
    for name,record in cases.items():
        t=Terrain(**record['geometry'])
        if not t.side or not record['food']:
            continue
        with np.load(PREV/name/'BODY.npz',allow_pickle=False) as saved:
            trace=saved['trace']
        inspected=[]
        for before,after in zip(trace[:-32:32],trace[32::32],strict=True):
            bx,by=before[1:3]
            ax,ay=after[1:3]
            if ax<=bx or ax<t.near-.24 or ax>t.near-.02 or                     min(clearance(t,by),clearance(t,ay))>=CFG.foot_margin_cm:
                continue
            if after[0]>=record['first_fall_s']:
                continue
            first=Snapshot(float(bx),float(by),0.)
            last=Snapshot(float(ax),float(ay),0.)
            old=.70*(t.distance(np.array((bx,by)))-.985*t.distance(np.array((ax,ay))))
            new=cost(t,first,last,crossed_before=False,crossed_now=False,success=False,
                     terminal=False,truncated=False,speed=0.,turn=0.)
            reflected=cost(replace(t,side=-t.side,goal_y=-t.goal_y),replace(first,y=-first.y),
                           replace(last,y=-last.y),crossed_before=False,crossed_now=False,
                           success=False,terminal=False,truncated=False,speed=0.,turn=0.)
            assert reflected==new
            inspected.append(dict(time_s=float(after[0]),advance_mm=float((ax-bx)*10.),
                                  clearance_mm=float(clearance(t,ay)*10.),old_shaping=float(old),
                                  new_shaping=float(new.route+new.unsafe_advance+new.overhang)))
        if inspected:
            rows.append(dict(name=name,samples=inspected))
    flat=[v for row in rows for v in row['samples']]
    aggregate=dict(n=len(flat),
        old_positive=sum(row['old_shaping']>0 for row in flat),
        new_positive=sum(row['new_shaping']>0 for row in flat),
        old_mean=float(np.mean([row['old_shaping'] for row in flat])),
        new_mean=float(np.mean([row['new_shaping'] for row in flat])))
    result=dict(aggregate=aggregate,paired_windows=rows,synthetic_probes=smoke())
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(aggregate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    evaluate(parser.parse_args().output)
