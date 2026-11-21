"""Reconstruct the frozen head's post-meal walking gate from recorded inputs."""
import argparse
import json
from pathlib import Path

import numpy as np


def examine(directory):
    facts=json.loads((directory/'RESULT.json').read_text())
    observations=[json.loads(row) for row in (directory/'EYES.jsonl').read_text().splitlines()]
    meals=np.load(directory/'MEAL.npy',allow_pickle=False)
    trace=np.load(directory/'TRACE.npy',allow_pickle=False)
    with np.load(directory/'NEURAL.npz',allow_pickle=False) as neural:
        gate=neural['head'][:,4].astype(bool)
    indexes=np.flatnonzero(meals[:,2]>0.)
    if not len(indexes):
        raise ValueError('Expected a measured meal in the ground extension')
    last=float(meals[indexes[-1],0])
    taste=0.
    blocked_with_visible_low=[]
    state=[]
    launch=facts['takeoff_s']
    for row,allow in zip(observations,gate,strict=True):
        t=row['time_s']
        launched=launch is not None and t>launch
        taste=taste+.02 if row['contact'] and not launched else max(0.,taste-.25*.02)
        could_see_ground=row['seen'] and row['height_probability']<.8 and not launched
        assert bool(could_see_ground and taste<.115)==allow
        if t>=last and could_see_ground and taste>=.115:
            blocked_with_visible_low.append(t)
        state.append(taste)
    time_after=[i for i,row in enumerate(observations) if row['time_s']>=last]
    resumed=[observations[i]['time_s'] for i in time_after if gate[i]]
    post_trace=trace[trace[:,0]>=last]
    travel=10.*np.linalg.norm(post_trace[:,1:3]-post_trace[0,1:3],axis=1).max()
    return dict(last_intake_s=last,seconds_post_meal=facts['elapsed_s']-last,
                first_post_meal_walk_allowed_s=resumed[0] if resumed else None,
                latch_at_first_post_meal_packet=state[time_after[0]],latch_at_last_packet=state[-1],
                visible_low_but_walk_gate_closed_packets=len(blocked_with_visible_low),
                inferred_remaining_decay_s=max(0.,(state[-1]-.115)/.25),
                max_horizontal_move_after_last_intake_mm=float(travel),
                support_count_final=trace[-1,4:7].astype(int).tolist(),
                mass_positive_food_sources=[name for name,value in facts['intake'].items() if value>0.])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(examine(args.case),indent=2)+'\n')
