"""Freeze the full previously-passed ground regression set before execution."""
from pathlib import Path
import json
from adapters import HERE

jobs=[]
def add(suite,name):
    jobs.append((suite,name))
for name in ('held_cruise_0','held_cruise_1','held_fast_0','held_fast_1',
             'held_arc_left_0','held_arc_left_1','held_arc_right_0','held_arc_right_1',
             'held_back_0','held_back_1','held_brake_0','held_brake_1','held_cut_steer',
             'held_walk','held_brake','held_push','held_cut','held_restore'):
    add('body',name)
for target in ('stride_left','stride_right','bluebell','web_b'):
    for variant in ('held0','held1'):
        for mode in ('open','cut'):
            add('ground',f'{target}__{variant}__{mode}')
for seed in (271,362):
    for name in ('cruise','fast','a01_left','a01_right','a02_left','a02_right'):
        add('eon',f'{seed}_{name}__open')
    for name in ('a01_left','a01_right','a02_left','a02_right'):
        add('eon',f'{seed}_{name}__steer')
for name in ('181_off__open','271_cruise__speed','271_a02_left__all'):
    add('eon',name)
with (HERE/'FORMAL_JOBS.txt').open('x') as stream:
    for suite,name in jobs:
        stream.write(suite+' '+name+'\n')
print(len(jobs),'whole episodes')
