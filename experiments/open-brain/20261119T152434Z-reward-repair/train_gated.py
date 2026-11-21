"""Train a branch that cannot gain any food credit via unsupported proxy crossing."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import train_long as trial
from bridge_reward import Snapshot, cost as base_cost, smoke as baseline_smoke

class GatedSAC(trial.learn.SAC):
    def save(self,path,step,identity,rng):
        return super().save(path,step,identity|{'reward':'cross_gated_v2'},rng)


def gated(t,old,new,*,crossed_before,crossed_now,success,terminal,truncated,speed,turn):
    raw=base_cost(t,old,new,crossed_before=crossed_before,crossed_now=crossed_now,
                  success=success,terminal=terminal,truncated=truncated,speed=speed,turn=turn)
    unsafe_exit=bool(t.side and not crossed_now)
    if not unsafe_exit:
        return raw
    # Never pay for eating via the point-body simulator's unsupported route.
    # A meal that terminates after such a shortcut loses a separate bonus-sized cost.
    shortcut=4.5 if terminal and not new.fall else 0.
    return replace(raw,food=0.,overhang=raw.overhang-shortcut)


def smoke():
    model=SimpleNamespace(side=1,near=.77,far=1.045,edge=.13,goal_x=1.87,goal_y=.05)
    a=Snapshot(1.7,.04,.12)
    b=Snapshot(1.71,.06,.22)
    args=dict(crossed_before=False,crossed_now=False,success=False,terminal=True,
              truncated=False,speed=.04,turn=.02)
    bare=base_cost(model,a,b,**args)
    denied=gated(model,a,b,**args)
    assert bare.food>0. and denied.food==0. and denied.value<-4.5
    success=args|dict(crossed_before=True,crossed_now=True,success=True)
    included=gated(model,a,b,**success)
    assert included==base_cost(model,a,b,**success)
    return baseline_smoke()|dict(blocked_shortcut=asdict(denied)|dict(total=denied.value),
                                allowed_meal=asdict(included)|dict(total=included.value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--steps',type=int,default=180000)
    args=p.parse_args()
    trial.cost=gated
    trial.smoke=smoke
    trial.learn.SAC=GatedSAC
    trial.run(args)
