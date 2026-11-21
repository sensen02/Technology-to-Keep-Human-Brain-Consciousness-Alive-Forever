"""Reward and evaluation from held physical samples, independent of sensory inputs."""
import json
from pathlib import Path
import numpy as np


def analyze(folder,phase):
    result=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'BODY.npz') as b,np.load(folder/'TRAIN_MEASURES.npz') as m:
        trace=b['trace']; decisions=b['observation']; ledger=b['meal']
        oral=m['oral']; feet=m['feet'].astype(bool)
    assert len(oral)==len(trace)==len(feet) and np.allclose(oral[:,0],trace[:,0],atol=1e-8)
    t=trace[:,0]; dist=oral[:,4]; xy=trace[:,1:3]*10
    standing=(trace[:,4]>=4)&(trace[:,6]>.94)&(np.linalg.norm(trace[:,7:9],axis=1)<=.08)
    upright=(trace[:,4]>=3)&(trace[:,6]>.88)
    start_window=(t>=.08)&(t<.16)
    initialized=bool(standing[start_window].all() and start_window.any())
    walk_window=(t>=.16)&(t<=1.2)
    walk_feet=feet[walk_window]
    # A complete unloading/reloading sequence, starting from stance.
    completed=0
    for stance in walk_feet.T:
        loaded=np.flatnonzero(stance)
        if len(loaded)>=2 and not stance[loaded[0]:loaded[-1]].all():
            completed+=1
    advance=np.linalg.norm(xy[walk_window][-1]-xy[walk_window][0])
    g1=bool(initialized and result['duration_s']>=1.2 and advance>=2.5 and
            upright[walk_window].mean()>=.9 and completed>=2 and
            (result['first_fall_s'] is None or result['first_fall_s']>1.2))
    d0=float(dist[np.searchsorted(t,.16)])
    stages=np.maximum.accumulate(np.floor(np.maximum(0.,d0-dist)+1e-8)).astype(int)
    count_steps=np.maximum.accumulate(np.floor(np.linalg.norm(xy-xy[np.searchsorted(t,.16)],axis=1))).astype(int)
    meals=np.interp(t,ledger[:,0],ledger[:,1])
    assert abs(result['intake']-result['stock_loss'])<1e-9 and result['native_extra_force']==0
    trials=[]; totals=np.zeros(4)
    for i,row in enumerate(decisions):
        start=row[0]; last=i==len(decisions)-1
        end=result['duration_s'] if last else decisions[i+1,0]
        left=np.searchsorted(t,start); right=np.searchsorted(t,end)
        if right<=left:
            continue
        prior=max(0,left-1); k=right-1
        fall=result['first_fall_s'] is not None and start<=result['first_fall_s']<end+1e-9
        # Count the final shortened interval at its physical duration.
        stability=.2*(upright[left:right].mean()-.80)*(end-start)/.08
        stability+=.15*(min(5,count_steps[k])-min(5,count_steps[prior]))-3.*fall
        if start<1.2<=end+1e-9 and g1:
            stability+=1.
        approach=.65*(stages[k]-stages[prior])
        # The global runner stops at actual ingestion threshold and its standing gate.
        feeding=12.*(meals[k]-meals[prior])/.18
        success=4.*bool(last and result['meal_pass'])
        components=np.array([stability,approach,feeding,success])
        reward=stability+(approach if phase>=2 else 0.)+(feeding+success if phase>=3 else 0.)
        totals+=components
        terminal=bool(last and (fall or result['first_fall_s'] is not None or result['meal_pass']))
        # A time limit has no measured successor at the next decision instant.
        # Drop only that last incomplete transition, rather than bootstrap on an invented observation.
        if last and not terminal:
            continue
        nxt=row[1:15] if last else decisions[i+1,1:15]
        trials.append((row[1:15],row[15:17],reward,nxt,float(terminal)))
    summary=dict(phase=phase,initial_stand=initialized,walk_gate=g1,walk_advance_mm=float(advance),
                 reloaded_legs=completed,upright_fraction=float(upright[walk_window].mean()),
                 rings=int(stages[-1]),first_distance_mm=d0,min_distance_mm=float(dist.min()),
                 time_s=result['duration_s'],fall_s=result['first_fall_s'],intake=result['intake'],
                 meal_pass=result['meal_pass'],standing=result['ended_standing'],
                 native_steps=result['native_steps'],component_returns=totals.tolist(),transitions=len(trials))
    return trials,summary
