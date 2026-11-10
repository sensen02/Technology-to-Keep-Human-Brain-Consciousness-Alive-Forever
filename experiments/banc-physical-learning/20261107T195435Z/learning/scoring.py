"""Privileged training score only; no actor inputs, phase generator or physics."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype/tools')
import dynamic_support_metrics as dsm
WEIGHT=24.446354755878005
NATIVE_DT=0.0000125


def heading_from_points(head,thorax):
    delta=np.asarray(head,dtype=float)[:2]-np.asarray(thorax,dtype=float)[:2]
    norm=float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm<=1e-9:raise ValueError('Initial head-thorax heading is degenerate')
    return delta/norm


def cycle_score(loads5ms):
    loads=np.asarray(loads5ms,float)
    if loads.ndim!=2 or loads.shape[1]!=6 or not np.isfinite(loads).all():raise ValueError('Expected finite (T,6) foot loads')
    qualified=[];completed=[]
    for leg in range(6):
        labels=np.where(loads[:,leg]>.05*WEIGHT,1,np.where(loads[:,leg]<.005*WEIGHT,-1,0))
        runs=[];start=0
        while start<len(labels):
            end=start+1
            while end<len(labels) and labels[end]==labels[start]:end+=1
            if labels[start]!=0 and end-start>=4:runs.append(dict(state='stance' if labels[start]==1 else 'air',start_bin=int(start),end_bin_exclusive=int(end),duration_ms=int(end-start)*5))
            start=end
        stage=0
        for run in runs:
            wanted=('stance','air','stance')[min(stage,2)]
            if run['state']==wanted:stage+=1
            if stage==3:break
        qualified.append(runs);completed.append(stage==3)
    return dict(completed_by_leg=completed,completed_legs=int(sum(completed)),qualified_runs=qualified)


def score_episode(native,slip,loads5ms,rates5ms,start_com,end_com,heading,assist,slip_dt):
    native=np.asarray(native,float);slip=np.asarray(slip,float);rates=np.asarray(rates5ms,float)
    if len(native)<10 or not all(np.isfinite(x).all() for x in (native,slip,rates,start_com,end_com,heading)):
        raise ValueError('Finite nonempty scoring traces required')
    duration=len(native)*NATIVE_DT
    displacement=np.asarray(end_com)-np.asarray(start_com);heading=np.asarray(heading)
    if heading.shape!=(2,) or not np.isclose(np.linalg.norm(heading),1,rtol=0,atol=1e-12):raise ValueError('Heading must be a unit horizontal vector')
    forward=float(displacement[:2]@heading);lateral=float(displacement[:2]@np.array([-heading[1],heading[0]]))
    weighted_slip=float(slip[:,7:13].sum()/WEIGHT*slip_dt)
    minimum=min(float(start_com[2]),float(native[:,10].min()))
    height_drop=max(0.,float(start_com[2])-minimum-.05)
    nonfoot=float(native[:,7].sum()/WEIGHT*NATIVE_DT)
    deficit=float(np.maximum(0.,.6-native[:,1:7].sum(axis=1)/WEIGHT).sum()*NATIVE_DT)
    rate_energy=float(np.mean(rates*rates)*duration) if rates.size else 0.
    cycles=cycle_score(loads5ms)
    terms=dict(forward_displacement=forward,weighted_slip_distance=weighted_slip,lateral_displacement=lateral,height_drop_excess=height_drop,positive_nonfoot_impulse_overW=nonfoot,support_deficit_integral=deficit,rate_squared_time=rate_energy,completed_cycle_legs=cycles['completed_legs'])
    score=forward-2*weighted_slip-.5*abs(lateral)-5*height_drop-5*nonfoot-5*deficit-.0005*rate_energy+.03*cycles['completed_legs']
    gate=dsm.evaluate_native(assist,native)
    gate['checks']['D1_no_nonfoot_ground_force']=bool(native[:,7].max()==0)
    gate['passed']=bool(all(gate['checks'].values()))
    return dict(reward=float(score),reward_terms=terms,cycles=cycles,physical_gate=gate,duration_s=duration,slip_sampling_s=float(slip_dt),slip_status='5ms_training_approximation' if slip_dt==.005 else '0.5ms_final_validation',feasible=bool(gate['passed']))
