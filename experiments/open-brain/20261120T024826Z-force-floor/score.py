"""Compare already written measurements to hidden references; calculate behavior agreement."""
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

D=Path(__file__).resolve().parent
L='LF LM LH RF RM RH'.split()
def rows(rel):
    with (D/rel).open() as f:return list(csv.DictReader(f))
def stats(a):
    a=np.asarray(a,float)
    return dict(n=a.size,median=float(np.median(a)),p90=float(np.percentile(a,90)),mean=float(np.mean(a)),max=float(np.max(a))) if a.size else dict(n=0)
def score_bench(name):
    truth=json.loads((D/name/'private.json').read_text())
    by=[[] for _ in truth]
    for r in rows(name+'/patches.csv'):
        by[int(r['frame'])].append([float(r[k]) for k in ('x_mm','y_mm','fz_uN')])
    force=[];location=[];control=[];holds=[]
    stages=[i for i,q in enumerate(truth) if i==0 or truth[i-1]!=q]+[len(truth)]
    for s,e in zip(stages,stages[1:]):
        t=np.array(truth[s]).reshape(-1,3);stable=[]
        for i in range(s+3,e):
            pred=np.array(by[i]).reshape(-1,3)
            assert len(t)==len(pred),(name,i,t,pred)
            if len(t):
                a,b=linear_sum_assignment(np.linalg.norm(t[:,None,:2]-pred[None,:,:2],axis=-1))
                force.extend(abs(t[a,2]-pred[b,2]));location.extend(np.linalg.norm(t[a,:2]-pred[b,:2],axis=-1))
                stable.append(pred[b,2].tolist())
        # First capture within 10% of each target or 0.25 µN for low loads.
        hit=None
        for i in range(s,e):
            pp=sorted(by[i],key=lambda r:r[0])
            if len(pp)==len(t) and all(abs(v[2]-q[2])<=max(.25,.1*q[2]) for v,q in zip(pp,t)):
                hit=2*(i-s+1);break
        holds.append(dict(begin_ms=2*s,force_uN=t[:,2].tolist(),measured_mean_uN=np.mean(stable,axis=0).tolist() if stable else [],
                          first_in_band_ms=hit))
    retained=abs(holds[2]['measured_mean_uN'][0]-holds[3]['measured_mean_uN'][0])
    return dict(holds=holds,stable_abs_error_uN=stats(force),stable_xy_error_mm=stats(location),
                withdrawal_other_change_uN=retained,pass_gate=bool(max(force)<.5 and max(location)<.2 and retained<.5))
def behavior(folder):
    truth=np.load(D/folder/'private_walk.npz');feet=np.load(D/folder/'observed_feet_mm.npy')
    body=truth['body']
    assert np.isfinite(feet).all()
    # Matched behavior reference only. Analyze active movement from 0.24 s.
    stance=truth['forces'][:,:6]>.5
    durations=[];strides=[];speeds=[];touchdowns=0
    for k in range(6):
        switch=np.flatnonzero(np.diff(stance[:,k].astype(int))==1)+1
        switch=[t for t in switch if t>=120 and t+2<len(stance) and stance[t:t+3,k].all() and not stance[t-2:t,k].any()]
        touchdowns+=len(switch)
        for a,b in zip(switch,switch[1:]):
            strides.append(np.linalg.norm(feet[b,k,:2]-feet[a,k,:2]))
        for a in switch:
            end=np.flatnonzero(~stance[a:,k])
            if len(end):durations.append(2*end[0])
        speed=np.linalg.norm(np.diff(feet[:,k,:2],axis=0),axis=1)/.002
        speeds.extend(speed[(stance[1:,k]&stance[:-1,k]&(np.arange(len(speed))>=120))])
    return dict(dx_mm=float(body[-1,0]-body[0,0]),touchdowns=touchdowns,step_mm=stats(strides),
                load_duration_ms=stats(durations),loaded_toe_xy_speed_mm_s=stats(speeds))
def main():
    audit={n:score_bench(n) for n in ('blind','punches')}
    merged=[r for r in rows('merged/patches.csv') if 62<=int(r['time_ms'])<=90]
    audit['merged']=dict(spacing_mm=.10,frames=15,detected_patches=len(merged),
                        mean_combined_uN=float(np.mean([float(r['fz_uN']) for r in merged])))
    assert len(set(r['frame'] for r in merged))==len(merged)==15
    audit['hover']=dict(frames=71,patches=len(rows('hover/patches.csv')))
    true=np.load(D/'insert/private_walk.npz')
    obs=np.array([float(r['fz_uN']) for r in rows('insert/feet.csv')]).reshape(-1,6)
    ref=true['forces'][:,:6]
    valid=np.isfinite(obs)
    tp=np.sum((ref>=.5)&(obs>=.5));fp=np.sum((ref<.5)&(obs>=.5));fn=np.sum((ref>=.5)&(obs<.5))
    err=np.abs(ref-obs); active=ref>=.5
    patches=rows('insert/patches.csv')
    audit['walking']=dict(frames=len(obs),foot_samples=int(valid.sum()),coverage=float(valid.mean()),
                          patch_labels=Counter(r['attribution'] for r in patches),
                          load_episodes=len(rows('insert/load_episodes.csv')),
                          foot_absolute_error_uN=stats(err[valid]),loaded_absolute_error_uN=stats(err[valid&active]),
                          whole_body_raw_absolute_error_uN=stats(np.abs(np.loadtxt(D/'insert/total_uN.csv',skiprows=1)-true['total'])),
                          precision=float(tp/(tp+fp)),recall=float(tp/(tp+fn)),
                          signed_reference_mean_uN=float(true['total'].mean()),sensor_mean_uN=float(np.loadtxt(D/'insert/total_uN.csv',skiprows=1).mean()))
    audit['behavior']={n:behavior(n) for n in ('natural','insert')}
    (D/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
