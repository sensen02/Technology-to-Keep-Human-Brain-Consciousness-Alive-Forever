"""Matched-window and descriptive burst diagnostics from cached saved arrays only."""
import json
import sys
from pathlib import Path
import numpy as np
from collect import HERE,ROOT,BASE,digest,jsonable
sys.path.insert(0,str(BASE))
import student_wave as sw
from physics import W,LEGS


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def window(t,first=5800,last=10600):
    rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(first,last)]
    summary=sw.summarise(rec,(last-first)*80,initial_heading=np.array([1.,0.]))
    load=t['loads'][first:last];speed=t['speed'][first:last]
    vt=(load*speed).sum(0)/load.sum(0)
    return dict(summary=summary,per_leg_loadweighted_speed=dict(zip(LEGS,vt.tolist())),
                slip_denominator_netXY_speed=summary['forward_speed_native_per_s'],
                numerator_share_loaded_point1=dict(zip(LEGS,((load*speed*(load>=.1*W)).sum(0)/np.maximum((load*speed).sum(0),1e-30)).tolist())))


report={};hashes={}
for condition in ('left','right'):
    run=HERE/('COLLECT_'+condition);tp=run/'trace_1ms.npz';sp=run/'SAMPLES_5MS.npz'
    reference=ROOT.parent/'20261110T191449Z/teacher'/('left_full' if condition=='left' else 'right_full')
    teacherpath=reference/'TRACE_1MS.npz';samplepath=reference/'SAMPLES_5MS.npz'
    t=arrays(tp);s=arrays(sp);tt=arrays(teacherpath);ts=arrays(samplepath)
    for p in (tp,sp,teacherpath,samplepath):hashes[str(p)]=digest(p)
    sd=np.maximum(ts['X238'][:1920].std(0),.05)
    deviation=np.sqrt(np.mean(((s['X238']-ts['X238'][:1920])/sd)**2,axis=1))
    actionerr=np.sqrt(np.mean((s['student_action']-s['teacher_action'])**2,axis=1))
    segments=[]
    for b in range(24):
        k=960+40*b
        # Observations at k+5 are immediately after student25ms; k+40 after teacher175ms when present.
        item=dict(burst=b,time_start=float(s['time'][k]),start_observation_RMS=float(deviation[k]),student_end_observation_RMS=float(deviation[k+5]),
                  start_action_RMS=float(actionerr[k]),student_end_action_RMS=float(actionerr[k+5]))
        if k+40<len(s['time']):item.update(recovery_end_observation_RMS=float(deviation[k+40]),recovery_end_action_RMS=float(actionerr[k+40]))
        first=5800+b*200;middle=first+25;end=first+200
        for label,a,z in (('student',first,middle),('recovery',middle,end)):
            load=t['loads'][a:z];speeds=t['speed'][a:z]
            item[label+'_loadweighted_speed_mean']=float((load*speeds).sum()/load.sum())
            item[label+'_raw_loaded_slip_integral']=float((load*speeds*.001).sum())
        segments.append(item)
    complete=[v for v in segments if 'recovery_end_observation_RMS' in v]
    aggregate=dict(start_obs_RMS_mean=float(np.mean([v['start_observation_RMS'] for v in complete])),
                   after_student_obs_RMS_mean=float(np.mean([v['student_end_observation_RMS'] for v in complete])),
                   after_recovery_obs_RMS_mean=float(np.mean([v['recovery_end_observation_RMS'] for v in complete])),
                   recovery_obs_RMS_lower_than_student_end_count=sum(v['recovery_end_observation_RMS']<v['student_end_observation_RMS'] for v in complete),
                   complete_bursts_for_endpoint_comparison=len(complete),
                   note='Descriptive time-aligned deviation from original teacher path; different phases/durations and no matched no-recovery intervention prevent causal recovery claim.')
    report[condition]=dict(actual_mixed=window(t),matched_original_teacher_first4p8=window(tt),original_teacher_full9p6=window(tt,5800,15400),
                           per_burst=segments,descriptive_aggregate=aggregate,
                           teacher_root_balance_diagnostic=dict(max=float(s['root_balance_residual'].max()),above_point02_count=int((s['root_balance_residual']>.02).sum())))
result=dict(kind='matched4.8teacher control plus descriptive25ms/175ms segments',runs=report,physical_integration_steps=0,optimizer_updates=0,
            source_hashes={**hashes,str(HERE/'diagnose_segments.py'):digest(HERE/'diagnose_segments.py')},
            interpretation='Old4.8s mixed admission unchanged; a control failure does not retroactively admit data or establish recovery. No new physical run.')
with (HERE/'SEGMENT_DIAGNOSIS.json').open('x') as f:json.dump(jsonable(result),f,indent=2);f.write('\n')
print(json.dumps({k:dict(mixed_slip=v['actual_mixed']['summary']['slip_ratio_max'],matched_teacher_slip=v['matched_original_teacher_first4p8']['summary']['slip_ratio_max'],aggregate=v['descriptive_aggregate']) for k,v in report.items()}))
