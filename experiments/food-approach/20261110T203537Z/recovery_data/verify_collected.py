"""Independent saved-array replay: no Core construction or physical integration."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from contract import source_at,admit
from collect import HERE,ROOT,BASE,digest,jsonable
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import fast_motor as fm
import student_wave as sw
from identity import verify


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def one(run):
    r=json.loads((run/'RESULT.json').read_text());ident=json.loads((run/'IDENTITY.json').read_text())
    verify(ident);fm.verify_hashes(ident['config']['source_hashes'])
    t=arrays(run/'trace_1ms.npz');s=arrays(run/'SAMPLES_5MS.npz')
    checks={}
    def ck(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('failed:'+name)
    ck('trace_hash',digest(run/'trace_1ms.npz')==r['artifacts']['trace']['sha256'])
    ck('samples_hash',digest(run/'SAMPLES_5MS.npz')==r['artifacts']['samples']['sha256'])
    ck('complete_actual_counts',len(t['time'])==10600 and len(s['time'])==1920 and r['complete'])
    ck('all_values_finite',all(np.isfinite(v).all() for v in t.values()) and all(np.isfinite(v).all() for v in s.values()))
    src=np.array([source_at(k) for k in range(1920)])
    ck('source_schedule_exact',np.array_equal(s['source'],src) and np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.repeat(src,5)]))
    ck('actual_clock',np.max(np.abs(t['time']-np.arange(1,10601)*.001))<1e-8 and np.max(np.abs(s['time']-(1+np.arange(1920)*.005)))<1e-8)
    ck('zero_external',np.max(t['external_body_force_max'])==0 and np.max(t['external_generalized_force_max'])==0)
    ck('all_actions_bounded',np.max(np.abs(s['teacher_action']))<=1 and np.max(np.abs(s['student_action']))<=1 and np.max(np.abs(s['actually_applied_action']))<=1)
    ck('actual_action_choice',np.array_equal(s['actually_applied_action'],np.where((src==3)[:,None],s['student_action'],s['teacher_action'])))
    ck('actual_action_hold',np.array_equal(t['action'][1000:],np.repeat(s['actually_applied_action'],5,axis=0)))
    ck('actual_requested_equals_applied',np.array_equal(t['action'],t['actual_requested_action']))
    ck('previous_actual_action_input',np.array_equal(s['X238'][1:,195:237],s['actually_applied_action'][:-1]) and np.array_equal(s['X238'][0,195:237],np.zeros(42)))
    ck('base_targetspeed_unchanged',np.array_equal(s['X238'][:,-1],np.full(1920,.075)))
    ck('hidden_chain',np.array_equal(s['hidden_before'][0],np.zeros(128)) and np.array_equal(s['hidden_before'][1:],s['hidden_after'][:-1]))
    model,p=fm.load(ident['config']['cli']['checkpoint'],r['checkpoint_sha256'])
    ck('fixed3000',p['curve'][-1]['step']==3000)
    state=model.initial_state(1);means=[];hidden=[];actions=[]
    with torch.no_grad():
        for x,u in zip(s['X238'],s['command2']):
            mean,state=model.step(torch.tensor(x[None]),torch.tensor(u[None]),state)
            means.append(mean[0].numpy());hidden.append(state[0].numpy());actions.append(mean[0].tanh().numpy())
    ck('all1920_checkpoint_mean_exact',np.array_equal(means,s['student_mean']))
    ck('all1920_checkpoint_hidden_exact',np.array_equal(hidden,s['hidden_after']))
    ck('all1920_checkpoint_action_exact',np.array_equal(actions,s['student_action']))
    original=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
    oldt=arrays(original/'trace_1ms.npz');oldd=arrays(original/'neural_decisions.npz')
    ck('full5800_actual_original_prefix',all(np.array_equal(t[k][:5800],oldt[k][:5800]) for k in t if k in oldt))
    ck('actual960_original_neural_prefix',np.array_equal(s['X238'][:960],oldd['X238'][:960]) and np.array_equal(s['student_mean'][:960],oldd['mean'][:960]) and np.array_equal(s['hidden_after'][:960],oldd['hidden_after'][:960]))
    ck('teacher_query_counts',r['teacher_queries']==1920 and r['teacher_applied_decisions']==1800 and r['student_applied_decisions']==120)
    heading=np.asarray(r['collection']['initial_heading_diagnostics']['heading_world_xy'])
    rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,10600)]
    summary=sw.summarise(rec,4800*80,initial_heading=heading)
    ck('whole_collection_summary_exact',jsonable(summary)==r['collection'])
    ck('safety_admission_recomputed',admit(summary,True,r['early_stop'],None)==r['dataset_safety_admitted'])
    observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
    for i in range(5800,10600):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
    ck('cycle_events_recomputed',jsonable(sw.cycle_report(observer,events))==r['cycles_diagnostic'])
    boundaries=json.loads((run/'BOUNDARIES.json').read_text())
    ck('all48_boundary_snapshots_exact',len(boundaries)==48 and all(b['all_exact'] and b['before']==b['after'] for b in boundaries))
    expected_phase=np.arange(1920)*.005
    ck('phase_provenance_actualquery_clock',np.max(np.abs(s['teacher_phase_provenance_only']-expected_phase))<1e-12)
    # Teacher fullscale1 phase stays t. Smooth command normalization is reference's exact formula.
    z=np.clip((expected_phase-4.8),0,1);blend=z*z*(3-2*z)
    sign=1 if ident['config']['cli']['condition']=='left' else -1
    ck('ramped_actualcommand',np.max(np.abs(s['command2'][:,1]-sign*blend))<1e-14 and np.array_equal(s['command2'][:,0],np.zeros(1920)))
    # Actual-state coverage only; teacher episode is an observed reference, not a heldout split.
    name='left_full' if sign==1 else 'right_full'
    teacher=arrays(ROOT.parent/'20261110T191449Z/teacher'/name/'SAMPLES_5MS.npz')
    reference=torch.tensor(teacher['X238'][:1920],dtype=torch.float64)
    # Compute time-aligned actual base memories on the original teacher stream.
    st=model.initial_state(1);hhs=[]
    with torch.no_grad():
        for x in reference:
            _,st=model.base.step(x[None],st);hhs.append(st[0].numpy())
    sd=np.maximum(teacher['X238'][:1920].std(0),.05)
    deviation=np.sqrt(np.mean(((s['X238']-teacher['X238'][:1920])/sd)**2,axis=1))
    hd=np.sqrt(np.mean((s['hidden_after']-np.asarray(hhs))**2,axis=1))
    coverage={str(source):dict(rows=int((src==source).sum()),obs_standardized_RMS_mean=float(deviation[src==source].mean()),obs_standardized_RMS_max=float(deviation[src==source].max()),hidden_RMS_mean=float(hd[src==source].mean())) for source in (2,3,5)}
    verify(ident)
    return dict(integrity_all_pass=True,checks=checks,run=str(run),dataset_safety_admitted=r['dataset_safety_admitted'],
                samples_sha256=digest(run/'SAMPLES_5MS.npz'),trace_sha256=digest(run/'trace_1ms.npz'),result_sha256=digest(run/'RESULT.json'),checkpoint_sha256=r['checkpoint_sha256'],
                scope='independent actual saved mixed-control replay, not autonomous pass or recovery causality',collection=summary,
                coverage=coverage,teacher_label_scope='shape/bounds/sourceclock/actualquery provenance verified; independent teacher privateIK replay is separate',
                recovery_effect='NOT_ESTABLISHED: endpoints descriptive, no matched no-recovery counterfactual',physical_integration_steps=0,optimizer_updates=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    reports={name:one(HERE/('COLLECT_'+name)) for name in ('left','right')}
    for name,report in reports.items():
        with (HERE/('COLLECT_'+name)/'VERIFIED_SAMPLES.json').open('x') as f:json.dump(jsonable(report),f,indent=2);f.write('\n')
    with (HERE/'VERIFICATION.json').open('x') as f:json.dump(jsonable(dict(integrity_all_pass=all(r['integrity_all_pass'] for r in reports.values()),runs=reports)),f,indent=2);f.write('\n')
    print(json.dumps({name:dict(integrity_all_pass=r['integrity_all_pass'],dataset_safety_admitted=r['dataset_safety_admitted'],slip=r['collection']['slip_ratio_max']) for name,r in reports.items()}))
