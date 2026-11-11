"""Independent whole saved-run audit; explicit copy of ROOT/verify_fast_actual.py.
No Core construction or integration calls.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from eval3000 import HERE,ROOT,BASE,ORIGINAL,COMMANDS,pin,jsonable,condition_score,checked_load


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify(run):
    import torch
    torch.set_num_threads(1)
    sw,identity,fm=pin()
    run=Path(run).resolve()
    if not run.is_relative_to(HERE):raise ValueError('audit run outside continuation')
    result=json.loads((run/'RESULT.json').read_text());ident=json.loads((run/'IDENTITY.json').read_text())
    identity.verify(ident);fm.verify_hashes(result['source_hashes'])
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise ValueError('failed: '+name)
    check('result_identity_bound',result['identity_sha256']==ident['identity_sha256'])
    check('checkpoint_bound_to_execution',result['checkpoint_sha256']==ident['config']['cli']['sha256'] and result['checkpoint']==str(Path(ident['config']['cli']['checkpoint']).resolve()))
    check('sourcecapture_bound_to_execution',result['source_hashes']==ident['config']['evaluation_source_hashes'])
    for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('runtime_geometry.npz','runtime_geometry')]:check(filename+'_hash',fm.sha256(run/filename)==result[key]['sha256'])
    t=arrays(run/'trace_1ms.npz');d=arrays(run/'neural_decisions.npz');geometry=arrays(run/'runtime_geometry.npz')
    check('all_trace_values_finite',all(np.isfinite(v).all() for v in t.values()))
    check('all_neural_values_finite',all(np.isfinite(v).all() for v in d.values()))
    check('complete_endpoints',t['time'].shape==(15400,) and t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48))
    check('complete_decisions',d['X238'].shape==(2880,238) and d['hidden_before'].shape==(2880,128) and d['hidden_after'].shape==(2880,128) and d['mean'].shape==(2880,42))
    check('native_stepcount',result['physical_steps']==1232000)
    check('full_1ms_clock',np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(np.abs(np.diff(t['time'])-.001))<1e-9)
    check('decision_clock',np.max(np.abs(d['time']-(1.+np.arange(2880)*.005)))<1e-8)
    phase=np.r_[np.zeros(1000,dtype=int),np.full(4800,2),np.full(9600,3)]
    check('all_source_counts_and_boundaries',np.array_equal(t['control_source'],phase) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]))
    check('zero_external_forces',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('actual_requested_actions_equal_applied',np.array_equal(t['actual_requested_action'],t['action']))
    check('all_held_actions_match_decisions',np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)))
    check('zero_settle_actions',np.array_equal(t['action'][:1000],np.zeros((1000,42))))
    check('autonomous_direct_neural_actions',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]))
    check('actions_bounded',np.max(np.abs(d['actually_applied_action']))<=1.)
    check('previous_action_inputs',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.array_equal(d['X238'][0,195:237],np.zeros(42)))
    check('constant_original_targetspeed',np.array_equal(d['X238'][:,-1],np.full(2880,.075)))
    expected=np.r_[np.zeros((960,2)),np.tile(COMMANDS[result['condition']],(1920,1))]
    check('commands_and_neutral_priming',np.array_equal(d['command'],expected))
    check('stored_hidden_chain',np.array_equal(d['hidden_before'][0],np.zeros(128)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]))
    model,payload=checked_load(fm,result['checkpoint'],result['checkpoint_sha256'])
    check('fixed3000_checkpoint',payload['curve'][-1]['step']==3000)
    state=model.initial_state(1);allmeans=[];allstates=[];allactions=[]
    with torch.no_grad():
        for x,u in zip(d['X238'],d['command']):
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
            allmeans.append(mean[0].numpy());allstates.append(state[0].numpy());allactions.append(mean[0].tanh().numpy())
    check('all2880_actual_checkpoint_means_bitexact',np.array_equal(np.asarray(allmeans),d['mean']))
    check('all2880_actual_checkpoint_hidden_bitexact',np.array_equal(np.asarray(allstates),d['hidden_after']))
    check('all2880_actual_checkpoint_actions_bitexact',np.array_equal(np.asarray(allactions),d['policy_action']))
    handover=json.loads((run/'HANDOVER.json').read_text())
    check('handover_saved_matches_result',handover==result['transition'])
    check('all_handover_fields_exact',handover['all_exact'] and all(handover['exact'].values()) and handover['before_sha256']==handover['after_sha256'])
    # Independently derive all physically/neurally observable handover fields from saved endpoints.
    boundary={'qpos':t['qpos'][5799],'qvel':t['qvel'][5799],'ctrl':t['actualctrl'][5799],'target':t['target'][5799],
              'requested_rate':t['actual_requested_action'][5799]*2.,'previous_actual_action':d['actually_applied_action'][959],
              'time':np.asarray(t['time'][5799]),'hidden':d['hidden_before'][960:961]}
    for key,value in boundary.items():check('derived_handover_hash_'+key,hashlib.sha256(value.tobytes()).hexdigest()==handover['before_sha256'][key])
    check('teacher_calls_limited_to_prime',result['teacher_calls']==960 and result['teacher_calls_after_handover']==0)
    for key in ('geom_type','geom_rgba'):
        meta=ident['compiled_arrays'][key]
        check('runtime_'+key+'_compiled_hash',hashlib.sha256(geometry[key].tobytes()).hexdigest()==meta['sha256'] and list(geometry[key].shape)==meta['shape'])
    mesh=geometry['geom_type']==7
    check('all71_meshes_opaque',mesh.sum()==71 and np.all(geometry['geom_rgba'][mesh,3]==1.))
    rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)]
    heading=np.asarray(result['autonomous']['initial_heading_diagnostics']['heading_world_xy'])
    auto=sw.summarise(rec,768000,initial_heading=heading)
    auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600)
    check('whole9600_summary_exact',jsonable(auto)==result['autonomous'])
    observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
    for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
    cycles=sw.cycle_report(observer,events)
    check('all_sensed_cycle_events_exact',jsonable(cycles)==result['autonomous_cycles'])
    yaw=np.unwrap(d['yaw_after_each_1ms'])
    check('complete_yaw_record',yaw.shape==(15400,))
    delta=float(yaw[-1]-yaw[5799]);check('actual_yaw_exact',delta==result['yaw_change_rad'])
    ref=result['neutral_reference'];speed=None
    if result['condition'].startswith('speed'):
        check('neutral_reference_hash',fm.sha256(ref['path'])==ref['sha256'])
        nr=json.loads(Path(ref['path']).read_text());ni=json.loads((Path(ref['path']).parent/'IDENTITY.json').read_text());identity.verify(ni)
        check('neutral_qualified_same_checkpoint',nr['engineering_pass'] and nr['condition']=='neutral' and nr['checkpoint_sha256']==result['checkpoint_sha256'] and nr['checkpoint']==result['checkpoint'] and nr['source_hashes']==result['source_hashes'] and ni['compiled_arrays']==ident['compiled_arrays'])
        speed=nr['autonomous']['forward_signed_native_per_s'];check('neutral_speed_exact',ref['forward_signed_native_per_s']==speed)
    check('criteria_recomputed',condition_score(result['condition'],auto,cycles,delta,speed)==result['criteria'])
    check('pass_flag_recomputed',result['engineering_pass']==result['criteria']['eligible'])
    # Every condition must retain the original actual 5800-endpoint/960-decision prefix.
    oldd=arrays(ORIGINAL/'neural_decisions.npz');oldt=arrays(ORIGINAL/'trace_1ms.npz')
    for key in ('X238','hidden_before','hidden_after','mean','policy_action','actually_applied_action','control_source'):
        check('original_prefix_neural_'+key,np.array_equal(d[key][:960],oldd[key][:960]))
    for key in t:
        if key in oldt:check('original_prefix_physical_'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
    if result['condition']=='neutral':
        for key in ('X238','hidden_before','hidden_after','mean','policy_action','actually_applied_action','control_source'):check('neutral_original_'+key,np.array_equal(d[key],oldd[key]))
        for key in t:
            if key in oldt:check('neutral_original_physical_'+key,np.array_equal(t[key],oldt[key]))
    if result.get('native') is not None:
        check('native_recorder_complete',result['native']['native_samples']==1232000 and result['native']['pending_rows']==0)
        for chunk in result['native']['chunks']:check('native_chunk_'+Path(chunk['path']).name,fm.sha256(chunk['path'])==chunk['sha256'])
    identity.verify(ident);fm.verify_hashes(result['source_hashes']);checked_load(fm,result['checkpoint'],result['checkpoint_sha256'])
    return dict(integrity_all_pass=True,checks=checks,run=str(run),condition=result['condition'],checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],engineering_pass=result['engineering_pass'],physics_integration_steps=0,source_hashes={str(HERE/n):fm.sha256(HERE/n) for n in ('verify_actual3000.py','eval3000.py','EVAL_PREREG.json')},scope='independent saved checkpoint/endpoint audit; not complete native or biological certification')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if not Path(a.run).is_absolute() or not Path(a.out).is_absolute():raise ValueError('absolute audit paths required')
    if out.exists() or not out.is_relative_to(HERE):raise ValueError('fresh audit output within continuation root required')
    try:report=verify(a.run)
    except BaseException as exc:
        out.write_text(json.dumps(dict(integrity_all_pass=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');raise
    out.write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(integrity_all_pass=True,engineering_pass=report['engineering_pass'],checks=len(report['checks']),out=str(out))))

if __name__=='__main__':main()
