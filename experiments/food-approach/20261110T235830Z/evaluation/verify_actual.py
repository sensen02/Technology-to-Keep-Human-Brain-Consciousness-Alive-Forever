"""Independent saved actual-run audit. No Core construction, mj_step or simulation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from evaluate_motor import HERE,BASE,ORIGINAL,COMMANDS,pin,checked_load,flatten,command_at,score,jsonable,sha256


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify(run):
    import torch
    torch.set_num_threads(1);sw,identity,io=pin()
    run=Path(run).resolve();r=json.loads((run/'RESULT.json').read_text());ident=json.loads((run/'IDENTITY.json').read_text())
    identity.verify(ident)
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise ValueError('failed: '+name)
    check('identity_bound',r['identity_sha256']==ident['identity_sha256'])
    check('checkpoint_bound',r['checkpoint_sha256']==ident['config']['cli']['sha256'] and r['checkpoint']==str(Path(ident['config']['cli']['checkpoint']).resolve()))
    check('source_map_bound',r['source_hashes']==ident['config']['evaluation_source_hashes'])
    for p,h in r['source_hashes'].items():check('source_hash:'+p,sha256(p)==h)
    for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry')]:check(filename+'_hash',sha256(run/filename)==r[key]['sha256'])
    t=arrays(run/'trace_1ms.npz');d=arrays(run/'neural_decisions.npz');g=arrays(run/'geometry_1ms.npz');compiled=arrays(run/'runtime_geometry.npz')
    check('all_saved_values_finite',all(np.isfinite(v).all() for group in (t,d,g,compiled) for v in group.values()))
    check('full15400_endpoints',t['time'].shape==(15400,) and t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48))
    check('full2880_dualhidden_decisions',d['X238'].shape==(2880,238) and d['command'].shape==(2880,2) and d['hidden_before'].shape==(2880,192) and d['hidden_after'].shape==(2880,192) and d['mean'].shape==(2880,42))
    check('native_stepcount',r['physical_steps']==1232000)
    check('whole1ms_clock',np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(np.abs(np.diff(t['time'])-.001))<1e-9)
    check('whole5ms_clock',np.max(np.abs(d['time']-(1.+np.arange(2880)*.005)))<1e-8)
    sources=np.r_[np.zeros(1000,dtype=int),np.full(4800,2),np.full(9600,3)]
    check('sources_counts_boundaries',np.array_equal(t['control_source'],sources) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]))
    check('actual_extra_forces_zero',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
    check('actual_requested_equals_applied',np.array_equal(t['actual_requested_action'],t['action']))
    check('all5ms_held_actions',np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)))
    check('settle_zero',np.array_equal(t['action'][:1000],np.zeros((1000,42))))
    check('ANN_direct_outputs',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]))
    check('bounded_actions',np.max(np.abs(d['actually_applied_action']))<=1.)
    check('previous_actual_action_inputs',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.array_equal(d['X238'][0,195:237],np.zeros(42)))
    check('original_constant_targetspeed',np.array_equal(d['X238'][:,-1],np.full(2880,.075)))
    expected=np.asarray([command_at(r['condition'],k) for k in range(2880)])
    check('all2880_actual_ramp_commands',np.array_equal(d['command'],expected))
    check('full192_hidden_chain',np.array_equal(d['hidden_before'][0],np.zeros(192)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]))
    model,payload=checked_load(io,r['checkpoint'],r['checkpoint_sha256'])
    state=model.initial_state(1);means=[];hidden=[];actions=[]
    with torch.no_grad():
        for x,u in zip(d['X238'],d['command']):
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
            means.append(mean[0].numpy());hidden.append(flatten(state)[0].numpy());actions.append(mean[0].tanh().numpy())
    check('actual2880_checkpoint_means_exact',np.array_equal(np.asarray(means),d['mean']))
    check('actual2880_checkpoint_bothhidden_exact',np.array_equal(np.asarray(hidden),d['hidden_after']))
    check('actual2880_checkpoint_actions_exact',np.array_equal(np.asarray(actions),d['policy_action']))
    hand=json.loads((run/'HANDOVER.json').read_text())
    check('immediate_handover_bound',hand==r['transition'])
    check('handover_allfields_exact',hand['all_exact'] and all(hand['exact'].values()) and hand['before_sha256']==hand['after_sha256'] and hand['state_dims']==[128,64])
    boundary={'qpos':t['qpos'][5799],'qvel':t['qvel'][5799],'ctrl':t['actualctrl'][5799],'target':t['target'][5799],
              'requested_rate':t['actual_requested_action'][5799]*2.,'previous_actual_action':d['actually_applied_action'][959],
              'time':np.asarray(t['time'][5799]),'hidden':d['hidden_before'][960:961]}
    for key,value in boundary.items():check('derived_handover_'+key,hashlib.sha256(value.tobytes()).hexdigest()==hand['before_sha256'][key])
    check('teacher_only_prime',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0)
    for key in ('geom_type','geom_rgba'):
        meta=ident['compiled_arrays'][key]
        check('compiled_'+key,hashlib.sha256(compiled[key].tobytes()).hexdigest()==meta['sha256'] and list(compiled[key].shape)==meta['shape'])
    mesh=compiled['geom_type']==7
    check('complete71opaque',mesh.sum()==71 and np.all(compiled['geom_rgba'][mesh,3]==1.))
    check('geometry_fullclock_source',g['thorax_yaw'].shape==(15400,) and g['head_xyz'].shape==(15400,3) and g['head_rotation'].shape==(15400,3,3) and np.array_equal(g['time'],t['time']) and np.array_equal(g['control_source'],sources))
    audit=json.loads((ROOT_HEAD_AUDIT()).read_text());local=np.asarray(audit['sensor_offsets']['local_xyz'])
    sensors=np.asarray([pos+local@rot.T for pos,rot in zip(g['head_xyz'],g['head_rotation'])])
    check('actual_head_sensor_geometry',np.array_equal(sensors,g['sensor_xyz']) and np.array_equal(sensors.mean(axis=1),g['sensor_midpoint_xyz']))
    check('head_rotation_proper',np.max(np.abs(np.einsum('tji,tjk->tik',g['head_rotation'],g['head_rotation'])-np.eye(3)))<1e-10 and np.max(np.abs(np.linalg.det(g['head_rotation'])-1))<1e-10)
    rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)]
    heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy'])
    auto=sw.summarise(rec,768000,initial_heading=heading)
    auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600)
    check('whole9600_summary_exact',jsonable(auto)==r['autonomous'])
    observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
    for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
    cycles=sw.cycle_report(observer,events);check('whole_cycle_events_exact',jsonable(cycles)==r['autonomous_cycles'])
    yaw=np.unwrap(g['thorax_yaw']);delta=float(yaw[-1]-yaw[5799]);check('actual_rootyaw_exact',delta==r['yaw_change_rad'])
    q=t['qpos'][:,3:7];quat_yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    check('rootyaw_savedqpos_consistent',np.max(np.abs(np.angle(np.exp(1j*(quat_yaw-g['thorax_yaw'])))))<1e-10)
    check('criteria_exact',score(r['condition'],auto,cycles,delta)==r['criteria'])
    check('engineering_flag_exact',r['engineering_pass']==r['criteria']['eligible'])
    oldd=arrays(ORIGINAL/'neural_decisions.npz');oldt=arrays(ORIGINAL/'trace_1ms.npz')
    for key in ('X238','mean','policy_action','actually_applied_action','control_source'):check('original_neural_prime_'+key,np.array_equal(d[key][:960],oldd[key][:960]))
    for key in ('hidden_before','hidden_after'):check('original_basehidden_prime_'+key,np.array_equal(d[key][:960,:128],oldd[key][:960]))
    for key in t:
        if key in oldt:check('original5800_physical_'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
    if r['condition']=='neutral':
        for key in ('X238','mean','policy_action','actually_applied_action','control_source'):check('neutral_original_'+key,np.array_equal(d[key],oldd[key]))
        for key in ('hidden_before','hidden_after'):check('neutral_original_base_'+key,np.array_equal(d[key][:,:128],oldd[key]))
        for key in t:
            if key in oldt:check('neutral_original_physical_'+key,np.array_equal(t[key],oldt[key]))
    if r.get('native') is not None:
        check('native_recorder_complete',r['native']['native_samples']==1232000 and r['native']['pending_rows']==0)
        for chunk in r['native']['chunks']:check('native_chunk_'+Path(chunk['path']).name,sha256(chunk['path'])==chunk['sha256'])
    identity.verify(ident);checked_load(io,r['checkpoint'],r['checkpoint_sha256'])
    for p,h in r['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('source changed '+p)
    return dict(integrity_all_pass=True,engineering_pass=r['engineering_pass'],condition=r['condition'],run=str(run),checkpoint_sha256=r['checkpoint_sha256'],checks=checks,
                physics_integration_steps=0,optimizer_updates=0,source_hashes={str(HERE/n):sha256(HERE/n) for n in ('verify_actual.py','evaluate_motor.py','EVAL_PREREG.json')},scope='independent full saved actual checkpoint/endpoint audit; no Core construction or integration; not fullnative/food certification')


def ROOT_HEAD_AUDIT():
    from evaluate_motor import HEAD_AUDIT
    return HEAD_AUDIT


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if out.exists() or not out.is_relative_to(HERE):raise ValueError('fresh audit output inside evaluation required')
    try:r=verify(a.run)
    except BaseException as exc:
        out.write_text(json.dumps(dict(integrity_all_pass=False,error=repr(exc),physics_integration_steps=0),indent=2)+'\n');raise
    out.write_text(json.dumps(jsonable(r),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(integrity_all_pass=True,engineering_pass=r['engineering_pass'],checks=len(r['checks']),out=str(out))))

if __name__=='__main__':main()
