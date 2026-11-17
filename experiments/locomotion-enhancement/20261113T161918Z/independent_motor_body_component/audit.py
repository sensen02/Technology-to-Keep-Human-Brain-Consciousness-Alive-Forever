"""Parent savedbody + freshNN352 replay audit; no Core/MuJoCo construction."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from common import ROOT,BASE,FORMAT,CONDITIONS,SOURCE_CANDIDATE,read,save,sha,verify,cpu1,load_motor


def independent_pivot_counts(loads,padz,W):
    # Independent interval scan of actual saved contact, not common.PivotCycles.
    counts=[]
    for leg in range(6):
        phase=0;loaded_run=0;low_run=0;reload=0;peak=0.;start=None;completed=0
        for i in range(len(loads)):
            grounded=loads[i,leg]/W>=.1 and padz[i,leg]<=.004
            low=loads[i,leg]/W<=.05
            if phase==0:
                loaded_run=loaded_run+1 if grounded else 0
                if loaded_run>=10:phase=1
                continue
            if phase==1:
                low_run=low_run+1 if low else 0;peak=max(peak,float(padz[i,leg])) if low else 0.
                if low_run<5:continue
                start=i-4;phase=2
            elif low:peak=max(peak,float(padz[i,leg]))
            reload=reload+1 if grounded else 0
            if i-start+1>4000 or (grounded and peak<=.015):
                phase=0;loaded_run=low_run=reload=0;peak=0.;start=None
            elif peak>.015 and reload>=10:
                completed+=1;phase=1;loaded_run=10;low_run=reload=0;peak=0.;start=None
        counts.append(completed)
    return counts


def legacy_walking_counts(times,loads,padz,relativefoot,boundary):
    # This metric module imports NumPy, never Core/physics/MuJoCo.
    path=BASE/'cycle_reward.py';name='_component_audit_saved_walking_cycles'
    spec=importlib.util.spec_from_file_location(name,path);cr=importlib.util.module_from_spec(spec);sys.modules[name]=cr;spec.loader.exec_module(cr)
    observer=cr.CycleTracker(cr.CycleConfig(timeout=4.)).reset(start_time=boundary['time']);events=[]
    for t,load,z,foot in zip(times,loads,padz,relativefoot):events.extend(observer.update(dict(time=float(t),loads=load,padz=z,relativefoot=foot))['events'])
    events.extend(observer.finalize()['events'])
    return [sum(e['type']=='complete' and e['leg']==leg for e in events) for leg in cr.LEG_ORDER]


def independent_metrics(t,g,b,L,W,condition,native):
    import numpy as np
    selected=np.flatnonzero(t['control_source']==SOURCE_CANDIDATE)
    if len(selected)==0:return None,None,None
    com=t['com'][selected];loads=t['loads'][selected];nf=t['nonfoot'][selected];times=t['time'][selected];first=int(selected[0])
    gg=g['control_source']==SOURCE_CANDIDATE
    angles=np.unwrap(np.concatenate(([b['yaw']],g['thorax_yaw'][gg])));dyaw=float(angles[-1]-angles[0]);duration=float(times[-1]-b['time'])
    offset=com[:,:2]-np.asarray(b['com'])[:2];dist=np.sqrt(np.sum(offset**2,axis=1));drift=float(max(dist)/L)
    # Independent per-sample rotation, shared explicitly frozen1ms definition.
    world_delta=np.diff(np.vstack((np.asarray(b['com']),com)),axis=0)
    body_velocity=np.array([rotation.T.dot(delta/.001)/L for rotation,delta in zip(g['thorax_rotation'][gg],world_delta)])
    late=times>=b['time']+.5-1e-8
    rms=math.sqrt(float(np.mean(body_velocity[late,0]**2+body_velocity[late,1]**2))) if np.any(late) else None
    counts=independent_pivot_counts(t['loads'][max(0,first-10):selected[-1]+1],t['padz'][max(0,first-10):selected[-1]+1],W)
    walking=legacy_walking_counts(times,loads,t['padz'][selected],t['relativefoot'][selected],b)
    if len(times)>=2:
        dt=float(times[-1]-times[0]);netdelta=com[-1,:2]-com[0,:2];net_speed=float(np.linalg.norm(netdelta)/dt);den=loads.sum(0);vt=(loads*t['speed'][selected]).sum(0)/np.maximum(den,1e-12);ratios=vt/max(net_speed,1e-9)
        original=dict(support=float(loads.sum(1).mean()/W)>=.9,nonfoot=float(nf.max())<=.2*W,com=float(com[:,2].min())>=1.2,forward=float(netdelta[0]/dt)>=.05,slip=float(ratios.max())<=.5)
        legacy_scores=dict(forward_signed_native_per_s=float(netdelta[0]/dt),slip_ratio_max=float(ratios.max()))
    else:original=None;legacy_scores=None
    gates=dict(support=float(loads.sum(1).mean()/W)>=.9,nonfoot_1ms=float(nf.max())==0.,com=float(com[:,2].min())>=1.2,native_nonfoot=native['post_nonfoot_peak_native']==0.,native_extra_force=native['extra_force_max']==0.,native_COM=native['post_COM_min']>=1.2,full_window=native['post_native_steps']==768000 and native['actual_native_steps']==1232000)
    if condition=='walk':gates.update(forward=original is not None and original['forward'],legacy_slip=original is not None and original['slip'],all6walkingcycles=all(n>=1 for n in walking))
    elif condition=='stop':gates.update(drift=drift<=.1,post_transition_XY_RMS=rms is not None and rms<=.01,yaw_abs=abs(math.degrees(dyaw))<=5.)
    else:gates.update(drift=drift<=.5,signed_yaw=math.degrees(dyaw if condition=='left_pivot' else -dyaw)>=45.,all6pivotcycles=all(n>=1 for n in counts))
    denominator=loads.sum(0);tangent=(loads*t['speed'][selected]).sum(0)/np.maximum(denominator,1e-12)/L
    values=dict(duration_s=duration,yaw_change_rad=dyaw,yaw_change_deg=math.degrees(dyaw),max_COM_xy_drift_bodylength=drift,final_COM_xy_drift_bodylength=float(dist[-1]/L),COM_xy_path_bodylength=float(np.sqrt(np.sum(world_delta[:,:2]**2,axis=1)).sum()/L),support_mean_over_weight=float(loads.sum(1).mean()/W),COM_min_native=float(com[:,2].min()),nonfoot_1ms_peak=float(nf.max()),post_transition_body_XY_velocity_RMS_bodylength_per_s=rms,pivot_cycle_counts=counts,walk_cycle_counts=walking,per_leg_loaded_tangent_speed_bodylength_per_s=[float(v) if d>1e-12 else None for v,d in zip(tangent,denominator)],per_leg_load_integral_native_force_s=(denominator*.001).tolist(),gates=gates,task_pass_candidate=bool(all(gates.values())))
    return values,original,legacy_scores


def compare_value(expected,actual,tolerance=1e-10):
    if expected is None:return actual is None
    if isinstance(expected,dict):return isinstance(actual,dict) and all(key in actual and compare_value(value,actual[key],tolerance) for key,value in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(expected)==len(actual) and all(compare_value(a,b,tolerance) for a,b in zip(expected,actual))
    if isinstance(expected,bool):return type(actual) is bool and expected==actual
    if isinstance(expected,int):return expected==actual
    return isinstance(actual,(int,float)) and math.isfinite(float(actual)) and abs(float(expected)-float(actual))<=tolerance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=tuple(CONDITIONS),required=True);parser.add_argument('--execute-parent-saved-audit',action='store_true');a=parser.parse_args()
    if not a.execute_parent_saved_audit:parser.error('Explicit parent savedbody/NN audit required')
    cpu1();folder=ROOT/'trials'/a.condition;out=folder/'VERIFY.json'
    if out.exists():raise FileExistsError(out)
    r=read(folder/'RESULT.json');binding=read(ROOT/'INPUT_BINDINGS.json');verify(r['source_hashes'])
    if r['kind']!=FORMAT or r['condition']!=a.condition or r['checkpoint']!=binding['checkpoint'] or r['checkpoint_sha256']!=binding['checkpoint_sha256'] or r['state_dims']!=[128,64,64,96] or r['state_total']!=352:raise ValueError('Actual component result/binding identity')
    # Strict final600 loader before even savedmetric module imports, no physical stack.
    m,payload=load_motor()
    import torch
    import numpy as np
    checks={}
    def weights_digest():
        digest=hashlib.sha256()
        for name,value in sorted(m.state_dict().items()):
            digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_weights_digest=weights_digest()
    for key in ('trace','neural_decisions','initial_state','runtime_geometry','rng_initial'):
        item=r[key];checks[key+'_SHA']=sha(item['path'])==item['sha256']
    for key in ('geometry','body_length_artifact','handover'):
        if key in r:checks[key+'_SHA']=sha(r[key]['path'])==r[key]['sha256']
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
    t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');compiled=arrays(folder/'runtime_geometry.npz')
    g=arrays(folder/'geometry_1ms.npz') if 'geometry' in r else dict(time=np.empty(0),control_source=np.empty(0,dtype=int))
    checks['all_actual_arrays_finite']=all(np.isfinite(value).all() for data in (t,d,g,compiled) for value in data.values())
    checks['trace1ms_clock']=np.max(np.abs(t['time']-np.arange(1,len(t['time'])+1)*.001))<1e-8
    checks['neural5ms_clock']=np.max(np.abs(d['time']-np.arange(len(d['time']))*.005))<1e-8 and np.array_equal(d['physical_step_ms'],np.arange(len(d['time']))*5)
    checks['NN352_shapes']=d['hidden_before'].shape==d['hidden_after'].shape==(len(d['time']),352) and d['X238'].shape==(len(d['time']),238) and d['command'].shape==(len(d['time']),2)
    checks['NN352_state_chain']=np.array_equal(d['hidden_before'][0],np.zeros(352)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1])
    checks['new96_zero_entire_warm']=bool(np.all(d['hidden_before'][:1160,256:]==0.) and np.all(d['hidden_after'][:1160,256:]==0.))
    checks['no_extra_appliedforces']=bool(np.max(t['external_body_force_max'])==0. and np.max(t['external_generalized_force_max'])==0.)
    checks['bounded_42_actions']=d['actually_applied_action'].shape==(len(d['time']),42) and np.max(np.abs(d['actually_applied_action']))<=1.+1e-12
    checks['actually_applied5ms_hold']=np.array_equal(t['action'],np.repeat(d['actually_applied_action'],5,axis=0)[:len(t['time'])])
    checks['actual_source5ms_hold']=np.array_equal(t['control_source'],np.repeat(d['control_source'],5)[:len(t['time'])])
    checks['geometry_aftersettle_clock']=np.array_equal(g['time'],t['time'][1000:])
    checks['all71opaque']=int(np.sum(compiled['geom_type']==7))==71 and bool(np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1.))
    replay={key:[] for key in ('mean','rawmean','oldmean','residual','hidden_before','hidden_after','policy_action')};state=m.initial_state(1)
    with torch.no_grad():
        for k,x in enumerate(d['X238']):
            ms=k*5;hb=torch.cat(state,-1)[0].numpy().copy();xt=torch.tensor(x[None],dtype=torch.float64);command=d['command'][k];ut=torch.tensor(command[None],dtype=torch.float64)
            if ms<1000:
                if not np.array_equal(command,np.zeros(2)):raise ValueError('Settle records no consumed task')
                mean=torch.zeros((1,42),dtype=torch.float64);residual=torch.zeros_like(mean)
            elif ms<5800:
                if not np.array_equal(command,np.array([1.,0.])):raise ValueError('Warm must prime old256 with independent[1,0]')
                mean,state=m.prime_old(xt,ut,state);residual=torch.zeros_like(mean)
            else:
                if not np.array_equal(command,np.asarray(CONDITIONS[a.condition])):raise ValueError('Autonomous current task changed')
                mean,state,residual=m.step_details(xt,ut,state)
            values=dict(mean=mean[0].numpy().copy(),rawmean=mean[0].numpy().copy(),oldmean=(mean-residual)[0].numpy().copy(),residual=residual[0].numpy().copy(),hidden_before=hb,hidden_after=torch.cat(state,-1)[0].numpy().copy(),policy_action=mean[0].tanh().numpy().copy())
            for key,value in values.items():replay[key].append(value)
    for key,values in replay.items():checks['freshNN352_replay_'+key]=np.array_equal(np.asarray(values),d[key])
    candidate_neural=d['physical_step_ms']>=5800
    checks['ANN_only_applied_afterhandover']=np.array_equal(d['actually_applied_action'][candidate_neural],d['policy_action'][candidate_neural])
    expected_source=np.where(d['physical_step_ms']<1000,0,np.where(d['physical_step_ms']<5800,2,SOURCE_CANDIDATE))
    checks['phase_source_labels']=np.array_equal(d['control_source'],expected_source)
    checks['warm_teacher_calls']=r['teacher_calls']==int(np.sum((d['physical_step_ms']>=1000)&(d['physical_step_ms']<5800))) and r['teacher_calls_after_handover']==0
    checks['native_counts_bound']=r['actual_ms']==len(t['time']) and r['actual_native_steps']==len(t['time'])*80==r['native']['actual_native_steps']<=1232000
    checks['no_sensory_brain_neck_fields']=not r['brain_or_sensory_models_loaded'] and not r['neck_ports_present'] and r['training_updates']==0
    if len(g['time']):
        rot_yaw=np.arctan2(g['thorax_rotation'][:,1,0],g['thorax_rotation'][:,0,0]);checks['yaw_from_actual_matrix']=bool(np.max(np.abs(np.angle(np.exp(1j*(rot_yaw-g['thorax_yaw'])))))<1e-10)
        q=t['qpos'][1000:,3:7];q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T;root_yaw=np.arctan2(2*(w*z+x*y),1.-2*(y*y+z*z));checks['yaw_from_saved_actual_root']=bool(np.max(np.abs(np.angle(np.exp(1j*(root_yaw-g['thorax_yaw'])))))<1e-8)
    independent=None
    if r['candidate_start'] is not None:
        b=r['candidate_start'];L=r['body_length']['length_native'];W=r['weight_native']
        independent,legacy,legacy_scores=independent_metrics(t,g,b,L,W,a.condition,r['native'])
        checks['independent_core_metrics_and_task_gates']=compare_value(independent,r['candidate_metrics'])
        old=r['original_five'];count=int(np.sum(t['control_source']==SOURCE_CANDIDATE))
        if count>=2:checks['old_originalfive_independent']=old['gates']==legacy and old['all_gates']==bool(all(legacy.values())) and compare_value(legacy_scores,old)
        else:checks['old_originalfive_shortwindow']=old==dict(all_gates=False,sample_count=count,reason='less than two samples')
        hand=read(folder/'HANDOVER.json');checks['handover_actual_and_neural_continuous']=hand==r['transition'] and hand['no_state_reset'] and hand['new96_zero'] and hand['actual_state_exact'] and hand['neural_state_exact'] and hand['physical_before_sha256']==hand['physical_after_sha256'] and hand['hidden_before_sha256']==hand['hidden_after_sha256']
    else:checks['earlywarmstop_no_task_claim']=r['early_stop'] is not None and r['candidate_metrics'] is None
    ident=read(folder/'IDENTITY.json');canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
    checks['identity_canonical_digest']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ident['identity_sha256']==r['identity_sha256']
    checks['identity_allfiles_unchanged']=all(sha(path)==item['sha256'] for path,item in ident['files'].items())
    checks['finite_PD_actual_torque']=bool(np.all(np.abs(t['actualctrl'])<=np.asarray(ident['actuator']['torque_limit'])[None]+1e-9))
    checks['no_physics_Core_backend_loaded']=not any(name in sys.modules for name in ('mujoco','physics','flygym','student_wave'))
    checks['all_frozen_weights_unchanged_by_replay']=weights_digest()==initial_weights_digest
    m.verify_old();verify(r['source_hashes'])
    save(out,dict(kind=FORMAT+'_SAVED_AUDIT',condition=a.condition,integrity_all_pass=bool(all(checks.values())),checks=checks,result_sha256=sha(folder/'RESULT.json'),checkpoint_sha256=r['checkpoint_sha256'],state_total=352,independent_metrics=independent,neural_records=len(d['time']),physical_steps=0,Core_constructions=0,scope='Fresh checkpoint352 NN replay and independent saved-state metrics; native substep peaks accepted from recordedsummary, not new physics or fullD1-D8. Direct bodycomponent, no neck deployment.'))
    if not all(checks.values()):raise ValueError('SavedNN/body auditfailed; evidence retained, no automaticphysicalrerun')
    print(json.dumps(dict(condition=a.condition,integrity_all_pass=True,checks=len(checks),independent_metrics=independent)),flush=True)


if __name__=='__main__':main()
