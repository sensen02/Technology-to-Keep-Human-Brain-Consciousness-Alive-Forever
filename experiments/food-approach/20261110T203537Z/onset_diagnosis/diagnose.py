"""Saved-evidence analysis only; imports Torch/NumPy, never physical backend."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OLD=ROOT.parent/'20261110T191449Z'
sys.path.insert(0,str(ROOT))
import fast_motor as fm
W=24.446354755878005
LEGS=['LF','LH','LM','RF','RH','RM']
CACHE={};INPUTS={}
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_npz(p):
    p=Path(p).resolve();key=str(p)
    if key not in CACHE:
        with np.load(p,allow_pickle=False) as z:CACHE[key]={k:z[k].copy() for k in z.files}
        INPUTS[key]=digest(p)
    return CACHE[key]
def load_json(p):
    p=Path(p).resolve();INPUTS[str(p)]=digest(p)
    return json.loads(p.read_text())
def rms(a):return float(np.sqrt(np.mean(np.square(a))))
def errors(a,b):
    d=np.asarray(a)-np.asarray(b)
    return dict(RMS=rms(d),MAE=float(np.abs(d).mean()),max_abs=float(np.abs(d).max()))
def relative_series(a,b):
    axes=tuple(range(1,a.ndim))
    return np.sqrt(np.mean((a-b)**2,axis=axes))/np.maximum(np.sqrt(np.mean(b*b,axis=axes)),1e-12)
def crossing(x,threshold):
    hit=np.flatnonzero(x>threshold)
    return None if not len(hit) else float((hit[0]+1)*.001)
def summary_series(a,b):
    s=relative_series(a,b)
    return dict(at_5ms=float(s[4]),at_50ms=float(s[49]),at_100ms=float(s[99]),at_250ms=float(s[249]),at_500ms=float(s[499]),at_1s=float(s[999]),maximum_first1s=float(s[:1000].max()),first_over_1e_6_s=crossing(s,1e-6),first_over_1e_3_s=crossing(s,1e-3),first_over_1e_2_s=crossing(s,.01),first_over_1e_1_s=crossing(s,.1))
def segment(z,start,end):
    b=start-1;dur=float(z['time'][end-1]-z['time'][b]);v=np.linalg.norm(z['com'][end-1,:2]-z['com'][b,:2])/dur
    loads=z['loads'][start:end];speed=z['speed'][start:end];num=np.sum(loads*speed,axis=0);den=np.sum(loads,axis=0)
    vt=num/np.maximum(den,1e-12);rat=vt/max(v,1e-9)
    return dict(duration_s=dur,net_xy_speed=float(v),signed_x_speed=float((z['com'][end-1,0]-z['com'][b,0])/dur),load_weighted_tangential_speed=dict(zip(LEGS,map(float,vt))),slip_ratio=dict(zip(LEGS,map(float,rat))),slip_ratio_max=float(rat.max()),foot_support_mean_W=float(loads.sum(1).mean()/W),foot_support_min_W=float(loads.sum(1).min()/W),nonfoot_peak_native=float(z['nonfoot'][start:end].max()),nonfoot_positive_ms=int(np.sum(z['nonfoot'][start:end]>0)),COM_min=float(z['com'][start:end,2].min()),carrying_legs_median=float(np.median((loads>=.1*W).sum(1))))

def main():
    torch.set_num_threads(1)
    conditions={'left':(OLD/'teacher/left_full',np.array([0.,1.])), 'right':(OLD/'teacher/right_full',np.array([0.,-1.])), 'speed085':(OLD/'deceleration_moderate/straight_085',np.array([-.15,0.])), 'speed090':(OLD/'deceleration_moderate/straight_090',np.array([-.10,0.]))}
    results={};model=None;checkpoint=None;sha=None
    for name,(td,command) in conditions.items():
        actual_dir=ROOT/f'EVAL_FAST_{name}';r=load_json(actual_dir/'RESULT.json');t=load_npz(td/'SAMPLES_5MS.npz');tt=load_npz(td/'TRACE_1MS.npz');a=load_npz(actual_dir/'neural_decisions.npz');at=load_npz(actual_dir/'trace_1ms.npz')
        if model is None:
            checkpoint=Path(r['checkpoint']);sha=r['checkpoint_sha256'];model,payload=fm.load(checkpoint,sha);INPUTS[str(checkpoint)]=sha
        if r['checkpoint_sha256']!=sha or Path(r['checkpoint'])!=checkpoint:raise ValueError('mixed checkpoints')
        for file,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions')]:
            if INPUTS[str(actual_dir/file)]!=r[key]['sha256']:raise ValueError('actual artifact hash drift')
        tr=load_json(td/'RESULT.json')
        if INPUTS[str(td/'SAMPLES_5MS.npz')]!=tr['samples']['sha256'] or INPUTS[str(td/'TRACE_1MS.npz')]!=tr['trace']['sha256']:raise ValueError('teacher hash drift')
        prefix={key:bool(np.array_equal(tt[key][:5800],at[key][:5800])) for key in ('qpos','qvel','com','action','target','actualctrl')}
        prefix['first960_X_exact']=bool(np.array_equal(t['X238'][:960],a['X238'][:960]));prefix['first960_applied_Y_exact']=bool(np.array_equal(t['Y42'][:960],a['actually_applied_action'][:960]))
        prefix['first_autonomous_X_exact']=bool(np.array_equal(t['X238'][960],a['X238'][960]))
        if not all(prefix.values()):raise ValueError('unmatched actual/teacher prefix '+name)
        x=torch.tensor(t['X238'],dtype=torch.float64);u=torch.tensor(t['command2'],dtype=torch.float64);state=model.initial_state(1);base_means=[];states=[]
        with torch.no_grad():
            for obs in x[:1160]:
                mean,state=model.base.step(obs[None],state);base_means.append(mean[0]);states.append(state[0])
            b=torch.stack(base_means);h=torch.stack(states)
            feats,g=model.features(x[960:1160],h[960:1160],u[960:1160]);ramp=(b[960:1160]+model.correction(feats,g)).tanh().numpy()
            full=torch.tensor(np.repeat(command[None],200,axis=0),dtype=torch.float64);feats,g=model.features(x[960:1160],h[960:1160],full);step=(b[960:1160]+model.correction(feats,g)).tanh().numpy()
        hidden_exact=bool(np.array_equal(h[959].numpy(),a['hidden_before'][960]))
        with torch.no_grad():
            production_mean,_=model.step(x[960:961],torch.tensor(command[None],dtype=torch.float64),h[959:960])
            production_first=production_mean[0].tanh().numpy()
        firstprod=bool(np.array_equal(production_first,a['policy_action'][960]))
        batch_delta=errors(step[0],production_first)
        if not hidden_exact or not firstprod:raise ValueError('first production output/hidden mismatch '+json.dumps(dict(condition=name,hidden=hidden_exact,production=firstprod,batch_delta=batch_delta)))
        def fit(window):
            target=t['Y42'][960:1160][window]
            return dict(ramp_target=errors(ramp[window],target),step_target=errors(step[window],target),step_minus_ramp=errors(step[window],ramp[window]))
        normalize=np.array([.15,1.]);du=(np.repeat(command[None],200,axis=0)-t['command2'][960:1160])/normalize
        perstep=np.sqrt(np.mean((step-ramp)**2,axis=1));target=t['Y42'][960:1160]
        elapsed=np.arange(200)*.005;blend=elapsed**2*(3.-2.*elapsed)
        expected_commands=blend[:,None]*command
        command_formula_error=errors(t['command2'][960:1160],expected_commands)
        if command_formula_error['max_abs']>1e-12:raise ValueError('teacher ramp label mismatch')
        offline=dict(first_autonomous_teacher_command=t['command2'][960].tolist(),first_actual_command=a['command'][960].tolist(),first_normalized_command_gap=(du[0]).tolist(),normalized_command_gap_first1s=errors(du,np.zeros_like(du)),command_gap_norm_first1s_mean=float(np.linalg.norm(du,axis=1).mean()),
            teacher_command_at_250_500_995ms=t['command2'][[1010,1060,1159]].tolist(),teacher_label_smoothstep_formula_error=command_formula_error,first200ms=fit(slice(0,40)),first1s=fit(slice(None)),first_step=dict(actual_vs_teacher_target=errors(a['policy_action'][960],target[0]),ramp_vs_target=errors(ramp[0],target[0]),step_vs_ramp=errors(step[0],ramp[0]),offline_abrupt_matches_saved_production_bitexact=firstprod,teacher_causal_base_hidden_matches_actual_initial_hidden_bitexact=hidden_exact,original_neutral_base_vs_teacherY=errors(b[960].tanh().numpy(),target[0]),batched_vs_single_row_inference_rounding=batch_delta),
            action_gap_RMS_at_0_100_250_500_995ms=perstep[[0,20,50,100,199]].tolist(),ramp_is_offline_counterfactual_only=True)
        auto=slice(5800,15400)
        fields={'qpos_all':(at['qpos'][auto],tt['qpos'][auto]),'qvel_all':(at['qvel'][auto],tt['qvel'][auto]),'root_translation':(at['qpos'][auto,:3],tt['qpos'][auto,:3]),'joint_qpos':(at['qpos'][auto,7:],tt['qpos'][auto,7:]),'COM':(at['com'][auto],tt['com'][auto])}
        div={k:summary_series(v[0],v[1]) for k,v in fields.items()}
        qa=at['qpos'][auto,3:7];qt=tt['qpos'][auto,3:7];dots=np.abs(np.sum(qa*qt,axis=1)/(np.linalg.norm(qa,axis=1)*np.linalg.norm(qt,axis=1)));angles=2*np.arccos(np.clip(dots,0,1))
        div['orientation_angle_rad']=dict(at_5ms=float(angles[4]),at_100ms=float(angles[99]),at_1s=float(angles[999]),max_first1s=float(angles[:1000].max()))
        div['absolute_first1s']={k:errors(v[0][:1000],v[1][:1000]) for k,v in fields.items()}
        results[name]=dict(shared_prefix=prefix,offline_teacher_state=offline,actual_saved_physical_divergence=div,
            actual_windows={'first200ms':segment(at,5800,6000),'first1s':segment(at,5800,6800),'after1s':segment(at,6800,15400)},teacher_windows={'first200ms':segment(tt,5800,6000),'first1s':segment(tt,5800,6800),'after1s':segment(tt,6800,15400)},actual_final_gates=r['autonomous']['gates'],actual_final_slip=r['autonomous']['slip_ratio_max'])
        # Read each archive once; release materialized arrays after paired analysis.
        for path in (td/'SAMPLES_5MS.npz',td/'TRACE_1MS.npz',actual_dir/'neural_decisions.npz',actual_dir/'trace_1ms.npz'):CACHE.pop(str(path.resolve()))
    sources={str(p):digest(p) for p in (Path(__file__),HERE/'ANALYSIS_PLAN.json',ROOT/'fast_motor.py',ROOT/'fast_data.py',ROOT/'evaluate_fast.py',OLD/'teacher/turning_teacher.py',OLD/'deceleration_moderate/api.py')}
    report=dict(status='SAVED_EVIDENCE_ONLY_NOT_CAUSAL_INTERVENTION',checkpoint=str(checkpoint),checkpoint_sha256=sha,conditions=results,source_hashes=sources,input_hashes=INPUTS,
        npz_policy='Every NPZ materialized once, reused cached in loop then released; no repeated decompression',physics_steps=0,physics_backend_imported=any(k in sys.modules for k in ('mujoco','physics','student_wave')),optimizer_updates=0,
        relative_RMS_definition='sqrt(mean((actual-teacher)^2))/max(sqrt(mean(teacher^2)),1e-12), per-time coordinate vector; translation has arbitrary world-origin normalization, so absoluteRMS also supplied',
        caution='Offline ramp and step use same ACTUAL TEACHER state and causalbasehidden, NOT body states from a ramp physical intervention. Saved actual body is abrupt-only. Association cannot establish onset as cause of long-window failure. Segment slip net-displacement denominator may amplify turns/lowprogress; threshold is not changed or reinterpreted.',
        recommendation='After selected3000 existing fixed5reports finish, separately preregister matched actual1s smooth highlevel command onset vs abrupt reference with identicalphysicalprefix/checkpoint/state/timing; retain abruptfailures, fullwindow and originalgates. Highlevel command schedule diagnostic is not scripted gait, not food deploymentqualification. No experiment launched by this diagnosis.')
    if report['physics_backend_imported']:raise ValueError('unexpected physical backend import')
    with (HERE/'DIAGNOSIS.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:dict(firststep_action_gap_RMS=v['offline_teacher_state']['first_step']['step_vs_ramp']['RMS'],first1s_rampfit_RMS=v['offline_teacher_state']['first1s']['ramp_target']['RMS'],first1s_stepfit_RMS=v['offline_teacher_state']['first1s']['step_target']['RMS'],joint_divergence_1percent_s=v['actual_saved_physical_divergence']['joint_qpos']['first_over_1e_2_s']) for k,v in results.items()},indent=2))
if __name__=='__main__':main()
