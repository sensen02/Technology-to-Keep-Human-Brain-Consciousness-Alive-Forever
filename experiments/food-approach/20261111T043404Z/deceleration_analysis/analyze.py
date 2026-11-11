"""Read existing teacher arrays and neural sources; no Core/physics/optimizer."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach')
TEACHER=ROOT/'20261110T191449Z/deceleration_moderate'
MODEL_IO=ROOT/'20261111T010218Z/train/model_io.py'
CHECKPOINT=MODEL_IO.parent/'TRAIN/checkpoint_002000.pt'
SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def rms(x):return float(np.sqrt(np.mean(np.square(x))))

def main():
    torch.set_num_threads(1)
    before={};verification=json.loads((TEACHER/'VERIFICATION.json').read_text())
    assert verification['all_integrity_checks_passed']
    cases={};samples={};traces={}
    for name in ('straight_control_100','straight_085','straight_090'):
        folder=TEACHER/name;result=json.loads((folder/'RESULT.json').read_text())
        for filename in ('RESULT.json','SAMPLES_5MS.npz','TRACE_1MS.npz','IDENTITY.json'):
            p=folder/filename;before[str(p)]=digest(p)
            if str(p) in verification['artifact_sha256']:assert before[str(p)]==verification['artifact_sha256'][str(p)]
        assert result['all_gates'] and result['all_functional_gates'] and result['cycles']['all_legs_completed'] and result['nonfoot_force_max_native']==0
        assert result['physical']==dict(native_dt_s=result['physical']['native_dt_s'],actual_native_steps=1232000,nq=49,nv=48,actuators=42)
        assert abs(result['physical']['native_dt_s']-.0000125)<1e-15
        t=read_arrays(folder/'TRACE_1MS.npz');s=read_arrays(folder/'SAMPLES_5MS.npz');traces[name]=t;samples[name]=s
        assert t['qpos'].shape==(15400,49) and s['X238'].shape==(2880,238)
        assert np.array_equal(s['post_prime'],np.arange(2880)>=960)
        assert not np.any(t['external_body_force_max']) and not np.any(t['external_generalized_force_max'])
        ev=result['cycles']['events'];events={leg:[e for e in ev if e.get('leg')==leg and e.get('type')=='complete'] for leg in ('LF','LH','LM','RF','RH','RM')}
        periods={leg:[float(e['duration']) for e in event] for leg,event in events.items()}
        cases[name]=dict(forward=result['forward_signed_native_per_s'],forward_gate_margin=result['forward_signed_native_per_s']-.05,slip=result['slip_ratio_max'],support=result['foot_support_mean'],COM_min=result['com_height_min'],strict_nonfoot_zero=True,all5=True,cycles=result['cycles']['complete_per_leg'],cycle_duration_s=periods,max_cycle_duration_s=max(d for ds in periods.values() for d in ds),phase_span_post=float(s['source_phase'][-1]-s['source_phase'][960]),steady_command=s['command2'][-1].tolist(),post_target_speed_observation_unique=np.unique(s['X238'][960:,-1]).tolist(),full_original_native_acceptance=result['original_acceptance'])
    control=samples['straight_control_100'];ct=traces['straight_control_100'];prefix={}
    for name in ('straight_085','straight_090'):
        s=samples[name];t=traces[name];assert all(np.array_equal(t[k][:5800],ct[k][:5800]) for k in t)
        assert all(np.array_equal(s[k][:960],control[k][:960]) for k in s)
        prefix[name]=dict(all_trace_fields_first5800_exact=True,all_sample_fields_first960_exact=True)
        phase=s['source_phase'][960:];cphase=control['source_phase']
        # Interpolation is descriptive alignment of saved teacher data, not a deployed phase input.
        aligned=np.column_stack([np.interp(phase,cphase,control['X238'][:,j]) for j in range(238)])
        aligned_y=np.column_stack([np.interp(phase,cphase,control['Y42'][:,j]) for j in range(42)])
        actual=s['X238'][960:];same=control['X238'][960:]
        cases[name]['descriptive_error_vs_control']=dict(same_time_obs_joint0_42_RMS=rms(actual[:,:42]-same[:,:42]),phase_aligned_obs_joint0_42_RMS=rms(actual[:,:42]-aligned[:,:42]),same_time_normalized_target126_168_RMS=rms(actual[:,126:168]-same[:,126:168]),phase_aligned_normalized_target126_168_RMS=rms(actual[:,126:168]-aligned[:,126:168]),same_time_action_RMS=rms(s['Y42'][960:]-control['Y42'][960:]),phase_aligned_action_RMS=rms(s['Y42'][960:]-aligned_y),action_RMS=rms(s['Y42'][960:]),scope='phase alignment uses privileged teacher metadata ONLY for offline diagnosis, never actor inputs; not causal isolation')
        cases[name]['fraction_of_teacher_matched_control']=cases[name]['forward']/cases['straight_control_100']['forward']
    # Neural inference only: parent receives [0,0] on all actual straight teacher histories.
    spec=importlib.util.spec_from_file_location('_deceleration_readonly_parent_io',MODEL_IO);io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
    model,payload=io.load_checked(CHECKPOINT,SHA)
    invariant={}
    with torch.no_grad():
        for name,s in samples.items():
            state=model.initial_state(1);plain=model.parent.initial_state(1);equal_mean=equal_state=True;max_error=0.;saved_mean=[]
            for row in s['X238']:
                x=torch.tensor(row[None],dtype=torch.float64);u=torch.zeros((1,2),dtype=torch.float64)
                mean,state=model.step(x,u,state);other,plain=model.parent.step(x,u,plain)
                equal_mean &= torch.equal(mean,other);equal_state &= all(torch.equal(a,b) for a,b in zip(state,plain));max_error=max(max_error,float((mean-other).abs().max()));saved_mean.append(mean[0].numpy().copy())
            saved_mean=np.asarray(saved_mean);target=s['Y42'];post=np.arange(2880)>=960
            residual=np.arctanh(np.clip(target[post],-.999999,.999999))-saved_mean[post]
            cases[name]['frozen_parent_teacher_distribution_fit']=dict(bounded_action_MSE=float(np.mean((np.tanh(saved_mean[post])-target[post])**2)),absolute_required_latent_percentiles=dict(zip(('p50','p90','p95','p99','max'),map(float,np.percentile(np.abs(residual),[50,90,95,99,100])))),teacher_action_fraction_abs_ge_point999=float(np.mean(np.abs(target[post])>=.999)),scope='Descriptive teacher-distribution additive latent residual, not fitted speed branch or physical recovery proof')
            invariant[name]=dict(all2880_parent_means_exact=bool(equal_mean),all2880_192_memories_exact=bool(equal_state),maximum_mean_error=max_error,scope='existing teacher observations replayed with parent command[0,0]; no physical trajectory or new speed policy')
    assert all(v['all2880_parent_means_exact'] and v['all2880_192_memories_exact'] for v in invariant.values())
    sources=[MODEL_IO,ROOT/'20261110T235830Z/train_two_only_control/model_io.py',ROOT/'20261110T191449Z/residual_candidate/residual_motor.py',ROOT/'20261110T203537Z/fast_motor.py',ROOT/'20261110T203537Z/correction_fit5000/FINAL_REPORT.md',Path(io.BASE)/'models.py',Path(io.BASE)/'physics.py',ROOT/'20261110T235830Z/evaluation/evaluate_motor.py',ROOT/'20261110T235830Z/evaluation/verify_actual.py',TEACHER/'REPORT.md',TEACHER/'api.py',TEACHER/'VERIFICATION.json',TEACHER/'CURRICULUM_ADMISSION.json',CHECKPOINT]
    for p in sources:before[str(p)]=digest(p)
    assert not any(n in sys.modules for n in ('mujoco','physics','student_wave'))
    unchanged=all(digest(p)==h for p,h in before.items());assert unchanged
    report=dict(kind='READONLY_DECELERATION_ANALYSIS_NOT_QUALIFICATION',cases=cases,shared_prefix=prefix,forward_zero_parent_neural_replay=invariant,neural_time_constants=dict(base_alpha=float(model.base.ALPHA),base_tau_s=float(model.base.NEURAL_INTEGRATION_TAU_S),turn_residual_alpha=float(model.parent.ALPHA),policy_dt_s=.005),parent_checkpoint=str(CHECKPOINT),parent_sha256=SHA,actor_target_feedback=dict(physical_mask_channels=int(model.base.physical_mask.sum()),masked_ranges=[[42,84],[126,168],[195,237]],target_position_direct_inhibitory=True,minimum_gain=float(model.base.gain().min()),maximum_gain=float(model.base.gain().max())),inputs_unchanged=unchanged,input_hashes=before,Core_constructions=0,physics_integration_steps=0,optimizer_updates=0,new_training_datasets_created=0,scope='Teacher algorithm data and existing-model inference analysis only. No speed branch, modified weights, new qualification or motor deployment.')
    (HERE/'METRICS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_3_teacher_gates_verified=True,prefix_exact=True,parent_neural_replay_exact=True,base_alpha=report['neural_time_constants']['base_alpha'],forwards={k:v['forward'] for k,v in cases.items()},Core_constructions=0,physics_steps=0,optimizer_updates=0)))
if __name__=='__main__':main()
