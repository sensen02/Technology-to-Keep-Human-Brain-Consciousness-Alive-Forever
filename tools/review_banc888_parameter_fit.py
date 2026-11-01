"""Independent native-loss/SPSA/selection audit; no new fitting or rollout."""
from pathlib import Path
import json,hashlib,math
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-closedloop-parameter-fit/20261101T222509Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-parameter-fit-review/20261101T222509Z')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def main():
    O.mkdir(parents=True,exist_ok=True)
    output=O/('RESULT_WITH_TIMING.json' if (O/'RESULT.json').exists() else 'RESULT.json')
    assert not output.exists(), 'Never overwrite a completed review'
    if not (O/'PREREGISTRATION.json').exists():
        dump(O/'PREREGISTRATION.json',dict(scope='Post-run independent saved-array verification and descriptive time-attribution only',source=str(S),selection='Do not refit, reselect or change primary sufficient-improvement criterion',checks='Reconstruct every native physical loss component, quaternion-invariant orientation via atan2, regularization, held controls, all SPSA updates and argmin selection, identical integration initial states.',additional='Partition candidate-minus-neutral per-step physical loss before/after the common first nonfoot event. Descriptive attribution not causal ablation or a new selection criterion.',calls=dict(model_load=0,network=0,body_step=0,neural_step=0,training=0)))
    context=json.loads((S/'LOSS_CONTEXT.json').read_text());h0=float(context['h0']);W=float(context['W']);q0=np.array(context['q0']);history=json.loads((S/'TRAINING_HISTORY.json').read_text());selection=json.loads((S/'SELECTED_CANDIDATE.json').read_text());result=json.loads((S/'RESULT.json').read_text())
    assert len(history)==46 and all(not r['failed'] for r in history)
    compiled=np.load(S/'COMPILED_MODEL_ARRAYS.npz');initial=np.load(S/'INITIAL_CHECKPOINT.npz');neutral_body=np.load(S/'CACHED_NEUTRAL_150MS.npz')
    assert abs(math.fsum(map(float,compiled['body_mass']))*abs(float(compiled['gravity'][2]))-W)<1e-12
    assert float(neutral_body['com'][0,2])==h0 and np.array_equal(initial['body_qpos'][3:7],q0)
    for f in ('qpos','qvel','act','ctrl','foot_force','com'):assert json.loads((S/'GATE_RESULT.json').read_text())['checks'][f]['bitwise']
    init0=None;maxerror=0.;checks=[]
    def terms(a,x):
        n=len(a['native_COMz']);assert n>0
        foot=np.array([math.fsum(map(float,row))/W for row in a['native_foot_Fz']]);nf=a['native_nonfoot_positive_Fz']/W
        qq=a['native_root_quaternion'];scalar=qq[:,0]*q0[0]+qq[:,1:]@q0[1:]
        vector=q0[0]*qq[:,1:]-qq[:,0,None]*q0[1:]-np.cross(qq[:,1:],q0[1:]);angle2=(2*np.arctan2(np.linalg.norm(vector,axis=1),abs(scalar)))**2
        assert abs(angle2-a['native_root_angle_squared']).max()<1e-10
        per=np.column_stack([((a['native_COMz']-h0)/h0)**2,(foot-1)**2,nf,.01*angle2,.001*a['native_ctrl_mean_squared']])
        reg=.001*(math.log(x[0])**2+math.log(x[1])**2+(x[2]/3)**2)
        values=dict(zip(['com_height','foot_support','nonfoot_positive','root_rotation','control'],[math.fsum(map(float,per[:,j]))/n for j in range(5)]));values['regularization']=reg
        return values,per
    for record in history+[json.loads((S/'GATE_20MS/RESULT.json').read_text()),json.loads((S/'VALIDATION_150MS/RESULT.json').read_text())]:
        path=S/record['label'];p=json.loads((path/'PARAMETERS.json').read_text());a=np.load(path/'TRAJECTORY.npz');x=p['x'];z=p['z']
        assert .5<=x[0]<=2 and .5<=x[1]<=2 and -3<=x[2]<=3
        np.testing.assert_allclose(x,[2**z[0],2**z[1],3*z[2]],atol=1e-14,rtol=0)
        initial=np.load(path/'INITIAL_INTEGRATION_STATE.npz')['state']
        if init0 is None:init0=initial.copy()
        else:assert np.array_equal(init0,initial)
        assert not a['qfrc_applied'].any() and not a['xfrc_applied'].any() and not a['warning'].any()
        assert np.isfinite(a['native_foot_Fz']).all() and np.isfinite(a['native_nonfoot_positive_Fz']).all()
        assert (a['native_nonfoot_positive_Fz']>=0).all()
        mean_ctrl=np.mean(a['commands']**2,axis=1);assert abs(np.repeat(mean_ctrl,80)-a['native_ctrl_mean_squared']).max()<1e-14
        values,_=terms(a,x);error=max(abs(values[k]-record['terms'][k]) for k in values);maxerror=max(maxerror,error)
        assert error<1e-10 and abs(math.fsum(values.values())-record['loss'])<1e-10
        checks.append(dict(label=record['label'],loss=record['loss'],max_term_error=error))
    rng=np.random.default_rng(1413);z=np.zeros(3)
    for k in range(15):
        u=json.loads((S/f'UPDATE_{k:02d}.json').read_text());delta=rng.choice([-1.,1.],size=3);a=.1/(k+1)**.602;c=.05/(k+1)**.101
        np.testing.assert_array_equal(delta,u['delta']);np.testing.assert_array_equal(z,u['old_z'])
        plus=history[3*k+1];minus=history[3*k+2];center=history[3*k+3]
        np.testing.assert_array_equal(np.clip(z+c*delta,-1,1),plus['z']);np.testing.assert_array_equal(np.clip(z-c*delta,-1,1),minus['z'])
        gradient=(plus['loss']-minus['loss'])/(2*c)*delta;np.testing.assert_array_equal(gradient,u['gradient']);z=np.clip(z-a*gradient,-1,1)
        np.testing.assert_array_equal(z,center['z']);np.testing.assert_array_equal(z,u['new_z'])
    best=min(history,key=lambda r:r['loss']);assert best['label']==selection['selected']['label']==result['selected']
    val=np.load(S/'VALIDATION_150MS/TRAJECTORY.npz');cached=np.load(S/'CACHED_NEUTRAL_LOSS_INPUTS.npz');x=selection['selected']['x'];p0,_=terms(cached,[1,1,0]);pc,stepc=terms(val,x);_,step0=terms(cached,[1,1,0])
    assert abs(sum(p0.values())-result['neutral_validation_loss'])<1e-10
    assert abs(sum(pc.values())-result['validation_loss'])<1e-10
    np.testing.assert_array_equal(val['qpos'][:101],np.load(S/best['label']/'TRAJECTORY.npz')['qpos'])
    nf0=cached['native_nonfoot_positive_Fz'];nfc=val['native_nonfoot_positive_Fz'];first0=int(np.flatnonzero(nf0>0)[0]);firstc=int(np.flatnonzero(nfc>0)[0]);common=min(first0,firstc);diff=stepc.sum(1)-step0.sum(1);n=len(diff)
    attribution=dict(common_first_nonfoot_index=common,neutral_first_nonfoot_s=(first0+1)*.0000125,candidate_first_nonfoot_s=(firstc+1)*.0000125,weighted_loss_delta_before_first_event=float(diff[:common].sum()/n),weighted_loss_delta_from_first_event=float(diff[common:].sum()/n),regularization_delta=pc['regularization'],total_delta=float(diff.mean()+pc['regularization']))
    assert abs(attribution['total_delta']-(result['validation_loss']-result['neutral_validation_loss']))<1e-10
    neutral50=np.load(S/'TRAIN_000_NEUTRAL/TRAJECTORY.npz');trained50=np.load(S/best['label']/'TRAJECTORY.npz')
    pool=json.loads((S/'MAPPING.json').read_text());motor=sorted({i for row in pool['rows'] for i in row['local_indices']})
    def first_difference_time(aa,bb,dt,endpoint):
        diff=np.any(aa!=bb,axis=tuple(range(1,aa.ndim)));where=np.flatnonzero(diff)
        return None if not len(where) else float((where[0]+int(endpoint))*dt)
    onset=dict(native_physical_loss_fields_exact_before_first_nonfoot=all(np.array_equal(val[k][:common],cached[k][:common]) for k in ['native_COMz','native_foot_Fz','native_nonfoot_positive_Fz','native_root_quaternion','native_ctrl_mean_squared']),first_all_network_spike_difference_s=first_difference_time(trained50['spikes'],neutral50['spikes'],.0005,True),first_readout_MN_spike_difference_s=first_difference_time(trained50['spikes'][:,motor],neutral50['spikes'][:,motor],.0005,True),first_applied_command_difference_s=first_difference_time(trained50['commands'],neutral50['commands'],.001,False),first_shared_qpos_difference_s=first_difference_time(trained50['qpos'],neutral50['qpos'],.0005,False),interpretation='Labels use neural end-step times, command interval starts and body endpoint grid respectively; this is a timing audit, not a new causal ablation.')
    report=dict(passed=True,checked_trials=len(checks),max_loss_term_error=maxerror,spsa_updates_exact=15,initial_integration_states_identical=True,selected=best['label'],selected_parameters=x,training_loss_baseline=history[0]['loss'],training_loss_selected=best['loss'],validation_loss_baseline=result['neutral_validation_loss'],validation_loss_selected=result['validation_loss'],primary_sufficient_improvement=result['sufficient_improvement'],time_attribution=attribution,change_onset=onset,trial_loss_checks=checks,calls=dict(model_load=0,network=0,body_step=0,neural_step=0,training=0),limits='Independent arithmetic and source-order review; native force values were acquired by the primary physics experiment. Time attribution is observational within this deterministic paired initial condition, not a causal mechanism proof.')
    if output.name=='RESULT_WITH_TIMING.json':
        previous=json.loads((O/'RESULT.json').read_text())
        for key,value in previous.items():assert value==report[key],key
    dump(output,report);print(json.dumps({k:v for k,v in report.items() if k!='trial_loss_checks'},indent=2))
if __name__=='__main__':main()
