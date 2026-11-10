"""Independent saved-data/readout/neural replay; imports no trainer or physics runner.

No fitting, optimization, MuJoCo import or physical steps occur here. LocalCircuit
is intentionally reused for deterministic neural replay; source/labels/readout
and constrained-ridge optimality are independently reconstructed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DESKTOP = Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z')
SOURCE = Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
sys.path[:0] = [str(BASE/'encoder'), str(BASE/'dynamics')]
from encoder import SensoryEncoder
from module import LocalCircuit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


class Verification:
    def __init__(self,mode):
        self.mode=mode
        self.records=[]
        self.errors={}
        self.steps=0
        self.started=time.monotonic()
        self.provenance={}

    def check(self,name,condition,**details):
        ok=bool(condition)
        self.records.append(dict(name=name,passed=ok,**details))
        if not ok:
            raise AssertionError(name)

    def equal(self,name,a,b,tolerance=0.):
        a,b=np.asarray(a),np.asarray(b)
        self.check(name+'_shape',a.shape==b.shape)
        if a.dtype.kind in 'USO' or b.dtype.kind in 'USO':
            self.check(name,np.array_equal(a,b))
            return
        self.check(name+'_finite',np.isfinite(a).all() and np.isfinite(b).all())
        error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        self.errors[name]=max(self.errors.get(name,0.),error)
        self.check(name,(np.array_equal(a,b) if tolerance==0 else error<=tolerance),max_abs_error=error,tolerance=tolerance)

    def hashed(self,path,expected=None):
        path=Path(path)
        value=sha(path)
        if expected is not None:
            self.check('hash_'+str(path),value==expected)
        self.provenance[str(path)]=value
        return value

    def summarize(self,**extra):
        return dict(status='PASS_ENGINEERING_REPLAY_NOT_CALIBRATION',mode=self.mode,
                    checks_passed=len(self.records),checks=self.records,max_abs_errors=self.errors,
                    actual_neural_replay_steps=self.steps,new_learning_updates=0,new_physical_steps=0,
                    source_hashes=self.provenance,wall_s=time.monotonic()-self.started,
                    runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__),
                    biological_calibration='CALIBRATION_FAILED; fixed .02 fallback only',
                    walking_claim=False,**extra)


def initialize(v):
    v.hashed(HERE/'PREREG.json')
    v.hashed(Path(__file__))
    enc=SensoryEncoder()
    circuit=LocalCircuit(total_conductance_uS=.02,dt_ms=.5)
    v.equal('local_root_order',enc.root_ids,circuit.root_ids)
    v.check('qualified_ports_subset',enc.qualified_count==556 and np.isin(enc.local_indices,circuit.allowed_sensory_local).all())
    chosen=read_json(BASE/'dynamics/CHOSEN_CONFIG.json')
    v.check('frozen_fallback_not_calibration',chosen['total_conductance_uS']==.02 and chosen['dt_ms']==.5 and chosen['status']=='CALIBRATION_FAILED')
    for path in (BASE/'dynamics/module.py',BASE/'dynamics/CHOSEN_CONFIG.json',BASE/'encoder/encoder.py',BASE/'LEARNING_PREREGISTRATION.json'):
        v.hashed(path)
    v.check('no_physics_or_trainer_import',not any(x in sys.modules for x in ('mujoco','train_readout','run_closed_loop')))
    return enc,circuit


def model_checks(v,model,enc,circuit):
    w=model['weights']
    v.check('readout_weight_shape',w.shape==(323,42))
    v.check('readout_finite_bounded',np.isfinite(w).all() and np.all(np.abs(w)<=1))
    v.equal('MN_local_identity',model['mn_local'],circuit.mn_local)
    v.equal('MN_root_identity',model['mn_root_ids'],circuit.root_ids[circuit.mn_local])
    v.equal('MN_leg_identity',model['mn_leg'],circuit.mn_leg)
    v.equal('actual42_joint_order',model['joint_names'],enc.joint_names)
    v.check('engineering_scales',float(model['feature_hz_scale'])==100. and float(model['rate_cap'])==30.)
    v.equal('zero_bias',model['bias'],np.zeros(42))
    jointlegs=np.array([name.removeprefix('joint_')[:2].lower() for name in enc.joint_names])
    allowed=circuit.mn_leg[:,None]==jointlegs[None,:]
    v.check('same_leg_only_mask',np.all(w[~allowed]==0),allowed_parameter_count=int(allowed.sum()))
    v.equal('silent_MN_zero_action',30*np.tanh(np.zeros(323)@w+model['bias']),np.zeros(42))
    return w


def source_inputs(body,native,initial,indices):
    q=body['qpos'][indices][:,initial['qa']]
    vel=body['qvel'][indices][:,initial['va']]
    foot=np.zeros((len(indices),6))
    nonzero=indices>0
    foot[nonzero]=native[indices[nonzero]*80-1,1:7]
    return np.concatenate((q,vel,foot),axis=1)


def replay(v,enc,circuit,inputs,currents,mn_rates,mn_voltages,handover=None):
    zero=np.zeros(20000)
    maxima={'encoder_current_nA':0.,'MN_start_rates_hz':0.,'MN_start_voltage_mV':0.}
    vmin=0.;vmax=-1e10
    for i,(physical,recorded) in enumerate(zip(inputs,currents)):
        q,vel,load=physical[:42],physical[42:84],physical[84:]
        current=enc.encode(q,vel,load)
        maxima['encoder_current_nA']=max(maxima['encoder_current_nA'],float(np.max(np.abs(current[enc.local_indices]-recorded))))
        maxima['MN_start_rates_hz']=max(maxima['MN_start_rates_hz'],float(np.max(np.abs(circuit.filtered_rates_hz[circuit.mn_local]-mn_rates[i]))))
        maxima['MN_start_voltage_mV']=max(maxima['MN_start_voltage_mV'],float(np.max(np.abs(circuit.net.v[circuit.mn_local]-mn_voltages[i]))))
        if not all(np.isfinite(x).all() for x in (current,recorded,mn_rates[i],mn_voltages[i])):
            raise AssertionError('Nonfinite replay data')
        if np.any(current[~enc.allowed_mask]!=0):
            raise AssertionError('Off-port current')
        if handover is not None and i==200:
            for key,value in (('mn_rates_hz',circuit.filtered_rates_hz[circuit.mn_local]),('neural_v',circuit.net.v),('neural_ge',circuit.net.ge),('neural_gi',circuit.net.gi),('neural_adaptation',circuit.net.adaptation),('neural_delay_ring',circuit.net._ring),('neural_ring_i',circuit.net._ring_i)):
                v.equal('handover_'+key,handover[key],value)
        # Replay the RECORDED current, independently already compared to encoder.
        stimulus=np.zeros(20000)
        stimulus[enc.local_indices]=recorded
        circuit.advance(stimulus,zero,5.)
        v.steps=circuit.net.steps
        vmin=min(vmin,float(circuit.net.v.min()))
        vmax=max(vmax,float(circuit.net.v.max()))
        if (i+1)%200==0:
            print(json.dumps(dict(mode=v.mode,samples_replayed=i+1,actual_neural_steps=v.steps,wall_s=time.monotonic()-v.started)),flush=True)
    for name,error in maxima.items():
        v.errors[name]=error
        v.check(name+'_entire_trace_exact',error==0.,max_abs_error=error)
    return [vmin,vmax]


def snapshot(c):
    return dict(v=c.net.v,ge=c.net.ge,gi=c.net.gi,adaptation=c.net.adaptation,
                refractory_until_ms=c.net.refractory_until_ms,spike_count=c.net.spike_count,
                delay_ring=c.net._ring,ring_i=c.net._ring_i,filtered_rates_hz=c.filtered_rates_hz,
                steps=c.net.steps)


def verify_readout(v):
    folder=DESKTOP/'readout'
    result=read_json(folder/'RESULT.json')
    for path,expected in result['source_hashes'].items():v.hashed(path,expected)
    for name,expected in result['files'].items():v.hashed(folder/name,expected)
    v.hashed(folder/'RESULT.json')
    model,trace=load(folder/'MODEL.npz'),load(folder/'TRAIN_TRACE.npz')
    enc,circuit=initialize(v)
    w=model_checks(v,model,enc,circuit)
    body,native=load(SOURCE/'BODY.npz'),load(SOURCE/'NATIVE.npz')['rows']
    commands,initial=load(SOURCE/'COMMANDS.npz')['rows'],load(SOURCE/'INITIAL.npz')
    v.hashed(SOURCE/'INITIAL.npz')
    n=len(commands)//5
    v.check('sample_count_1817',n==1817 and len(trace['time_s'])==n)
    indices=np.arange(n)*5
    inputs=source_inputs(body,native,initial,indices)
    v.equal('exact_source_q_velocity_foot_rows',trace['physical_local_input'],inputs)
    v.equal('source_start_time',trace['time_s'],body['time'][indices])
    labels=np.stack([np.mean(commands[start:start+5,1:43],axis=0) for start in indices])
    v.equal('independent_five_ms_mean_applied_labels',trace['target_rate'],labels)
    v.equal('qualified_current_local_indices',trace['sensory_local_indices'],enc.local_indices)
    v.check('current_shape',trace['sensory_current_nA'].shape==(1817,556))
    v.check('current_bounds',np.isfinite(trace['sensory_current_nA']).all() and np.all((trace['sensory_current_nA']>=0)&(trace['sensory_current_nA']<=.06)))
    times=body['time'][indices]
    train=(times>=.2-1e-9)&(times<6.-1e-9)
    temporal=times>=6.-1e-9
    v.equal('train_rows',train,trace['train_mask'])
    v.equal('temporal_rows',temporal,trace['temporal_mask'])
    v.check('temporal_split_counts',train.sum()==1160 and temporal.sum()==617 and not np.any(train&temporal))
    v.check('discarded_tail_three_ms',len(commands)-5*n==3 and result['discarded_tail_ms']==3)
    x=trace['mn_rates_hz']/100.
    predict=30*np.tanh(x@w)
    v.equal('independent_readout_predictions',predict,trace['predicted_rate'],1e-10)
    rms=lambda a:float(np.sqrt(np.mean(a*a)))
    metrics=dict(train_RMS_rad_s=rms(predict[train]-labels[train]),temporal_RMS_rad_s=rms(predict[temporal]-labels[temporal]),zero_baseline_temporal_RMS_rad_s=rms(labels[temporal]))
    for key,value in metrics.items():v.equal('metric_'+key,value,result[key],1e-10)
    v.equal('per_joint_temporal_RMS',np.sqrt(np.mean((predict[temporal]-labels[temporal])**2,axis=0)),result['per_joint_temporal_RMS_rad_s'],1e-10)
    kkt=[]
    for j,name in enumerate(enc.joint_names):
        leg=name.removeprefix('joint_')[:2].lower()
        idx=np.flatnonzero(model['mn_leg']==leg)
        design=x[train][:,idx]
        target=np.arctanh(np.clip(labels[train,j]/30.,-.999,.999))
        beta=w[idx,j]
        residual=design@beta-target
        gradient=design.T@residual+beta
        projected=float(np.max(np.abs(beta-np.clip(beta-gradient,-1,1))))
        lower=beta<=-1+1e-7;upper=beta>=1-1e-7
        violation=np.where(lower,np.minimum(gradient,0),np.where(upper,np.maximum(gradient,0),gradient))
        raw=float(np.max(np.abs(violation)))
        cost=.5*(float(residual@residual)+float(beta@beta))
        v.check('KKT_projected_gradient_'+name,projected<=1e-6,residual=projected,tolerance=1e-6)
        v.equal('ridge_objective_'+name,cost,result['fit_solves'][j]['cost'],1e-10)
        kkt.append(dict(joint=name,projected_gradient_residual=projected,raw_active_set_KKT_residual=raw,bound_active_cells=int(np.count_nonzero(lower|upper)),objective=cost))
    vrange=replay(v,enc,circuit,inputs,trace['sensory_current_nA'],trace['mn_rates_hz'],trace['mn_voltages_mV'])
    v.check('exact_replay_neural_step_count',v.steps==18170==result['neural_steps'])
    v.equal('neural_time_ms',circuit.net.t_ms,result['neural_time_ms'])
    v.equal('full_interval_end_voltage_range',vrange,result['voltage_range_mV'])
    for key,value in dict(MN_spikes=int(circuit.net.spike_count[circuit.mn_local].sum()),all_spikes=int(circuit.net.spike_count.sum()),active_MN_count=int(np.count_nonzero(circuit.net.spike_count[circuit.mn_local]))).items():v.equal(key,value,result[key])
    for leg in np.unique(circuit.mn_leg):
        v.equal('active_MN_'+str(leg),np.count_nonzero(circuit.net.spike_count[circuit.mn_local][circuit.mn_leg==leg]),result['active_MN_by_leg'][str(leg)])
    np.savez_compressed(HERE/'REPLAY_READOUT_FINAL.npz',**snapshot(circuit))
    return v.summarize(samples=n,metrics=metrics,kkt=kkt,max_projected_gradient_residual=max(item['projected_gradient_residual'] for item in kkt),
                       voltage_range_mV=vrange,voltage_minus90_sanity_pass=vrange[0]>=-90.,
                       temporal_better_than_zero=metrics['temporal_RMS_rad_s']<metrics['zero_baseline_temporal_RMS_rad_s'],
                       reference_status='failed slow4; temporal sameepisode only',
                       replay_final_state_note='Readout did not archive full neural final state; replay-derived snapshot is saved independently, not claimed as original final-state equality.')


def verify_closed_loop(v):
    folder=DESKTOP/'closed_loop'
    result,protocol=read_json(folder/'RESULT.json'),read_json(folder/'PROTOCOL.json')
    for name,expected in result['files'].items():v.hashed(folder/name,expected)
    v.hashed(folder/'RESULT.json')
    for path,key in ((BASE/'run_closed_loop.py','runner_sha256'),(DESKTOP/'readout/MODEL.npz','readout_sha256'),(BASE/'dynamics/module.py','network_sha256'),(BASE/'encoder/encoder.py','encoder_sha256')):v.hashed(path,protocol[key])
    policy,model=load(folder/'POLICY.npz'),load(DESKTOP/'readout/MODEL.npz')
    body,native,initial=load(folder/'BODY.npz'),load(folder/'NATIVE.npz')['rows'],load(folder/'INITIAL.npz')
    commands=load(folder/'COMMANDS.npz')['rows']
    enc,circuit=initialize(v)
    w=model_checks(v,model,enc,circuit)
    v.check('policy600',len(policy['time_s'])==600)
    indices=np.arange(600)*5
    inputs=source_inputs(body,native,initial,indices)
    v.equal('closed_actual_source_q_velocity_load',inputs,policy['local_physical_input'])
    v.equal('closed_start_time',body['time'][indices],policy['time_s'])
    v.equal('closed_teacher_phase_flag',policy['teacher_warmup'],indices<1000)
    prediction=30*np.tanh((policy['mn_rates_hz']/100.)@w)
    v.equal('closed_independent_predictions',prediction,policy['predicted_rate'],1e-10)
    source_commands=load(SOURCE/'COMMANDS.npz')['rows']
    v.equal('closed_exact_warmup_source_rates',commands[:1000,1:43],source_commands[:1000,1:43])
    held=np.repeat(prediction[200:],5,axis=0)
    v.equal('closed_posthandover_MN_only_commands',commands[1000:,1:43],held,1e-10)
    vrange=replay(v,enc,circuit,inputs,policy['sensory_current_nA'],policy['mn_rates_hz'],policy['mn_voltages_mV'],load(folder/'HANDOVER.npz'))
    final=load(folder/'NEURAL_FINAL.npz')
    for key,value in snapshot(circuit).items():v.equal('full_neural_final_'+key,value,final[key])
    v.check('closed_exact_neural_steps6000',v.steps==6000==result['neural_steps'])
    v.equal('closed_voltage_range',vrange,result['neural_voltage_range_mV'])
    v.equal('closed_MN_spikes',circuit.net.spike_count[circuit.mn_local].sum(),result['MN_spikes'])
    v.equal('closed_active_MN',np.count_nonzero(circuit.net.spike_count[circuit.mn_local]),result['active_MN_count'])
    return v.summarize(samples=600,voltage_range_mV=vrange,voltage_minus90_sanity_pass=vrange[0]>=-90.,
                       physical_replay='Not performed here; parent owns independent native physical replay.',
                       preserved_parent_physical_gate=result['physical_gate'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['readout','closed_loop'])
    args=parser.parse_args()
    v=Verification(args.mode)
    output=HERE/('VERIFY_'+args.mode.upper()+'.json')
    if output.exists():
        raise FileExistsError('Preserve existing verification output: '+str(output))
    try:
        result=verify_readout(v) if args.mode=='readout' else verify_closed_loop(v)
    except Exception:
        failure=v.summarize(error=traceback.format_exc())
        failure['status']='VERIFICATION_FAILED'
        write_json(HERE/('FAILED_'+args.mode.upper()+'.json'),failure)
        raise
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','mode','checks_passed','actual_neural_replay_steps','wall_s')},indent=2),flush=True)


if __name__=='__main__':main()
