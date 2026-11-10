"""Supervised engineering MN readout on declared failed-reference physical data.
Neurons receive qualified local afferents only; no clocks/teacher/root signals.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy.optimize import lsq_linear
HERE=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z')
sys.path[:0]=[str(HERE/'dynamics'),str(HERE/'encoder')]
from module import LocalCircuit
from encoder import SensoryEncoder
SOURCE=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
    out=OUT/'readout';out.mkdir(parents=True,exist_ok=False)
    circuit=LocalCircuit();enc=SensoryEncoder();assert np.array_equal(circuit.root_ids,enc.root_ids)
    assert np.isin(enc.local_indices,circuit.allowed_sensory_local).all()
    b=load(SOURCE/'BODY.npz');n=load(SOURCE/'NATIVE.npz')['rows'];commands=load(SOURCE/'COMMANDS.npz')['rows'];init=load(SOURCE/'INITIAL.npz')
    rows=[];stimuli=[];inputs=[];voltages=[];before_spikes=circuit.net.spike_count.copy();zero=np.zeros(20000);start=time.monotonic();vmin=0.;vmax=-1e10
    total=len(commands)//5
    for i in range(total):
        k=i*5;q=b['qpos'][k,init['qa']];v=b['qvel'][k,init['va']];foot=np.zeros(6) if k==0 else n[k*80-1,1:7]
        current=enc.encode(q,v,foot)
        rows.append(np.r_[b['time'][k],circuit.filtered_rates_hz[circuit.mn_local],commands[k:k+5,1:43].mean(0)])
        inputs.append(np.r_[q,v,foot]);stimuli.append(current[enc.local_indices]);voltages.append(circuit.net.v[circuit.mn_local].copy())
        circuit.advance(current,zero,5.)
        vmin=min(vmin,float(circuit.net.v.min()));vmax=max(vmax,float(circuit.net.v.max()))
        if (i+1)%200==0:print(json.dumps(dict(samples=i+1,neural_s=circuit.net.t_ms/1000,wall_s=time.monotonic()-start)),flush=True)
    rows=np.asarray(rows);features=rows[:,1:324]/100.;labels=rows[:,324:366];times=rows[:,0]
    mask=(times>=.2-1e-9)&(times<6.-1e-9);valid=times>=6.-1e-9
    assert mask.sum()==1160 and valid.sum()==617
    names=np.asarray(enc.joint_names);W=np.zeros((323,42));fit=[]
    for j,name in enumerate(names):
        leg=name.split('_')[1][:2].lower();indices=np.flatnonzero(circuit.mn_leg==leg);x=features[mask][:,indices]
        y=np.arctanh(np.clip(labels[mask,j]/30.,-.999,.999));A=np.vstack([x,np.eye(len(indices))]);rhs=np.r_[y,np.zeros(len(indices))]
        sol=lsq_linear(A,rhs,bounds=(-1.,1.),tol=1e-9,max_iter=100)
        assert sol.success;W[indices,j]=sol.x;fit.append(dict(joint=str(name),cells=len(indices),status=int(sol.status),cost=float(sol.cost),optimality=float(sol.optimality)))
    predict=30*np.tanh(features@W)
    assert np.isfinite(predict).all() and np.abs(W).max()<=1 and np.array_equal(30*np.tanh(np.zeros(323)@W),np.zeros(42))
    np.savez_compressed(out/'MODEL.npz',weights=W,mn_root_ids=circuit.root_ids[circuit.mn_local],mn_local=circuit.mn_local,mn_leg=circuit.mn_leg,joint_names=names,feature_hz_scale=100.,rate_cap=30.,bias=np.zeros(42))
    np.savez_compressed(out/'TRAIN_TRACE.npz',time_s=times,mn_rates_hz=rows[:,1:324],target_rate=labels,predicted_rate=predict,physical_local_input=np.asarray(inputs),sensory_current_nA=np.asarray(stimuli),sensory_local_indices=enc.local_indices,mn_voltages_mV=np.asarray(voltages),train_mask=mask,temporal_mask=valid)
    spikes=circuit.net.spike_count-before_spikes
    result=dict(status='READOUT_FIT_NOT_WALKING_ACCEPTANCE',samples=len(rows),train_samples=int(mask.sum()),temporal_samples=int(valid.sum()),neural_steps=circuit.net.steps,neural_time_ms=circuit.net.t_ms,
      active_MN_count=int(np.count_nonzero(spikes[circuit.mn_local])),active_MN_by_leg={str(l):int(np.count_nonzero(spikes[circuit.mn_local][circuit.mn_leg==l])) for l in np.unique(circuit.mn_leg)},MN_spikes=int(spikes[circuit.mn_local].sum()),all_spikes=int(spikes.sum()),voltage_range_mV=[vmin,vmax],
      train_RMS_rad_s=float(np.sqrt(np.mean((predict[mask]-labels[mask])**2))),temporal_RMS_rad_s=float(np.sqrt(np.mean((predict[valid]-labels[valid])**2))),zero_baseline_temporal_RMS_rad_s=float(np.sqrt(np.mean(labels[valid]**2))),
      per_joint_temporal_RMS_rad_s=np.sqrt(np.mean((predict[valid]-labels[valid])**2,axis=0)),fitted_readout_parameters=323*7,fit_solves=fit,
      network='fixed strict real-ID conductance network; receptor/sign/normalization/pointparameters uncalibrated',source_quality='failed slow4 reference; temporal sameepisode diagnostics only',descending_current='zero',physical_steps=0,discarded_tail_ms=3,
      source_hashes={str(p):sha(p) for p in [HERE/'LEARNING_PREREGISTRATION.json',HERE/'dynamics/module.py',HERE/'dynamics/CHOSEN_CONFIG.json',HERE/'encoder/encoder.py',SOURCE/'BODY.npz',SOURCE/'NATIVE.npz',SOURCE/'COMMANDS.npz']},files={p.name:sha(p) for p in out.iterdir() if p.is_file()},wall_s=time.monotonic()-start)
    dump(out/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','neural_steps','active_MN_count','train_RMS_rad_s','temporal_RMS_rad_s','zero_baseline_temporal_RMS_rad_s','wall_s']},indent=2))
if __name__=='__main__':main()
