"""Hold a parent-selected real saved sensory sample; execute no physical model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from module import LocalCircuit, HERE


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    if (HERE/'RECORDED_SAMPLE_RESULT.json').exists():raise RuntimeError('preserve previous evidence')
    ep=HERE.parent/'encoder/encoder.py'
    verification=HERE.parent/'encoder/VERIFICATION.json'
    assert json.loads(verification.read_text())['status']=='PASS_NUMERICAL_ENGINEERING_ONLY'
    spec=importlib.util.spec_from_file_location('verified_encoder',ep)
    em=importlib.util.module_from_spec(spec);spec.loader.exec_module(em)
    e=em.SensoryEncoder()
    source=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
    with np.load(source/'INITIAL.npz') as f:qa,va=f['qa'],f['va']
    with np.load(source/'BODY.npz') as f:q=f['qpos'][1000,qa];v=f['qvel'][1000,va];t=float(f['time'][1000])
    with np.load(source/'NATIVE.npz') as f:load=f['rows'][79999,1:7]
    current=e.encode(q,v,load);zero=np.zeros(20000)
    c=LocalCircuit();assert np.isin(np.flatnonzero(current),e.local_indices).all();assert np.isin(e.local_indices,c.allowed_sensory_local).all()
    np.savez_compressed(HERE/'RECORDED_SAMPLE_INPUT.npz',q42_rad=q,qvel42_rad_s=v,load6_native=load,sensory_current_nA=current,descending_current_nA=zero,source_time_s=t,local_indices=e.local_indices)
    results=[];state0=None;actual_steps=0
    for label,dt in [('nominal',.5),('refined',.25),('nominal_replay',.5)]:
        c=LocalCircuit(dt_ms=dt);counts=[];rates=[];vrange=[];arange=[]
        for _ in range(round(200/dt)):
            rates.append(c.advance(current,zero,dt))
            counts.append(c.net.spike_count[c.mn_local].copy())
            vrange.append([float(c.net.v.min()),float(c.net.v.max())])
            arange.append([float(c.net.adaptation.min()),float(c.net.adaptation.max())])
        actual_steps+=c.net.steps;count=c.net.spike_count[c.mn_local]
        state={k:getattr(c.net,k).copy() for k in ['v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring']}
        state['filtered_rates_hz']=c.filtered_rates_hz.copy()
        if label=='nominal':state0=state;rates0=np.array(rates)
        replay_equal=None if label!='nominal_replay' else bool(all(np.array_equal(x,state[k]) for k,x in state0.items()) and np.array_equal(rates0,rates))
        if replay_equal is not None:assert replay_equal
        result={'label':label,'dt_ms':dt,'neural_steps':c.net.steps,'MN_spikes':int(count.sum()),'MN_recruited':int(np.count_nonzero(count)),'MN_mean_Hz':float(count.mean()/ .2),'MN_max_cell_mean_Hz':float(count.max()/ .2),'voltage_min_mV':float(np.min(vrange)),'voltage_max_mV':float(np.max(vrange)),'max_adaptation_nA':float(np.max(arange)),'exact_replay':replay_equal,'per_leg':{str(leg):{'spikes':int(count[c.mn_leg==leg].sum()),'recruited':int(np.count_nonzero(count[c.mn_leg==leg]))} for leg in np.unique(c.mn_leg)}}
        result['all_legs_propagated']=all(r['spikes']>0 for r in result['per_leg'].values())
        result['original_conservative_voltage_floor_pass']=result['voltage_min_mV']>=-90
        np.savez_compressed(HERE/('recorded_sample_'+label+'.npz'),MN_filtered_Hz=np.array(rates),MN_spike_count=np.array(counts),voltage_range_mV=np.array(vrange),adaptation_range_nA=np.array(arange),**state)
        results.append(result);print(json.dumps(result),flush=True)
    report={'status':'EXECUTED_FIXED_CONFIG','config_uS':.02,'source_sample':1000,'source_time_s':t,'load_sample_row':79999,'source_hashes':{str(p):sha(p) for p in [source/'INITIAL.npz',source/'BODY.npz',source/'NATIVE.npz',ep,verification,HERE/'RECORDED_SAMPLE_PREREGISTRATION.json']},'actual_neural_steps':actual_steps,'physical_steps':0,'training_updates':0,'input_current_nonzero_cells':int(np.count_nonzero(current)),'input_current_min_max_nA':[float(current.min()),float(current.max())],'results':results,'aggregate_rate_relative_difference':abs(results[0]['MN_mean_Hz']-results[1]['MN_mean_Hz'])/max(results[0]['MN_mean_Hz'],1e-12),'meaning':'one real recorded sensory sample held constant; not ongoing closed-loop physical feedback or brain output; original broad calibration FAILED remains unchanged'}
    (HERE/'RECORDED_SAMPLE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
