"""Fixed-condition conductance-adaptation probes and exact recorded-input replay."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from module import LocalCircuit, HERE, OLD

OLD_DYNAMICS=OLD.parent
TRAIN=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def state(c):
    n=c.net
    result={k:getattr(n,k).copy() for k in ['v','ge','gi','g_adapt','refractory_until_ms','spike_count','_ring','last_spikes']}
    result.update(filtered_rates_hz=c.filtered_rates_hz.copy(),t_ms=np.array(n.t_ms),steps=np.array(n.steps),ring_i=np.array(n._ring_i),dt_ms=np.array(n.dt),delay_steps=np.array(n.delay_steps))
    return result

def summary(c,mins,maxs,duration_ms):
    counts=c.net.spike_count[c.mn_local]
    rate=counts.mean()*1000/duration_ms;maximum=counts.max()*1000/duration_ms
    legs={str(leg):{'spikes':int(counts[c.mn_leg==leg].sum()),'recruited':int(np.count_nonzero(counts[c.mn_leg==leg]))} for leg in np.unique(c.mn_leg)}
    result={'neural_steps':c.net.steps,'duration_ms':duration_ms,'dt_ms':c.dt_ms,'all_spikes':int(c.net.spike_count.sum()),'MN_spikes':int(counts.sum()),'MN_recruited':int(np.count_nonzero(counts)),'MN_mean_Hz':float(rate),'MN_max_cell_mean_Hz':float(maximum),'voltage_min_mV':float(np.min(mins)),'voltage_max_mV':float(np.max(maxs)),'g_adapt_max_final_uS':float(c.net.g_adapt.max()),'per_leg':legs,'finite_final':all(np.isfinite(v).all() for v in state(c).values())}
    result.update(voltage_band_pass=result['voltage_min_mV']>=-90 and result['voltage_max_mV']<=-49,
                  derived_lower_bound_pass=result['voltage_min_mV']>=-80-1e-10,
                  propagation_rate_pass=bool(np.count_nonzero(counts)>=16 and 0<rate<100 and maximum<200 and all(v['spikes']>0 for v in legs.values())))
    return result


def main():
    start=time.monotonic()
    if (HERE/'PROBE_RESULTS.json').exists():raise RuntimeError('refuse to overwrite prior evidence')
    hashes={str(p):sha(p) for p in [HERE/'module.py',HERE/'PREREGISTRATION.json',OLD,OLD.parent/'CHOSEN_CONFIG.json',Path('/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py'),TRAIN,OLD_DYNAMICS/'RECORDED_SAMPLE_INPUT.npz']}
    with np.load(OLD_DYNAMICS/'RECORDED_SAMPLE_INPUT.npz') as z:current=z['sensory_current_nA'];indices=z['local_indices']
    c=LocalCircuit();zero=np.zeros(20000)
    assert len(indices)==556 and np.isin(indices,c.allowed_sensory_local).all()
    assert not np.any(current[np.setdiff1d(np.arange(20000),indices)]) and np.all(current>=0)
    results={};actual_steps=0;baseline=None
    for name,dt,drive,noedge in [('held_nominal',.5,current,False),('held_refined',.25,current,False),('held_replay',.5,current,False),('quiet',.5,zero,False),('legal_noedge_software_control',.5,current,True)]:
        c=LocalCircuit(dt_ms=dt)
        if noedge:c.net.We.data[:]=0;c.net.Wi.data[:]=0
        nsteps=round(200/dt);mnrate=np.empty((nsteps,323));mins=np.empty(nsteps);maxs=mins.copy();allspikes=np.empty(nsteps,int);mnspikes=allspikes.copy();gmax=mins.copy()
        for k in range(nsteps):
            mnrate[k]=c.advance(drive,zero,dt)
            mins[k]=c.net.v.min();maxs[k]=c.net.v.max();gmax[k]=c.net.g_adapt.max()
            allspikes[k]=c.net.last_spikes.sum();mnspikes[k]=c.net.last_spikes[c.mn_local].sum()
        actual_steps+=c.net.steps;final=state(c)
        results[name]=summary(c,mins,maxs,200)
        if name=='held_nominal':baseline=(mnrate.copy(),final)
        if name=='held_replay':
            results[name]['bitwise_replay']=bool(np.array_equal(mnrate,baseline[0]) and all(np.array_equal(v,final[k]) for k,v in baseline[1].items()))
            assert results[name]['bitwise_replay']
        if name=='quiet':assert not c.net.spike_count.any() and not c.filtered_rates_hz.any()
        if noedge:assert not c.net.spike_count[c.mn_local].any()
        np.savez_compressed(HERE/(name+'.npz'),mn_rates_hz=mnrate,voltage_min_mV=mins,voltage_max_mV=maxs,all_spike_count_step=allspikes,mn_spike_count_step=mnspikes,g_adapt_max_uS=gmax,**final)
        dump(name+'.json',results[name]);dump('RUN_PROGRESS.json',{'actual_neural_steps':actual_steps,'results':results})
        print(name,json.dumps(results[name]),flush=True)
    # Recorded currents copied exactly; no call to encoder and no source target/time input to circuit.
    with np.load(TRAIN) as z:
        time_s=z['time_s'].copy();currents=z['sensory_current_nA'].copy();trace_indices=z['sensory_local_indices'].copy()
    assert time_s.shape==(1817,) and currents.shape==(1817,556) and np.array_equal(trace_indices,indices)
    assert np.isfinite(currents).all() and np.all(currents>=0)
    c=LocalCircuit();bins=len(time_s);substeps=10;N=bins*substeps
    rates=np.empty((bins,323));voltage=np.empty_like(rates)
    mins=np.empty(N);maxs=mins.copy();gmax=mins.copy();spikes=np.empty(N,np.int64);mnspikes=spikes.copy();packed=np.empty((N,2500),dtype=np.uint8)
    for b in range(bins):
        rates[b]=c.filtered_rates_hz[c.mn_local];voltage[b]=c.net.v[c.mn_local]
        drive=np.zeros(20000);drive[indices]=currents[b]
        for j in range(substeps):
            k=b*substeps+j;c.advance(drive,zero,.5)
            mins[k]=c.net.v.min();maxs[k]=c.net.v.max();gmax[k]=c.net.g_adapt.max();spikes[k]=c.net.last_spikes.sum();mnspikes[k]=c.net.last_spikes[c.mn_local].sum();packed[k]=np.packbits(c.net.last_spikes,bitorder='little')
        if (b+1)%400==0:print('full_trace_bins',b+1,'steps',c.net.steps,flush=True)
    actual_steps+=c.net.steps
    full=summary(c,mins,maxs,9085);results['full_source_trace']=full
    np.savez_compressed(HERE/'SOURCE_TRACE.npz',time_s=time_s,mn_rates_hz=rates,mn_voltages_mV=voltage,sensory_current_nA=currents,sensory_local_indices=indices,every_step_time_ms=(np.arange(N)+1)*.5,voltage_min_mV=mins,voltage_max_mV=maxs,g_adapt_max_uS=gmax,all_spike_count_step=spikes,mn_spike_count_step=mnspikes,spikes_packed=packed,root_ids=c.root_ids,mn_local=c.mn_local,mn_leg=c.mn_leg)
    np.savez_compressed(HERE/'FULL_FINAL_STATE.npz',**state(c))
    with np.load(HERE/'SOURCE_TRACE.npz') as z:
        assert np.array_equal(z['sensory_current_nA'],currents) and np.array_equal(z['sensory_local_indices'],trace_indices)
        reconstructed=np.unpackbits(z['spikes_packed'],axis=1,bitorder='little',count=20000).sum(axis=0)
        assert np.array_equal(reconstructed,c.net.spike_count)
    assert all(sha(Path(p))==digest for p,digest in hashes.items())
    sensitivity={'MN_aggregate_rate_relative_difference':abs(results['held_nominal']['MN_mean_Hz']-results['held_refined']['MN_mean_Hz'])/max(results['held_nominal']['MN_mean_Hz'],1e-12),'MN_recruited_difference':abs(results['held_nominal']['MN_recruited']-results['held_refined']['MN_recruited'])}
    sensitivity['pass']=sensitivity['MN_aggregate_rate_relative_difference']<=.25 and sensitivity['MN_recruited_difference']<=.25*323
    report={'status':'EXECUTED_FIXED_HYPOTHESIS','source_hashes':hashes,'results':results,'held_timestep_sensitivity':sensitivity,'actual_neural_steps_this_script':actual_steps,'earlier_smoke_steps':10,'physical_steps':0,'training_updates':0,'source_currents_exact':True,'source_trace_sha256':sha(HERE/'SOURCE_TRACE.npz'),'full_final_state_sha256':sha(HERE/'FULL_FINAL_STATE.npz'),'all_source_hashes_unchanged':True,'wall_seconds':time.monotonic()-start,'meaning':'engineering bounds and propagation checks only; adaptation reversal and physiology unmeasured; prior current-model failures untouched'}
    dump('PROBE_RESULTS.json',report);dump('RUN_PROGRESS.json',{'actual_neural_steps':actual_steps,'results':results,'complete':True})
    print('FINAL',json.dumps(report),flush=True)

if __name__=='__main__':main()
