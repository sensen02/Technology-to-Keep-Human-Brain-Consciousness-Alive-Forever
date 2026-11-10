"""Six preregistered fixed-input neural probes; never imports a physical runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
FOUNDATION=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text());checks=json.loads((HERE/'SOURCE_CHECK.json').read_text())
    assert checks['status']=='PASS'
    assert all(sha(p)==h for p,h in checks['source_hashes'].items())
    assert sha(HERE/'INPUTS.npz')==checks['INPUTS_sha256']
    assert sha(HERE/'command.py')==prereg['command_sha256']
    rt=module('_round11_probe_foundation_runtime',FOUNDATION/'runtime.py')
    cmd=module('_round11_named_command',HERE/'command.py')
    a=load(HERE/'INPUTS.npz');initial=load(FOUNDATION/'NEURAL_START.npz')
    c=rt.load_circuit();dn=np.array([1414,13170]);mn=a['all_mn_local'];sensory=a['sensory_local']
    guards=[]
    for bad in [None,True,False,np.bool_(True),'0.025',[.025],np.array(.025),np.array([.025]),.01+0j,float('nan'),float('inf'),-float('inf'),-.001,.05000000001]:
        try:cmd.named_current(bad)
        except ValueError:guards.append(repr(bad))
        else:raise AssertionError('Invalid command accepted: '+repr(bad))
    for dose in [0.,.025,.05]:
        d=cmd.named_current(dose);expected=np.zeros(20000);expected[dn]=dose
        assert d.dtype==np.float64 and np.array_equal(d,expected)
        d[0]=1.;assert cmd.named_current(dose)[0]==0
    for role,idx,value in [('sensory',1414,.01),('descending',0,.01),('sensory',int(sensory[0]),-.01),('descending',1414,-.01)]:
        s=np.zeros(20000);d=s.copy();(s if role=='sensory' else d)[idx]=value
        before=rt.snapshot(c)
        try:c.advance(s,d,.5)
        except ValueError:pass
        else:raise AssertionError('Invalid port accepted')
        assert all(np.array_equal(v,rt.snapshot(c)[k]) for k,v in before.items())
    dump(HERE/'API_GUARDS.json',{'status':'PASS','invalid_command_rejections':guards,'invalid_current_port_rejections':4,'complete_state_unchanged':True,'neural_steps':0,'physical_steps':0})
    results=[];total=0;started=time.monotonic()
    run_sources={str(p):sha(p) for p in [Path(__file__),HERE/'command.py',HERE/'PREREG.json',HERE/'SOURCE_CHECK.json',HERE/'INPUTS.npz',FOUNDATION/'runtime.py',FOUNDATION/'NEURAL_START.npz']}
    np.savez_compressed(HERE/'NEURAL_INITIAL.npz',**initial)
    dump(HERE/'RUN_IDENTITY.json',{'sources':run_sources,'physical_steps':0,'environment':prereg['environment']})
    for mode in prereg['input_modes']:
        for dose in prereg['doses_nA']:
            name=mode+'_dose'+str(int(round(dose*1000))).zfill(3)
            folder=HERE/'conditions'/name
            if folder.exists():raise FileExistsError('Refuse to overwrite condition '+str(folder))
            folder.mkdir(parents=True)
            rt.restore_circuit(c)
            assert all(np.array_equal(v,rt.snapshot(c)[k]) for k,v in initial.items())
            start_steps=c.net.steps;failure=None
            step=[];spikes=[];bins=[];hashes=[]
            current_rows=a['sensory_held_nA'] if mode=='held_first' else a['sensory_recorded_nA']
            descending=cmd.named_current(dose)
            try:
                for b in range(100):
                    s=np.zeros(20000);s[sensory]=current_rows[b]
                    hashes.append(digest(s+descending))
                    rec={'start_time_ms':c.net.t_ms,'mn_start_hz':c.filtered_rates_hz[mn].copy(),'mn_start_mV':c.net.v[mn].copy(),'mn_start_counts':c.net.spike_count[mn].copy()}
                    for sub in range(10):
                        c.advance(s,descending,.5)
                        step.append(np.r_[c.net.t_ms,c.net.v.min(),c.net.v.max(),c.net.g_adapt.min(),c.net.g_adapt.max(),c.net.v[dn],c.net.g_adapt[dn],c.filtered_rates_hz[dn],c.net.last_spikes[dn],c.net.spike_count[dn]])
                        spikes.append(np.packbits(c.net.last_spikes,bitorder='little'))
                        assert c.net.v.min()>=-80-1e-10 and c.net.v.max()<=-50+1e-10
                        assert all(np.isfinite(x).all() and np.all(x>=0) for x in [c.net.ge,c.net.gi,c.net.g_adapt,c.filtered_rates_hz])
                    rec.update(end_time_ms=c.net.t_ms,mn_end_hz=c.filtered_rates_hz[mn].copy(),mn_end_mV=c.net.v[mn].copy(),mn_end_counts=c.net.spike_count[mn].copy())
                    bins.append(rec)
            except Exception:
                failure=traceback.format_exc()
            completed=int(c.net.steps-start_steps);total+=completed
            trace={'step_rows':np.asarray(step),'packed_spikes':np.asarray(spikes,dtype=np.uint8),'sensory_current_nA':current_rows.copy(),
                   'descending_current_nA':np.repeat(descending[dn][None,:],100,axis=0),'full_current_sha256':np.array(hashes),
                   'all_mn_local':mn,'dn_local':dn,'sensory_local':sensory}
            if bins:trace.update({k:np.asarray([x[k] for x in bins]) for k in bins[0]})
            np.savez_compressed(folder/'TRACE.npz',**trace)
            np.savez_compressed(folder/'FINAL_STATE.npz',**rt.snapshot(c))
            result={'condition':name,'input_mode':mode,'amplitude_nA':dose,'status':'COMPLETE' if failure is None else 'FAILED_PRESERVED',
                    'neural_steps':completed,'physical_steps':0,'training_updates':0,'start_clock_ms':1000.,'end_clock_ms':float(c.net.t_ms),
                    'DN_spikes':(c.net.spike_count[dn]-initial['spike_count'][dn]).tolist(),
                    'all391_MN_spikes':int((c.net.spike_count[mn]-initial['spike_count'][mn]).sum()),
                    'exported323_MN_spikes':int((c.net.spike_count[c.mn_local]-initial['spike_count'][c.mn_local]).sum()),
                    'all391_active_MN':int(np.count_nonzero(c.net.spike_count[mn]-initial['spike_count'][mn])),
                    'voltage_min_mV':float(np.min(trace['step_rows'][:,1])) if step else None,
                    'voltage_max_mV':float(np.max(trace['step_rows'][:,2])) if step else None,
                    'gA_max_uS':float(np.max(trace['step_rows'][:,4])) if step else None,
                    'failure':failure,'TRACE_sha256':sha(folder/'TRACE.npz'),'FINAL_STATE_sha256':sha(folder/'FINAL_STATE.npz')}
            dump(folder/'RESULT.json',result);results.append(result)
            dump(HERE/'RUN_COUNTS.json',{'primary_neural_steps':total,'physical_steps':0,'training_updates':0,'conditions_completed':len(results),'conditions':results})
            print(json.dumps(result),flush=True)
            if failure:
                (folder/'FAILED_RUN_SOURCE.py').write_bytes(Path(__file__).read_bytes())
                raise RuntimeError('Software/model error preserved; aborting study\n'+failure)
    assert total==6000 and all(x['neural_steps']==1000 for x in results)
    assert all(sha(p)==h for p,h in run_sources.items())
    dump(HERE/'RESULT.json',{'status':'SIX_NEURAL_PROBES_COMPLETE_REPLAY_PENDING','primary_amplitude_nA':.05,
        'primary_neural_steps':total,'physical_steps':0,'training_updates':0,'new_synapses':0,'wall_s':time.monotonic()-started,
        'step_columns':['time_ms','v_min_mV','v_max_mV','gA_min_uS','gA_max_uS','DN1414_v_mV','DN13170_v_mV','DN1414_gA_uS','DN13170_gA_uS','DN1414_filtered_hz','DN13170_filtered_hz','DN1414_spiked','DN13170_spiked','DN1414_cumulative_spikes','DN13170_cumulative_spikes'],
        'conditions':results,'source_hashes_unchanged':True,'interpretation':'Engineering constant candidate-pair injection; no brain output, no learning or walking claim.'})

if __name__=='__main__':main()
