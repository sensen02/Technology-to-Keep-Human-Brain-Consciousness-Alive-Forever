"""Independent replay: imports frozen core, never command/runner/foundation runtime.
Manual complete restore, direct net.step, hand filter; no equation replacement.
Also checks saved current supports/source identity and archived zero reference.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
CORE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
CORE_SHA='c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250'
NET_FIELDS=('v','ge','gi','g_adapt','refractory_until_ms','spike_count','_ring','last_spikes')
C_FIELDS=('root_ids','cache_indices','allowed_sensory_local','source_sensory_local','excluded_ambiguous_sensory_local','descending_local','ascending_local','mn_local','mn_leg','strict_source_edge_rows','_sensory_mask','_descending_mask')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def state(c):
    d={k:getattr(c.net,k).copy() for k in NET_FIELDS}
    d.update({k:getattr(c,k).copy() for k in C_FIELDS})
    d.update(filtered_rates_hz=c.filtered_rates_hz.copy(),_ring_i=np.array(c.net._ring_i,dtype=np.int64),steps=np.array(c.net.steps,dtype=np.int64),t_ms=np.array(c.net.t_ms,dtype=float),neural_module_sha256=np.array(CORE_SHA),parameter_contract_json=np.array(json.dumps({'point':asdict(c.net.p),'adaptation':asdict(c.net.adaptation_params),'dt_ms':c.net.dt,'delay_steps':c.net.delay_steps,'filter_tau_ms':c.filter_tau_ms,'total_conductance_uS':c.total_conductance_uS},sort_keys=True,separators=(',',':'))))
    return d

def main():
    assert sha(CORE)==CORE_SHA
    checks=json.loads((HERE/'SOURCE_CHECK.json').read_text())
    assert all(sha(p)==h for p,h in checks['source_hashes'].items())
    a=load(HERE/'INPUTS.npz');initial=load(HERE/'NEURAL_INITIAL.npz')
    ports=json.loads((HERE/'PORTS.json').read_text())
    source=load(ports['source_hashes']['source_connectome']['path']);local=load(ports['source_hashes']['local_candidate']['path'])
    dn=np.array([1414,13170]);mn=a['all_mn_local'];sense=a['sensory_local'];positions=np.array([np.flatnonzero(mn==x)[0] for x in a['mn323_local']])
    for li,rid in [(1414,720575941433155799),(13170,720575941566493282)]:
        gi=int(local['cache_indices'][li]);assert int(local['root_ids'][li])==rid==int(source['root_ids'][gi])
        assert source['ann_primary_type'][gi]=='DNp09' and source['ann_nt_verified'][gi]=='acetylcholine'
    assert not np.intersect1d(dn,sense).size
    support_checks=[]
    for mode in ['held_first','recorded100']:
        for dose in [0.,.025,.05]:
            name=mode+'_dose'+str(int(round(dose*1000))).zfill(3);t=load(HERE/'conditions'/name/'TRACE.npz')
            expected=a['sensory_held_nA'] if mode=='held_first' else a['sensory_recorded_nA']
            assert np.array_equal(t['sensory_current_nA'],expected)
            assert np.array_equal(t['descending_current_nA'],np.full((100,2),dose))
            for b in range(100):
                manual=np.zeros(20000);manual[sense]=expected[b];manual[dn]=dose
                assert np.isfinite(manual).all() and np.all(manual>=0)
                assert np.all(manual[np.setdiff1d(np.arange(20000),np.r_[sense,dn])]==0)
                assert digest(manual)==t['full_current_sha256'][b]
            support_checks.append(name)
    baseline=load(HERE/'conditions/recorded100_dose000/TRACE.npz')
    historical={'start_rates':bool(np.array_equal(baseline['mn_start_hz'][:,positions],a['old_zero_mn_start_hz'])),
                'start_voltages':bool(np.array_equal(baseline['mn_start_mV'][:,positions],a['old_zero_mn_start_mV'])),
                'final_rates_t1_5':bool(np.array_equal(baseline['mn_end_hz'][-1,positions],a['old_zero_mn_final_hz'])),
                'final_voltages_t1_5':bool(np.array_equal(baseline['mn_end_mV'][-1,positions],a['old_zero_mn_final_mV']))}
    assert all(historical.values())
    spec=importlib.util.spec_from_file_location('_round11_independent_neural_core',CORE)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    c=mod.LocalCircuit();results=[];total=0
    for mode in ['held_first','recorded100']:
        folder=HERE/'replay'/mode
        if folder.exists():raise FileExistsError('Refuse to overwrite independent replay')
        folder.mkdir(parents=True)
        for k in C_FIELDS:assert np.array_equal(getattr(c,k),initial[k])
        for k in NET_FIELDS:setattr(c.net,k,initial[k].copy())
        c.net._ring_i=int(initial['_ring_i']);c.net.steps=int(initial['steps']);c.net.t_ms=float(initial['t_ms']);c.filtered_rates_hz=initial['filtered_rates_hz'].copy()
        assert all(np.array_equal(v,state(c)[k]) for k,v in initial.items())
        expected=load(HERE/'conditions'/(mode+'_dose050')/'TRACE.npz');end=load(HERE/'conditions'/(mode+'_dose050')/'FINAL_STATE.npz')
        currents=a['sensory_held_nA'] if mode=='held_first' else a['sensory_recorded_nA']
        steps=[];packed=[];bins=[];hashes=[];failure=None
        try:
            for b in range(100):
                current=np.zeros(20000);current[sense]=currents[b];current[1414]=.05;current[13170]=.05
                hashes.append(digest(current))
                rec={'start_time_ms':c.net.t_ms,'mn_start_hz':c.filtered_rates_hz[mn].copy(),'mn_start_mV':c.net.v[mn].copy(),'mn_start_counts':c.net.spike_count[mn].copy()}
                for j in range(10):
                    event=c.net.step(current)
                    c.filtered_rates_hz*=np.exp(-.5/20.)
                    c.filtered_rates_hz+=(1000./20.)*event
                    steps.append(np.array([c.net.t_ms,c.net.v.min(),c.net.v.max(),c.net.g_adapt.min(),c.net.g_adapt.max(),c.net.v[1414],c.net.v[13170],c.net.g_adapt[1414],c.net.g_adapt[13170],c.filtered_rates_hz[1414],c.filtered_rates_hz[13170],event[1414],event[13170],c.net.spike_count[1414],c.net.spike_count[13170]]))
                    packed.append(np.packbits(event,bitorder='little'))
                rec.update(end_time_ms=c.net.t_ms,mn_end_hz=c.filtered_rates_hz[mn].copy(),mn_end_mV=c.net.v[mn].copy(),mn_end_counts=c.net.spike_count[mn].copy());bins.append(rec)
        except Exception:failure=traceback.format_exc()
        completed=c.net.steps-2000;total+=completed
        replay={'step_rows':np.asarray(steps),'packed_spikes':np.asarray(packed,dtype=np.uint8),'sensory_current_nA':currents.copy(),
                'descending_current_nA':np.full((100,2),.05),'full_current_sha256':np.array(hashes),'all_mn_local':mn,'dn_local':dn,'sensory_local':sense}
        if bins:replay.update({k:np.asarray([x[k] for x in bins]) for k in bins[0]})
        final=state(c)
        np.savez_compressed(folder/'TRACE.npz',**replay);np.savez_compressed(folder/'FINAL_STATE.npz',**final)
        comparisons={k:bool(np.array_equal(expected[k],replay[k])) for k in expected if k in replay}
        final_comparisons={k:bool(np.array_equal(end[k],final[k])) for k in end}
        ok=failure is None and set(expected)==set(replay) and all(comparisons.values()) and set(end)==set(final) and all(final_comparisons.values()) and completed==1000
        result={'mode':mode,'status':'PASS' if ok else 'FAIL','neural_steps':completed,'physical_steps':0,'failure':failure,'trace_exact':comparisons,'all_final_fields_exact':final_comparisons,'final_clock_ms':c.net.t_ms,'final_steps':c.net.steps}
        dump(folder/'CHECK.json',result);results.append(result)
        dump(HERE/'REPLAY_COUNTS.json',{'replay_neural_steps':total,'physical_steps':0,'results':results})
        print(json.dumps({'mode':mode,'status':result['status'],'neural_steps':completed,'trace_fields':len(comparisons),'full_final_fields':len(final_comparisons)}),flush=True)
        if not ok:
            (folder/'FAILED_REPLAY_SOURCE.py').write_bytes(Path(__file__).read_bytes())
            raise RuntimeError('Replay failed; preserved '+str(folder))
    assert total==2000
    assert all(sha(p)==h for p,h in checks['source_hashes'].items())
    dump(HERE/'INDEPENDENT_CHECK.json',{'status':'PASS','neural_steps':total,'physical_steps':0,'training_updates':0,
        'replay_source_sha256':sha(__file__),'frozen_core_sha256':sha(CORE),'manual_restore':True,'manual_current_no_command_import':True,
        'direct_net_step_and_manual_filter':True,'independent_equation_rederivation':False,'frozen_equations_previously_verified':True,
        'saved_current_support_all600bins_exact':support_checks,'historical_zero_reference':historical,
        'historical_reference_limit':'Only exported323 MN start-bin rates/volts and finalt1.5 rates/volts available; no historical full20k t1.5 checkpoint claim.',
        'replays':results,'source_hashes_unchanged':True})

if __name__=='__main__':main()
