"""Offline independent saved spike-event reconstruction; no network imports/steps."""
from pathlib import Path
import hashlib
import json
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')


def main():
    with np.load(HERE/'SOURCE_TRACE.npz') as z:trace={k:z[k] for k in z.files}
    with np.load(HERE/'FULL_FINAL_STATE.npz') as z:final={k:z[k] for k in z.files}
    with np.load(OLD) as z:
        assert all(np.array_equal(trace[k],z[k]) for k in ['time_s','sensory_current_nA','sensory_local_indices'])
    packed=trace['spikes_packed'];mn=trace['mn_local'];N=len(packed);assert N==18170
    rates=np.zeros(20000);ga=rates.copy();counts=np.zeros(20000,dtype=np.int64);refrac=rates.copy()
    start_error=0.;delay_queue=[]
    for k,row in enumerate(packed):
        if k%10==0:start_error=max(start_error,float(np.max(np.abs(rates[mn]-trace['mn_rates_hz'][k//10]))))
        sp=np.unpackbits(row,bitorder='little',count=20000).astype(bool)
        rates*=np.exp(-.5/20);rates+=50.*sp
        ga*=np.exp(-.5/100);ga[sp]+=.002/15
        counts+=sp;refrac[sp]=(k+1)*.5+2.
        assert sp.sum()==trace['all_spike_count_step'][k]
        assert sp[mn].sum()==trace['mn_spike_count_step'][k]
        assert np.isclose(ga.max(),trace['g_adapt_max_uS'][k],rtol=0,atol=1e-14)
        if k>=N-5:delay_queue.append(sp)
    errors={'start_MN_rate_Hz':start_error,'final_filtered_rate_Hz':float(np.max(np.abs(rates-final['filtered_rates_hz']))),'final_g_adapt_uS':float(np.max(np.abs(ga-final['g_adapt']))),'final_refractory_ms':float(np.max(np.abs(refrac-final['refractory_until_ms'])))}
    assert all(x==0 for x in errors.values())
    assert np.array_equal(counts,final['spike_count'])
    ring_i=int(final['ring_i'])
    for offset,sp in enumerate(delay_queue):assert np.array_equal(sp,final['_ring'][(ring_i+offset)%5])
    assert np.array_equal(delay_queue[-1],final['last_spikes'])
    assert int(final['steps'])==N and float(final['t_ms'])==9085. and int(final['delay_steps'])==4
    assert np.array_equal(trace['every_step_time_ms'],(np.arange(N)+1)*.5)
    report={'status':'PASS_SAVED_EVENT_RECONSTRUCTION','network_steps':0,'physical_steps':0,'training_updates':0,'saved_event_iterations':N,'checks':['exact original currents, indices and source time','start-of-bin MNrate from saved spikes','every-step all/MN spike counts','every-step adaptation maximum','complete final gA/filter/count/refractory','absolute final time and all5delayqueue slots','lastspikes and every-stepclock'],'errors_max':errors,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'SOURCE_TRACE.npz',HERE/'FULL_FINAL_STATE.npz',OLD]},'limitation':'This checks saved event/filter/adaptation/queue integrity for all18170steps; it does not re-solve full18170step voltage trajectory. Separate independent solver verification covers both timesteps on100ms legal input.'}
    (HERE/'SAVED_TRACE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
