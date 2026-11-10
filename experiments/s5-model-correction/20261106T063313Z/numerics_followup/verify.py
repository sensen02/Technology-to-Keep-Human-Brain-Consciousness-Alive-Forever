"""Independent saved-spike/filter/clock/event/math verification (no neural stepping)."""
import sys,pickle
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy import sparse
import run as r

def main():
    summary=r.js(r.O/'SUMMARY.json');ident=r.js(r.OLD/'TRANSLATOR_IDENTITIES.json');matrix=np.array(ident['old_matrix']);limits=np.array(ident['limits']);pools=ident['pools'];details=[]
    orig=r.cache(.5)[1.3];base=r.arrays(r.OLD/'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz')
    for row in summary['results']:
        folder=Path(row['folder']);z=r.arrays(folder/'NEURAL.npz');dt=row['dt_ms'];n=round(1/dt);sp=np.unpackbits(z['spikes12000_packed'],axis=1,count=12000)
        with (folder/'INITIAL_STATE.pkl').open('rb') as f:initial=pickle.load(f)
        with (folder/'FINAL_STATE.pkl').open('rb') as f:final=pickle.load(f)
        a=initial['net'];b=orig['net'];end=final['net']
        for k in r.FIELDS:assert np.array_equal(getattr(a,k),getattr(b,k))
        assert a.t_ms==b.t_ms and a.steps==round(b.t_ms/dt)
        assert a._ring_i==0 and np.array_equal(initial['rates'],orig['rates'])
        # Independent event dictionary indexed by absolute physical timestamp.
        events_old={b.t_ms+k*b.dt:tuple(np.flatnonzero(b._ring[(b._ring_i+k)%len(b._ring)])) for k in range(len(b._ring)) if b._ring[(b._ring_i+k)%len(b._ring)].any()}
        events_new={a.t_ms+k*dt:tuple(np.flatnonzero(a._ring[k])) for k in range(len(a._ring)) if a._ring[k].any()}
        assert events_old==events_new
        for key in ['We','Wi']:
            old=getattr(b,key);new=getattr(a,key);assert np.array_equal(old.indices,new.indices) and np.array_equal(old.indptr,new.indptr)
            expected=old.data.copy()
            if key=='We':expected[np.isin(np.repeat(np.arange(12000),np.diff(old.indptr)),ident['RH_cells'])]*=.375
            assert np.array_equal(expected,new.data)
        # Independent pool averaging, scipy linear filter and causal command reconstruction.
        drive=np.array([sp[:,ids].mean(axis=1) for ids in pools]).T/(dt/1000)
        decay=np.exp(-dt/5);filtered=lfilter([1-decay],[1,-decay],drive,axis=0,zi=(decay*orig['rates'])[None,:])[0]
        rates=filtered[n-1::n];source=np.vstack([orig['rates'],rates[:-1]])
        raw=source@matrix.T*limits/100;clipped=np.clip(raw,-limits,limits)
        errors={key:float(np.max(np.abs(x-z[key]))) for key,x in [('pool_rates',rates),('command_source_rates',source),('raw_command',raw),('clipped_command',clipped)]}
        assert max(errors.values())<1e-9,errors
        counts=sp.reshape(200,n,12000).sum(1);assert np.array_equal(counts,z['counts12000'])
        assert np.array_equal(end.spike_count-a.spike_count,counts.sum(0))
        assert np.array_equal(z['sensory_current'],base['sensory_current'])
        assert np.array_equal(z['neural_clock'][:,0],a.t_ms+np.arange(1,201))
        assert np.array_equal(z['neural_clock'][:,1],a.steps+np.arange(1,201)*n)
        assert np.array_equal(z['time_s'],1.3+np.arange(1,201)/1000)
        for key in r.FIELDS:assert np.array_equal(z[key][-1],getattr(end,key))
        # Analytic adaptation recurrence independent of engine implementation.
        adapt=lfilter([a.p.adaptation_increment_nA],[1,-np.exp(-dt/a.p.adaptation_tau_ms)],sp,axis=0,zi=(np.exp(-dt/a.p.adaptation_tau_ms)*a.adaptation)[None,:])[0]
        adapt_err=float(np.max(np.abs(adapt[n-1::n]-z['adaptation'])));assert adapt_err<1e-10
        details.append(dict(label=row['label'],filter_command_errors=errors,adaptation_error=adapt_err,counts_exact=True,inputs_exact=True,transfer_events_exact=True,RH_once=True,clocks_exact=True))
    # Independent re-evaluation of all window gates and values.
    gate_errors=[]
    for block in ['main','supplement']:
        for method,item in summary[block].items():
            zs=[r.arrays(r.O/'RUNS'/(method+'_'+str(dt).replace('.','p')+'ms')/'NEURAL.npz') for dt in item['dts_ms']]
            for w,win in item['windows'].items():
                passed=[]
                for key in ['pool_rates','raw_command','clipped_command','counts12000']:
                    x,y,z=[a[key][:int(w)].astype(float) for a in zs];norm=lambda a,b:np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-9*np.sqrt(b.size))
                    coarse=norm(x,y);fine=norm(y,z);saved=win['metrics'][key]
                    gate_errors.extend([abs(coarse-saved['coarse']['normalized_RMS']),abs(fine-saved['fine']['normalized_RMS'])])
                    ok=fine<=.15 and (fine/max(coarse,1e-15)<=.9 or max(coarse,fine)<=.02)
                    assert ok==saved['passed']
                    if key!='counts12000':passed.append(ok)
                assert all(passed)==win['passed']
    assert max(gate_errors)<1e-12
    r.checkhash()
    r.dump(r.O/'VERIFICATION.json',dict(passed=True,cases=len(details),details=details,max_gate_error=max(gate_errors),historical_hashes_unchanged=True,source_freeze_unchanged=all(r.sha(p)==h for p,h in r.js(r.O/'SOURCE_FREEZE.json').items()),physical_steps=0))
    print('VERIFIED',len(details),flush=True)
if __name__=='__main__':main()
