"""Scoped S1/S5 neural-only local refinement. Never writes sealed numerics."""
import sys, json, pickle, copy, hashlib, time, traceback
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0]=[str(R),str(R/'tools')]
from engine.neural_cond import ConductanceNetwork, ConductanceParams
O=Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_followup')
OLD=O.parent/'numerics'
SRC=Path(__file__).resolve().parent
DTS=[.125,.0625,.03125]; METHODS=['backward_euler','frozen_exact']; WINDOWS=[5,10,20,50,100,200]
FIELDS=['v','ge','gi','adaptation','refractory_until_ms','spike_count']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def dump(p,v):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=default)+'\n')
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def cache(dt):
    with (OLD/'CONFIGS'/('dt_'+str(dt).replace('.','p')+'ms')/'WARM_CACHE.pkl').open('rb') as f:return pickle.load(f)
def transfer(old,dt,method):
    ratio=old.dt/dt;assert abs(ratio-round(ratio))<1e-12
    new=ConductanceNetwork(old.n,old.p,dt,method,old.threshold_offset_mV)
    new.set_connectivity(old.We,old.Wi)
    for k in FIELDS:setattr(new,k,getattr(old,k).copy())
    new.t_ms=old.t_ms;new.steps=round(old.t_ms/dt)
    assert new.steps*dt==old.t_ms
    for k in range(len(old._ring)):
        idx=round(k*old.dt/dt);assert idx<len(new._ring)
        new._ring[idx]=old._ring[(old._ring_i+k)%len(old._ring)]
    new._ring_i=0
    return new

def tests():
    records=[]
    for dt in [.5]+DTS+[.015625]:
        old=ConductanceNetwork(2,dt_ms=.5)
        old.steps=20;old.t_ms=10.;old._ring_i=3
        old._ring[:]=False
        old._ring[(old._ring_i+1)%5,0]=True
        old._ring[(old._ring_i+4)%5,1]=True
        old.We=sparse.csr_matrix(np.diag([.001,.002]));old.Wi=sparse.csr_matrix(np.diag([.003,.004]))
        new=transfer(old,dt,'backward_euler');arrivals=[]
        for j in range(round(3/dt)):
            start=new.t_ms;ge=new.ge.copy();gi=new.gi.copy();new.step(-100.)
            e=new.ge/np.exp(-dt/3)-ge;i=new.gi/np.exp(-dt/8)-gi
            if np.max(abs(e))+np.max(abs(i))>1e-12:arrivals.append([start,e.tolist(),i.tolist()])
        assert np.allclose([a[0] for a in arrivals],[10.5,12.])
        assert np.allclose([a[1] for a in arrivals],[[.001,0],[0,.002]])
        assert np.allclose([a[2] for a in arrivals],[[.003,0],[0,.004]])
        records.append(dict(dt_ms=dt,arrivals=arrivals))
    c=cache(.5)[1.3];old=copy.deepcopy(c['net']);new=transfer(old,.5,'backward_euler')
    currents=arrays(OLD/'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz')['sensory_current']
    for current in currents[:10]:
        for _ in range(2):
            assert np.array_equal(old.step(current),new.step(current))
            for k in FIELDS:assert np.array_equal(getattr(old,k),getattr(new,k)),k
            assert np.array_equal(np.roll(old._ring,-old._ring_i,axis=0),np.roll(new._ring,-new._ring_i,axis=0))
    dump(O/'MECHANISM_TESTS.json',dict(passed=True,same_dt_full_network_steps=20,future_events=records,exact_absolute_timestamps=True))

def prereg():
    O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
    dump(O/'PREREGISTRATION.json',dict(scope='S1/S5 neural-only local diagnostics; manual research, not product',methods=METHODS,main_dt_ms=DTS,baseline_dt_ms=.5,conditional_dt_ms=.015625,maximum_cases=9,conditional='For each method whose main triple fails any of six windows on any of pool/raw/clipped, run its fourth fine grid once. No further refinement.',windows_ms=WINDOWS,gate='At each window and each pool/raw/clipped: fine nRMS <= .15 AND (fine/coarse nRMS <= .9 OR both nRMS <= .02). First5/10ms reported separately from longer windows. Engineering tolerance only.',normalization='RMS(coarse-fine)/max(RMS(fine),1e-9); absolute RMS and reference RMS retained',initial='Identical baseline .5ms warm state at physical1.3s (neural1100ms), full v/ge/gi/adaptation/refractory/spike_count and absolute pending event times transferred. Identical initial pool rates. No fine-grid warmup.',RH='unscaled cache; multiply RH postsynaptic We rows by .375 exactly once AFTER transfer',input='same actual200x12000 float64 sensory_current baseline .5ms release, held1ms',readout='same pools/matrix/limits; causal previous completed rates; dt-aware exact exponential5ms filter;20ms aggregates diagnostic only',units='C nF / g uS = ms; voltage mV, current nA; illustrative electrical parameters, not physiological calibration',frozen_exact='Exact voltage only with coefficients frozen at interval start, split conductance/adaptation decay and endpoint threshold/reset remain',physical_runs=0,new_warm_runs=0,selected_model=None))
    paths=list(OLD.rglob('*'))+list((SRC.parent/'numerics').glob('*.py'))+[R/'engine/neural_cond.py']+list(SRC.glob('*.py'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in paths if p.is_file()})
    dump(O/'CHANGE_LEDGER.json',dict(old_source=str(SRC.parent/'numerics/run.py'),old_sha256=sha(SRC.parent/'numerics/run.py'),new_source=str(SRC/'run.py'),new_sha256=sha(SRC/'run.py'),reason='Separate local state transfer from long-warm trajectory differences; no engine/global dt mutation'))
    tests()
    dump(O/'SOURCE_FREEZE.json',{str(p):sha(p) for p in SRC.glob('*.py')})
    print('PREREGISTERED TESTED FROZEN',flush=True)

def distribution(x):return dict(quantiles=dict(zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]))),mean=float(np.mean(x)))
def tau_diagnostics():
    records=[];saved={}
    for dt in [.5,.25,.125,.0625]:
        cc=cache(dt);assert len(cc)==17;taus=[]
        for phase,c in sorted(cc.items()):
            n=c['net'];tau=n.p.capacitance_nF/(n.p.leak_uS+n.ge+n.gi);ratio=dt/tau
            # Also include immediately pending arrivals: these enter BEFORE voltage update.
            ev=n._ring[n._ring_i];tstart=n.p.capacitance_nF/(n.p.leak_uS+n.ge+n.gi+n.We@ev+n.Wi@ev)
            records.append(dict(dt_ms=dt,phase_s=phase,tau_ms=distribution(tau),dt_over_tau=distribution(ratio),dt_over_tau_after_pending_arrival=distribution(dt/tstart),fraction_ratio_above_1=float(np.mean(ratio>1)),fraction_ratio_above_point1=float(np.mean(ratio>.1))))
            saved[str(dt)+'_'+str(phase)]=np.stack([tau,ratio,tstart]);taus.append(tau)
        records.append(dict(dt_ms=dt,all17=True,tau_ms=distribution(np.concatenate(taus)),dt_over_tau=distribution(dt/np.concatenate(taus))))
    np.savez_compressed(O/'TAU_ALL_CELLS.npz',**saved)
    dump(O/'TAU_DIAGNOSTIC.json',dict(parameters=vars(ConductanceParams()),resting_tau_ms=20.,units='nF/uS = ms; not seconds; dt in ms. Snapshot caches, not complete warm-time distribution. All12000cells x17states x4dt.',records=records))

def identities():
    ident=js(OLD/'TRANSLATOR_IDENTITIES.json');pools=[np.asarray(p) for p in ident['pools']]
    poolmat=sparse.csr_matrix((np.concatenate([np.full(len(p),1/len(p)) for p in pools]),(np.repeat(np.arange(66),[len(p) for p in pools]),np.concatenate(pools))),shape=(66,12000))
    return ident,poolmat

def case(dt,method):
    label=method+'_'+str(dt).replace('.','p')+'ms';folder=O/'RUNS'/label;folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();c=cache(.5)[1.3];old=c['net'];net=transfer(old,dt,method)
    assert net.We.nnz==180604 and net.Wi.nnz==148811
    ident,poolmat=identities();rows=np.repeat(np.arange(net.n),np.diff(net.We.indptr));mask=np.isin(rows,ident['RH_cells']);net.We.data[mask]*=.375
    assert np.array_equal(net.We.data[~mask],old.We.data[~mask]) and np.array_equal(net.We.data[mask],old.We.data[mask]*.375)
    matrix=np.asarray(ident['old_matrix']);limits=np.asarray(ident['limits']);rates=c['rates'].copy();initial=rates.copy()
    base=arrays(OLD/'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz');currents=base['sensory_current'];assert currents.shape==(200,12000) and currents.dtype==np.float64
    with (folder/'INITIAL_STATE.pkl').open('wb') as f:pickle.dump(dict(net=net,rates=rates),f,protocol=5)
    n=round(1/dt);decay=np.exp(-dt/5);packed=[];pool=[];raw=[];clip=[];sources=[];counts=[];states={k:[] for k in FIELDS};clock=[]
    for current in currents:
        sources.append(rates.copy());cmd=limits/100*(matrix@rates);raw.append(cmd);clip.append(np.clip(cmd,-limits,limits));cnt=np.zeros(12000,np.int32)
        for _ in range(n):
            sp=net.step(current);packed.append(np.packbits(sp));cnt+=sp
            rates=decay*rates+(1-decay)*np.asarray(poolmat@sp,float)/(dt/1000)
        pool.append(rates.copy());counts.append(cnt);clock.append([net.t_ms,net.steps])
        for k in FIELDS:states[k].append(getattr(net,k).copy())
    with (folder/'FINAL_STATE.pkl').open('wb') as f:pickle.dump(dict(net=net,rates=rates),f,protocol=5)
    z=dict(spikes12000_packed=packed,pool_rates=pool,raw_command=raw,clipped_command=clip,command_source_rates=sources,counts12000=counts,initial_pool_rates=initial,sensory_current=currents,neural_clock=clock,neural_dt_ms=dt,time_s=1.3+np.arange(1,201)/1000,command_time_s=1.3+np.arange(200)/1000,**states)
    np.savez_compressed(folder/'NEURAL.npz',**z)
    baseline=None
    if dt==.5:
        baseline={k:np.array_equal(np.asarray(z[k]),base[b]) for k,b in [('pool_rates','rates'),('raw_command','raw_command'),('clipped_command','applied'),('command_source_rates','source_rates')]}
        baseline['spikes12000_packed']=np.array_equal(np.asarray(packed),base['spikes12000_packed'].reshape(-1,1500));assert all(baseline.values()),baseline
    result=dict(label=label,dt_ms=dt,method=method,neural_steps=200*n,wall_s=time.monotonic()-start,baseline_exact=baseline,folder=str(folder),total_spikes=int(np.sum(counts)),mean_hz=float(np.sum(counts)/12000/.2))
    dump(folder/'RESULT.json',result);print('CASE',label,result['wall_s'],flush=True);return result

def metrics(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);rms=lambda x:float(np.sqrt(np.mean(x*x)))
    absolute=rms(a-b);ref=rms(b)
    return dict(absolute_RMS=absolute,reference_RMS=ref,normalized_RMS=absolute/max(ref,1e-9))
def compare(method,dts):
    zz=[arrays(O/'RUNS'/(method+'_'+str(dt).replace('.','p')+'ms')/'NEURAL.npz') for dt in dts];out={}
    for w in WINDOWS:
        entries={}
        for key in ['pool_rates','raw_command','clipped_command','counts12000']:
            a,b,c=[z[key][:w] for z in zz];x=metrics(a,b);y=metrics(b,c);ratio=y['normalized_RMS']/max(x['normalized_RMS'],1e-15)
            entries[key]=dict(coarse=x,fine=y,contraction=ratio,absolute_contraction=y['absolute_RMS']/max(x['absolute_RMS'],1e-15),passed=y['normalized_RMS']<=.15 and (ratio<=.9 or max(x['normalized_RMS'],y['normalized_RMS'])<=.02),cumulative_cell_counts=metrics(a.sum(0),c.sum(0)) if key=='counts12000' else None)
        out[str(w)]=dict(metrics=entries,passed=all(entries[k]['passed'] for k in ['pool_rates','raw_command','clipped_command']))
    #20ms nonoverlapping diagnostic averages; never used by actual decoder.
    agg={key:metrics(zz[-2][key].reshape(10,20,-1).mean(1),zz[-1][key].reshape(10,20,-1).mean(1)) for key in ['pool_rates','raw_command','clipped_command','counts12000']}
    return dict(method=method,dts_ms=dts,windows=out,all_windows_passed=all(v['passed'] for v in out.values()),aggregate20ms_diagnostic=agg)

def checkhash():
    before=js(O/'INPUT_HASHES_BEFORE.json');after={p:sha(p) for p in before};assert before==after;return after

def run():
    checkhash();assert js(O/'MECHANISM_TESTS.json')['passed'];tau_diagnostics()
    results=[case(.5,'backward_euler')];main={};supp={}
    for method in METHODS:
        for dt in DTS:results.append(case(dt,method))
        main[method]=compare(method,DTS)
        dump(O/'PROGRESS.json',dict(results=results,main=main))
    for method in METHODS:
        if not main[method]['all_windows_passed']:
            results.append(case(.015625,method));supp[method]=compare(method,DTS[1:]+[.015625])
    matched={}
    for dt in DTS+[.015625]:
        paths=[O/'RUNS'/(m+'_'+str(dt).replace('.','p')+'ms')/'NEURAL.npz' for m in METHODS]
        if all(p.exists() for p in paths):
            a,b=map(arrays,paths);matched[str(dt)]={str(w):{key:metrics(a[key][:w],b[key][:w]) for key in ['pool_rates','raw_command','clipped_command','counts12000']} for w in WINDOWS}
    dump(O/'SUMMARY.json',dict(results=results,main=main,supplement=supp,matched_methods=matched,neural_steps=sum(r['neural_steps'] for r in results),physical_runs=0,new_warm_runs=0,selected_model=None,biological_validity='NOT_ESTABLISHED'))
    dump(O/'INPUT_HASHES_AFTER.json',checkhash());print('DONE',flush=True)
if __name__=='__main__':
    try:{'prereg':prereg,'run':run}[sys.argv[1]]()
    except BaseException:
        O.mkdir(parents=True,exist_ok=True);dump(O/('FAILURE_'+sys.argv[1]+'_'+str(time.time_ns())+'.json'),dict(traceback=traceback.format_exc()));raise
