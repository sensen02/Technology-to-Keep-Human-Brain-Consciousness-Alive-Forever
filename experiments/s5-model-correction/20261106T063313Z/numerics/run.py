"""S1/P2 actual neural dt refinement, fixed connectivity and physical clocks.
No original module is changed. No performance-based model selection.
"""
import os,sys,json,copy,pickle,time,resource
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0]=[str(BASE),str(R),str(R/'tools')]
import runtime as rt
from translator import SpikeHistory
from engine.neural_cond import ConductanceNetwork,ConductanceParams
O=Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics')
SRC=Path(__file__).resolve().parent
S=Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
rt.O=O;rt.b.O=O
js,sha,dump=rt.js,rt.sha,rt.dump
PHASES=[1.225,1.275,1.3,1.325,1.375]
DTS=[.0005,.00025,.000125]
CDT=.001;PDT=.0000125

def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def exact(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if sparse.issparse(a):return sparse.issparse(b) and a.shape==b.shape and all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['data','indices','indptr'])
    if isinstance(a,dict):return set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return bool(a==b)
def cleanup():
    p=O/'COVERAGE.json'
    if p.exists():
        dest=O/'CLEANUP_LEDGER.json';v=js(dest) if dest.exists() else {'deleted':[]}
        v['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,importance='可再生临时',reason='Inspected constructor write-only output; pool identities retained in TRANSLATOR_IDENTITIES; historical coverage references unchanged.'))
        dump(dest,v);p.unlink()
def checkhash():
    v=js(O/'INPUT_HASHES_BEFORE.json');a={p:sha(p) for p in v};assert a==v;return a

def prereg():
    O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
    cfg=dict(stage='S1/P2 bounded numerical convergence diagnostic; no optimization',neural_dt_s=DTS,command_dt_s=CDT,physics_dt_s=PDT,params=vars(ConductanceParams()),rate_filter_tau_s=.005,
      currents=str(S/'TRUE_CURRENTS.npz'),physical_cache=str(S/'DATA/PHYSICAL_CACHE.pkl'),warm='1200 actual float64 1ms held currents .2..1.4 physical seconds; each candidate starts rest; all17 physical states shared, all neural states candidate-specific; RH .375 release-only',
      immutable='Original matrices, encoder, neuron parameters, readout matrix/limits, physical body, assistance states. Explicit self.neural_dt; rt.NDT untouched. Correct-dt 5/20/50ms history retained, zero actor only.',
      baseline='Exact .5ms full network vars, rates/history at all17 caches and full teacher204 spikes; physical1.3 J1.5903704331878943 tolerance1e-9 before finer releases',
      releases=dict(phases=PHASES,duration_s=.2,main=15,controls='.125ms repeat+motor cut+sensory cut at1.3 =3',conditional='.0625ms warm+same5 releases ONLY if neural OR physical main convergence gates fail; no additional refinement',maximum=23),
      comparisons=dict(boundaries='Pool/leg rates at common1ms END boundaries; raw/clipped held commands at interval START from prior completed rates (first command zero); actual same causal ordering during physical releases',windows={'full':[0,1200],'primary_exclude_initial100ms':[100,1200]},
       normalized_rms='RMS(coarse-fine)/max(RMS(fine),1e-9); flatten time and channels, also report per-channel max',
       neural_primary=['pool_rates','leg_rates','raw_command','clipped_command'],fine_difference_max=.15,trend_ratio_max=.9,near_zero_pair_max=.02,
       neural_gate='BOTH warm and matched-input RH.375 release replay: All4 primary traces fine normalized RMS<=.15 AND (fine/coarse<=.9 OR both<=.02). Full window descriptive; primary excludes first100ms. No exact-spike requirement across dts. Engineering tolerances, not biological proof.',
       controlled_release='From each dt own1.3 warm state, apply RH.375 once and hold same200 baseline .5ms physical1.3 actualfloat64 currents each1ms. Baseline raw packed spikes/rates/commands exact. No additional physics. Save raw and normalized differences plus reference RMS. Main and supplementary gate both include this comparison.',
       physical_gate='Every paired phase: abs(deltaJ)<=.2, abs(delta foot mean)<=.1W, abs(delta COM min)<=.1, abs(delta first positive nonfoot)<=.02s (None=.2 right censored), identical all D booleans. Descriptive cross-phase trends also reported.',
       descriptive='per-cell/window meanHz, population/leg rates, saturation, temporal variation, all-neuron count; no meanHz-only convergence claim'),
      clock_tests='Synthetic2-cell ring arrival and refractory interval, exact taus/physical ms for every dt including supplementary; conducted only after parent GO',
      selection='None; do not optimize J or replace model; report unresolved nonconvergence at cap',units='absolute force/SI and biological validity NOT_ESTABLISHED',importance='核心必需')
    dump(O/'PREREGISTRATION.json',cfg)
    paths=list(dict.fromkeys(rt.files()+[R/'run_integrated_repaired_closure.py',S/'TRUE_CURRENTS.npz',S/'DATA/PHYSICAL_CACHE.pkl',S/'DATA/true_teacher/TEACHER.npz',S/'DATA/true_teacher/BODY.npz',S/'WARM_DATA_HASHES.json',O/'PREREGISTRATION.json']+list(SRC.glob('*.py'))))
    paths.append(S/'SUMMARY.json')
    for row in js(S/'SUMMARY.json')['results']:
        if row['candidate']=='baseline' and row['release_time_s'] in PHASES and ('_train_guard_' in row['label'] or '_dev_' in row['label']):
            paths.extend(Path(row['folder'])/f for f in ['BODY.npz','NATIVE.npz','NEURAL.npz'])
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in paths})
    dump(O/'CHANGE_LEDGER.json',dict(old=[dict(path=str(p),sha256=sha(p)) for p in paths if p.parent!=SRC and p.parent!=O],new=[dict(path=str(p),sha256=sha(p),reason='Explicit numerical clock runtime and independent verification, no changes to frozen modules') for p in SRC.glob('*.py')]))
    print('PREREGISTERED',flush=True)

class NumericalRuntime(rt.Runtime):
    def __init__(self):
        rt.O=O;rt.b.O=O;super().__init__();cleanup();self.set_dt(DTS[0])
        rows=js(rt.b.B/'RUN/MAPPING.json')['rows']
        self.legcells=[np.array(sorted({int(c) for row in rows if row['leg']==leg for c in row['local_indices']})) for leg in rt.b.LEGS]
        dump(O/'ANATOMICAL_IDENTITIES.json',dict(legs=rt.b.LEGS,cells=self.legcells,definition='unique mapped motor cells per anatomical leg; overlaps retained explicitly, not all sensory cells'))
    def set_dt(self,dt):
        self.neural_dt=dt;self.n=round(CDT/dt);assert abs(self.n*dt-CDT)<1e-15
        self.history=SpikeHistory(self.cells,dt=dt)
    def integrate(self,net,current,rates):
        assert net.dt==self.neural_dt*1000
        decay=np.exp(-self.neural_dt/.005);counts=np.zeros(12000,np.int32);self.last_spikes=[];self.last_full=[]
        for _ in range(self.n):
            sp=net.step(current_nA=current);counts+=sp;self.last_full.append(np.packbits(sp));self.last_spikes.append(sp[self.cells].copy());self.history.consume(sp)
            rates=decay*rates+(1-decay)*np.asarray(self.poolmat@sp,float)/self.neural_dt
        return rates,counts,current
    def observe_neural(self,d,net,enc,rates):
        from run_neural_muscle_closure import observe_quantities
        current=np.zeros(12000) if self.zero_sensory else np.asarray(enc.encode(observe_quantities(self.m,d,self.q0,self.joints,self.geom))[0])
        return self.integrate(net,current,rates)
    def episode(self,label,ph,config,cut=False,sensory_cut=False):
        import train_dynamic_support as dst
        import dynamic_support_metrics as dsm
        folder=O/'RUNS'/label;folder.mkdir(parents=True,exist_ok=False)
        dump(folder/'PREREGISTRATION.json',dict(label=label,parent=str(O/'PREREGISTRATION.json'),parent_sha256=sha(O/'PREREGISTRATION.json'),parameters=str(config),parameters_sha256=sha(config),release_time_s=ph,duration_s=.2,neural_dt_s=self.neural_dt,neural_steps_per_command=self.n,command_dt_s=CDT,physics_dt_s=PDT,history_dt_s=self.history.dt,actor='identically zero; no translator',RH_exc_gain=.375,cut=cut,sensory_cut=sensory_cut))
        c=self.cache[ph];net=copy.deepcopy(c['net']);enc=copy.deepcopy(c['enc']);rates=c['rates'].copy();self.history.load_state_dict(c['history']);self.zero_sensory=sensory_cut
        rowids=np.repeat(np.arange(net.n),np.diff(net.We.indptr));net.We.data[np.isin(rowids,self.rh)]*=.375
        d=self.mj.MjData(self.m);self.mj.mj_setState(self.m,d,c['integration'],self.mj.mjtState.mjSTATE_INTEGRATION);self.mj.mj_forward(self.m,d)
        q=d.qpos.copy();v=d.qvel.copy();d.ctrl[:]=0 if cut else self.old_command(rates);d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;self.mj.mj_forward(self.m,d)
        assert np.array_equal(q,d.qpos) and np.array_equal(v,d.qvel)
        body=[self.bodyrow(d)];native=[self.contact(d)];logs=[];start=time.monotonic()
        try:
            for k in range(200):
                features=self.history.features();old=self.old_command(rates);raw=self.limits/100*(self.matrix@rates);ctrl=np.zeros(42) if cut else old.copy();source=rates.copy()
                rates,counts,cur=self.observe_neural(d,net,enc,rates);d.ctrl[:]=ctrl
                for j in range(80):
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                    self.mj.mj_step(self.m,d);self.mj.mj_forward(self.m,d);native.append(self.contact(d))
                    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and not d.warning.number.any()
                    if (j+1)%40==0:body.append(self.bodyrow(d))
                logs.append(dict(time=float(d.time),features=features,source_rates=source,rates=rates.copy(),raw_command=raw,old_torque=old,applied=ctrl,spikes204=np.asarray(self.last_spikes),spikes12000_packed=np.asarray(self.last_full),sensory_current=cur.copy(),sensory_current_norm=float(np.linalg.norm(cur)),net_t_ms=net.t_ms,net_steps=net.steps))
            arr=np.asarray(native);scored=arr.copy();scored[:,0]-=ph-1.3
            old_duration=dst.RELEASE;dst.RELEASE=.2
            try:J,parts=dst.objective(scored,c['assist'])
            finally:dst.RELEASE=old_duration
            gate=dsm.evaluate_native(c['assist'],arr);gate['checks']['D1_no_nonfoot_ground_force']=bool(np.max(arr[:,7])==0);gate['passed']=all(gate['checks'].values())
            pos=np.flatnonzero(arr[:,7]>1e-9)
            result=dict(label=label,dt_s=self.neural_dt,J=J,parts=parts,validation=gate,release_time_s=ph,duration_s=.2,folder=str(folder),cut=cut,sensory_cut=sensory_cut,first_positive_nonfoot_s=float(arr[pos[0],0]-ph) if len(pos) else None,wall_s=time.monotonic()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,warnings=int(d.warning.number.sum()),release_aux_max=0.,release_external_max=0.,walking_certified=False)
            dump(folder/'RESULT.json',result)
        except BaseException:
            import traceback
            dump(folder/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
        finally:
            np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(native));np.savez_compressed(folder/'BODY.npz',**{k:np.array([r[k] for r in body]) for k in body[0]})
            if logs:np.savez_compressed(folder/'NEURAL.npz',**{k:np.array([r[k] for r in logs]) for k in logs[0]})
            self.zero_sensory=False
        print('RELEASE',label,J,gate['passed'],flush=True);return result

def name(dt):return 'dt_'+str(dt*1000).replace('.','p')+'ms'
def clock_tests():
    results=[]
    for dt in DTS+[.0000625]:
        ms=dt*1000;n=ConductanceNetwork(2,dt_ms=ms);w=sparse.csr_matrix(([.001],([1],[0])),shape=(2,2));n.set_connectivity(w,sparse.csr_matrix((2,2)))
        assert n.step([100.,0.])[0];emission=n.t_ms;arrival=None
        for k in range(round(3/ms)):
            start=n.t_ms;n.step(0.)
            if n.ge[1]>0 and arrival is None:arrival=start
        assert abs(arrival-emission-2)<1e-12
        n=ConductanceNetwork(1,dt_ms=ms);times=[]
        for k in range(round(10/ms)):
            if n.step(100.)[0]:times.append(n.t_ms)
        assert np.allclose(np.diff(times),2+ms,rtol=0,atol=1e-12)
        dec=ConductanceNetwork(1,dt_ms=ms);dec.ge[:]=.01;dec.gi[:]=.02;dec.adaptation[:]=.03;dec.step(0.)
        measured=[-ms/np.log(dec.ge[0]/.01),-ms/np.log(dec.gi[0]/.02),-ms/np.log(dec.adaptation[0]/.03)]
        assert np.allclose(measured,[3.,8.,100.],rtol=0,atol=1e-9)
        results.append(dict(dt_ms=ms,delay_ms=arrival-emission,refractory_inactive_ms=2.,earliest_end_spike_interval_ms=2+ms,observed_intervals_ms=np.diff(times),measured_taus_ms=measured,filter_tau_ms=5.,ring_size=round(2/ms)+1,voltage_method=n.voltage_method))
    dump(O/'CLOCK_TESTS.json',dict(passed=True,results=results,interpretation='End spike -> arrival at interval START after exactly2ms; refractory blocks starts for2ms, next emitted spike is at least2ms+dt later. Quantization changes with dt as intended.'))

def warm(t,dt,physical,currents):
    start=time.monotonic();t.set_dt(dt);folder=O/'CONFIGS'/name(dt);folder.mkdir(parents=True,exist_ok=False)
    cfg=folder/'PARAMETERS.json';dump(cfg,dict(neural_dt_s=dt,steps_per_command=t.n,command_dt_s=CDT,physics_dt_s=PDT,params=vars(ConductanceParams()),history_dt_s=dt,rate_filter_tau_s=.005,connectivity='original',RH_release_only=.375))
    net=ConductanceNetwork(12000,dt_ms=dt*1000);net.set_connectivity(*[sparse.load_npz(rt.b.AB/'A'/('ACTUAL_'+k+'.npz')) for k in ['We','Wi']])
    rates=np.zeros(66);t.history.reset();cache={};packed=[];rr=[];legs=[];cell_filtered=np.zeros(12000);cell_rates=[];checks=[];decay=np.exp(-dt/.005)
    captures={round((ph-.2)/CDT):ph for ph in physical}
    for k,current in enumerate(currents):
        rates,counts,_=t.integrate(net,current,rates);packed.extend(t.last_full);rr.append(rates.copy())
        for s in t.last_full:cell_filtered=decay*cell_filtered+(1-decay)*np.unpackbits(s,count=12000)/dt
        cell_rates.append(cell_filtered.copy());legs.append([cell_filtered[c].mean() for c in t.legcells])
        if k+1 in captures:
            ph=captures[k+1];c=dict(physical[ph]);c.update(net=copy.deepcopy(net),rates=rates.copy(),history=t.history.state_dict());cache[ph]=c
            if dt==DTS[0]:
                assert exact(vars(net),vars(physical[ph]['net'])),f'network {ph}'
                assert exact(rates,physical[ph]['rates']) and exact(c['history'],physical[ph]['history'])
                checks.append(dict(phase=ph,fullnet_rates_history_exact=True))
    packed=np.asarray(packed);spikes=np.unpackbits(packed,axis=1,count=12000)[:,t.cells]
    if dt==DTS[0]:assert np.array_equal(spikes,arrays(S/'DATA/true_teacher/TEACHER.npz')['spikes204'])
    rr=np.asarray(rr);source_rates=np.vstack([np.zeros(66),rr[:-1]]);raw=source_rates@t.matrix.T*t.limits/100
    np.savez_compressed(folder/'WARM.npz',spikes12000_packed=packed,spikes204=spikes,cells=t.cells,pool_rates=rr,command_source_rates=source_rates,leg_rates=legs,cell_filtered_rates=cell_rates,raw_command=raw,clipped_command=np.clip(raw,-t.limits,t.limits),time=.2+np.arange(1,1201)*CDT,neural_dt_s=dt)
    with (folder/'WARM_CACHE.pkl').open('wb') as f:pickle.dump(cache,f,protocol=5)
    dump(folder/'WARM_IDENTITY.json',dict(dt_s=dt,current_source=str(S/'TRUE_CURRENTS.npz'),current_sha256=sha(S/'TRUE_CURRENTS.npz'),physical_source=str(S/'DATA/PHYSICAL_CACHE.pkl'),physical_sha256=sha(S/'DATA/PHYSICAL_CACHE.pkl'),neural_steps=len(packed),wall_s=time.monotonic()-start,baseline_exact_checks=checks,cache_sha256=sha(folder/'WARM_CACHE.pkl')))
    t.cache=cache;print('WARM',dt,len(packed),len(cache),flush=True);return cfg

def controlled_release(t,dt,results):
    """Same actual baseline release currents; own warm state; RH .375 applied once."""
    baseline=next(r for r in results if r['dt_s']==DTS[0] and r['release_time_s']==1.3 and 'paired' in r['label'])
    source=Path(baseline['folder'])/'NEURAL.npz';base=arrays(source);c=t.cache[1.3];net=copy.deepcopy(c['net']);rates=c['rates'].copy();t.history.load_state_dict(c['history'])
    rows=np.repeat(np.arange(net.n),np.diff(net.We.indptr));net.We.data[np.isin(rows,t.rh)]*=.375
    warmdata=arrays(O/'CONFIGS'/name(dt)/'WARM.npz');cell=warmdata['cell_filtered_rates'][1099].copy();decay=np.exp(-dt/.005)
    packed=[];pool=[];legs=[];raws=[];commands=[];sources=[];cellrates=[]
    for current in base['sensory_current']:
        sources.append(rates.copy());raws.append(t.limits/100*(t.matrix@rates));commands.append(t.old_command(rates))
        rates,_,_=t.integrate(net,current,rates);pool.append(rates.copy());packed.extend(t.last_full)
        for p in t.last_full:cell=decay*cell+(1-decay)*np.unpackbits(p,count=12000)/dt
        cellrates.append(cell.copy());legs.append([cell[ids].mean() for ids in t.legcells])
    packed=np.asarray(packed);pool=np.asarray(pool);commands=np.asarray(commands)
    if dt==DTS[0]:
        assert np.array_equal(packed,base['spikes12000_packed'].reshape(-1,1500))
        assert np.array_equal(pool,base['rates']) and np.array_equal(np.asarray(sources),base['source_rates']) and np.array_equal(commands,base['applied']) and np.array_equal(np.asarray(raws),base['raw_command'])
    folder=O/'CONFIGS'/name(dt)
    np.savez_compressed(folder/'CONTROLLED_RELEASE.npz',spikes12000_packed=packed,pool_rates=pool,leg_rates=legs,cell_filtered_rates=cellrates,command_source_rates=sources,raw_command=raws,clipped_command=commands,time=1.3+np.arange(1,201)*CDT,neural_dt_s=dt)
    dump(folder/'CONTROLLED_RELEASE_IDENTITY.json',dict(source=str(source),sha256=sha(source),RH_applied_once=.375,own_warm_state_s=1.3,neural_steps=200*t.n,baseline_exact=dt==DTS[0],clock='same 200 actual float64 baseline currents held1ms, commands causal previous rates'))
    print('CONTROLLED RELEASE',dt,flush=True)

def nrm(a,b):return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(b*b)),1e-9))
def trace_comparison(dts,file):
    z=[arrays(O/'CONFIGS'/name(dt)/file) for dt in dts];windows={};primary=True
    for window,start in [('full',0),('primary_exclude_initial100ms',100)]:
        rows={}
        for key in ['pool_rates','leg_rates','raw_command','clipped_command']:
            a,b,c=[v[key][start:] for v in z];x=nrm(a,b);y=nrm(b,c);ratio=y/max(x,1e-15);ok=y<=.15 and (ratio<=.9 or max(x,y)<=.02)
            rms=lambda v:float(np.sqrt(np.mean(v*v)))
            rows[key]=dict(coarse_difference=x,fine_difference=y,ratio=ratio,coarse_absolute_rms=rms(a-b),fine_absolute_rms=rms(b-c),middle_reference_rms=rms(b),fine_reference_rms=rms(c),absolute_contraction_ratio=rms(b-c)/max(rms(a-b),1e-15),passed=ok,per_channel_fine_nrm=np.sqrt(np.mean((b-c)**2,axis=0))/np.maximum(np.sqrt(np.mean(c*c,axis=0)),1e-9))
            if window.startswith('primary'):primary=primary and ok
        windows[window]=rows
    return windows,bool(primary)
def compare(dts,results):
    windows,warm_passed=trace_comparison(dts,'WARM.npz');controlled,controlled_passed=trace_comparison(dts,'CONTROLLED_RELEASE.npz');primary=warm_passed and controlled_passed
    pairs=[]
    for ph in PHASES:
        a,b,c=[next(r for r in results if r['dt_s']==dt and r['release_time_s']==ph and not r['cut'] and not r['sensory_cut'] and 'paired' in r['label']) for dt in dts]
        first=lambda r:.2 if r['first_positive_nonfoot_s'] is None else r['first_positive_nonfoot_s']
        delta=dict(J=abs(b['J']-c['J']),foot_mean=abs(b['parts']['foot_support_mean']-c['parts']['foot_support_mean']),com_min=abs(b['parts']['com_z_min']-c['parts']['com_z_min']),first_nonfoot=abs(first(b)-first(c)))
        ok=all(delta[k]<=v for k,v in dict(J=.2,foot_mean=.1,com_min=.1,first_nonfoot=.02).items()) and b['validation']['checks']==c['validation']['checks']
        pairs.append(dict(phase=ph,fine_delta=delta,coarse_deltaJ=abs(a['J']-b['J']),same_D=b['validation']['checks']==c['validation']['checks'],passed=ok))
    return dict(dts=dts,neural_windows=windows,warm_neural_passed=warm_passed,controlled_release_windows=controlled,controlled_release_passed=controlled_passed,neural_passed=bool(primary),physical_pairs=pairs,physical_passed=all(p['passed'] for p in pairs),passed=bool(primary and all(p['passed'] for p in pairs)))

def run():
    checkhash();assert all(sha(p)==h for p,h in js(S/'WARM_DATA_HASHES.json').items());clock_tests();t=NumericalRuntime()
    with (S/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
    cur=arrays(S/'TRUE_CURRENTS.npz');currents=cur['currents'];assert currents.shape==(1200,12000) and currents.dtype==np.float64 and len(physical)==17
    assert np.max(abs(cur['time']-(.2+np.arange(1200)*CDT)))<1e-11
    results=[];warmsteps=0
    for dt in DTS:
        cfg=warm(t,dt,physical,currents);warmsteps+=1200*t.n
        # Establish historical physical anchor first.
        for ph in [1.3]+[v for v in PHASES if v!=1.3]:
            r=t.episode(name(dt)+'_paired_'+str(ph),ph,cfg);results.append(r)
            if dt==DTS[0]:
                if ph==1.3:assert abs(r['J']-1.5903704331878943)<1e-9
                old=next(x for x in js(S/'SUMMARY.json')['results'] if x['candidate']=='baseline' and x['release_time_s']==ph and ('_train_guard_' in x['label'] or '_dev_' in x['label']))
                for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
                    a=arrays(Path(old['folder'])/file);b=arrays(Path(r['folder'])/file)
                    keys=list(a) if file!='NEURAL.npz' else ['spikes204','rates','source_rates','applied','features','old_torque','sensory_current_norm','time']
                    assert all(np.array_equal(a[key],b[key]) for key in keys),(ph,file)
                dump(Path(r['folder'])/'BASELINE_EXACT.json',dict(old_folder=old['folder'],all_BODY_NATIVE_exact=True,shared_NEURAL_exact=True))
        controlled_release(t,dt,results)
        dump(O/'PROGRESS.json',dict(results=results,warm_steps=warmsteps))
    main=compare(DTS,results);dump(O/'MAIN_CONVERGENCE.json',main)
    for label,kw in [('repeat',{}),('motor_cut',{'cut':True}),('sensory_cut',{'sensory_cut':True})]:results.append(t.episode(name(DTS[-1])+'_'+label,1.3,cfg,**kw))
    supplement=None
    if not main['passed']:
        dt=.0000625;cfg=warm(t,dt,physical,currents);warmsteps+=1200*t.n
        for ph in PHASES:results.append(t.episode(name(dt)+'_paired_'+str(ph),ph,cfg))
        controlled_release(t,dt,results)
        supplement=compare(DTS[1:]+[dt],results);dump(O/'SUPPLEMENTARY_CONVERGENCE.json',supplement)
    assert len(results)<=23 and rt.NDT==.0005 and rt.b.NDT==.0005
    summary=dict(results=results,main=main,supplement=supplement,physical_releases=len(results),physical_seconds=.2*len(results),new_teacher_runs=0,warm_neural_steps=warmsteps,controlled_release_neural_steps=sum(round(.2/dt) for dt in DTS+([.0000625] if supplement else [])),release_neural_steps=sum(round(.2/r['dt_s']) for r in results),selected_model=None,convergence_established=(supplement or main)['passed'],walking_certified=False,biological_validity='NOT_ESTABLISHED')
    dump(O/'SUMMARY.json',summary);dump(O/'INPUT_HASHES_AFTER.json',checkhash());print('DONE',len(results),'convergence',summary['convergence_established'],flush=True)
if __name__=='__main__':
    action=sys.argv[1]
    try:{'prereg':prereg,'run':run}[action]()
    except BaseException:
        import traceback
        O.mkdir(parents=True,exist_ok=True);dump(O/('FAILURE_'+action+'_'+str(time.time_ns())+'.json'),dict(traceback=traceback.format_exc()));raise
