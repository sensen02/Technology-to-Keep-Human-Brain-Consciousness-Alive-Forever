"""Bounded R: replace anatomical pool filter, no controller fitting."""
import os,sys,json,pickle,copy,time
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0]=[str(BASE),str(R),str(R/'tools')]
import runtime as rt
O=Path('/home/sensen/Desktop/engine修复验收/s5-robust-readout/20261106T031010Z')
SRC=Path(__file__).resolve().parent
rt.O=O;rt.b.O=O
js,sha,dump=rt.js,rt.sha,rt.dump
ARCH=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
AUD=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural/SENSORY_CURRENTS.npz')
TEACHER=ARCH/'DATA/train_offset_0/TEACHER.npz'
TAUS=[.005,.020,.050,.100];PHASES=[1.225,1.275,1.3,1.325,1.375];FINAL=[1.212,1.262,1.312,1.362]
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def prereg():
    O.mkdir(parents=True,exist_ok=True)
    assert not (O/'PREREGISTRATION.json').exists()
    dump(O/'PREREGISTRATION.json',dict(stage='S5 R bounded temporal readout replacement',taus_s=TAUS,phases=PHASES,release_s=.2,final=FINAL,final_status='previously used archived states, withheld only within this experiment; not unseen',change='only pool filter tau; same 66 anatomical pools, signed matrix, limits, SNN and RH incoming excitation .375 at release; no fitted controller, no residual, no body/time/phase inputs',warm='reconstruct causal memory from all teacher spikes up to release, verify 5ms against cached rates; no release reset',replay=dict(source=str(AUD),steps=1200,neural_dt_s=.0005,edge_sigmas=[.001,.01],seeds=[1,2,3],RH=.375,note='same sensory segment as fragility; old fragility omitted RH, not exact old network reproduction'),robustness='maximum across six perturbations RMS(command difference)/RMS(unperturbed command) <= .15, entire .6s AND post-.1s window; diagnostic strengths never deployed',information='report command temporal standard deviation, total variation and raw torque variation relative to 5ms; no claim these prove motor intent',selection='candidate must pass robustness and 1.3 guard: J<=baseline, foot mean/COM min/first nonfoot>=baseline (1e-9); rank full D passes then worst J then mean J across five states; baseline fallback. All 3 candidates evaluated regardless replay result. Only selected nonbaseline gets four final pairs; repeat selected and motor/sensory cuts at 1.3.',budget='7 replay networks x1200 steps, 20 paired releases +3 controls, optional8 final; no new warmup or training',thresholds='unchanged D1-D8 and objective; absolute force/SI and intent NOT_ESTABLISHED'))
    inputs=list(dict.fromkeys(rt.files()+[AUD,TEACHER,ARCH/'DATA/CACHE_IDENTITY.json',Path(js(ARCH/'DATA/CACHE_IDENTITY.json')['file']),ARCH/'MANIFEST_SHA256.json',SRC/'run.py']))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs})
    dump(O/'CLEANUP_LEDGER.json',dict(deleted=[]))
    for tau in TAUS:
        d=O/'CONFIGS'/f'tau_{tau:.3f}';d.mkdir(parents=True)
        dump(d/'PARAMETERS.json',dict(tau_s=tau,old_tau_s=.005,RH=.375,matrix='unchanged',neural_strengths='unchanged except diagnostic replay noise',residual='identically zero',reason='replace fast filter, retain signed anatomical mapping'))
    print('PREREGISTERED',flush=True)
class RobustRuntime(rt.Runtime):
    tau=.005
    def observe_neural(self,d,net,enc,rates):
        from run_neural_muscle_closure import observe_quantities
        current=np.zeros(12000) if self.zero_sensory else np.asarray(enc.encode(observe_quantities(self.m,d,self.q0,self.joints,self.geom))[0])
        decay=np.exp(-rt.NDT/self.tau);counts=np.zeros(12000,dtype=np.int32);self.last_spikes=[]
        for _ in range(2):
            sp=net.step(current_nA=current);counts+=sp;self.last_spikes.append(sp[self.cells].copy());self.history.consume(sp)
            rates=decay*rates+(1-decay)*np.asarray(self.poolmat@sp,float)/rt.NDT
        return rates,counts,current

def replay(t):
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    seq=np.repeat(arrays(AUD)['currents'].astype(float),2,axis=0)[:1200]
    we=sparse.load_npz(rt.b.AB/'A/ACTUAL_We.npz').tocsr();wi=sparse.load_npz(rt.b.AB/'A/ACTUAL_Wi.npz').tocsr()
    row=np.repeat(np.arange(12000),np.diff(we.indptr));we.data[np.isin(row,t.rh)]*=.375
    rows=[];refs={};P=t.poolmat[:,t.cells].toarray()
    for sigma,seed in [(0.,0)]+[(s,k) for s in [.001,.01] for k in [1,2,3]]:
        rng=np.random.default_rng(seed);mats=[]
        for m in [we,wi]:
            n=m.copy()
            if sigma:n.data*=np.exp(rng.normal(0,sigma,n.nnz))
            mats.append(n)
        net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=.5);net.set_connectivity(*mats)
        spikes=[];population=[]
        for cur in seq:
            sp=net.step(current_nA=cur);spikes.append(sp[t.cells]);population.append(sp.sum())
        spikes=np.asarray(spikes);data=dict(spikes204=spikes,population_counts=population)
        for tau in TAUS:
            decay=np.exp(-.0005/tau);rates=np.zeros(66);rr=[]
            for sp in spikes:
                rates=decay*rates+(1-decay)*(P@sp)/.0005;rr.append(rates.copy())
            rr=np.asarray(rr)[1::2];raw=(rr@t.matrix.T)*t.limits/100;cmd=np.clip(raw,-t.limits,t.limits)
            key=f'{tau:.3f}';data['rates_'+key]=rr;data['raw_'+key]=raw;data['cmd_'+key]=cmd
            if sigma==0:refs[tau]=(cmd,raw)
            ref=refs[tau][0]
            change=lambda a,b:float(np.sqrt(np.mean((a-b)**2))/(np.sqrt(np.mean(b**2))+1e-12))
            rows.append(dict(sigma=sigma,seed=seed,tau=tau,relative_rms=change(cmd,ref),relative_rms_post100ms=change(cmd[100:],ref[100:]),temporal_std=float(np.sqrt(np.mean(np.var(cmd[100:],axis=0)))),raw_temporal_std=float(np.sqrt(np.mean(np.var(raw[100:],axis=0)))),total_variation=float(np.mean(np.abs(np.diff(cmd[100:],axis=0)))),saturation=float(np.mean(abs(cmd)>=t.limits)),rate_hz=float(np.sum(population)/.6/12000)))
        np.savez_compressed(O/f'REPLAY_sigma{sigma}_seed{seed}.npz',**data)
        print('REPLAY',sigma,seed,flush=True)
    dump(O/'ROBUSTNESS.json',rows)
    return rows

def main():
    if sys.argv[1]=='prereg':prereg();return
    before=js(O/'INPUT_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in before.items())
    t=RobustRuntime()
    cov=O/'COVERAGE.json'
    # Inspected runtime + legacy constructor: current COVERAGE is write-only; rt.files refers PRIOR/COVERAGE instead.
    if cov.exists():
        dump(O/'CLEANUP_LEDGER.json',dict(deleted=[dict(path=str(cov),sha256=sha(cov),bytes=cov.stat().st_size,importance='可再生临时',reason='constructor write-only table; runtime/episode/new runner have no current-path reader; identities preserve pools/matrix')]))
        cov.unlink()
    ident=js(ARCH/'DATA/CACHE_IDENTITY.json');assert sha(ident['file'])==ident['sha256']
    with Path(ident['file']).open('rb') as f:original=pickle.load(f)
    teacher=arrays(TEACHER);assert np.array_equal(teacher['cells'],t.cells)
    P=t.poolmat[:,t.cells].toarray();warm={};alignment=[]
    for tau in TAUS:
        decay=np.exp(-.0005/tau);rr=np.zeros(66);history=[rr.copy()]
        for sp in teacher['spikes204']:
            rr=decay*rr+(1-decay)*(P@sp)/.0005;history.append(rr.copy())
        for phase,c in original.items():
            count=round((phase-.2)/.001)*2;rates=history[count];warm[tau,phase]=rates
            if tau==.005:
                error=float(np.max(abs(rates-c['rates'])));assert error<1e-10;alignment.append(dict(phase=phase,spikes_used=count,max_error=error))
    dump(O/'WARM_ALIGNMENT.json',alignment)
    np.savez_compressed(O/'WARM_FILTERS.npz',**{f'tau{tau:.3f}_phase{ph:.3f}':v for (tau,ph),v in warm.items()})
    rob=replay(t);results={};all_results=[]
    actor=lambda x,stochastic:(np.zeros(42,np.float32),0.,0.)
    def evaluate(tau,phase,label,**kw):
        t.tau=tau;t.cache={ph:dict(c,rates=warm[tau,ph].copy()) for ph,c in original.items()}
        config=O/'CONFIGS'/f'tau_{tau:.3f}'/'PARAMETERS.json'
        r,_=t.episode(label,phase,actor,config,**kw);r['tau_s']=tau
        dump(Path(r['folder'])/'RESULT.json',r);all_results.append(r)
        return r
    for tau in TAUS:
        results[tau]=[evaluate(tau,ph,f'tau{tau:.3f}_dev{ph}') for ph in PHASES]
    assert abs(results[.005][2]['J']-1.5903704331878943)<1e-9
    base=results[.005][2];rank=[]
    for tau in TAUS:
        r=results[tau][2];p=r['parts'];b=base['parts']
        guard=r['J']<=base['J']+1e-9 and all(p[k]>=b[k]-1e-9 for k in ['foot_support_mean','com_z_min']) and (p['first_nonfoot_s'] if p['first_nonfoot_s'] is not None else .2)>=(b['first_nonfoot_s'] if b['first_nonfoot_s'] is not None else .2)-1e-9
        maxr=max(max(x['relative_rms'],x['relative_rms_post100ms']) for x in rob if x['tau']==tau)
        rs=results[tau];rank.append(dict(tau=tau,guard=bool(guard),robust=maxr<=.15,max_relative_rms=maxr,passes=sum(x['validation']['passed'] for x in rs),worst_J=max(x['J'] for x in rs),mean_J=float(np.mean([x['J'] for x in rs]))))
    eligible=[x for x in rank if x['tau']!=.005 and x['guard'] and x['robust']]
    selected=min(eligible,key=lambda x:(-x['passes'],x['worst_J'],x['mean_J']))['tau'] if eligible else .005
    dump(O/'SELECTION.json',dict(ranking=rank,selected_tau=selected,final_not_yet_evaluated=True))
    if selected!=.005:
        for ph in FINAL:
            for tau in [.005,selected]:evaluate(tau,ph,f'final_tau{tau:.3f}_{ph}')
    evaluate(selected,1.3,'selected_repeat');evaluate(selected,1.3,'selected_motor_cut',cut=True);evaluate(selected,1.3,'selected_sensory_cut',sensory_cut=True)
    dump(O/'SUMMARY.json',dict(selected_tau=selected,ranking=rank,results=all_results,physical_releases=len(all_results),physical_seconds=.2*len(all_results),neural_replay_steps=8400,new_warmups=0,trained_parameters=0,final_skipped=selected==.005))
    after={p:sha(p) for p in before};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
    print('DONE',selected,len(all_results),flush=True)
if __name__=='__main__':main()
