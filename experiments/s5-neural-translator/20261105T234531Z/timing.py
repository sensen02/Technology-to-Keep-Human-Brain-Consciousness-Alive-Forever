"""Frozen translator: causal-history time constants x bounded residual amplitude.
No future features, body state, teacher or phase enters live translation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,time,copy,pickle
from pathlib import Path
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0,str(R/'experiments/s5-neural-translator/20261105T231405Z'))
import continuation as previous
rt=previous.rt;FIRST=previous.OLD;PREV=previous.O
from translator import SpikeHistory,NumpyTranslator
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T234531Z');rt.O=O;rt.b.O=O
js=rt.js;dump=rt.dump;sha=rt.sha
DEV=[1.225,1.275,1.3,1.325,1.375];FINAL=[1.206,1.256,1.306,1.356]
TAUS={'original':np.array([.005,.020,.050]),'half':np.array([.0025,.010,.025]),'fifth':np.array([.001,.004,.010]),'double':np.array([.010,.040,.100])}
GAINS=[.05,.1,.25,.5,.75,1.]
MODEL=Path(js(PREV/'SUMMARY.json')['best_nonzero']['model'])

def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def inputs():return [PREV/'MANIFEST_SHA256.json',PREV/'SUMMARY.json',PREV/'DATA/CACHE.pkl',FIRST/'DATA/train_offset_0/TEACHER.npz',MODEL,Path(previous.__file__),Path(rt.__file__),Path(__file__)]+rt.b.sources()
def prereg():
    O.mkdir(parents=True,exist_ok=False);(O/'DATA').mkdir();(O/'CONFIGS').mkdir()
    dump(O/'PREREGISTRATION.json',dict(stage='S5 frozen neural translator timing and amplitude diagnosis with known-state regression guard',
        frozen='Weights of selected refit_1 MLP;12000SNN/RH0.375/old174 readout/sensory encoder/physical plant/limits unchanged; no training or new body inputs',
        temporal='Replace only translator204-cell exponential filters; reconstruct initial memory from actual archived warmup spikes up to the release boundary, never from future spikes or reset at release. Original66-pool5ms decoder remains unchanged.',
        variants=dict(taus_s={k:v for k,v in TAUS.items()},residual_gains=GAINS,zero_baseline='gain0 original filters'),
        force='tau=clip(old_torque+2L*gain*tanh(MLP(features)), -L,L); implemented through atanh for the fixed runtime, with original gain1 shortcut exact and gain0 exact zero',
        offline_alignment='Compare recorded correction against hypothetical reference correction on saved body states only; evaluate lag -20,-10,-5,-2,-1,0,1,2,5,10,20ms within strict pre-first-contact prefix. prediction[t] vs reference[t+lag], negative best lag suggests correspondence to past reference, not proof of a physical delay.',
        selection='Protect1.300s: J must not rise and mean foot load/minCOM/first nonfoot time must not decrease relative to zero gain, tolerance1e-9. Among guard-passing variants rank full D-pass count, worstJ, meanJ across5 states. Also freeze best nonzero unguarded variant to expose unresolved tradeoffs.',
        phases=dict(dev=DEV,new_final=FINAL),duration_s=.2,
        evaluations='25 variants x5 phases=125; frozen zero/guarded/best-nonzero on4 new final states, reuse same variants; repeat and3 cut controls of best nonzero. No selection using final results.',
        limits='Changing filter constants also changes feature distributions; amplitude and time-factor controls do not by themselves uniquely isolate delay. No intention semantics inferred from activity magnitude.',
        resources='Single heavy process, reuse isolated torchCPU environment',units='NOT_ESTABLISHED',importance='核心必需'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(PREV/'PREREGISTRATION.json'),old_sha256=sha(PREV/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='User continue: separate neural filter times and correction amplitude; protect known1.3s state, no direct body compensation.'))
    dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))

def cleanup(p,reason):
    if p.exists():
        d=js(O/'CLEANUP_LEDGER.json');d['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,classification='可再生临时',reason=reason));p.unlink();dump(O/'CLEANUP_LEDGER.json',d)
def runtime(prepared=False):
    t=rt.Runtime();cleanup(O/'COVERAGE.json','Unused legacy dominant coverage; current identity mapping remains in TRANSLATOR_IDENTITIES.')
    p=(O if prepared else PREV)/'DATA/CACHE.pkl';assert sha(p)==js(p.with_name('CACHE_IDENTITY.json'))['sha256']
    with p.open('rb') as f:t.cache=pickle.load(f)
    return t

def diagnose(t):
    prior=js(PREV/'SUMMARY.json');cases=prior['best_nonzero']['results']+[prior['representative']];out=[]
    for r in cases:
        folder=Path(r['folder']);n=arrays(folder/'NEURAL.npz');b=arrays(folder/'BODY.npz');a=arrays(folder/'NATIVE.npz')['rows'];count=len(n['time']);indices=np.arange(count)*2
        target=previous.correction(t,b,indices);old=n['old_torque'];actual=n['applied'];wanted=(target-old)/t.limits;pred=(actual-old)/t.limits
        touch=np.flatnonzero(a[:,7]>0);first=a[touch[0],0] if len(touch) else a[-1,0]+.001;clean=np.array([b['time'][2*i]+.001<first-1e-12 and np.min(a[80*i:80*i+81,10])>=.9658022579833524 for i in range(count)])
        olderr=np.mean(((old-target)/t.limits)**2,axis=1);newerr=np.mean(((actual-target)/t.limits)**2,axis=1);bad=np.flatnonzero(clean&(newerr>olderr+1e-6));lagcurve=[]
        for lag in [-20,-10,-5,-2,-1,0,1,2,5,10,20]:
            ii=np.arange(count);jj=ii+lag;valid=(jj>=0)&(jj<count);ii=ii[valid];jj=jj[valid];valid=clean[ii]&clean[jj];ii=ii[valid];jj=jj[valid]
            if len(ii)>=10:lagcurve.append(dict(lag_ms=lag,n=len(ii),MSE=float(np.mean((pred[ii]-wanted[jj])**2))))
        d=dict(phase=r['release_time_s'],folder=str(folder),strict_prefix_samples=int(clean.sum()),first_contact_s=r['parts']['first_nonfoot_s'],first_reference_worsening_s=None if len(bad)==0 else float(b['time'][2*bad[0]]-r['release_time_s']),initial_old_reference_MSE=float(olderr[0]),initial_new_reference_MSE=float(newerr[0]),strict_old_reference_MSE=None if not clean.any() else float(olderr[clean].mean()),strict_new_reference_MSE=None if not clean.any() else float(newerr[clean].mean()),lag_curve=lagcurve,best_lag=None if not lagcurve else min(lagcurve,key=lambda z:z['MSE']))
        out.append(d);np.savez_compressed(O/'DATA'/('ALIGNMENT_'+str(r['release_time_s'])+'.npz'),time=b['time'][indices],target=target,old=old,actual=actual,strict_prefix=clean,old_error=olderr,new_error=newerr)
    dump(O/'TIMING_DIAGNOSIS.json',dict(cases=out,interpretation='Reference labels are hypothetical; lag correlation does not prove delay. Variable overlap per lag explicitly reported. No future data is used online.'))
    print('TIMING',[(r['phase'],r['strict_prefix_samples'],r['first_reference_worsening_s'],None if r['best_lag'] is None else r['best_lag']['lag_ms']) for r in out],flush=True)

def prepare():
    t=runtime();diagnose(t);t.warmup('fresh_final_cache',0.,FINAL)
    with (O/'DATA/CACHE.pkl').open('wb') as f:pickle.dump(t.cache,f,protocol=5)
    dump(O/'DATA/CACHE_IDENTITY.json',dict(file=str(O/'DATA/CACHE.pkl'),sha256=sha(O/'DATA/CACHE.pkl'),owned_local_pickle=True))
    cleanup(O/'DATA/fresh_final_cache/TEACHER.npz','Duplicate warm teacher artifact not used; causal spike-history source is original archived teacher file.')

def make_variant(name,gain,taus):
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z';folder=O/'CONFIGS'/(stamp+'_'+name);folder.mkdir();weights=arrays(MODEL);path=folder/'MODEL.npz';np.savez_compressed(path,**weights,residual_gain=float(gain),taus_s=np.asarray(taus))
    dump(folder/'PREREGISTRATION.json',dict(name=name,gain=gain,taus_s=taus,source_file=str(MODEL),source_sha256=sha(MODEL),new_file=str(path),new_sha256=sha(path),reason='Only declared causal neural filters and output amplitude change; all learned matrices identical.'))
    return path

def parameters(path):
    d=arrays(path);return float(d['residual_gain']),np.asarray(d['taus_s'])
def actor(path):
    m=NumpyTranslator(path);gain,_=parameters(path)
    def infer(x,stochastic):
        assert not stochastic
        if gain==0:u=np.zeros(42,np.float32)
        elif gain==1:u=m(x)
        else:u=np.arctanh(gain*np.tanh(m(x)))
        return u,0.,0.
    return infer

def histories(t,base,taus):
    spikes=arrays(FIRST/'DATA/train_offset_0/TEACHER.npz')['spikes204'];hist=SpikeHistory(t.cells);hist.decay=np.exp(-rt.NDT/taus)[:,None];steps={round((phase-.2)/rt.NDT):phase for phase in base};result={}
    for i,sample in enumerate(spikes):
        full=np.zeros(12000,bool);full[t.cells]=sample;hist.consume(full)
        if i+1 in steps:result[steps[i+1]]=hist.state_dict()
    assert set(result)==set(base)
    if np.array_equal(taus,TAUS['original']):assert all(np.array_equal(result[p]['rates'],base[p]['history']['rates']) for p in base)
    return result

def rank(rs):return(sum(r['validation']['passed'] for r in rs),-max(r['J'] for r in rs),-np.mean([r['J'] for r in rs]))
def guard(rs,baseline):
    r=next(x for x in rs if x['release_time_s']==1.3);b=next(x for x in baseline if x['release_time_s']==1.3)
    return r['J']<=b['J']+1e-9 and all(r['parts'][k]>=b['parts'][k]-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s'])

def run():
    t=runtime(True);base=t.cache.copy();h={name:histories(t,base,taus) for name,taus in TAUS.items()};records=[];candidates=[]
    def evaluate(path,name,phases,**flags):
        gain,taus=parameters(path);typename=next(k for k,v in TAUS.items() if np.array_equal(v,taus));t.history.decay=np.exp(-rt.NDT/taus)[:,None]
        t.cache={p:dict(c,history=h[typename][p]) for p,c in base.items()};results=[]
        for phase in phases:
            r,_=t.episode(name+'_'+str(phase),phase,actor(path),path,**flags);results.append(r);records.append(r);dump(O/'PROGRESS.json',records)
        return results
    zero=make_variant('zero',0.,TAUS['original']);baseline=evaluate(zero,'zero',DEV);candidates.append(dict(name='zero',model=str(zero),gain=0.,tau_mode='original',results=baseline,rank=rank(baseline),guard_pass=True))
    for mode,taus in TAUS.items():
        for gain in GAINS:
            name=mode+'_gain_'+str(gain);path=make_variant(name,gain,taus);rs=evaluate(path,name,DEV);item=dict(name=name,model=str(path),gain=gain,tau_mode=mode,results=rs,rank=rank(rs),guard_pass=guard(rs,baseline));candidates.append(item);dump(O/'CANDIDATES.json',candidates)
            if mode=='original' and gain==1:
                old=js(PREV/'SUMMARY.json')['representative'];new=next(r for r in rs if r['release_time_s']==1.3);aa=arrays(Path(old['folder'])/'BODY.npz');bb=arrays(Path(new['folder'])/'BODY.npz');assert old['J']==new['J'] and all(np.array_equal(aa[k],bb[k]) for k in aa)
    safe=max((v for v in candidates if v['guard_pass']),key=lambda v:v['rank']);nonzero=max(candidates[1:],key=lambda v:v['rank'])
    dump(O/'FROZEN_SELECTION.json',dict(guarded=safe,best_nonzero=nonzero,reference_zero=candidates[0],final_not_used=True))
    final=[]
    for phase in FINAL:
        by_model={}
        for variant in [candidates[0],safe,nonzero]:
            if variant['model'] not in by_model:by_model[variant['model']]=evaluate(variant['model'],'FINAL_'+variant['name'],[phase])[0]
        final.append(dict(phase=phase,zero=by_model[candidates[0]['model']],guarded=by_model[safe['model']],best_nonzero=by_model[nonzero['model']]))
    rep=next(r for r in nonzero['results'] if r['release_time_s']==1.3);repeat=evaluate(nonzero['model'],'REPEAT',[1.3])[0]
    aa=arrays(Path(rep['folder'])/'BODY.npz');bb=arrays(Path(repeat['folder'])/'BODY.npz');assert rep['J']==repeat['J'] and all(np.array_equal(aa[k],bb[k]) for k in aa)
    motor=evaluate(nonzero['model'],'ALL_MOTOR_CUT',[1.3],cut=True)[0];sensory=evaluate(nonzero['model'],'SENSORY_CUT',[1.3],sensory_cut=True)[0];branch=evaluate(nonzero['model'],'NEW_BRANCH_CUT',[1.3],translator_cut=True)[0]
    assert branch['J']==next(r['J'] for r in baseline if r['release_time_s']==1.3)
    dump(O/'SUMMARY.json',dict(candidates=candidates,guarded=safe,best_nonzero=nonzero,final_pairs=final,representative=rep,repeat=repeat,motor_cut=motor,sensory_cut=sensory,translator_cut=branch,total=len(records),walking_certified=False,intent_preserved='NOT_ESTABLISHED',body_state_input_added=False))
    print('COMPLETE',len(records),'guarded',safe['name'],'nonzero',nonzero['name'],flush=True)

def main():
    mode=sys.argv[1]
    if mode=='prereg':prereg();return
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in inputs()}
    try:
        if mode=='prepare':prepare()
        elif mode=='run':run()
    except BaseException:
        import traceback
        dump(O/('FAILURE_'+mode+'.json'),dict(traceback=traceback.format_exc()));raise
    finally:
        after={str(p):sha(p) for p in inputs()};dump(O/('INPUT_HASHES_AFTER_'+mode+'.json'),after);assert before==after
if __name__=='__main__':main()
