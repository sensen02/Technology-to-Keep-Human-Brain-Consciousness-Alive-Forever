"""S5 registered regression fit; only reusable runtime/translator are imported.
All online inputs are causal neural history. Offline reference labels are hypotheses.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,pickle,copy,hashlib,traceback,time
from pathlib import Path
import numpy as np
import torch
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
C=Path(__file__).resolve().parent
BASE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path.insert(0,str(BASE))
import runtime as rt
from translator import Translator,NumpyTranslator
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator')
FIRST=ROOT/'20261105T200021Z';PREV=ROOT/'20261105T231405Z';LAST=ROOT/'20261105T234531Z'
O=ROOT/C.name
rt.O=O;rt.b.O=O
js=rt.js;sha=rt.sha
TAUS=np.array([.010,.040,.100]);ORIGINAL=np.array([.005,.020,.050])
DEV=[1.225,1.275,1.3,1.325,1.375];FINAL=[1.214,1.264,1.314,1.364]
PENALTIES=[0.,.1,.5,2.,10.,50.];FLOOR=.9658022579833524
DATASETS=['closed_train','aggregate_1','aggregate_2','closed_dev']
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def dump(p,d):
    p=Path(p).resolve();assert p.is_relative_to(O)
    rt.dump(p,d)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(p,d):
    p=Path(p).resolve();assert p.is_relative_to(O);assert not p.exists();np.savez_compressed(p,**d)
def frozen_inputs():
    # Protect all sealed archive entries, not merely files used by this fit.
    result={}
    for root in [FIRST,PREV,LAST]:
        manifest=root/'MANIFEST_SHA256.json';result[str(manifest)]=sha(manifest)
        for p,h in js(manifest).items():
            assert sha(p)==h,('sealed input mismatch',p)
            if p in result:assert result[p]==h
            result[p]=h
    for p in rt.b.sources()+[BASE/'runtime.py',BASE/'translator.py',Path(__file__),C/'verify.py',C/'report.py']:
        result[str(p)]=sha(p)
    for p,h in js(LAST/'INPUT_HASHES_BEFORE.json').items():
        assert sha(p)==h,('upstream input mismatch',p)
        result[p]=h
    return result

def cleanup(p,reason):
    p=Path(p).resolve();assert p.is_relative_to(O)
    if p.exists():
        ledger=js(O/'CLEANUP_LEDGER.json');ledger['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,importance='可再生临时',reason=reason));p.unlink();dump(O/'CLEANUP_LEDGER.json',ledger)

def export(path,model,gain=1.):
    weights={f'W{i}':m.weight.detach().cpu().numpy().copy() for i,m in enumerate(m for m in model.net if isinstance(m,torch.nn.Linear))}
    save(path,dict(kind=np.array('mlp'),taus_s=TAUS,residual_gain=np.array(gain),**weights))
    z=arrays(path);assert set(z)=={'kind','taus_s','residual_gain','W0','W1','W2'}
    assert np.array_equal(z['taus_s'],TAUS) and float(z['residual_gain'])==gain
    restored=Translator();restored.restore_export(path)
    for a,b in zip(model.parameters(),restored.parameters()):assert torch.equal(a,b)
    xx=np.random.default_rng(12).normal(size=(7,612)).astype(np.float32)
    np.testing.assert_allclose(NumpyTranslator(path)(xx),model(torch.from_numpy(xx)).detach().numpy(),atol=1e-5,rtol=1e-4)

def prereg():
    O.mkdir(parents=True,exist_ok=False)
    for name in ['DATA','CONFIGS','RUNS']:(O/name).mkdir()
    last=js(LAST/'SUMMARY.json');initial=last['best_nonzero']['model']
    assert last['best_nonzero']['name']=='double_gain_1.0'
    protection=next(r for r in last['candidates'][0]['results'] if r['release_time_s']==1.3)
    config=dict(stage='S5 neural-only regression-protected translator',importance='核心必需',
        initial_model=initial,initial_sha256=sha(initial),protection_result=protection,
        frozen='SNN12000, encoder, RH0.375, original matrix, old66-pool5ms, full physical model and torque limits unchanged. No body-state/time/phase input, no online teacher/PD/CPG.',
        filter_taus_s=TAUS,history='Warm spikes from .2s to release (round((release-.2)/.0005)); then each completed NEURAL.spikes204 pair. Features immediately before each action. Explicitly assign decay before episode; never restore decay implicitly.',
        correction_sources=[str(PREV/'DATA'/f'{n}.npz') for n in DATASETS],
        correction_mask='Strict continuous initial prefix: every preceding and next action native window has no positive nonfoot contact and COM>=.9658022579833524. Preserve all excluded rows. Closed train+aggregate1+aggregate2 only; closed_dev whole trajectories only for validation. No warm anchor in this finite experiment.',
        protection='ONLY known-good OLD zero-residual release1.300s strict continuous initial safe prefix; target residual0 and target applied torque=old. Not all warm gait states. This is FIT data, not holdout.',
        loss='correction mean((clip(old/L+2*tanh(MLP(x)),-1,1)-teacher/L)^2) + lambda*0.5*(mean((2*tanh(MLP(protect_x)))^2)+mean((clip(protect_old/L+2*tanh(MLP(protect_x)),-1,1)-protect_old/L)^2))',
        fit=dict(penalties=PENALTIES,initialization='same frozen refit_1 weights every fit',seed=20261106,Adam_lr=.0003,weight_decay=.0001,max_epochs=120,patience=12,batch=256,gradient_clip=1.,epoch_selection='whole closed_dev eligible trajectory MSE + lambda*protection FIT loss; initial epoch included; tolerance1e-7'),
        dev=DEV,final=FINAL,duration_s=.2,
        guard='At1.3 J<=zero+1e-9 and footmean/minCOM/firstnonfoot>=zero-1e-9.',
        selection='Among guard-passing candidates rank full D pass count, lower worst J, lower mean J across5dev. Separately best_nonzero and unconstrained lambda0; final never used for selection.',
        trials='2 exact engineering reproductions (reuse their dev results); 8 candidates*5 dev=40 total including those2; final 4*(2or3 deduplicated models); nonzero repeat+3 real cuts; separate zero repeat1.3. All starts, including failures, in TRIALS.json.',
        final_cache='New Runtime.warmup captures all4 final times before candidates, full warm body/native/spikes retained; no previous holdout reused.',
        limitations='Teacher labels offline hypothetical, not necessary actions; no information absence or walking-intention claim. Any full D gate failure remains behavioral failure. SI NOT_ESTABLISHED.',
        output_root=str(O),source_root=str(C))
    dump(O/'PREREGISTRATION.json',config)
    dump(O/'INPUT_HASHES_BEFORE.json',frozen_inputs())
    dump(O/'ARCHIVE_PATHS_BEFORE.json',{str(root):sorted(str(p) for p in root.rglob('*') if p.is_file()) for root in [FIRST,PREV,LAST]})
    dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))
    dump(O/'CHANGE_LEDGER.json',dict(old_file=initial,old_sha256=sha(initial),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Matched10/40/100ms neural training and explicit old1.3 safety-prefix regression targets.'))
    print('REGISTERED',O,flush=True)

def history(spikes,taus,stops):
    decay=np.exp(-.0005/taus)[:,None];state=np.zeros((3,204));out={0:state.copy()}
    for i,s in enumerate(spikes,1):
        state=decay*state+(1-decay)*np.asarray(s,float)[None,:]/.0005
        if i in stops:out[i]=state.copy()
    assert set(stops)<=set(out)
    return out

def strict_mask(native,count):
    local=np.array([np.max(native[80*i:80*i+81,7])==0 and np.min(native[80*i:80*i+81,10])>=FLOOR for i in range(count)])
    return np.logical_and.accumulate(local)

def rebuild_source(folder,warmsp):
    n=arrays(folder/'NEURAL.npz');b=arrays(folder/'BODY.npz');a=arrays(folder/'NATIVE.npz')['rows'];cfg=js(folder/'PREREGISTRATION.json')
    release=float(cfg['release_time_s']);count=len(n['time']);stop=round((release-.2)/.0005)
    assert n['spikes204'].shape==(count,2,204)
    assert np.max(abs(b['time'][::2][:-1]-(n['time']-.001)))<1e-9
    both=[]
    for taus in [ORIGINAL,TAUS]:
        rates=history(warmsp[:stop],taus,{stop})[stop];decay=np.exp(-.0005/taus)[:,None];features=[]
        for pair in n['spikes204']:
            features.append((rates/100).astype(np.float32).ravel())
            for sp in pair:rates=decay*rates+(1-decay)*sp[None,:]/.0005
        both.append(np.array(features))
    assert np.array_equal(both[0],n['features']),('source-decay mismatch',str(folder))
    return both[1],strict_mask(a,count),n,b,release

def prepare_data(t):
    warmsp=arrays(FIRST/'DATA/train_offset_0/TEACHER.npz')['spikes204'];assert warmsp.shape==(2400,204)
    sourcecache={};reports=[];allsets={};train_sources=set();dev_sources=set()
    for name in DATASETS:
        d=arrays(PREV/'DATA'/f'{name}.npz');assert set(d)=={'x','old','y','eligible','time','source','body_index'}
        out={k:v.copy() for k,v in d.items()};out['original_eligible']=out.pop('eligible');out['original_x']=out.pop('x');out['x']=np.zeros_like(d['x']);out['eligible']=np.zeros(len(d['x']),bool)
        for src in np.unique(d['source']):
            if src not in sourcecache:sourcecache[src]=rebuild_source(Path(src),warmsp)
            x,mask,n,b,release=sourcecache[src];ids=np.flatnonzero(d['source']==src);bi=d['body_index'][ids];assert np.all(bi%2==0);ii=bi//2
            assert release in (rt.DEV if name=='closed_dev' else rt.TRAIN)
            assert np.array_equal(d['x'][ids],n['features'][ii]) and np.array_equal(d['old'][ids],n['old_torque'][ii])
            assert np.array_equal(d['time'][ids],b['time'][bi]);assert np.all(np.isfinite(d['y'][ids]))
            out['x'][ids]=x[ii];out['eligible'][ids]=mask[ii]
            (dev_sources if name=='closed_dev' else train_sources).add(src)
        # Existing eligibility used a slightly stricter floor; all selected points must satisfy it too.
        assert np.all(out['original_eligible'][out['eligible']])
        save(O/'DATA'/f'{name}.npz',out);allsets[name]=out
        reports.append(dict(name=name,total=len(out['x']),strict_prefix=int(out['eligible'].sum()),original_eligible=int(d['eligible'].sum())))
    assert not train_sources&dev_sources
    cfg=js(O/'PREREGISTRATION.json');src=Path(cfg['protection_result']['folder']);x,mask,n,b,release=rebuild_source(src,warmsp);assert release==1.3
    assert mask.any() and np.array_equal(np.flatnonzero(mask),np.arange(mask.sum()))
    protect=dict(x=x,old=n['old_torque'],y=n['old_torque'].copy(),zero_residual=np.zeros_like(n['old_torque']),eligible=mask,time=b['time'][::2][:-1],body_index=np.arange(len(x))*2,source=np.array([str(src)]*len(x)))
    save(O/'DATA/protection_old1.300.npz',protect)
    dump(O/'DATA/PROVENANCE.json',dict(datasets=reports,protection=dict(source=str(src),release=release,prefix_rows=int(mask.sum()),first_excluded_action=int(mask.sum()),target='old torque and zero residual',not_holdout=True),sources_original_decay_exact=len(sourcecache)+1,train_sources=sorted(train_sources),dev_sources=sorted(dev_sources),no_final_sources=True,original_features_retained_for_audit=True))
    return allsets,protect,warmsp

def actor(path):
    z=arrays(path);m=NumpyTranslator(path);gain=float(z['residual_gain']);assert gain in [0.,1.] and np.array_equal(z['taus_s'],TAUS)
    def call(x,stochastic):
        assert not stochastic
        return (np.zeros(42,np.float32) if gain==0 else m(x)),0.,0.
    return call

def config_folder(name):
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z'
    folder=O/'CONFIGS'/(stamp+'_'+name);folder.mkdir(exist_ok=False);return folder

def fit(name,penalty,sets,protect,limits,initial):
    folder=config_folder(name)
    dump(folder/'PREREGISTRATION.json',dict(name=name,penalty=penalty,initial=str(initial),initial_sha256=sha(initial),parent_sha256=sha(O/'PREREGISTRATION.json'),source_sha256=sha(__file__)))
    def tensor(d):
        mask=d['eligible'];return tuple(torch.tensor(d[k][mask]/(limits if k!='x' else 1.),dtype=torch.float32) for k in ['x','old','y'])
    train={k:np.concatenate([sets[n][k] for n in DATASETS[:-1]]) for k in ['x','old','y','eligible']}
    x,old,y=tensor(train);vx,vo,vy=tensor(sets['closed_dev']);px,po,py=tensor(protect)
    torch.manual_seed(20261106);rng=np.random.default_rng(20261106);m=Translator();m.restore_export(initial);opt=torch.optim.Adam(m.parameters(),lr=.0003,weight_decay=.0001)
    def correction(a,b,c):return ((torch.clamp(b+2*torch.tanh(m(a)),-1,1)-c)**2).mean()
    def guardloss():
        residual=2*torch.tanh(m(px));return .5*(residual.square().mean()+(torch.clamp(po+residual,-1,1)-po).square().mean())
    best=float('inf');beststate=None;patience=0;trace=[]
    for epoch in range(-1,120):
        if epoch>=0:
            for _ in range(int(np.ceil(len(x)/256))):
                ii=rng.integers(len(x),size=256);loss=correction(x[ii],old[ii],y[ii])+penalty*guardloss();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        with torch.no_grad():cv=float(correction(vx,vo,vy));pv=float(guardloss());val=cv+penalty*pv
        trace.append(dict(epoch=epoch,closed_dev_MSE=cv,protection_fit_loss=pv,selection_loss=val))
        if val<best-1e-7:best=val;beststate=copy.deepcopy(m.state_dict());patience=0;bestepoch=epoch
        else:patience+=1
        if patience>=12:break
    m.load_state_dict(beststate);path=folder/'MODEL.npz';export(path,m)
    dump(folder/'FIT.json',dict(history=trace,best_epoch=bestepoch,model_sha256=sha(path),train_rows=len(x),dev_rows=len(vx),protection_fit_rows=len(px),penalty=penalty))
    print('FIT',name,'epoch',bestepoch,'loss',best,flush=True);return path

def rank(rs):return (sum(r['validation']['passed'] for r in rs),-max(r['J'] for r in rs),-float(np.mean([r['J'] for r in rs])))
def guarded(rs,baseline):
    a=next(r for r in rs if r['release_time_s']==1.3);b=next(r for r in baseline if r['release_time_s']==1.3)
    return a['J']<=b['J']+1e-9 and all(a['parts'][k]>=b['parts'][k]-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s'])
def exact(a,b):
    assert a['J']==b['J']
    for name in ['BODY.npz','NATIVE.npz']:
        aa=arrays(Path(a['folder'])/name);bb=arrays(Path(b['folder'])/name);assert all(np.array_equal(aa[k],bb[k]) for k in aa),(name,a['folder'],b['folder'])

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in before.items())
    assert rt.O==O and rt.b.O==O
    t=rt.Runtime();cleanup(O/'COVERAGE.json','Unused legacy constructor coverage; actual mapping retained in TRANSLATOR_IDENTITIES.json; no readers.')
    assert rt.O==O and rt.b.O==O
    cp=LAST/'DATA/CACHE.pkl';assert sha(cp)==js(LAST/'DATA/CACHE_IDENTITY.json')['sha256']
    with cp.open('rb') as f:t.cache=pickle.load(f)
    base=t.cache.copy();sets,protect,warmsp=prepare_data(t)
    initial=Path(js(O/'PREREGISTRATION.json')['initial_model']);z=arrays(initial);assert np.array_equal(z['taus_s'],TAUS) and float(z['residual_gain'])==1
    refit=arrays(js(PREV/'SUMMARY.json')['best_nonzero']['model']);assert all(np.array_equal(z[k],refit[k]) for k in ['W0','W1','W2'])
    model=Translator();model.restore_export(initial);baseline_paths={}
    for name,gain in [('zero',0.),('unchanged',1.)]:
        folder=config_folder(name);dump(folder/'PREREGISTRATION.json',dict(name=name,gain=gain,taus_s=TAUS,initial=str(initial),initial_sha256=sha(initial)));export(folder/'MODEL.npz',model,gain);baseline_paths[name]=folder/'MODEL.npz'
    # Demonstrate original decay reconstruction for every cached release; retain original cache objects unchanged.
    stops={round((p-.2)/.0005) for p in base};original=history(warmsp,ORIGINAL,stops)
    assert all(np.array_equal(original[round((p-.2)/.0005)],c['history']['rates']) for p,c in base.items())
    matched=history(warmsp,TAUS,stops)
    for p,c in base.items():
        assert c['history']['samples']==round((p-.2)/.0005)
    dump(O/'PREFLIGHT.json',dict(passed=True,source_decay_exact=True,complete_cached_states=len(base),datasets_valid=True,export_load_metadata=True,output_paths_redirected=True,imported_drivers=False,protection_rows=int(protect['eligible'].sum()),final_not_evaluated=True))
    records=[]
    def evaluate(path,label,phase,cut=False,sensory_cut=False,translator_cut=False):
        assert Path(path).resolve().is_relative_to(O);assert rt.O==O and rt.b.O==O
        t.history.decay=np.exp(-.0005/TAUS)[:,None]
        state=base[phase];t.cache={phase:dict(state,history=dict(state['history'],rates=matched[round((phase-.2)/.0005)].copy()))}
        trial=dict(index=len(records)+1,label=label,release=phase,model=str(path),cut=cut,sensory_cut=sensory_cut,translator_cut=translator_cut,status='started')
        records.append(trial);dump(O/'TRIALS.json',records)
        try:
            result,_=t.episode(label,phase,actor(path),path,stochastic=False,cut=cut,sensory_cut=sensory_cut,translator_cut=translator_cut)
            trial.update(status='completed',result=result);dump(O/'TRIALS.json',records);return result
        except BaseException:
            trial.update(status='failed',traceback=traceback.format_exc());dump(O/'TRIALS.json',records);raise
    zero=baseline_paths['zero'];unchanged=baseline_paths['unchanged']
    zr=evaluate(zero,'ENGINEERING_zero',1.3);ur=evaluate(unchanged,'ENGINEERING_unchanged',1.3)
    prior=js(LAST/'SUMMARY.json');exact(zr,js(O/'PREREGISTRATION.json')['protection_result']);exact(ur,next(r for r in prior['best_nonzero']['results'] if r['release_time_s']==1.3))
    dump(O/'ENGINEERING_REPRODUCTION.json',dict(zero=zr,unchanged=ur,BODY_NATIVE_J_exact=True))
    # All final caches are captured before any fitted candidate; retain complete actual new warmup.
    t.cache=base.copy();t.history.decay=np.exp(-.0005/ORIGINAL)[:,None];t.warmup('fresh_final_warmup',0.,FINAL);base=t.cache.copy()
    freshsp=arrays(O/'DATA/fresh_final_warmup/TEACHER.npz')['spikes204'];assert np.array_equal(freshsp,warmsp)
    stops={round((p-.2)/.0005) for p in base};matched=history(freshsp,TAUS,stops)
    with (O/'DATA/CACHE.pkl').open('wb') as f:pickle.dump(base,f,protocol=5)
    dump(O/'DATA/CACHE_IDENTITY.json',dict(file=str(O/'DATA/CACHE.pkl'),sha256=sha(O/'DATA/CACHE.pkl')))
    candidates=[]
    def assess(path,name,penalty,reuse=None):
        rs=[reuse if p==1.3 and reuse is not None else evaluate(path,name+'_'+str(p),p) for p in DEV]
        item=dict(name=name,model=str(path),penalty=penalty,results=rs,rank=rank(rs));item['guard_pass']=True if not candidates else guarded(rs,candidates[0]['results']);candidates.append(item);dump(O/'CANDIDATES.json',candidates);return item
    assess(zero,'zero',None,zr);assess(unchanged,'unchanged',None,ur)
    for penalty in PENALTIES:
        name='lambda_'+str(penalty);path=fit(name,penalty,sets,protect,t.limits,initial);assess(path,name,penalty)
    safe=max((v for v in candidates if v['guard_pass']),key=lambda v:v['rank']);nonzero=max(candidates[1:],key=lambda v:v['rank'])
    selection=dict(guarded=safe,best_nonzero=nonzero,unconstrained=next(c for c in candidates if c['name']=='lambda_0.0'),zero=candidates[0],final_not_used=True)
    dump(O/'FROZEN_SELECTION.json',selection);final=[]
    for phase in FINAL:
        unique={}
        for c in [candidates[0],safe,nonzero]:
            if c['model'] not in unique:unique[c['model']]=evaluate(c['model'],'FINAL_'+c['name']+'_'+str(phase),phase)
        final.append(dict(phase=phase,zero=unique[str(zero)],guarded=unique[safe['model']],best_nonzero=unique[nonzero['model']]))
    rep=next(r for r in nonzero['results'] if r['release_time_s']==1.3);repeat=evaluate(nonzero['model'],'NONZERO_repeat',1.3);exact(rep,repeat)
    motor=evaluate(nonzero['model'],'NONZERO_motor_cut',1.3,cut=True);sensory=evaluate(nonzero['model'],'NONZERO_sensory_cut',1.3,sensory_cut=True);branch=evaluate(nonzero['model'],'NONZERO_translator_cut',1.3,translator_cut=True);exact(zr,branch)
    zerorepeat=evaluate(zero,'ZERO_repeat',1.3);exact(zr,zerorepeat)
    summary=dict(candidates=candidates,**selection,final_pairs=final,representative=rep,repeat=repeat,motor_cut=motor,sensory_cut=sensory,translator_cut=branch,zero_repeat=zerorepeat,total=len(records),failed_starts=sum(r['status']=='failed' for r in records),engineering_runs=2,warmups=1,simulated_seconds=1.4+.2*len(records),behavioral_passes=sum(r['result']['validation']['passed'] for r in records if r['status']=='completed'),walking_certified=False,intent_preserved='NOT_ESTABLISHED',body_state_input_added=False)
    dump(O/'SUMMARY.json',summary)
    print('COMPLETE',len(records),'guarded',safe['name'],'bestnonzero',nonzero['name'],flush=True)

def main():
    mode=sys.argv[1]
    if mode=='prereg':prereg();return
    try:run()
    except BaseException:dump(O/'FAILURE_run.json',dict(traceback=traceback.format_exc()));raise
    finally:
        before=js(O/'INPUT_HASHES_BEFORE.json');after={p:sha(p) for p in before};dump(O/'INPUT_HASHES_AFTER_run.json',after);assert before==after
if __name__=='__main__':main()
