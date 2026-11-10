"""Closed-loop distribution audit and offline corrective-label data aggregation.
Teacher is evaluated on archived states with mj_forward only; never applied online.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,time,pickle,copy,json
from pathlib import Path
import numpy as np
import torch
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
C=Path(__file__).parent;OLD_CODE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path.insert(0,str(OLD_CODE))
import runtime as rt
from translator import Translator,NumpyTranslator
OLD=rt.O;O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T231405Z');rt.O=O;rt.b.O=O
js=rt.js;dump=rt.dump;sha=rt.sha
torch.set_num_threads(1);torch.set_num_interop_threads(1)
FRESH=[1.218,1.268,1.318,1.368]

def protected():
    return [OLD/'MANIFEST_SHA256.json',OLD/'DATA/CACHE.pkl',OLD/'DATA/CACHE_IDENTITY.json',OLD/'TRAIN_HISTORY.json',OLD/'VALIDATION_HISTORY.json',OLD/'SUMMARY.json',OLD/'MODELS/zero.npz',OLD/'MODELS/offline_mlp.npz',OLD_CODE/'runtime.py',OLD_CODE/'translator.py',Path(__file__)]+rt.b.sources()

def prereg():
    O.mkdir(parents=True,exist_ok=False);(O/'MODELS').mkdir();(O/'DATA').mkdir()
    dump(O/'PREREGISTRATION.json',dict(stage='S5 neural-only translator closed-loop distribution/alias audit and corrective data aggregation',
        frozen='Same 612->64->64->42 bias-free residual MLP; same12000SNN/RH0.375/old174 readout/limits/sensory encoder; no direct body compensation input or online teacher',
        sources='Previous160 exploratory TRAIN phases only for training; previous initial zero/linear/MLP DEVELOPMENT phases only for label validation; previous final phases excluded from all fit/selection',
        label='At saved 1ms action-window start, reconstruct qpos/qvel/actual previous ctrl, mj_forward only; evaluate previously used warmup PD gait reference at actual time and current state, clip to original teacher and actuator limits. Hypothetical reference correction, not proven necessary torque or guaranteed recovery.',
        fit_mask='Entire next1ms has zero positive nonfoot force and COM >=0.9658022579833524; do not teach recovery from collapsed/contact-corrupted states. Keep excluded rows for diagnostics.',
        temporal='Use saved source neural features before action; every2ms sampling for historical training/label validation; complete1ms samples for later aggregated trajectories; source time from BODY[2*k] checked against NEURAL endtime minus1ms',
        teacher_alignment='Warm teacher labels recomputed at the same window-start convention rather than mixing old averaged next-window targets with instantaneous closed-state targets',
        audit='Nearest-neighbor standardized feature RMS relative to warm training only; scale=max(train std,.05); nearest heldout warm distribution establishes reference; compare closed pre-contact and post-contact separately. Similar-input reference-label disagreement is diagnostic, not an information-theoretic bound.',
        fitting=dict(seed=20261105,initialization='previous offline MLP',closed_weights=[.25,.5,.75],max_epochs=100,patience=12,batch=256,adam_lr=.0003,weight_decay=.0001,normalization='torque/L',loss='(1-w)*mean warm MSE + w*mean closed eligible MSE',dev_early_stop='same mixture of whole-trajectory warm-dev and closed-dev; no closed final test labels'),
        aggregation='Choose by4 dev physical phases; collect chosen nonzero translator at5 train phases, append only eligible offline correction labels, refit at its fixed weight; repeat twice. Always retain zero baseline and previous incumbent; no final-test feedback.',
        selection='count full D gates then lower worst devJ then lower meanJ; independently identify best nonzero model on dev to exhibit failed alternative if zero retained',
        phases=dict(train=rt.TRAIN,dev=rt.DEV,final_fresh=FRESH),
        duration_s=.2,controls='old zero exact replay; frozen best/zero on4 fresh phases; representative repeat/new-branch cut/sensory cut/all-motor cut of trained model; actual selected model stated separately',
        intent='No independently validated walk/stop condition exists in current subnetwork; no intent-preservation claim; stance alone not walking',
        resource='One serial heavy process, reuse previous isolated torchCPU environment; all parameter variants timestamped configuration records before physical evaluation',importance='核心必需',units='NOT_ESTABLISHED'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in protected()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(OLD/'PREREGISTRATION.json'),old_sha256=sha(OLD/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='User continue: diagnose train/closed-loop shift and reference-action alias; improve training coverage while keeping inference neural-only.'))
    dump(O/'CLEANUP_LEDGER.json',dict(importance='重要辅助',deleted=[]))

def cleanup_generated():
    p=O/'COVERAGE.json'
    if p.exists():
        ledger=js(O/'CLEANUP_LEDGER.json');ledger['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,classification='可再生临时',reason='Unused legacy dominant coverage, actual mapping in TRANSLATOR_IDENTITIES; no new code reads this artifact.'));p.unlink();dump(O/'CLEANUP_LEDGER.json',ledger)

def runtime(fresh=False):
    t=rt.Runtime();cleanup_generated()
    p=O/'DATA/CACHE.pkl' if fresh else OLD/'DATA/CACHE.pkl'
    expected=js(p.with_name('CACHE_IDENTITY.json'))['sha256'];assert sha(p)==expected
    with p.open('rb') as f:t.cache=pickle.load(f)
    return t

def archived(path):
    # Hash every actual archived input before use, against its sealed manifest.
    path=Path(path)
    if str(path).startswith(str(OLD)+'/'):
        manifest=js(OLD/'MANIFEST_SHA256.json');assert sha(path)==manifest[str(path)]
    with np.load(path) as z:return {k:z[k] for k in z.files}

def correction(t,body,indices,offset=0.):
    d=t.mj.MjData(t.m);out=[]
    for i in indices:
        d.qpos[:]=body['qpos'][i];d.qvel[:]=body['qvel'][i];d.ctrl[:]=body['ctrl'][i];t.mj.mj_forward(t.m,d)
        when=float(body['time'][i]);ramp=np.clip((when-.15)/.1,0,1);phi=(max(0,when-.15)*2*np.pi*5+offset)%(2*np.pi)
        qg=t.gait(phi);desired=(1-ramp)*t.q0[t.qa]+ramp*qg
        velocity=ramp*t.gait(phi,1)*2*np.pi*5+(qg-t.q0[t.qa])*(10. if .15<when<.25 else 0.)
        tau=t.gains['kp']*(desired-d.qpos[t.qa])+np.asarray(t.gains['kd'])*(velocity-d.qvel[t.va])+d.qfrc_bias[t.va]-d.qfrc_passive[t.va]
        out.append(np.clip(np.clip(tau,-t.gains['clip'],t.gains['clip']),-t.limits,t.limits))
    return np.array(out,dtype=np.float32)

def warm_dataset(t,prefix):
    rows=[];paths=sorted((OLD/'DATA').glob(prefix+'*/TEACHER.npz'))
    for path in paths:
        n=archived(path);b=archived(path.parent/'BODY.npz');offset=js(path.parent/'PREREGISTRATION.json')['phase_offset_rad'];ids=np.flatnonzero(n['time']>=.3-1e-9)
        ix=np.array([np.argmin(abs(b['time']-tt)) for tt in n['time'][ids]]);assert np.max(abs(b['time'][ix]-n['time'][ids]))<1e-9
        target=correction(t,b,ix,offset);rows.append(dict(x=n['features'][ids],old=n['old_torque'][ids],y=target,eligible=np.ones(len(ids),bool),time=n['time'][ids],source=np.array([str(path)]*len(ids)),body_index=ix))
    return combine(rows)

def closed_dataset(t,results,stride=2):
    rows=[]
    for r in results:
        folder=Path(r['folder']);n=archived(folder/'NEURAL.npz');b=archived(folder/'BODY.npz');a=archived(folder/'NATIVE.npz')['rows'];ids=np.arange(0,len(n['time']),stride);ix=2*ids
        assert np.max(abs(b['time'][ix]-(n['time'][ids]-.001)))<1e-9
        target=correction(t,b,ix);eligible=np.array([np.max(a[80*i:80*i+81,7])==0 and np.min(a[80*i:80*i+81,10])>=.9658022579833524 for i in ids])
        rows.append(dict(x=n['features'][ids],old=n['old_torque'][ids],y=target,eligible=eligible,time=b['time'][ix],source=np.array([str(folder)]*len(ids)),body_index=ix))
    return combine(rows)

def combine(rows):return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
def save(name,data):np.savez_compressed(O/'DATA'/(name+'.npz'),**data)
def load(name):
    with np.load(O/'DATA'/(name+'.npz')) as z:return {k:z[k] for k in z.files}

def distances(query,reference,scale):
    ref=np.asarray(reference/scale,np.float32);qn=np.asarray(query/scale,np.float32);rn=np.sum(ref*ref,axis=1);ds=[];ids=[]
    for start in range(0,len(qn),256):
        q=qn[start:start+256];dd=np.maximum(0,np.sum(q*q,axis=1)[:,None]+rn[None,:]-2*q@ref.T);i=np.argmin(dd,axis=1);ids.extend(i);ds.extend(np.sqrt(dd[np.arange(len(i)),i]/q.shape[1]))
    return np.array(ds),np.array(ids)

def prepare():
    t=runtime();warm=warm_dataset(t,'train_offset_');warmdev=warm_dataset(t,'dev_offset_')
    training=js(OLD/'TRAIN_HISTORY.json');dev=[r for r in js(OLD/'VALIDATION_HISTORY.json') if r['label'].startswith('initial_')]
    closed=closed_dataset(t,training);closeddev=closed_dataset(t,dev)
    for name,ds in [('warm_train',warm),('warm_dev',warmdev),('closed_train',closed),('closed_dev',closeddev)]:save(name,ds)
    scale=np.maximum(np.std(warm['x'],axis=0),.05);wd,wi=distances(warmdev['x'],warm['x'],scale);cd,ci=distances(closed['x'],warm['x'],scale);dd,di=distances(closeddev['x'],warm['x'],scale)
    limit=t.limits
    def stats(dist,mask):
        z=dist[mask];return dict(n=len(z),median=float(np.median(z)),p90=float(np.quantile(z,.9)),beyond_warm_dev_p95=float(np.mean(z>np.quantile(wd,.95)))) if len(z) else dict(n=0)
    eligible=closed['eligible'];valelig=closeddev['eligible']
    # Compare dev points with nearest training points; do not compare adjacent frames to themselves.
    unionx=np.concatenate([warm['x'],closed['x'][eligible]]);uniony=np.concatenate([warm['y'],closed['y'][eligible]])/limit
    near,nn=distances(closeddev['x'][valelig],unionx,scale);yd=closeddev['y'][valelig]/limit;yn=uniony[nn];gaps=np.sqrt(np.mean((yd-yn)**2,axis=1));conflicts=np.mean(((yd*yn)<0)&(abs(yd)>.1)&(abs(yn)>.1),axis=1)
    audit=dict(warm_dev_to_train=stats(wd,np.ones(len(wd),bool)),closed_train_before_failure=stats(cd,eligible),closed_train_excluded=stats(cd,~eligible),closed_dev_before_failure=stats(dd,valelig),closed_dev_excluded=stats(dd,~valelig),reference_label_conflicts=dict(n=len(near),nearest_distance_median=float(np.median(near)),normalized_target_RMS_gap_median=float(np.median(gaps)),opposite_sign_joint_fraction_mean=float(np.mean(conflicts)),note='Hypothetical teacher-reference discrepancies; not proof of unique necessary action or unobservable intent.'),eligible_train=int(eligible.sum()),eligible_dev=int(valelig.sum()),all_train=len(eligible),all_dev=len(valelig),sources_exclude_previous_final=True)
    dump(O/'DISTRIBUTION_AUDIT.json',audit);np.savez_compressed(O/'DATA/DISTANCES.npz',scale=scale,warmdev_distance=wd,closed_distance=cd,closeddev_distance=dd,closed_nearest_warm=ci,dev_neighbor_distance=near,dev_target_gap=gaps,dev_sign_conflict=conflicts)
    # Fresh final times are captured from a continuous untouched physical/SNN warmup.
    t.warmup('fresh_final_cache',0.,FRESH)
    with (O/'DATA/CACHE.pkl').open('wb') as f:pickle.dump(t.cache,f,protocol=5)
    dump(O/'DATA/CACHE_IDENTITY.json',dict(file=str(O/'DATA/CACHE.pkl'),sha256=sha(O/'DATA/CACHE.pkl'),owned_local_pickle=True))
    # New warm teacher duplicate is not used for fit; body/native are required by replay/reference audit.
    p=O/'DATA/fresh_final_cache/TEACHER.npz';ledger=js(O/'CLEANUP_LEDGER.json');ledger['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,classification='可再生临时',reason='Fresh cache warmup produces duplicate teacher features; explicitly excluded from training, no verifier uses it. Original teacher trajectory remains archived.'));p.unlink();dump(O/'CLEANUP_LEDGER.json',ledger)
    dump(O/'DATA/SPLIT.json',dict(train_sources=sorted(set(closed['source'].tolist())),dev_sources=sorted(set(closeddev['source'].tolist())),train_warm_sources=sorted(set(warm['source'].tolist())),dev_warm_sources=sorted(set(warmdev['source'].tolist())),final_times=FRESH))
    print('AUDIT',json.dumps(audit),flush=True)

def actor(path):
    m=NumpyTranslator(path);return lambda x,stochastic:(m(x),0.,0.)
def score(rs):return(sum(r['validation']['passed'] for r in rs),-max(r['J'] for r in rs),-np.mean([r['J'] for r in rs]))

def fit(name,weight,closed,initial):
    warm=load('warm_train');warmdev=load('warm_dev');closeddev=load('closed_dev');limit=np.array(js(O/'TRANSLATOR_IDENTITIES.json')['limits'],np.float32)
    def tensors(ds):
        ids=ds['eligible'];return tuple(torch.tensor(ds[k][ids]/(limit if k!='x' else 1.),dtype=torch.float32) for k in ['x','old','y'])
    wx,wo,wy=tensors(warm);cx,co,cy=tensors(closed);vx,vo,vy=tensors(warmdev);dx,do,dy=tensors(closeddev)
    torch.manual_seed(20261105);rng=np.random.default_rng(20261105);m=Translator();m.restore_export(initial);opt=torch.optim.Adam(m.parameters(),lr=.0003,weight_decay=.0001);best=float('inf');state=None;patience=0;hist=[]
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+str(time.time_ns()%1000000000).zfill(9)+'Z';folder=O/'CONFIGS'/(stamp+'_'+name);folder.mkdir(parents=True)
    dump(folder/'PREREGISTRATION.json',dict(name=name,closed_weight=weight,initial_model=str(initial),initial_sha256=sha(initial),closed_rows=int(closed['eligible'].sum()),teacher_use='offline only',source_sha256=sha(__file__)))
    def mse(x,old,y):return ((torch.clamp(old+2*torch.tanh(m(x)),-1,1)-y)**2).mean()
    for epoch in range(100):
        for _ in range(max(1,int(np.ceil(max(len(wx),len(cx))/256)))):
            wi=rng.integers(len(wx),size=256);ci=rng.integers(len(cx),size=256);loss=(1-weight)*mse(wx[wi],wo[wi],wy[wi])+weight*mse(cx[ci],co[ci],cy[ci]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        with torch.no_grad():wv=float(mse(vx,vo,vy));cv=float(mse(dx,do,dy));val=(1-weight)*wv+weight*cv
        hist.append(dict(epoch=epoch,warm_dev_mse=wv,closed_dev_mse=cv,weighted_dev_mse=val))
        if val<best-1e-6:best=val;state=copy.deepcopy(m.state_dict());patience=0
        else:patience+=1
        if patience>=12:break
    m.load_state_dict(state);path=folder/'MODEL.npz';m.export(path);np.testing.assert_allclose(NumpyTranslator(path)(warmdev['x']),m(vx).detach().numpy(),atol=1e-5,rtol=1e-4)
    dump(folder/'FIT.json',dict(history=hist,best_weighted_dev_mse=best,model_sha256=sha(path),closed_weight=weight));print('FIT',name,best,flush=True);return path

def run():
    t=runtime(fresh=True);zero=OLD/'MODELS/zero.npz';ref=js(OLD/'SUMMARY.json')['representative'];reproduced,_=t.episode('zero_exact_replay',1.3,actor(zero),zero)
    a=archived(Path(ref['folder'])/'BODY.npz');b=archived(Path(reproduced['folder'])/'BODY.npz');assert ref['J']==reproduced['J'] and all(np.array_equal(a[k],b[k]) for k in a)
    baseline=[r for r in js(OLD/'VALIDATION_HISTORY.json') if r['label'].startswith('initial_zero_')]
    candidates=[dict(name='zero',model=str(zero),weight=0.,results=baseline,score=score(baseline))];history=[];closed=load('closed_train')
    def assess(path,name,weight):
        rs=[t.episode(name+'_'+str(phase),phase,actor(path),path)[0] for phase in rt.DEV];history.extend(rs);item=dict(name=name,model=str(path),weight=weight,results=rs,score=score(rs));candidates.append(item);dump(O/'CANDIDATES.json',candidates);return item
    for weight in [.25,.5,.75]:
        name='mixed_'+str(weight);path=fit(name,weight,closed,OLD/'MODELS/offline_mlp.npz');assess(path,name,weight)
    best_nonzero=max(candidates[1:],key=lambda c:c['score'])
    for round_no in [1,2]:
        collected=[t.episode('aggregate_'+str(round_no)+'_'+str(phase),phase,actor(best_nonzero['model']),best_nonzero['model'])[0] for phase in rt.TRAIN];history.extend(collected)
        added=closed_dataset(t,collected,stride=1);save('aggregate_'+str(round_no),added);closed=combine([closed,added]);save('training_union_'+str(round_no),closed)
        name='refit_'+str(round_no);path=fit(name,best_nonzero['weight'],closed,best_nonzero['model']);assess(path,name,best_nonzero['weight']);best_nonzero=max(candidates[1:],key=lambda c:c['score'])
    incumbent=max(candidates,key=lambda c:c['score']);dump(O/'FROZEN_SELECTION.json',dict(incumbent=incumbent,best_nonzero=best_nonzero,final_data_not_used=True))
    final=[]
    for phase in FRESH:
        old=t.episode('FINAL_zero_'+str(phase),phase,actor(zero),zero)[0];trained=t.episode('FINAL_trained_'+str(phase),phase,actor(best_nonzero['model']),best_nonzero['model'])[0];final.append(dict(phase=phase,old=old,trained=trained))
    path=best_nonzero['model'];base=t.episode('REPRESENTATIVE_trained',1.3,actor(path),path)[0];repeat=t.episode('REPRESENTATIVE_repeat',1.3,actor(path),path)[0]
    aa=archived(Path(base['folder'])/'BODY.npz');bb=archived(Path(repeat['folder'])/'BODY.npz');assert base['J']==repeat['J'] and all(np.array_equal(aa[k],bb[k]) for k in aa)
    cut=t.episode('TRAINED_all_motor_cut',1.3,actor(path),path,cut=True)[0];sens=t.episode('TRAINED_sensory_cut',1.3,actor(path),path,sensory_cut=True)[0];newcut=t.episode('TRAINED_new_branch_cut',1.3,actor(path),path,translator_cut=True)[0]
    assert newcut['J']==ref['J']
    dump(O/'SUMMARY.json',dict(candidates=candidates,incumbent=incumbent,best_nonzero=best_nonzero,final_pairs=final,representative=base,repeat=repeat,motor_cut=cut,sensory_cut=sens,translator_cut=newcut,zero_reproduction=reproduced,walking_certified=False,intent_preservation='NOT_ESTABLISHED',body_compensation_added=False))
    print('COMPLETE selected',incumbent['name'],'best nonzero',best_nonzero['name'],flush=True)

def main():
    mode=sys.argv[1]
    if mode=='prereg':prereg();return
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in protected()}
    try:
        if mode=='prepare':prepare()
        elif mode=='run':run()
    except BaseException:
        import traceback
        dump(O/('FAILURE_'+mode+'.json'),dict(traceback=traceback.format_exc()));raise
    finally:
        after={str(p):sha(p) for p in protected()};dump(O/('INPUT_HASHES_AFTER_'+mode+'.json'),after);assert before==after
if __name__=='__main__':main()
