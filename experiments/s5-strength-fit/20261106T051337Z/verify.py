"""Saved-array independent verification; no neural or physics stepping."""
import sys,pickle
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
O=r.O
W=24.446354755878005;G=9801.;H0=1.2072528224791905

def main():
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    model=_load_mjcf(r.rt.b.P/'MODEL_JOINT_TORQUE.xml').compile()
    assert abs(model.body_mass.sum()*abs(model.opt.gravity[2])-W)<1e-12 and abs(model.opt.gravity[2])==G and model.opt.timestep==.0000125
    summary=r.js(O/'SUMMARY.json');ident=r.js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(ident['cells']);limits=np.array(ident['limits']);matrix=np.array(ident['old_matrix']);groups=r.arrays(O/'GRAPH_GROUPS.npz')['groups']
    P=np.zeros((66,204))
    for i,pool in enumerate(ident['pools']):
        for c in pool:P[i,np.flatnonzero(cells==c)[0]]=1/len(pool)
    with (O/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
    mats=[sparse.load_npz(r.rt.b.AB/'A'/('ACTUAL_'+k+'.npz')).tocsr() for k in ['We','Wi']]
    warm={};details=[];config_checks=[];max_filter=0.;max_command=0.;max_J=0.
    for folder in sorted((O/'CONFIGS').iterdir()):
        if not folder.is_dir():continue
        cfg=r.js(folder/'PARAMETERS.json');x=np.array(cfg['x']);assert x.shape==(20,) and np.max(abs(x))<=1
        perturb=cfg['perturbation'];rng=np.random.default_rng(perturb['seed']) if perturb else None
        for i,m in enumerate(mats):
            n=sparse.load_npz(folder/('We.npz' if i==0 else 'Wi.npz')).tocsr();row=np.repeat(np.arange(12000),np.diff(m.indptr));factor=1+.1*x[groups[row]+10*i]
            if perturb:factor=np.clip(factor*np.exp(rng.normal(0,perturb['sigma'],m.nnz)),.9,1.1)
            assert np.array_equal(n.indptr,m.indptr) and np.array_equal(n.indices,m.indices) and np.array_equal(n.data,m.data*factor)
            ratio=n.data/m.data;assert ratio.min()>=.9-1e-14 and ratio.max()<=1.1+1e-14 and np.all(n.data>0)
        spikes=r.arrays(folder/'WARM_SPIKES.npz')['spikes204'];a=np.exp(-.0005/.005);filtered=lfilter([(1-a)/.0005],[1,-a],spikes@P.T,axis=0)
        with (folder/'WARM_CACHE.pkl').open('rb') as f:cache=pickle.load(f)
        for ph,c in cache.items():
            count=round((ph-.2)*2000);rr=filtered[count-1];err=float(np.max(abs(rr-c['rates'])));assert err<1e-9;max_filter=max(max_filter,err);warm[folder.name,ph]=rr
            assert np.array_equal(c['integration'],physical[ph]['integration']) and c['assist']==physical[ph]['assist']
            assert np.array_equal(c['net'].We.data,sparse.load_npz(folder/'We.npz').data)
            assert np.array_equal(c['net'].Wi.data,sparse.load_npz(folder/'Wi.npz').data)
            for j,tau in enumerate([.005,.020,.050]):
                decay=np.exp(-.0005/tau);history=lfilter([(1-decay)/.0005],[1,-decay],spikes[:count],axis=0)[-1]
                assert np.max(abs(history-c['history']['rates'][j]))<1e-9
            assert c['net'].steps==count and c['net'].t_ms==count*.5 and c['history']['samples']==count
        config_checks.append(dict(candidate=folder.name,all_edges_exact_expected=True,warm_states=len(cache),causal_rates_history=True))
    warmbody=r.arrays(O/'DATA/true_teacher/BODY.npz')
    for res in summary['results']:
        folder=Path(res['folder']);cfg=r.js(folder/'PREREGISTRATION.json');z=r.arrays(folder/'NEURAL.npz');body=r.arrays(folder/'BODY.npz');native=r.arrays(folder/'NATIVE.npz')['rows'];ph=res['release_time_s'];duration=res['duration_s'];name=res['candidate'];a=np.exp(-.0005/.005)
        initial=warm[name,ph];flat=z['spikes204'].reshape(-1,204)@P.T
        allrates=lfilter([(1-a)/.0005],[1,-a],flat,axis=0,zi=(a*initial)[None,:])[0]
        ends=allrates[1::2];starts=np.vstack([initial,ends[:-1]])
        error=max(float(np.max(abs(ends-z['rates']))),float(np.max(abs(starts-z['source_rates']))));assert error<1e-9;max_filter=max(max_filter,error)
        cmd=np.clip((starts@matrix.T)*limits/100,-limits,limits);assert np.max(abs(cmd-z['old_torque']))<1e-9
        if cfg['cut']:cmd[:]=0
        ce=float(np.max(abs(cmd-z['applied'])));assert ce<1e-9;max_command=max(max_command,ce)
        assert not z['residual'].any() and not z['action'].any()
        assert np.max(abs(body['ctrl'][1:]-np.repeat(cmd,2,axis=0)))<1e-9
        idx=np.argmin(abs(warmbody['time']-ph));assert np.array_equal(body['qpos'][0],warmbody['qpos'][idx]) and np.array_equal(body['qvel'][0],warmbody['qvel'][idx])
        assert native.shape[0]==round(duration/.0000125)+1 and np.max(abs(np.diff(native[:,0])-.0000125))<1e-12
        foot=native[:,1:7].sum(axis=1)/W;nf=native[:,8];total=native[:,1:7].sum(axis=1)+nf;share=np.divide(nf,total,out=np.zeros_like(nf),where=abs(total)>1e-9)
        acc=np.convolve(np.gradient(np.gradient(native[:,10],native[:,0]),native[:,0]),np.ones(9)/9,'same');free=((total/W<.2)&(acc<=-.5*G)).mean();balance=np.percentile(abs((W/G)*acc-(total-W))/W,95)
        assist=res['validation']['assist'];commin=native[:,10].min();bearing=np.median((native[:,1:7]>1e-9).sum(axis=1));pos=np.flatnonzero(nf>1e-9);first=float(native[pos[0],0]-ph) if len(pos) else None
        J=2*share.mean()+max(0,(assist['foot_support_mean']-foot.mean())/max(assist['foot_support_mean'],1e-9))+max(0,(assist['com_z_mean']-commin)/max(assist['com_z_mean'],1e-9))+(0 if first is None else max(0,1-first/duration))+free
        je=abs(J-res['J']);assert je<1e-9;max_J=max(max_J,je)
        checks=[native[:,7].max()==0,free==0,foot.mean()>=assist['foot_support_mean']-.10,np.percentile(foot,10)>=.5,share.mean()<=.2,commin>=assist['com_z_mean']-.10,balance<=.15,bearing>=3,commin>=.8*H0]
        assert list(map(bool,checks))==list(res['validation']['checks'].values())
        if cfg['sensory_cut']:assert not z['sensory_current_norm'].any()
        assert res['release_aux_max']==res['release_external_max']==res['warnings']==0
        positive=np.flatnonzero(native[:,7]>1e-9);first_positive=float(native[positive[0],0]-ph) if len(positive) else None;assert first_positive==res['first_positive_nonfoot_s']
        details.append(dict(folder=str(folder),filter_error=error,command_error=ce,J_error=je,gates=list(map(bool,checks)),J=float(J),first_positive_nonfoot_s=first_positive,first_signed_nonfoot_s=first,signed_positive_same=first==first_positive,foot_support_mean=float(foot.mean()),com_z_min=float(commin),passed=all(checks)))
    selected=summary['selected'];normal=next(x for x in summary['results'] if x['candidate']==selected and x['release_time_s']==1.3 and '_train_guard_' in x['label']);repeat=next(x for x in summary['results'] if x['label']=='selected_repeat')
    for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
        a=r.arrays(Path(normal['folder'])/file);b=r.arrays(Path(repeat['folder'])/file);assert all(np.array_equal(a[k],b[k]) for k in a)
    # Recompute ranking and selection from independently scored native arrays.
    scored={v['folder']:v for v in details}
    result=lambda name,ph,stage:next(v for v in summary['results'] if v['candidate']==name and v['release_time_s']==ph and stage in v['label'])
    b=scored[result('baseline',1.3,'_train_guard_')['folder']]
    rankings={}
    for name in ['baseline']+[f'g{g}_c{i}' for g in range(3) for i in range(8)]:
        gr=scored[result(name,1.3,'_train_guard_')['folder']];first=lambda v:v['first_positive_nonfoot_s'] if v['first_positive_nonfoot_s'] is not None else .2
        diffs=[gr['J']-b['J'],b['foot_support_mean']-gr['foot_support_mean'],b['com_z_min']-gr['com_z_min'],first(b)-first(gr)]
        ok=max(diffs)<=1e-9;violation=sum(max(0,v) for v in diffs);tr=[scored[result(name,ph,'_train_guard_')['folder']] for ph in r.TRAIN]
        x=np.array(r.js(O/'CONFIGS'/name/'PARAMETERS.json')['x']);pen=.05*np.mean(x*x);passes=sum(v['passed'] for v in tr);worst=max(v['J'] for v in tr);mean=np.mean([v['J'] for v in tr])
        rankings[name]=(not ok,violation if not ok else 0,-passes,worst+pen,mean)
        saved=next(v for v in summary['candidate_metrics'] if v['candidate']==name);assert saved['guard']==ok and saved['passes']==passes and abs(saved['worst_J']-worst)<1e-9 and abs(saved['mean_J']-mean)<1e-9
    dev=[]
    for name in ['baseline']+summary['selection']['finalists']:
        rr=[scored[result(name,ph,'_dev_')['folder']] for ph in r.DEV];pen=.05*np.mean(np.array(r.js(O/'CONFIGS'/name/'PARAMETERS.json')['x'])**2)
        dev.append(dict(name=name,guard=not rankings[name][0],passes=sum(v['passed'] for v in rr),worst=max(v['J'] for v in rr),mean=np.mean([v['J'] for v in rr]),pen=pen))
    bd=dev[0];eligible=[v for v in dev if v['guard'] and v['passes']>=bd['passes'] and v['worst']<=bd['worst']+1e-9 and v['mean']<=bd['mean']+1e-9]
    assert min(eligible,key=lambda v:(-v['passes'],v['worst']+v['pen'],v['mean']))['name']==summary['selected']
    # Independently regenerate complete ES transcript from logged ranks and formula.
    rng=np.random.default_rng(r.SEED);mean=np.zeros(20);var=np.full(20,.35**2)
    for gen in range(3):
        g=r.js(O/f'GENERATION_{gen}_AFTER.json');z=rng.normal(size=(4,20));proposal=mean+np.sqrt(var)*np.concatenate([z,-z]);proposal=1-np.abs((proposal+1)%4-2)
        assert np.array_equal(proposal,np.array(g['population']))
        for i,x in enumerate(proposal):assert np.array_equal(x,np.array(r.js(O/'CONFIGS'/f'g{gen}_c{i}'/'PARAMETERS.json')['x']))
        assert sorted(range(8),key=lambda i:rankings[f'g{gen}_c{i}'])==g['ranking']
        elite=proposal[np.array(g['ranking'][:4])];w=np.array([.4,.3,.2,.1]);target=w@elite;spread=w@((elite-mean)**2);mean=.5*mean+.5*target;var=np.clip(.7*var+.3*spread,.05**2,.6**2)
        assert np.array_equal(mean,np.array(g['optimizer_after']['mean'])) and np.array_equal(var,np.array(g['optimizer_after']['diagonal_variance']))
    after=r.checkhash();r.dump(O/'INPUT_HASHES_AFTER.json',after)
    r.dump(O/'VERIFICATION.json',dict(passed=True,independent_method='scipy lfilter from saved raw warm/release spikes, expanded native J/D gates; all edge expected multipliers, all physical handovers, deterministic ES regeneration',runs=len(details),configs=config_checks,max_filter_error=max_filter,max_command_error=max_command,max_J_error=max_J,exact_repeat=True,source_hashes_unchanged=True,details=details))
    print('VERIFIED',len(details),max_filter,max_command,max_J,flush=True)
if __name__=='__main__':main()
