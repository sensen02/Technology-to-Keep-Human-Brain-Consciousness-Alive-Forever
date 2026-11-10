"""Independent saved-array physics scoring, warm filters, selection and optimizer replay."""
import sys,pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.signal import lfilter
import run as r
from optimizers import CLASSES
O=r.O;W=24.446354755878005;G=9801.;H0=1.2072528224791905

def main():
    summary=r.js(O/'SUMMARY.json');ident=r.js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(ident['cells']);limits=np.array(ident['limits']);matrix=np.array(ident['old_matrix']);groups=r.arrays(r.OLD/'GRAPH_GROUPS.npz')['groups'];P=np.zeros((66,204))
    for i,pool in enumerate(ident['pools']):
        for c in pool:P[i,np.flatnonzero(cells==c)[0]]=1/len(pool)
    with (r.OLD/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
    mats=[sparse.load_npz(r.rt.b.AB/'A'/('ACTUAL_'+k+'.npz')).tocsr() for k in ['We','Wi']]
    warm={};configs=[];maxfilter=0.;maxcmd=0.;details=[]
    for folder in sorted((O/'CONFIGS').iterdir()):
        cfg=r.js(folder/'PARAMETERS.json');x=np.array(cfg['x']);assert x.shape==(20,) and np.max(abs(x))<=1;pert=cfg['perturbation'];rng=np.random.default_rng(pert['seed']) if pert else None
        for i,m in enumerate(mats):
            n=sparse.load_npz(folder/('We.npz' if i==0 else 'Wi.npz'));row=np.repeat(np.arange(12000),np.diff(m.indptr));factor=1+.1*x[groups[row]+10*i]
            if pert:factor=np.clip(factor*np.exp(rng.normal(0,pert['sigma'],m.nnz)),.9,1.1)
            assert np.array_equal(n.indptr,m.indptr) and np.array_equal(n.indices,m.indices) and np.array_equal(n.data,m.data*factor)
            assert np.all(n.data>0) and (n.data/m.data).min()>=.9-1e-14 and (n.data/m.data).max()<=1.1+1e-14
        spikes=r.arrays(folder/'WARM_SPIKES.npz')['spikes204'];a=np.exp(-.0005/.005);filtered=lfilter([(1-a)/.0005],[1,-a],spikes@P.T,axis=0)
        with (folder/'WARM_CACHE.pkl').open('rb') as f:cache=pickle.load(f)
        for ph,c in cache.items():
            count=round((ph-.2)*2000);err=float(np.max(abs(filtered[count-1]-c['rates'])));assert err<1e-9;maxfilter=max(maxfilter,err);warm[folder.name,ph]=filtered[count-1]
            assert r.s.exact(c['integration'],physical[ph]['integration'])
            for k in c:
                if k not in ['net','rates','history']:
                    a0,b0=c[k],physical[ph][k]
                    if hasattr(a0,'__dict__'):assert r.s.exact(vars(a0),vars(b0))
                    else:assert r.s.exact(a0,b0)
            assert c['assist']==physical[ph]['assist']
            for i,k in enumerate(['We','Wi']):assert np.array_equal(getattr(c['net'],k).data,sparse.load_npz(folder/(k+'.npz')).data)
            for j,tau in enumerate([.005,.020,.050]):
                decay=np.exp(-.0005/tau);history=lfilter([(1-decay)/.0005],[1,-decay],spikes[:count],axis=0)[-1];assert np.max(abs(history-c['history']['rates'][j]))<1e-9
            assert c['net'].steps==count and c['history']['samples']==count
            if folder.name=='baseline':assert r.s.exact(vars(c['net']),vars(physical[ph]['net'])) and r.s.exact(c['history'],physical[ph]['history'])
        configs.append(dict(candidate=folder.name,edges_exact=True,states=len(cache),causal_history=True))
    warmbody=r.arrays(r.OLD/'DATA/true_teacher/BODY.npz')
    for res in summary['results']:
        folder=Path(res['folder']);cfg=r.js(folder/'PREREGISTRATION.json');z=r.arrays(folder/'NEURAL.npz');body=r.arrays(folder/'BODY.npz');native=r.arrays(folder/'NATIVE.npz')['rows'];ph=res['release_time_s'];duration=res['duration_s'];name=res['candidate'];a=np.exp(-.0005/.005)
        initial=warm[name,ph];flat=z['spikes204'].reshape(-1,204)@P.T;rates=lfilter([(1-a)/.0005],[1,-a],flat,axis=0,zi=(a*initial)[None,:])[0];ends=rates[1::2];starts=np.vstack([initial,ends[:-1]])
        error=max(float(np.max(abs(ends-z['rates']))),float(np.max(abs(starts-z['source_rates']))));assert error<1e-9;maxfilter=max(maxfilter,error)
        cmd=np.clip((starts@matrix.T)*limits/100,-limits,limits);assert np.max(abs(cmd-z['old_torque']))<1e-9
        if cfg['cut']:cmd[:]=0
        ce=float(np.max(abs(cmd-z['applied'])));assert ce<1e-9;maxcmd=max(maxcmd,ce);assert not z['residual'].any() and not z['action'].any()
        assert np.max(abs(body['ctrl'][1:]-np.repeat(cmd,2,axis=0)))<1e-9
        idx=np.argmin(abs(warmbody['time']-ph));assert np.array_equal(body['qpos'][0],warmbody['qpos'][idx]) and np.array_equal(body['qvel'][0],warmbody['qvel'][idx])
        assert len(native)==round(duration/.0000125)+1 and np.max(abs(np.diff(native[:,0])-.0000125))<1e-12
        foot=native[:,1:7].sum(axis=1)/W;nf=native[:,8];total=native[:,1:7].sum(axis=1)+nf;share=np.divide(nf,total,out=np.zeros_like(nf),where=abs(total)>1e-9)
        acc=np.convolve(np.gradient(np.gradient(native[:,10],native[:,0]),native[:,0]),np.ones(9)/9,'same');free=((total/W<.2)&(acc<=-.5*G)).mean();balance=np.percentile(abs((W/G)*acc-(total-W))/W,95)
        assist=res['validation']['assist'];commin=native[:,10].min();bearing=np.median((native[:,1:7]>1e-9).sum(axis=1));pos=np.flatnonzero(nf>1e-9);first=float(native[pos[0],0]-ph) if len(pos) else None
        J=2*share.mean()+max(0,(assist['foot_support_mean']-foot.mean())/max(assist['foot_support_mean'],1e-9))+max(0,(assist['com_z_mean']-commin)/max(assist['com_z_mean'],1e-9))+(0 if first is None else max(0,1-first/duration))+free
        assert abs(J-res['J'])<1e-9
        checks=[native[:,7].max()==0,free==0,foot.mean()>=assist['foot_support_mean']-.10,np.percentile(foot,10)>=.5,share.mean()<=.2,commin>=assist['com_z_mean']-.10,balance<=.15,bearing>=3,commin>=.8*H0]
        assert list(map(bool,checks))==list(res['validation']['checks'].values())
        if cfg['sensory_cut']:assert not z['sensory_current_norm'].any()
        assert res['release_aux_max']==res['release_external_max']==res['warnings']==0
        positive=np.flatnonzero(native[:,7]>1e-9);firstpos=float(native[positive[0],0]-ph) if len(positive) else None;assert firstpos==res['first_positive_nonfoot_s']
        details.append(dict(folder=str(folder),J=float(J),first_positive_nonfoot_s=firstpos,foot_support_mean=float(foot.mean()),com_z_min=float(commin),passed=all(checks),gates=list(map(bool,checks))))
    scored={d['folder']:d for d in details};lookup=lambda name,ph,stage:scored[next(v['folder'] for v in summary['results'] if v['candidate']==name and v['release_time_s']==ph and stage in v['label'])];base=lookup('baseline',1.3,'_train_guard_');keys={};first=lambda v:.2 if v['first_positive_nonfoot_s'] is None else v['first_positive_nonfoot_s']
    for cm in summary['candidate_metrics']:
        name=cm['candidate'];gr=lookup(name,1.3,'_train_guard_');diffs=[gr['J']-base['J'],base['foot_support_mean']-gr['foot_support_mean'],base['com_z_min']-gr['com_z_min'],first(base)-first(gr)];ok=max(diffs)<=1e-9;violation=sum(max(0,v) for v in diffs);tr=[lookup(name,ph,'_train_guard_') for ph in r.TRAIN];x=np.array(r.js(O/'CONFIGS'/name/'PARAMETERS.json')['x']);pen=.05*np.mean(x*x);passes=sum(v['passed'] for v in tr);worst=max(v['J'] for v in tr);mean=float(np.mean([v['J'] for v in tr]));keys[name]=(not ok,violation if not ok else 0,-passes,worst+pen,mean)
        assert cm['guard']==ok and abs(cm['guard_violation']-violation)<1e-9 and cm['passes']==passes and abs(cm['worst_J']-worst)<1e-9 and abs(cm['mean_J']-mean)<1e-9
    dev=[]
    for name in ['baseline']+summary['selection']['finalists']:
        rr=[lookup(name,ph,'_dev_') for ph in r.DEV];pen=.05*np.mean(np.array(r.js(O/'CONFIGS'/name/'PARAMETERS.json')['x'])**2);dev.append(dict(name=name,guard=not keys[name][0],passes=sum(v['passed'] for v in rr),worst=max(v['J'] for v in rr),mean=float(np.mean([v['J'] for v in rr])),pen=pen))
    bd=dev[0];eligible=[v for v in dev if v['guard'] and v['passes']>=bd['passes'] and v['worst']<=bd['worst']+1e-9 and v['mean']<=bd['mean']+1e-9];assert min(eligible,key=lambda v:(-v['passes'],v['worst']+v['pen'],v['mean']))['name']==summary['selected']
    for arm,cls in CLASSES.items():
        opt=cls()
        for batch in range(3):
            log=r.js(O/f'{arm}_BATCH_{batch}_AFTER.json');pop=opt.ask();assert np.array_equal(pop,np.array(log['population']));kk=[keys[f'{arm}_g{batch}_c{i}'] for i in range(8)];assert sorted(range(8),key=lambda i:kk[i])==log['ranking']
            for i,x in enumerate(pop):assert np.array_equal(x,np.array(r.js(O/'CONFIGS'/f'{arm}_g{batch}_c{i}'/'PARAMETERS.json')['x']))
            opt.tell(pop,kk)
            for k,v in opt.state().items():
                if isinstance(v,np.ndarray):assert np.array_equal(v,np.asarray(log['after'][k]))
                elif k not in ['keys']:assert v==log['after'][k]
    selected=summary['selected'];normal=next(v for v in summary['results'] if v['candidate']==selected and v['release_time_s']==1.3 and '_train_guard_' in v['label']);repeat=next(v for v in summary['results'] if v['label']=='selected_repeat')
    for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
        a=r.arrays(Path(normal['folder'])/file);b=r.arrays(Path(repeat['folder'])/file);assert all(np.array_equal(a[k],b[k]) for k in a)
    r.checkhash();r.dump(O/'VERIFICATION.json',dict(passed=True,method='independent saved spikes scipy lfilter; expanded native J/D; edge formulas; warm physical continuity; independent metric/selection; exact replay optimizer transcripts',runs=len(details),configs=configs,max_filter_error=maxfilter,max_command_error=maxcmd,exact_repeat=True,source_hashes_unchanged=True,details=details));print('VERIFIED',len(details),flush=True)
if __name__=='__main__':main()
