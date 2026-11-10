"""Independent saved-array recomputation; no physics or neural stepping."""
import sys,json,pickle
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
O=r.O;dump=r.dump
W=24.446354755878005;G=9801.;H0=1.2072528224791905

def main():
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    model=_load_mjcf(r.rt.b.P/'MODEL_JOINT_TORQUE.xml').compile()
    assert abs(model.body_mass.sum()*abs(model.opt.gravity[2])-W)<1e-12 and abs(model.opt.gravity[2])==G and model.opt.timestep==.0000125
    summary=r.js(O/'SUMMARY.json');ident=r.js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(ident['cells']);limits=np.array(ident['limits']);matrix=np.array(ident['old_matrix'])
    P=np.zeros((66,204))
    for i,pool in enumerate(ident['pools']):
        for c in pool:P[i,np.flatnonzero(cells==c)[0]]=1/len(pool)
    teach=r.arrays(r.TEACHER)['spikes204'];warm={};details=[]
    for tau in r.TAUS:
        a=np.exp(-.0005/tau)
        filtered=lfilter([(1-a)/.0005],[1,-a],teach@P.T,axis=0)
        for ph in r.PHASES+r.FINAL:warm[tau,ph]=filtered[round((ph-.2)/.001)*2-1]
    # Read retained original cache only for physical continuity, no constructor.
    ci=r.js(r.ARCH/'DATA/CACHE_IDENTITY.json')
    with Path(ci['file']).open('rb') as f:cache=pickle.load(f)
    max_filter=0.;max_command=0.;max_J=0.;replay_checks=[]
    rob=r.js(O/'ROBUSTNESS.json');refs={}
    for sigma,seed in [(0.,0)]+[(s,k) for s in [.001,.01] for k in [1,2,3]]:
        z=r.arrays(O/f'REPLAY_sigma{sigma}_seed{seed}.npz')
        for tau in r.TAUS:
            key=f'{tau:.3f}';a=np.exp(-.0005/tau)
            rates=lfilter([(1-a)/.0005],[1,-a],z['spikes204']@P.T,axis=0)[1::2]
            err=np.max(abs(rates-z['rates_'+key]));assert err<1e-9
            raw=(rates@matrix.T)*limits/100;cmd=np.clip(raw,-limits,limits)
            assert np.max(abs(raw-z['raw_'+key]))<1e-9 and np.max(abs(cmd-z['cmd_'+key]))<1e-9
            if sigma==0:refs[tau]=cmd
            rr=next(v for v in rob if v['sigma']==sigma and v['seed']==seed and v['tau']==tau)
            windows=[]
            for start,label in [(0,'relative_rms'),(100,'relative_rms_post100ms')]:
                baseline=refs[tau][start:];den=float(np.sqrt(np.mean(baseline**2)));numer=float(np.sqrt(np.mean((cmd[start:]-baseline)**2)));ratio=numer/(den+1e-12)
                assert abs(ratio-rr[label])<1e-10
                windows.append(dict(start_command_index=start,rms_difference=numer,baseline_rms=den,ratio=ratio))
            replay_checks.append(dict(sigma=sigma,seed=seed,tau=tau,windows=windows,filter_error=float(err)))
    for res in summary['results']:
        folder=Path(res['folder']);cfg=r.js(folder/'PREREGISTRATION.json');z=r.arrays(folder/'NEURAL.npz');body=r.arrays(folder/'BODY.npz');native=r.arrays(folder/'NATIVE.npz')['rows'];tau=res['tau_s'];ph=res['release_time_s'];a=np.exp(-.0005/tau)
        initial=warm[tau,ph];flat=z['spikes204'].reshape(-1,204)@P.T
        allrates=lfilter([(1-a)/.0005],[1,-a],flat,axis=0,zi=(a*initial)[None,:])[0]
        ends=allrates[1::2];starts=np.vstack([initial,ends[:-1]])
        error=max(float(np.max(abs(ends-z['rates']))),float(np.max(abs(starts-z['source_rates']))));assert error<1e-9;max_filter=max(max_filter,error)
        cmd=np.clip((starts@matrix.T)*limits/100,-limits,limits)
        assert np.max(abs(cmd-z['old_torque']))<1e-9
        if cfg['cut']:cmd[:]=0
        ce=float(np.max(abs(cmd-z['applied'])));assert ce<1e-9;max_command=max(max_command,ce)
        assert not z['residual'].any() and not z['action'].any()
        assert np.max(abs(body['ctrl'][1:]-np.repeat(cmd,2,axis=0)))<1e-9
        oldbody=r.arrays(Path(cache[ph]['warm_folder'])/'BODY.npz');idx=np.argmin(abs(oldbody['time']-ph))
        assert np.array_equal(body['qpos'][0],oldbody['qpos'][idx]) and np.array_equal(body['qvel'][0],oldbody['qvel'][idx])
        assert native.shape[0]==16001 and np.max(abs(np.diff(native[:,0])-.0000125))<1e-12
        # Independently expand all native gates and objective (do not call project scorer).
        foot=native[:,1:7].sum(axis=1)/W;nf=native[:,8];total=native[:,1:7].sum(axis=1)+nf
        share=np.divide(nf,total,out=np.zeros_like(nf),where=abs(total)>1e-9)
        acc=np.convolve(np.gradient(np.gradient(native[:,10],native[:,0]),native[:,0]),np.ones(9)/9,'same')
        free=((total/W<.2)&(acc<=-.5*G)).mean();balance=np.percentile(abs((W/G)*acc-(total-W))/W,95)
        assist=res['validation']['assist'];commin=native[:,10].min();bearing=np.median((native[:,1:7]>1e-9).sum(axis=1))
        pos=np.flatnonzero(nf>1e-9);first=float(native[pos[0],0]-ph) if len(pos) else None
        J=2*share.mean()+max(0,(assist['foot_support_mean']-foot.mean())/max(assist['foot_support_mean'],1e-9))+max(0,(assist['com_z_mean']-commin)/max(assist['com_z_mean'],1e-9))+(0 if first is None else max(0,1-first/.2))+free
        je=abs(J-res['J']);assert je<1e-9;max_J=max(max_J,je)
        checks=[native[:,7].max()==0,free==0,foot.mean()>=assist['foot_support_mean']-.10,np.percentile(foot,10)>=.5,share.mean()<=.2,commin>=assist['com_z_mean']-.10,balance<=.15,bearing>=3,commin>=.8*H0]
        assert list(map(bool,checks))==list(res['validation']['checks'].values())
        if cfg['sensory_cut']:assert not z['sensory_current_norm'].any()
        assert res['release_aux_max']==res['release_external_max']==res['warnings']==0
        positive=np.flatnonzero(native[:,7]>1e-9);first_positive=float(native[positive[0],0]-ph) if len(positive) else None
        assert first_positive==first
        details.append(dict(folder=str(folder),filter_error=error,command_error=ce,J_error=je,gates=list(map(bool,checks)),J=float(J),first_positive_nonfoot_s=first_positive,first_signed_nonfoot_s=first))
    selected=summary['selected_tau'];normal=next(x for x in summary['results'] if x['tau_s']==selected and x['release_time_s']==1.3 and '_dev' in x['label']);repeat=next(x for x in summary['results'] if x['label']=='selected_repeat')
    for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
        a=r.arrays(Path(normal['folder'])/file);b=r.arrays(Path(repeat['folder'])/file);assert all(np.array_equal(a[k],b[k]) for k in a)
    before=r.js(O/'INPUT_HASHES_BEFORE.json');after={p:r.sha(p) for p in before};assert before==after
    dump(O/'INPUT_HASHES_AFTER.json',after)
    dump(O/'VERIFICATION.json',dict(passed=True,independent_method='scipy lfilter from saved raw spikes; expanded native force gates and J; no runtime scorer called',runs=len(details),max_filter_error=max_filter,max_command_error=max_command,max_J_error=max_J,exact_repeat=True,source_hashes_unchanged=True,details=details,replay=replay_checks))
    print('VERIFIED',len(details),max_filter,max_command,max_J,flush=True)
if __name__=='__main__':main()
