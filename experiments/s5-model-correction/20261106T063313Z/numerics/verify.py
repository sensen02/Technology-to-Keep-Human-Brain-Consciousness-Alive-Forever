"""Independent saved-array clock/filter/native D/J verification; no simulation."""
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
    summary=r.js(O/'SUMMARY.json');ident=r.js(O/'TRANSLATOR_IDENTITIES.json');cells=np.array(ident['cells']);limits=np.array(ident['limits']);matrix=np.array(ident['old_matrix'])
    P=np.zeros((66,204))
    for i,pool in enumerate(ident['pools']):
        for c in pool:P[i,np.flatnonzero(cells==c)[0]]=1/len(pool)
    with (r.S/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
    mats=[sparse.load_npz(r.rt.b.AB/'A'/('ACTUAL_'+k+'.npz')) for k in ['We','Wi']]
    max_filter=0.;max_command=0.;max_J=0.;warm={};warmhist={};configs=[];diagnostics=[]
    legs=r.js(O/'ANATOMICAL_IDENTITIES.json')['cells']
    for folder in sorted((O/'CONFIGS').iterdir()):
        cfg=r.js(folder/'PARAMETERS.json');dt=cfg['neural_dt_s'];n=round(.001/dt);a=np.exp(-dt/.005);z=r.arrays(folder/'WARM.npz');sp=np.unpackbits(z['spikes12000_packed'],axis=1,count=12000)
        assert sp.shape==(1200*n,12000) and np.array_equal(sp[:,cells],z['spikes204'])
        filtered=lfilter([(1-a)/dt],[1,-a],sp[:,cells]@P.T,axis=0);ends=filtered[n-1::n]
        err=float(np.max(abs(ends-z['pool_rates'])));assert err<1e-9;max_filter=max(max_filter,err)
        # Independent cell filters in chunks limit temporary memory.
        for low in range(0,12000,500):
            rates=lfilter([(1-a)/dt],[1,-a],sp[:,low:low+500],axis=0)[n-1::n]
            e=float(np.max(abs(rates-z['cell_filtered_rates'][:,low:low+500])));assert e<1e-9;max_filter=max(max_filter,e)
        leg=np.stack([z['cell_filtered_rates'][:,c].mean(axis=1) for c in legs],axis=1);assert np.max(abs(leg-z['leg_rates']))<1e-9
        source=np.vstack([np.zeros(66),ends[:-1]]);assert np.max(abs(source-z['command_source_rates']))<1e-9
        raw=source@matrix.T*limits/100;assert np.max(abs(raw-z['raw_command']))<1e-9 and np.max(abs(np.clip(raw,-limits,limits)-z['clipped_command']))<1e-9
        with (folder/'WARM_CACHE.pkl').open('rb') as f:cache=pickle.load(f)
        for ph,c in cache.items():
            count=round((ph-.2)/dt);idx=round((ph-.2)/.001)-1
            assert np.max(abs(filtered[count-1]-c['rates']))<1e-9
            assert r.exact(c['integration'],physical[ph]['integration']) and r.exact(c['assist'],physical[ph]['assist']) and r.exact(vars(c['enc']),vars(physical[ph]['enc']))
            assert all(r.exact(m,getattr(c['net'],key)) for m,key in zip(mats,['We','Wi']))
            assert c['net'].steps==count and abs(c['net'].t_ms-count*dt*1000)<1e-10 and c['history']['samples']==count
            assert c['net'].delay_steps*dt*1000==2 and c['net']._ring.shape==(round(.002/dt)+1,12000)
            for j,tau in enumerate([.005,.020,.050]):
                decay=np.exp(-dt/tau);h=lfilter([(1-decay)/dt],[1,-decay],sp[:count,cells],axis=0)[-1];assert np.max(abs(h-c['history']['rates'][j]))<1e-9
            if dt==.0005:assert r.exact(vars(c['net']),vars(physical[ph]['net'])) and r.exact(c['rates'],physical[ph]['rates']) and r.exact(c['history'],physical[ph]['history'])
            warm[dt,ph]=filtered[count-1];warmhist[dt,ph]=c['history']['rates'].copy()
        stats={}
        for label,start in [('full',0),('exclude100ms',100)]:
            ss=sp[start*n:];duration=(1200-start)*.001;cell=ss.sum(axis=0)/duration;cmd=z['clipped_command'][start:];raw=z['raw_command'][start:]
            stats[label]=dict(mean_allcell_hz=float(cell.mean()),cell_rate_hz=cell,cell_rate_quantiles=np.quantile(cell,[0,.1,.5,.9,1]),motor_mean_hz=float(cell[cells].mean()),perleg_mean_hz=[float(cell[c].mean()) for c in legs],saturation=float(np.mean(abs(raw)>=limits)),command_temporal_rms=float(np.sqrt(np.mean(np.diff(cmd,axis=0)**2))),raw_temporal_rms=float(np.sqrt(np.mean(np.diff(raw,axis=0)**2))),pool_temporal_rms=float(np.sqrt(np.mean(np.diff(z['pool_rates'][start:],axis=0)**2))))
        cr=r.arrays(folder/'CONTROLLED_RELEASE.npz');cs=np.unpackbits(cr['spikes12000_packed'],axis=1,count=12000);initial=warm[dt,1.3]
        cf=lfilter([(1-a)/dt],[1,-a],cs[:,cells]@P.T,axis=0,zi=(a*initial)[None,:])[0][n-1::n]
        assert np.max(abs(cf-cr['pool_rates']))<1e-9
        source=np.vstack([initial,cf[:-1]]);raw=source@matrix.T*limits/100
        assert np.max(abs(source-cr['command_source_rates']))<1e-9 and np.max(abs(raw-cr['raw_command']))<1e-9 and np.max(abs(np.clip(raw,-limits,limits)-cr['clipped_command']))<1e-9
        for low in range(0,12000,500):
            init=z['cell_filtered_rates'][1099,low:low+500]
            rates=lfilter([(1-a)/dt],[1,-a],cs[:,low:low+500],axis=0,zi=(a*init)[None,:])[0][n-1::n]
            assert np.max(abs(rates-cr['cell_filtered_rates'][:,low:low+500]))<1e-9
        leg=np.stack([cr['cell_filtered_rates'][:,c].mean(axis=1) for c in legs],axis=1);assert np.max(abs(leg-cr['leg_rates']))<1e-9
        identity=r.js(folder/'CONTROLLED_RELEASE_IDENTITY.json');assert r.sha(identity['source'])==identity['sha256'] and identity['RH_applied_once']==.375
        if dt==.0005:
            actual=r.arrays(identity['source']);assert np.array_equal(cr['spikes12000_packed'],actual['spikes12000_packed'].reshape(-1,1500)) and np.array_equal(cr['pool_rates'],actual['rates']) and np.array_equal(cr['clipped_command'],actual['applied'])
        diagnostics.append(dict(dt_s=dt,stats=stats,controlled_release=dict(allcell_mean_hz=float(cs.mean()/dt),cell_mean_hz=cs.sum(axis=0)/.2,saturation=float(np.mean(abs(cr['raw_command'])>=limits)),temporal_rms=float(np.sqrt(np.mean(np.diff(cr['clipped_command'],axis=0)**2))))));configs.append(dict(dt_s=dt,states=len(cache),filter_error=err,physical_cache_exact=True,connectivity_exact=True))
    bodywarm=r.arrays(r.S/'DATA/true_teacher/BODY.npz');details=[]
    for res in summary['results']:
        folder=Path(res['folder']);cfg=r.js(folder/'PREREGISTRATION.json');z=r.arrays(folder/'NEURAL.npz');body=r.arrays(folder/'BODY.npz');native=r.arrays(folder/'NATIVE.npz')['rows'];ph=res['release_time_s'];dt=res['dt_s'];n=round(.001/dt);a=np.exp(-dt/.005)
        assert cfg['neural_dt_s']==dt and cfg['neural_steps_per_command']==n and cfg['history_dt_s']==dt and cfg['command_dt_s']==.001 and cfg['physics_dt_s']==.0000125
        assert z['spikes204'].shape==(200,n,204)
        allsp=np.unpackbits(z['spikes12000_packed'].reshape(-1,1500),axis=1,count=12000);assert np.array_equal(allsp[:,cells],z['spikes204'].reshape(-1,204))
        hist=[]
        for j,tau in enumerate([.005,.020,.050]):
            decay=np.exp(-dt/tau);h=lfilter([(1-decay)/dt],[1,-decay],allsp[:,cells],axis=0,zi=(decay*warmhist[dt,ph][j])[None,:])[0][n-1::n]
            hist.append(np.vstack([warmhist[dt,ph][j],h[:-1]]))
        features=(np.stack(hist,axis=1)/100).astype(np.float32).reshape(200,-1)
        assert np.max(abs(features-z['features']))<1e-6
        initial=warm[dt,ph];flat=z['spikes204'].reshape(-1,204)@P.T
        filtered=lfilter([(1-a)/dt],[1,-a],flat,axis=0,zi=(a*initial)[None,:])[0];ends=filtered[n-1::n];starts=np.vstack([initial,ends[:-1]])
        err=max(float(np.max(abs(ends-z['rates']))),float(np.max(abs(starts-z['source_rates']))));assert err<1e-9;max_filter=max(max_filter,err)
        raw=starts@matrix.T*limits/100;cmd=np.clip(raw,-limits,limits);assert np.max(abs(raw-z['raw_command']))<1e-9 and np.max(abs(cmd-z['old_torque']))<1e-9
        if cfg['cut']:cmd[:]=0
        ce=float(np.max(abs(cmd-z['applied'])));assert ce<1e-9;max_command=max(max_command,ce)
        assert np.max(abs(body['ctrl'][1:]-np.repeat(cmd,2,axis=0)))<1e-9
        idx=np.argmin(abs(bodywarm['time']-ph));assert np.array_equal(body['qpos'][0],bodywarm['qpos'][idx]) and np.array_equal(body['qvel'][0],bodywarm['qvel'][idx])
        assert native.shape[0]==16001 and np.max(abs(np.diff(native[:,0])-.0000125))<1e-12 and np.max(abs(np.diff(z['time'])-.001))<1e-11
        assert np.array_equal(z['net_steps'],round((ph-.2)/dt)+np.arange(1,201)*n)
        assert np.max(abs(z['net_t_ms']-((ph-.2)*1000+np.arange(1,201))))<1e-9
        foot=native[:,1:7].sum(axis=1)/W;nf=native[:,8];total=native[:,1:7].sum(axis=1)+nf;share=np.divide(nf,total,out=np.zeros_like(nf),where=abs(total)>1e-9)
        acc=np.convolve(np.gradient(np.gradient(native[:,10],native[:,0]),native[:,0]),np.ones(9)/9,'same');free=((total/W<.2)&(acc<=-.5*G)).mean();balance=np.percentile(abs((W/G)*acc-(total-W))/W,95)
        assist=res['validation']['assist'];commin=native[:,10].min();bearing=np.median((native[:,1:7]>1e-9).sum(axis=1));pos=np.flatnonzero(nf>1e-9);first=float(native[pos[0],0]-ph) if len(pos) else None
        J=2*share.mean()+max(0,(assist['foot_support_mean']-foot.mean())/max(assist['foot_support_mean'],1e-9))+max(0,(assist['com_z_mean']-commin)/max(assist['com_z_mean'],1e-9))+(0 if first is None else max(0,1-first/.2))+free
        je=abs(J-res['J']);assert je<1e-9;max_J=max(max_J,je)
        checks=[native[:,7].max()==0,free==0,foot.mean()>=assist['foot_support_mean']-.10,np.percentile(foot,10)>=.5,share.mean()<=.2,commin>=assist['com_z_mean']-.10,balance<=.15,bearing>=3,commin>=.8*H0]
        assert list(map(bool,checks))==list(res['validation']['checks'].values()) and all(checks)==res['validation']['passed']
        if cfg['sensory_cut']:assert not z['sensory_current'].any()
        assert res['release_aux_max']==res['release_external_max']==res['warnings']==0
        pos=np.flatnonzero(native[:,7]>1e-9);firstpos=float(native[pos[0],0]-ph) if len(pos) else None;assert firstpos==res['first_positive_nonfoot_s']
        details.append(dict(folder=str(folder),dt_s=dt,phase=ph,J=float(J),filter_error=err,command_error=ce,J_error=je,foot_mean=float(foot.mean()),com_min=float(commin),first_nonfoot=firstpos,passed=all(checks),allcell_mean_hz=float(allsp.mean()/dt),percell_hz=allsp.sum(axis=0)/.2,saturation=float(np.mean(abs(z['raw_command'])>=limits)),temporal_rms=float(np.sqrt(np.mean(np.diff(z['applied'],axis=0)**2)))))
    normal=next(x for x in summary['results'] if x['dt_s']==.000125 and x['release_time_s']==1.3 and 'paired' in x['label']);repeat=next(x for x in summary['results'] if x['label'].endswith('_repeat'))
    for file in ['BODY.npz','NATIVE.npz','NEURAL.npz']:
        a=r.arrays(Path(normal['folder'])/file);b=r.arrays(Path(repeat['folder'])/file);assert all(np.array_equal(a[k],b[k]) for k in a)
    # Independently recompute convergence primary thresholds from recorded 1ms arrays.
    independent=[]
    for comparison in [summary['main']]+([summary['supplement']] if summary['supplement'] else []):
        zs=[r.arrays(O/'CONFIGS'/r.name(dt)/'WARM.npz') for dt in comparison['dts']];passed=[]
        for key in ['pool_rates','leg_rates','raw_command','clipped_command']:
            x,y,z=[v[key][100:] for v in zs];co=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-9*np.sqrt(y.size)));fi=float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-9*np.sqrt(z.size)));ok=fi<=.15 and (fi/max(co,1e-15)<=.9 or max(co,fi)<=.02)
            saved=comparison['neural_windows']['primary_exclude_initial100ms'][key];assert abs(co-saved['coarse_difference'])<1e-12 and abs(fi-saved['fine_difference'])<1e-12 and ok==saved['passed'];passed.append(ok)
        assert all(passed)==comparison['warm_neural_passed']
        controlled_pass=[];zs=[r.arrays(O/'CONFIGS'/r.name(dt)/'CONTROLLED_RELEASE.npz') for dt in comparison['dts']]
        for key in ['pool_rates','leg_rates','raw_command','clipped_command']:
            x,y,z=[v[key][100:] for v in zs];co=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-9*np.sqrt(y.size)));fi=float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-9*np.sqrt(z.size)));ok=fi<=.15 and (fi/max(co,1e-15)<=.9 or max(co,fi)<=.02)
            saved=comparison['controlled_release_windows']['primary_exclude_initial100ms'][key];assert abs(co-saved['coarse_difference'])<1e-12 and abs(fi-saved['fine_difference'])<1e-12 and ok==saved['passed'];controlled_pass.append(ok)
        assert all(controlled_pass)==comparison['controlled_release_passed'];passed+=controlled_pass
        assert all(passed)==comparison['neural_passed']
        physical_pass=[]
        for ph in r.PHASES:
            dtb,dtc=comparison['dts'][-2:]
            b,c=[next(x for x in details if x['dt_s']==dt and x['phase']==ph and 'paired' in x['folder']) for dt in [dtb,dtc]]
            first=lambda x:.2 if x['first_nonfoot'] is None else x['first_nonfoot']
            rb,rc=[next(x for x in summary['results'] if x['folder']==row['folder']) for row in [b,c]]
            ok=abs(b['J']-c['J'])<=.2 and abs(b['foot_mean']-c['foot_mean'])<=.1 and abs(b['com_min']-c['com_min'])<=.1 and abs(first(b)-first(c))<=.02 and rb['validation']['checks']==rc['validation']['checks']
            saved=next(x for x in comparison['physical_pairs'] if x['phase']==ph);assert ok==saved['passed'];physical_pass.append(ok)
        assert all(physical_pass)==comparison['physical_passed'] and (all(passed) and all(physical_pass))==comparison['passed']
        independent.append(dict(dts=comparison['dts'],neural_passed=all(passed),physical_passed=all(physical_pass)))
    r.dump(O/'DIAGNOSTICS.json',dict(warm=diagnostics,release=details))
    r.dump(O/'VERIFICATION.json',dict(passed=True,runs=len(details),configs=configs,details=[{k:v for k,v in d.items() if k!='percell_hz'} for d in details],max_filter_error=max_filter,max_command_error=max_command,max_J_error=max_J,repeat_exact=True,clocks_exact=True,independent_neural_gates=independent,hashes_unchanged=True))
    r.dump(O/'INPUT_HASHES_AFTER.json',r.checkhash());print('VERIFIED',len(details),max_filter,max_command,max_J,flush=True)
if __name__=='__main__':main()
