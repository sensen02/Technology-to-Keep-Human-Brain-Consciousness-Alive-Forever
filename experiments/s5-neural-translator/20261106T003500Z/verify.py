"""Independent saved-evidence verifier. No experiment driver import or episodes."""
import os,sys,json,pickle,hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype');BASE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0]=[str(BASE),str(R),str(R/'tools')]
import runtime as rt
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261106T003500Z');rt.O=O;rt.b.O=O
FIRST=O.parent/'20261105T200021Z';PREV=O.parent/'20261105T231405Z'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,d):rt.dump(O/name,d)
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def warmstates(spikes,taus):
    state=np.zeros((3,204));decay=np.exp(-.0005/np.asarray(taus))[:,None];out=[state.copy()]
    for s in spikes:
        state=decay*state+(1-decay)*s[None,:]/.0005;out.append(state.copy())
    return np.array(out)
def predict(w,x):return np.tanh(np.tanh(x@w['W0'].T)@w['W1'].T)@w['W2'].T

def main():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before==js(O/'INPUT_HASHES_AFTER_run.json')
    assert all(sha(p)==h for p,h in before.items())
    pathsets=js(O/'ARCHIVE_PATHS_BEFORE.json');assert all(paths==sorted(str(p) for p in Path(root).rglob('*') if p.is_file()) for root,paths in pathsets.items())
    cfg=js(O/'PREREGISTRATION.json');summary=js(O/'SUMMARY.json');trials=js(O/'TRIALS.json');ident=js(O/'TRANSLATOR_IDENTITIES.json')
    cells=np.array(ident['cells']);limits=np.array(ident['limits']);matrix=np.array(ident['old_matrix']);pools=[np.searchsorted(cells,p) for p in ident['pools']]
    cp=O/'DATA/CACHE.pkl';assert sha(cp)==js(O/'DATA/CACHE_IDENTITY.json')['sha256']
    with cp.open('rb') as f:cache=pickle.load(f)
    spikes=load(FIRST/'DATA/train_offset_0/TEACHER.npz')['spikes204'];warm10=warmstates(spikes,[.01,.04,.1]);warm5=warmstates(spikes,[.005,.02,.05]);decay=np.exp(-.0005/np.array([.01,.04,.1]))[:,None];decay5=np.exp(-.0005/.005)
    assert np.array_equal(load(O/'DATA/fresh_final_warmup/TEACHER.npz')['spikes204'],spikes)
    for phase,c in cache.items():
        stop=round((phase-.2)/.0005);assert np.array_equal(c['history']['rates'],warm5[stop]);assert c['history']['samples']==stop
        np.testing.assert_allclose([warm5[stop,0,p].mean() for p in pools],c['rates'],atol=1e-10,rtol=1e-12)
    import train_dynamic_support as dst
    import dynamic_support_metrics as dsm
    checked=[]
    for trial in trials:
        assert trial['status']=='completed';r=trial['result'];folder=Path(r['folder']);assert folder.is_relative_to(O)
        reg=js(folder/'PREREGISTRATION.json');assert all(reg[k]==trial[k] for k in ['cut','sensory_cut','translator_cut'])
        path=Path(reg['model_file']);assert path.is_relative_to(O) and sha(path)==reg['model_sha256'];w=load(path);assert np.array_equal(w['taus_s'],[.01,.04,.1]);gain=float(w['residual_gain']);assert gain in [0,1]
        phase=r['release_time_s'];c=cache[phase];stop=round((phase-.2)/.0005);rates=warm10[stop].copy();oldrates=warm5[stop,0].copy()
        n=load(folder/'NEURAL.npz');b=load(folder/'BODY.npz');a=load(folder/'NATIVE.npz')['rows']
        for i,pair in enumerate(n['spikes204']):
            x=(rates/100).astype(np.float32).ravel();assert np.array_equal(x,n['features'][i]);effective=np.zeros_like(x) if reg['translator_cut'] else x
            assert np.array_equal(effective,n['policy_features'][i]);action=np.zeros(42,np.float32) if gain==0 else predict(w,effective);assert np.array_equal(action,n['action'][i])
            np.testing.assert_allclose([oldrates[p].mean() for p in pools],n['source_rates'][i],atol=1e-10,rtol=1e-12)
            for sp in pair:
                rates=decay*rates+(1-decay)*sp[None,:]/.0005;oldrates=decay5*oldrates+(1-decay5)*sp/.0005
        old=np.array([np.clip(limits/100*(matrix@rate),-limits,limits) for rate in n['source_rates']]);residual=2*limits*np.tanh(n['action']);applied=np.zeros_like(old) if reg['cut'] else np.clip(old+residual,-limits,limits)
        assert np.array_equal(old,n['old_torque']) and np.array_equal(residual,n['residual']) and np.array_equal(applied,n['applied'])
        assert np.array_equal(applied,b['ctrl'][1::2]) and np.array_equal(applied,b['ctrl'][2::2])
        assert np.max(abs(b['time'][::2][:-1]-(n['time']-.001)))<1e-9
        if reg['sensory_cut']:assert not n['sensory_current_norm'].any()
        if reg['translator_cut']:assert not residual.any()
        assert r['release_aux_max']==r['release_external_max']==r['warnings']==0
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b['qpos']));assert np.max(abs(np.diff(a[:,0])-.0000125))<1e-12;assert abs(a[-1,0]-a[0,0]-.2)<1e-9
        shifted=a.copy();shifted[:,0]-=phase-1.3;J,parts=dst.objective(shifted,c['assist']);assert J==r['J'] and parts==r['parts']
        g=dsm.evaluate_native(c['assist'],a);g['checks']['D1_no_nonfoot_ground_force']=bool(np.max(a[:,7])==0);assert g['checks']==r['validation']['checks']
        warm=load(Path(c['warm_folder'])/'BODY.npz');ix=np.argmin(abs(warm['time']-phase));assert np.array_equal(warm['qpos'][ix],b['qpos'][0]) and np.array_equal(warm['qvel'][ix],b['qvel'][0])
        checked.append(dict(folder=str(folder),commands_exact=True,causal_features_exact=True,old5ms_pool_verified=True,scoring_exact=True,continuity_exact=True))
    assert len(checked)==summary['total']==len(list((O/'RUNS').iterdir()))
    # Independent reconstruction of each training/validation/protection source, masks and labels.
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    from scipy.interpolate import CubicSpline
    import mujoco
    m=_load_mjcf(rt.b.P/'MODEL_JOINT_TORQUE.xml').compile();d=mujoco.MjData(m);jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];q0=m.key_qpos[0]
    gaitfile=load(rt.b.W/'GAIT.npz');gait=CubicSpline(gaitfile['phase'],gaitfile['qpos_hinges'],axis=0,bc_type='periodic');gains=js(rt.b.W/'GAINS.json')
    dataset_checks=[];cached={};label_count=0
    for name in ['closed_train','aggregate_1','aggregate_2','closed_dev','protection_old1.300']:
        ds=load(O/'DATA'/f'{name}.npz');protection=name.startswith('protection');prior=None if protection else load(PREV/'DATA'/f'{name}.npz')
        if prior is not None:
            for k in ['old','y','source','time','body_index']:assert np.array_equal(ds[k],prior[k])
            assert np.array_equal(ds['original_x'],prior['x']) and np.array_equal(ds['original_eligible'],prior['eligible'])
        for source in np.unique(ds['source']):
            if source not in cached:
                p=Path(source);sn=load(p/'NEURAL.npz');sb=load(p/'BODY.npz');sa=load(p/'NATIVE.npz')['rows'];release=js(p/'PREREGISTRATION.json')['release_time_s'];state=warm10[round((release-.2)/.0005)].copy();features=[];prefix=[];safe=True
                for i,pair in enumerate(sn['spikes204']):
                    features.append((state/100).astype(np.float32).ravel());segment=sa[i*80:i*80+81];safe=bool(safe and not np.any(segment[:,7]>0) and np.min(segment[:,10])>=.9658022579833524);prefix.append(safe)
                    for sp in pair:state=decay*state+(1-decay)*sp[None,:]/.0005
                cached[source]=(sn,sb,np.array(features),np.array(prefix),release)
            sn,sb,features,prefix,release=cached[source];ii=np.flatnonzero(ds['source']==source);bi=ds['body_index'][ii];idx=bi//2
            assert np.array_equal(ds['x'][ii],features[idx]) and np.array_equal(ds['eligible'][ii],prefix[idx]);assert np.array_equal(ds['time'][ii],sb['time'][bi])
            if protection:
                assert release==1.3 and source==cfg['protection_result']['folder'];assert not ds['zero_residual'].any();assert np.array_equal(ds['y'],ds['old']);assert np.array_equal(np.flatnonzero(ds['eligible']),np.arange(ds['eligible'].sum()))
            else:
                # Re-evaluate every retained FIT/validation label at the actual BODY action boundary.
                for row in ii[ds['eligible'][ii]]:
                    j=int(ds['body_index'][row]);d.qpos[:]=sb['qpos'][j];d.qvel[:]=sb['qvel'][j];d.ctrl[:]=sb['ctrl'][j];mujoco.mj_forward(m,d)
                    when=float(sb['time'][j]);ramp=np.clip((when-.15)/.1,0,1);phi=(max(0,when-.15)*2*np.pi*5)%(2*np.pi);qg=gait(phi);desired=(1-ramp)*q0[qa]+ramp*qg;velocity=ramp*gait(phi,1)*2*np.pi*5+(qg-q0[qa])*(10. if .15<when<.25 else 0.)
                    tau=gains['kp']*(desired-d.qpos[qa])+np.asarray(gains['kd'])*(velocity-d.qvel[va])+d.qfrc_bias[va]-d.qfrc_passive[va];target=np.clip(np.clip(tau,-gains['clip'],gains['clip']),-limits,limits).astype(np.float32)
                    assert np.array_equal(target,ds['y'][row]);label_count+=1
        dataset_checks.append(dict(name=name,all_rows=len(ds['x']),eligible=int(ds['eligible'].sum()),features_exact=True,strict_prefix_exact=True,labels_verified=True))
    candidates=summary['candidates'];zero=candidates[0];zr=next(r for r in zero['results'] if r['release_time_s']==1.3)
    def rank(c):return (sum(r['validation']['passed'] for r in c['results']),-max(r['J'] for r in c['results']),-np.mean([r['J'] for r in c['results']]))
    for cand in candidates:
        r=next(r for r in cand['results'] if r['release_time_s']==1.3);guard=r['J']<=zr['J']+1e-9 and all((r['parts'][k] if r['parts'][k] is not None else r['duration_s']) >= (zr['parts'][k] if zr['parts'][k] is not None else zr['duration_s'])-1e-9 for k in ['foot_support_mean','com_z_min','first_nonfoot_s']);assert guard==cand['guard_pass']
    assert max([c for c in candidates if c['guard_pass']],key=rank)['name']==summary['guarded']['name'];assert max(candidates[1:],key=rank)['name']==summary['best_nonzero']['name']
    assert all(sha(p)==h for p,h in before.items());dump('VERIFICATION.json',dict(passed=True,run_count=len(checked),runs=checked,datasets=dataset_checks,offline_reference_labels_recomputed=label_count,input_hashes_verified=len(before),source_hashes_unchanged=True,selection_dev_only_verified=True,actual_cut_flags=True,full_D_passes=summary['behavioral_passes'],visual_inspection='NOT_COMPLETED_MODEL_HAS_NO_IMAGE_INPUT'))
    print('VERIFIED',len(checked),'runs',label_count,'labels',len(before),'input hashes',flush=True)
if __name__=='__main__':main()
