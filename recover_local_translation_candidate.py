"""S126/P1/S2-S4: one preregistered .99 contraction, strictly OFFLINE ONLY.
Reuses S125's literal FIR/clip/RK4 arithmetic, never its main or optimizer.
No simulator, neural runner, physical trial, promotion or checkpoint write.
"""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
from train_local_translation_candidate import (BASE, REF, S124, FORMAL, BINS,
    sha, js, dump, save, archive, integrate, gain_bias)

ROOT = Path(__file__).resolve().parent
S125 = BASE/'s125-local-translation-candidate/20261024T080500Z'
OUT = BASE/'s126-fixed-contraction/20261024T081500Z'


def main(out):
    assert not out.exists(), 'Fresh output required; no retry/overwrite'
    assert 'mujoco' not in sys.modules
    oldpre = js(S125/'PREREG.json')
    commitments = oldpre['input_sha256'].copy()
    assert commitments == js(S125/'INPUT_HASHES.json')['before']
    assert commitments == js(S125/'INPUT_HASHES.json')['after']
    for p,h in commitments.items():
        assert sha(p) == h, 'Historical input commitment mismatch: '+p
    for p,h in js(S125/'OUTPUT_HASHES.json').items():
        assert sha(p) == h, 'S125 output commitment mismatch: '+p
    inputs = {Path(p) for p in commitments}
    inputs.update(p for p in S125.rglob('*') if p.is_file())
    inputs.add(Path(__file__).resolve())
    before = {str(p):sha(p) for p in sorted(inputs)}
    # This registration precedes even loading candidate arrays or computing delta.
    pre = dict(task='S126/P1/S2-S4 OFFLINE ONLY',attempt=1,max_attempts=1,
        contraction_factor=.99,formula='delta_new = 0.99 * S125.delta; Wnew = S125.reference_weights + delta_new',
        solver_invocations=0,rescale_retries=0,reference=str(REF),formal_constraints=str(FORMAL),
        s125_directory=str(S125),s125_prereg=str(S125/'PREREG.json'),
        s125_chain=str(S125/'LOCAL_CHAIN.npz'),s125_csv=str(S125/'ALL_WEIGHT_DELTAS.csv'),
        bounds_and_tolerances='S125 unchanged; no relaxed gate; actual nonlinear recomputation',
        trust=oldpre['trust'],allowed_pools=oldpre['allowed_pools'],
        allowed_positive_original_entries=oldpre['allowed_positive_original_entries'],
        frozen_pool36=True,zero_mask_preserved=True,branch=oldpre['branch'],
        window_ms=oldpre['window_ms'],window_convention=oldpre['window_convention'],
        tolerances=dict(activation=1e-12,FD_absolute=1e-7,FD_scaled_relative=1e-5,
            trust_residual=1e-9,weight_l2=1e-12,weight_box=1e-12,branch_interior_margin=1e-10),
        acceptance='all166x90 actual ctrl <=.005+1e-9; all121 DOF6/7 actual delta <=.005+1e-9; unchanged S125 linear trust; L2<=.002+1e-12 and component box .002/sqrt(15)+1e-12; pools3/15/38 only; pool36 exact; zero mask; [0,4]; original clip branch and interior1e-10; baseline activation<=1e-12; S125 chain FD gates; mean_window actual DOF8 delta < -1e-10; iff all pass save candidate',
        derivative_semantics='LOCAL_CHAIN derivatives remain analytic at reference, NOT candidate',
        activation=oldpre['activation'],physical_admission=oldpre['physical_admission'],
        omissions=oldpre['omissions'],model_load=0,forward=0,step=0,neural_ticks=0,
        physical_trial=0,checkpoint_replacement=False,promotion=False,input_sha256=before)
    out.mkdir(parents=True)
    dump(out/'PREREG.json',pre)
    prereg_hash = sha(out/'PREREG.json')
    old = archive(S125/'LOCAL_CHAIN.npz')
    W = old['reference_weights']; olddelta = old['delta']
    assert W.shape == olddelta.shape == (66,5)
    with (S125/'ALL_WEIGHT_DELTAS.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 330
    seen = set()
    for r in rows:
        p,b = int(r['pool']),int(r['bin']); assert (p,b) not in seen
        seen.add((p,b))
        assert float(r['original']) == W[p,b]
        assert float(r['delta']) == olddelta[p,b]
        assert float(r['candidate']) == W[p,b]+olddelta[p,b]
    assert seen == {(p,b) for p in range(66) for b in range(5)}
    dump(out/'S125_EXACT_CROSSCHECK.json',dict(all330_exact=True,columns=['original','delta','candidate'],
        chain_sha256=sha(S125/'LOCAL_CHAIN.npz'),csv_sha256=sha(S125/'ALL_WEIGHT_DELTAS.csv')))
    delta = .99*olddelta
    candidate = W+delta  # Sole candidate construction; never clipped or rescaled.
    z = archive(REF/'TRAJECTORY.npz'); ini = archive(REF/'INITIALIZATION.npz')
    assert np.array_equal(W,z['weights']) and np.array_equal(W,ini['weights'])
    pool = z['actuator_pool']; mapped = pool>=0
    source = js(REF.parent/'PREREG.json')['source_hashes']
    manifests = [p for p in source if p.endswith('/MANIFEST.json') and 's4-endpoints' in p]
    assert len(manifests)==1
    groups = [tuple(sorted(r['local_indices'])) for r in js(manifests[0])['rows']]
    pools = sorted(set(g for g in groups if g)); lookup = {g:i for i,g in enumerate(pools)}
    assert np.array_equal(pool,[lookup.get(g,-1) for g in groups])
    hist = np.zeros((80,66)); features=[]
    for sp in np.concatenate((ini['spikes'],z['spikes'])):
        activity = np.array([sp[list(g)].mean()/.0005/100 for g in pools])
        hist = np.concatenate((activity[None],hist))[:80]
        features.append(np.stack([hist[lo:hi].mean(0) for lo,hi in BINS],axis=1))
    features = np.array(features)
    ferr = float(np.max(np.abs(features-np.concatenate((ini['features'],z['features'])))))
    def controls(w):
        raw = np.sum(features*w,axis=2); c = np.zeros((166,90))
        c[:,mapped] = np.clip(raw[:,pool[mapped]],0,1)
        return raw,c
    raw,c = controls(W); newraw,newc = controls(candidate)
    ctrlerr = float(max(np.max(np.abs(c[:160]-ini['decoded'])),np.max(np.abs(c[160:]-z['applied']))))
    assert not z['ctrl'][0].any() and not ini['body_act'].any()
    for i in range(1,121): np.testing.assert_array_equal(z['ctrl'][i],c[160+(i-1)//20])
    resolution = js(S124/'ACTUATOR_XML_RESOLUTION.json')
    params={}
    for key,m in [('dynprm',10),('gainprm',10),('biasprm',10),('lengthrange',2)]:
        a=np.zeros((90,m))
        for i,e in enumerate(resolution['all_actuators']):
            v=list(map(float,e['resolved'][key].split())); a[i,:len(v)]=v
        params[key]=a
    assert np.all(params['dynprm'][:,:3]==np.array([.0005,.0004,0]))
    act,_ = integrate(c[160:],params['dynprm'],ini['body_act'])
    anew,_ = integrate(newc[160:],params['dynprm'],ini['body_act'])
    aerr = float(np.max(np.abs(act-z['act'])))
    G=np.zeros((121,90)); B=G.copy()
    for t in range(121):
        for a in range(90):
            G[t,a],B[t,a],*_=gain_bias(z['actuator_length'][t,a],z['actuator_velocity'][t,a],params['lengthrange'][a],params['gainprm'][a],params['biasprm'][a])
    M=z['actuator_moment'][:,:,6:9]
    for k,v in [('reference_act',act),('gain',G),('bias',B),('moment678',M)]:
        np.testing.assert_array_equal(old[k],v)
    force=G*act+B; fnew=G*anew+B
    forceerr=float(np.max(np.abs(force-z['actuator_force'])))
    baseline=np.einsum('ta,tad->td',force,M)
    actual=np.einsum('ta,tad->td',fnew,M)
    change=actual-baseline
    # Historical proposal also recomputed nonlinearly; never scale its output.
    _,oldc=controls(W+olddelta)
    oldact,_=integrate(oldc[160:],params['dynprm'],ini['body_act'])
    np.testing.assert_array_equal(oldact,old['candidate_act'])
    oldchange=np.einsum('ta,tad->td',G*oldact+B,M)-baseline
    indices=old['weight_indices']; dv=np.array([delta[p,b] for p,b in indices])
    linear_change=np.einsum('tdj,j->td',old['torque_derivative'],dv)
    linear_ctrl=np.einsum('taj,j->ta',old['command_derivative'],dv)
    branch=np.where(raw<=0,newraw<=0,np.where(raw>=1,newraw>=1+1e-10,(newraw>=1e-10)&(newraw<=1-1e-10)))
    mask=np.zeros_like(W,dtype=bool)
    for p,b in oldpre['allowed_positive_original_entries']: mask[p,b]=True
    assert oldpre['allowed_pools']==[3,15,38] and int(mask.sum())==15
    window=np.arange(77,105)
    assert np.all(z['time'][window]*1000>=1.925-1e-12) and np.all(z['time'][window]*1000<2.625)
    ctrlmax=float(np.max(np.abs(newc-c))); d67=float(np.max(np.abs(change[:,:2])))
    decrease=float(change[window,2].mean())
    with (S125/'CHAIN_FINITE_DIFFERENCES.csv').open() as f: fd=list(csv.DictReader(f))
    fdpass=len(fd)==45 and all(float(r['max_abs_error'])<=1e-7 and float(r['scaled_relative_error'])<=1e-5 and r['passed']=='True' for r in fd)
    checks=dict(S125_all330_exact=True,FIR_reproduction=ferr<1e-12,reference_ctrl=ctrlerr<=1e-12,
        reference_activation=aerr<=1e-12,reference_force=forceerr<1e-10,S125_FD=fdpass,
        S125_solver_success=bool(js(S125/'TRUST_CONSTRAINTS.json')['solver_success']),
        finite_weights=bool(np.isfinite(candidate).all()),weight_range=bool(np.all((candidate>=0)&(candidate<=4))),
        mask=bool(np.all(delta[~mask]==0) and np.array_equal(candidate[~mask],W[~mask])),
        pool36_exact=bool(np.array_equal(candidate[36],W[36])),zero_mask=bool(np.array_equal(candidate==0,W==0)),
        L2=bool(np.linalg.norm(delta)<=.002+1e-12),box=bool(np.max(np.abs(delta))<=.002/np.sqrt(15)+1e-12),
        all166_commands=ctrlmax<=.005+1e-9,all166_raw_branches=bool(branch.all()),
        all121_actual_DOF67=d67<=.005+1e-9,linear_control=bool(np.max(np.abs(linear_ctrl))<=.005+1e-9),
        linear_DOF67=bool(np.max(np.abs(linear_change[:,:2]))<=.005+1e-9),mean_window_DOF8=decrease<-1e-10)
    admitted=all(checks.values())
    peaks=[]
    for label,v in [('S125_proposal',oldchange),('S126_fixed_099',change)]:
        for j,d in enumerate((6,7,8)):
            t=int(np.argmax(np.abs(v[:,j])))
            peaks.append(dict(proposal=label,dof=d,sample=t,time_ms=float(z['time'][t]*1000),
                signed_value=float(v[t,j]),absmax=float(abs(v[t,j])),raw_cap=.005,
                raw_margin=float(.005-abs(v[t,j])),gate_applies=d in (6,7)))
    save(out/'PEAK_TORQUE_MARGINS.csv',peaks)
    save(out/'ALL_WEIGHT_DELTAS.csv',[dict(pool=p,bin=b,original=repr(float(W[p,b])),delta=repr(float(delta[p,b])),candidate=repr(float(candidate[p,b])),allowed=bool(mask[p,b]),changed=bool(delta[p,b]!=0)) for p in range(66) for b in range(5)])
    save(out/'ALL166_COMMAND_SERIES.csv',[dict(tick=k,actuator_id=a,pool=int(pool[a]),reference=float(c[k,a]),candidate=float(newc[k,a]),delta=float(newc[k,a]-c[k,a])) for k in range(166) for a in range(90)])
    save(out/'ACTIVATION_RECONSTRUCTION.csv',[dict(sample=t,actuator_id=a,saved=float(z['act'][t,a]),reference=float(act[t,a]),candidate=float(anew[t,a]),error=float(act[t,a]-z['act'][t,a])) for t in range(121) for a in range(90)])
    save(out/'AGGREGATE_TORQUE_SERIES.csv',[dict(sample=t,time_ms=float(z['time'][t]*1000),in_objective_window=bool(t in window),**{f'{k}_DOF{d}':float(v[t,d-6]) for k,v in [('reference',baseline),('S125_delta',oldchange),('S126_delta',change),('S126_linear_delta',linear_change)] for d in (6,7,8)}) for t in range(121)])
    copied={k:v.copy() for k,v in old.items()}; copied['delta']=delta; copied['candidate_act']=anew
    np.savez_compressed(out/'LOCAL_CHAIN.npz',**copied)
    if admitted:
        np.savez_compressed(out/'CANDIDATE_WEIGHTS.npz',weights=candidate,reference_weights=W,delta=delta)
    summary=dict(task='S126/P1/S2-S4 OFFLINE ONLY',checks=checks,offline_candidate_saved=admitted,
        contraction_factor=.99,attempts=1,solver_invocations=0,rescale_retries=0,
        FIR_max_abs_error=ferr,reference_control_max_abs_error=ctrlerr,activation_max_abs_error=aerr,
        control_max_abs_change_all166=ctrlmax,DOF67_max_abs_nonlinear_change=d67,
        DOF67_raw_margin=.005-d67,mean_window_DOF8_change=decrease,
        weight_l2=float(np.linalg.norm(delta)),weight_max_abs=float(np.max(np.abs(delta))),
        linear_nonlinear_max_abs_discrepancy=float(np.max(np.abs(change-linear_change))),
        peaks=peaks,derivatives='unchanged S125 baseline analytic derivatives',prereg_sha256=prereg_hash,
        physical_admission=False,training_acceptance=False,promotion=False,checkpoint_replacement=False,
        S119_physical_constraints='NOT TESTED / NOT SATISFIED',physical_units='NOT_ESTABLISHED',
        model_load=0,forward=0,step=0,neural_ticks=0,physical_trial=0,input_files=len(before))
    dump(out/'TRUST_CONSTRAINTS.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    for j,ax in enumerate(axes[:3]):
        ax.plot(z['time']*1000,oldchange[:,j],label='S125 rejected proposal: actual nonlinear')
        ax.plot(z['time']*1000,change[:,j],'--',label='S126 fixed .99: actual nonlinear')
        if j<2:
            ax.axhline(.005,color='red',linestyle=':'); ax.axhline(-.005,color='red',linestyle=':')
        ax.set_ylabel(f'delta torque DOF{j+6}'); ax.legend(fontsize=8)
        ax.axvspan(1.925,2.625,color='gray',alpha=.12)
    axes[3].plot(z['time']*1000,.005-np.max(np.abs(oldchange[:,:2]),axis=1),label='S125 raw .005 cap margin')
    axes[3].plot(z['time']*1000,.005-np.max(np.abs(change[:,:2]),axis=1),'--',label='S126 raw .005 cap margin')
    axes[3].axhline(0,color='red',linestyle=':'); axes[3].legend(fontsize=8)
    axes[3].set_ylabel('.005 - max |delta torque67|'); axes[3].set_xlabel('release-relative time (ms)')
    fig.suptitle('S126 OFFLINE ONLY: fixed gain/moment/body/neural history; NO physical admission')
    fig.tight_layout(); fig.savefig(out/'FIXED_CONTRACTION_COMPARISON.png',dpi=150); plt.close(fig)
    # Source snapshot has the exact preregistered hash and cannot modify old evidence.
    (out/'RECOVERY_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'SOURCE_MANIFEST.json',dict(source=str(Path(__file__).resolve()),sha256=sha(__file__),
        snapshot=str(out/'RECOVERY_SOURCE.py'),snapshot_sha256=sha(out/'RECOVERY_SOURCE.py'),
        reused_functions=['train_local_translation_candidate.integrate','train_local_translation_candidate.rhs','audit_muscle_reconstructability.gain_bias'],
        never_called=['train_local_translation_candidate.main','scipy.optimize.minimize']))
    after={str(p):sha(p) for p in sorted(inputs)}
    assert before==after and sha(out/'PREREG.json')==prereg_hash
    assert 'mujoco' not in sys.modules
    dump(out/'INPUT_HASHES.json',dict(before=before,after=after,all_unchanged=True))
    summary['all_inputs_unchanged']=True
    dump(out/'SUMMARY.json',summary)
    (out/'REPORT.zh-CN.md').write_text('# S126 固定收缩离线候选（P1/S2-S4）\n\n'
        '先写PREREG，固定唯一尝试delta_new=.99*delta_S125，Wnew=Wref+delta_new。逐项330行CSV与NPZ精确核对；未求解、重缩放或重试。\n\n'
        f'实际重算166×90裁剪命令、实际6命令保持20步/25us RK4的121×90activation，冻结全部90 gain/bias与moment；非缩放旧力矩。基线activation误差 {aerr:.17g}。\n\n'
        f'全部门：{checks}。候选保存：{admitted}。全121 DOF6/7最大实际变化 {d67:.17g}；原始.005界余量 {.005-d67:.17g}。目标窗DOF8平均变化 {decrease:.17g} native。\n\n'
        'LOCAL_CHAIN原参考解析导数不变；只delta及candidate_act换新，供独立父核验。OFFLINE ONLY，不接纳S119物理条件，不提升活动权重/检查点；0模型/forward/step/神经/物理。身体/接触/感觉反馈未重算，不能称预热、站立或行走成功。绝对物理单位NOT_ESTABLISHED。全部输入前后哈希相同，旧目录零写入。\n')
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(__import__('json').dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    main(parser.parse_args().out)
