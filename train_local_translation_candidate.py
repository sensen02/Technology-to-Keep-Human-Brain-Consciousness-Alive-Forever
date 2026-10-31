"""S125 bounded OFFLINE local FIR/control/activation/frozen-body candidate.
No simulator, neural runner, model construction or physical trial is imported.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from audit_muscle_reconstructability import gain_bias

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/sensen/Desktop/engine修复验收')
REF=BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms'
S124=BASE/'s124-muscle-reconstructability/20261024T075500Z'
FORMAL=BASE/'s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json'
OUT=BASE/'s125-local-translation-candidate/20261024T080500Z'
BINS=((0,5),(5,10),(10,20),(20,40),(40,80))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def js(p): return json.loads(Path(p).read_text())
def dump(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def archive(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def rhs(a,u,prm):
    ac=np.clip(a,0,1); uc=np.clip(u,0,1); v=.5+1.5*ac; d=uc-a
    rising=d>0; tau=np.where(rising,prm[:,0]*v,prm[:,1]/v)
    dtau=np.where(rising,1.5*prm[:,0],-1.5*prm[:,1]/v**2)*((a>0)&(a<1))
    return d/tau, -1/tau-d*dtau/tau**2, 1/tau

def integrate(ctrl,prm,initial,du=None):
    """Literal classical RK4 and its analytic tangent, command held per20 steps."""
    a=initial.copy(); states=[a.copy()]; n=0 if du is None else du.shape[-1]
    tangent=np.zeros((90,n)); tangents=[tangent.copy()]; h=.000025
    for k in range(120):
        u=ctrl[k//20]; U=np.zeros((90,n)) if du is None else du[k//20]
        k1,fa,fu=rhs(a,u,prm); j1=fa[:,None]*tangent+fu[:,None]*U
        k2,fa,fu=rhs(a+h*k1/2,u,prm); j2=fa[:,None]*(tangent+h*j1/2)+fu[:,None]*U
        k3,fa,fu=rhs(a+h*k2/2,u,prm); j3=fa[:,None]*(tangent+h*j2/2)+fu[:,None]*U
        k4,fa,fu=rhs(a+h*k3,u,prm); j4=fa[:,None]*(tangent+h*j3)+fu[:,None]*U
        # Keep engine RK4 summation order as closely as possible; roundoff is measured.
        a=a+h*(k1/6+k2/3+k3/3+k4/6)
        tangent=tangent+h*(j1/6+j2/3+j3/3+j4/6)
        states.append(a.copy()); tangents.append(tangent.copy())
    return np.array(states),np.array(tangents)

def main(out):
    assert not out.exists(), 'Fresh output required'
    assert 'mujoco' not in sys.modules
    inputs={Path(__file__).resolve(),ROOT/'audit_muscle_reconstructability.py',ROOT/'run_extended_initialization_probe.py',FORMAL}
    inputs.update(p for r in (REF.parent,S124,FORMAL.parent) for p in r.rglob('*') if p.is_file())
    commitments=js(REF.parent/'INPUT_HASHES.json')
    commitments.update(js(REF/'RESULT.json')['hashes'])
    for p,h in js(S124/'INPUT_HASHES.json')['after'].items(): commitments[p]=h
    for p,h in commitments.items():
        assert sha(p)==h, 'Input commitment mismatch '+p
        inputs.add(Path(p))
    before={str(p):sha(p) for p in sorted(inputs)}
    z=archive(REF/'TRAJECTORY.npz'); ini=archive(REF/'INITIALIZATION.npz')
    W=z['weights']; pool=z['actuator_pool']; names=js(REF/'MOMENT_METADATA.json')['actuator_names']
    assert np.array_equal(W,ini['weights']) and W.shape==(66,5)
    allowed=sorted(set(int(p) for p in pool[:6] if p>=0 and p!=36))
    indices=[(p,b) for p in allowed for b in range(5) if W[p,b]>0]
    n=len(indices); scale=.002/np.sqrt(n)
    affected=np.flatnonzero(np.isin(pool,allowed))
    # One deterministic QP, no configuration scan; box implies ||delta W||2 <= .002.
    prereg=dict(task='S125/P1/S2-S4 OFFLINE ONLY',reference=str(REF),formal_constraints=str(FORMAL),
        objective='min mean_window(delta torque DOF8)/0.005 + 0.05*sum(x*x), deltaW=scale*x',
        window_ms=[1.925,2.625],window_convention='1.925 <= release-relative t_ms < 2.625; robust integer step77..104',
        solver='one scipy SLSQP invocation of strictly convex QP; x0=0, ftol=1e-12,maxiter=200',
        unique_global_solution='positive definite 0.1 I Hessian; no restart/backtrack/configuration scan',
        allowed_pools=allowed,allowed_positive_original_entries=indices,frozen_pool36=True,zero_mask_preserved=True,
        trust=dict(max_abs_weight=.002,max_l2_weight=.002,conservative_individual_box=scale,max_abs_ctrl=.005,
                   max_abs_dof6_or7_torque_change=.005,weights_range=[0,4]),
        linearization='analytic FIR+clip+ODE RK4 tangent; all90 frozen gain/moment; all affected shared mappings',
        branch='constrain every166 tick raw drive to same clip branch (interior margin1e-10)',
        acceptance='activation reproduction <=1e-12, all chain FD abs<=1e-7 and scaled rel<=1e-5; solver success; linear+nonlinear trust residual<=1e-9; mean actual frozen-body DOF8 delta<-1e-10; else no candidate file',
        activation='release body_act zero; no activation integration during160 initialization ticks; actual physical commands ticks160..165, samples1..20 etc; sample0 ctrl zero',
        physical_admission='S119 NOT SATISFIED / NOT TESTED: this is not a physical admission test',
        omissions=['body length/velocity/moment response','contact response','neural or sensory feedback changes','generalized acceleration and joint safety prediction'],
        model_load=0,forward=0,step=0,neural_ticks=0,physical_trial=0,checkpoint_replacement=False,
        input_sha256=before)
    out.mkdir(parents=True); dump(out/'PREREG.json',prereg)
    prereg_hash=sha(out/'PREREG.json')
    # Independently reconstruct all166 FIR ticks from actual binary spikes and manifest.
    source=js(REF.parent/'PREREG.json')['source_hashes']
    manifests=[p for p in source if p.endswith('/MANIFEST.json') and 's4-endpoints' in p]
    assert len(manifests)==1
    groups=[tuple(sorted(r['local_indices'])) for r in js(manifests[0])['rows']]
    pools=sorted(set(g for g in groups if g)); lookup={g:i for i,g in enumerate(pools)}
    assert np.array_equal(pool,[lookup.get(g,-1) for g in groups])
    hist=np.zeros((80,66)); features=[]
    for sp in np.concatenate((ini['spikes'],z['spikes'])):
        activity=np.array([sp[list(g)].mean()/.0005/100 for g in pools])
        hist=np.concatenate((activity[None],hist))[:80]
        features.append(np.stack([hist[lo:hi].mean(0) for lo,hi in BINS],axis=1))
    features=np.array(features)
    ferr=float(np.max(np.abs(features-np.concatenate((ini['features'],z['features'])))))
    assert ferr<1e-12
    mapped=pool>=0
    def controls(weights):
        raw=np.sum(features*weights,axis=2); c=np.zeros((166,90)); c[:,mapped]=np.clip(raw[:,pool[mapped]],0,1)
        return raw,c
    raw,c=controls(W)
    np.testing.assert_allclose(c[:160],ini['decoded'],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(c[160:],z['applied'],atol=1e-12,rtol=1e-12)
    assert not z['ctrl'][0].any() and not ini['body_act'].any()
    for i in range(1,121): np.testing.assert_array_equal(z['ctrl'][i],c[160+(i-1)//20])
    resolution=js(S124/'ACTUATOR_XML_RESOLUTION.json')
    params={}
    for key,m in [('dynprm',10),('gainprm',10),('biasprm',10),('lengthrange',2)]:
        a=np.zeros((90,m))
        for i,e in enumerate(resolution['all_actuators']):
            v=list(map(float,e['resolved'][key].split())); a[i,:len(v)]=v
        params[key]=a
    assert np.all(params['dynprm'][:,:3]==np.array([.0005,.0004,0]))
    act,_=integrate(c[160:],params['dynprm'],ini['body_act'])
    aerr=float(np.max(np.abs(act-z['act'])))
    save(out/'ACTIVATION_RECONSTRUCTION.csv',[dict(sample=i,time_ms=float(z['time'][i]*1000),actuator_id=a,saved=float(z['act'][i,a]),reconstructed=float(act[i,a]),error=float(act[i,a]-z['act'][i,a])) for i in range(121) for a in range(90)])
    verification=dict(FIR_max_abs_error=ferr,activation_max_abs_error=aerr,activation_tolerance=1e-12,
        status='ROUND_OFF_MATCH' if aerr<=1e-12 else 'APPROXIMATE_REJECTED',physical_sample0='zero ctrl, zero activation; release_decoded not applied',
        initialization_ticks=160,physical_ticks=6,total_history=166,activation_error_bound_measured_on_reference=aerr)
    dump(out/'HISTORICAL_RECONSTRUCTION.json',verification)
    assert aerr<=1e-12, 'Activation reproduction gate failed; no chain or solve'
    G=np.zeros((121,90)); B=G.copy()
    for t in range(121):
        for a in range(90):
            G[t,a],B[t,a],*_=gain_bias(z['actuator_length'][t,a],z['actuator_velocity'][t,a],params['lengthrange'][a],params['gainprm'][a],params['biasprm'][a])
    force=G*act+B; force_error=float(np.max(np.abs(force-z['actuator_force'])))
    assert force_error<1e-10
    M=z['actuator_moment'][:,:,6:9]
    baseline=np.einsum('ta,tad->td',force,M)
    draw=np.zeros((166,90,n))
    for j,(p,b) in enumerate(indices): draw[:,pool==p,j]=features[:,p,b,None]
    rawact=np.zeros((166,90)); rawact[:,mapped]=raw[:,pool[mapped]]
    dc=draw*((rawact>0)&(rawact<1))[:,:,None]
    _,da=integrate(c[160:],params['dynprm'],ini['body_act'],dc[160:])
    chain=np.einsum('taj,ta,tad->tdj',da,G,M)
    fdrows=[]
    for j,(p,b) in enumerate(indices):
        eps=1e-6; wp=W.copy(); wm=W.copy(); wp[p,b]+=eps; wm[p,b]-=eps
        _,cp=controls(wp); _,cm=controls(wm)
        ap,_=integrate(cp[160:],params['dynprm'],ini['body_act']); am,_=integrate(cm[160:],params['dynprm'],ini['body_act'])
        fdact=(ap-am)/(2*eps); fdctrl=(cp-cm)/(2*eps)
        fd=np.einsum('ta,ta,tad->td',fdact,G,M)
        for label,analytic,numeric in [('ctrl',dc[:,:,j],fdctrl),('activation',da[:,:,j],fdact),('torque678',chain[:,:,j],fd)]:
            err=float(np.max(np.abs(analytic-numeric))); rel=err/max(1e-8,float(np.max(np.abs(numeric))))
            fdrows.append(dict(pool=p,bin=b,component=label,epsilon=eps,max_abs_error=err,scaled_relative_error=rel,passed=err<=1e-7 and rel<=1e-5))
    save(out/'CHAIN_FINITE_DIFFERENCES.csv',fdrows)
    assert all(r['passed'] for r in fdrows), 'Chain FD failed; no solve'
    window=np.arange(77,105); objective=chain[window,2,:].mean(0)*scale/.005
    matrices=[]; lower=[]; upper=[]
    def constrain(A,lo,hi):
        A=np.asarray(A).reshape(-1,n); matrices.append(A); lower.extend(np.broadcast_to(lo,(len(A),))); upper.extend(np.broadcast_to(hi,(len(A),)))
    constrain(dc[:,affected,:]*scale,-.005,.005)
    constrain(chain[:,:2,:]*scale,-.005,.005)
    # Clip branch preservation including warmup: bounds are on raw drive, not linearized clipped command.
    for a in affected:
        r=rawact[:,a]; lo=np.where(r<=0,-np.inf,np.where(r>=1,1+1e-10,1e-10))-r
        hi=np.where(r<=0,0,np.where(r>=1,np.inf,1-1e-10))-r
        constrain(draw[:,a,:]*scale,lo,hi)
    A=np.vstack(matrices); lo=np.array(lower); hi=np.array(upper)
    wb=np.array([W[p,b] for p,b in indices]); bounds=list(zip(np.maximum(-1,-wb/scale),np.minimum(1,(4-wb)/scale)))
    result=minimize(lambda x:float(objective@x+.05*(x@x)),np.zeros(n),jac=lambda x:objective+.1*x,
        bounds=bounds,constraints=[LinearConstraint(A,lo,hi)],method='SLSQP',options=dict(ftol=1e-12,maxiter=200,disp=False))
    x=result.x; delta=np.zeros_like(W)
    for value,(p,b) in zip(scale*x,indices): delta[p,b]=value
    candidate=W+delta; rawnew,cnew=controls(candidate)
    anew,_=integrate(cnew[160:],params['dynprm'],ini['body_act'])
    fnew=G*anew+B; actual=np.einsum('ta,tad->td',fnew,M); linear=baseline+np.einsum('tdj,j->td',chain,scale*x)
    change=actual-baseline
    residual=float(max(0,np.max(lo-A@x),np.max(A@x-hi)))
    ctrlmax=float(np.max(np.abs(cnew[:,affected]-c[:,affected])))
    d67=float(np.max(np.abs(change[:,:2]))); decrease=float(change[window,2].mean())
    trust=dict(solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit),solve_invocations=1,
        linear_constraint_max_violation=residual,weight_max_abs=float(np.max(np.abs(delta))),weight_l2=float(np.linalg.norm(delta)),
        control_max_abs_change_all166=ctrlmax,DOF67_max_abs_nonlinear_change=d67,mean_window_DOF8_change=decrease,
        linear_nonlinear_max_abs_discrepancy=float(np.max(np.abs(actual-linear))),all90_force_reproduction_error=force_error,
        unchanged_pool36=bool(np.array_equal(candidate[36],W[36])),zero_mask_retained=bool(np.array_equal(candidate==0,W==0)))
    admitted=bool(result.success and residual<=1e-9 and ctrlmax<=.005+1e-9 and d67<=.005+1e-9 and np.linalg.norm(delta)<=.002+1e-12 and decrease<-1e-10)
    trust.update(offline_candidate_saved=admitted,physical_admission=False,S119_physical_constraints='NOT TESTED / NOT SATISFIED',checkpoint_replacement=False)
    dump(out/'TRUST_CONSTRAINTS.json',trust)
    save(out/'ALL_WEIGHT_DELTAS.csv',[dict(pool=p,bin=b,original=repr(float(W[p,b])),delta=repr(float(delta[p,b])),candidate=repr(float(candidate[p,b])),allowed=(p,b) in indices,changed=bool(delta[p,b]!=0)) for p in range(66) for b in range(5)])
    save(out/'ALL90_ACTUATOR_MAPPING.csv',[dict(actuator_id=a,name=names[a],pool=int(pool[a]),affected=bool(a in affected),all_shared_actuator_ids=json.dumps(np.flatnonzero(pool==pool[a]).tolist()) if pool[a]>=0 else '[]',unchanged_control=bool(np.array_equal(c[:,a],cnew[:,a]))) for a in range(90)])
    save(out/'ALL166_COMMAND_SERIES.csv',[dict(tick=k,stage='initialization_not_applied' if k<160 else 'physical_held',actuator_id=a,name=names[a],pool=int(pool[a]),reference=float(c[k,a]),candidate=float(cnew[k,a]),delta=float(cnew[k,a]-c[k,a])) for k in range(166) for a in range(90)])
    save(out/'ACTIVATION_FORCE_TORQUE_SERIES.csv',[dict(sample=t,time_ms=float(z['time'][t]*1000),actuator_id=a,name=names[a],pool=int(pool[a]),affected=bool(a in affected),reference_act=float(act[t,a]),candidate_act=float(anew[t,a]),reference_force=float(force[t,a]),candidate_force=float(fnew[t,a]),**{f'delta_tau{d}':float((fnew[t,a]-force[t,a])*M[t,a,d-6]) for d in (6,7,8)}) for t in range(121) for a in range(90)])
    save(out/'AGGREGATE_TORQUE_SERIES.csv',[dict(sample=t,time_ms=float(z['time'][t]*1000),in_objective_window=bool(t in window),**{f'{k}_DOF{d}':float(v[t,d-6]) for k,v in [('reference',baseline),('linear_prediction',linear),('nonlinear_frozen_prediction',actual),('delta',change)] for d in (6,7,8)}) for t in range(121)])
    np.savez_compressed(out/'LOCAL_CHAIN.npz',weight_indices=np.array(indices),command_derivative=dc,activation_derivative=da,torque_derivative=chain,reference_weights=W,delta=delta,reference_act=act,candidate_act=anew,gain=G,bias=B,moment678=M,time=z['time'])
    if admitted: np.savez_compressed(out/'CANDIDATE_WEIGHTS.npz',weights=candidate,reference_weights=W,delta=delta)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    for d,ax in zip((6,7,8),axes[:3]):
        ax.plot(z['time']*1000,change[:,d-6],label='nonlinear frozen-body chain')
        ax.plot(z['time']*1000,(linear-baseline)[:,d-6],'--',label='linear QP chain')
        ax.set_ylabel(f'delta torque DOF{d} (native)'); ax.legend(); ax.axvspan(1.925,2.625,color='gray',alpha=.15)
    for a in affected: axes[3].plot(z['time']*1000,anew[:,a]-act[:,a],label=f'{a} {names[a]} pool{pool[a]}')
    axes[3].legend(fontsize=7); axes[3].set_ylabel('delta activation'); axes[3].set_xlabel('release-relative time (ms)')
    fig.suptitle('S125 OFFLINE ONLY: fixed body/contact/neural history; NOT S119 physical admission')
    fig.tight_layout(); fig.savefig(out/'LOCAL_TRANSLATION_CANDIDATE.png',dpi=150); plt.close(fig)
    after={str(p):sha(p) for p in sorted(inputs)}; assert before==after and sha(out/'PREREG.json')==prereg_hash
    assert 'mujoco' not in sys.modules
    dump(out/'INPUT_HASHES.json',dict(before=before,after=after,all_unchanged=True))
    summary=dict(trust,activation_reconstruction=verification,chain_FD_passed=True,all_inputs_unchanged=True,input_files=len(before),
        affected_actuator_ids=affected.tolist(),allowed_weight_entries=n,prereg_sha256=prereg_hash,
        model_load=0,forward=0,step=0,neural_ticks=0,physical_trial=0,physical_units='NOT_ESTABLISHED',training_acceptance=False,standing_or_walking_claim=False)
    dump(out/'SUMMARY.json',summary)
    (out/'REPORT.zh-CN.md').write_text('# S125 有界离线局部翻译层候选\n\n'
        +'单次正定正则化凸QP；实际166tick FIR完整复现，初始化期间身体activation=0且不积分；只用实际6条物理命令，sample0不是release decoded。全部121×90激活误差 '+str(aerr)+'。\n\n'
        +'冻结参考身体长度/速度/力臂、神经事件及接触；局部链不是闭环梯度，未预测承重、姿态或关节安全。所有共享映射执行器均纳入，pool36全冻结。候选写入：'+str(admitted)+'；DOF8目标窗平均变化 '+str(decrease)+' native。\n\n'
        +'S119物理接纳未满足/未测试；没有新MuJoCo模型、forward/step、神经tick或物理试验。没有替换活动权重/checkpoint，没有宣称物理预热、站立或行走。绝对物理单位NOT_ESTABLISHED。\n')
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=OUT)
    main(parser.parse_args().out)
