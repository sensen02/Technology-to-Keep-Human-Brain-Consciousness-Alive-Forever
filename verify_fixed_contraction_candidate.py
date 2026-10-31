"""Independent S125 offline activation/local-torque verifier. No project/simulator imports.
Scope: saved features, fixed body states; not a closed-loop or behavioral prediction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
S124 = Path('/home/sensen/Desktop/engine修复验收/s124-muscle-reconstructability/20261024T075500Z')
COLS = [(p,b) for p in (3,15,38) for b in range(5)]
H = .000025


def rate(a, u, prm):
    """Literal 3.9 muscle RHS plus independent partials wrt RAW a and u.
    Endpoint clip derivative convention zero; nonsmooth endpoints flagged separately.
    """
    ac=np.clip(a,0,1); uc=np.clip(u,0,1)
    ca=((a>0)&(a<1)).astype(float); cu=((u>0)&(u<1)).astype(float)
    r=.5+1.5*ac; ta=prm[:,0]*r; td=prm[:,1]/r
    taa=1.5*prm[:,0]*ca; tda=-1.5*prm[:,1]*ca/(r*r)
    d=uc-a; width=prm[:,2]; hard=width<1e-15
    x=d/np.where(hard,1,width)+.5; xc=np.clip(x,0,1)
    s=xc*xc*xc*(3*xc*(2*xc-5)+10)
    sd=np.where((x>0)&(x<1),30*xc*xc*(xc-1)**2,0)/np.where(hard,1,width)
    s=np.where(hard,(d>0).astype(float),s); sd=np.where(hard,0,sd)
    tau=td+(ta-td)*s
    tau_a=tda+(taa-tda)*s-(ta-td)*sd
    tau_u=(ta-td)*sd*cu
    den=np.maximum(tau,1e-15); active=tau>1e-15
    fa=-1/den-d*np.where(active,tau_a,0)/(den*den)
    fu=cu/den-d*np.where(active,tau_u,0)/(den*den)
    return d/den,fa,fu


def rk(a,j,u,du,prm):
    def stage(x,z):
        f,fa,fu=rate(x,u,prm)
        return f,fa[:,None]*z+fu[:,None]*du
    k1,l1=stage(a,j)
    k2,l2=stage(a+H*.5*k1,j+H*.5*l1)
    k3,l3=stage(a+H*.5*k2,j+H*.5*l2)
    k4,l4=stage(a+H*k3,j+H*l3)
    # Match MuJoCo summation order, no clipping of intermediate RK states.
    av=np.zeros_like(a); jj=np.zeros_like(j)
    for k,l,b in [(k1,l1,1/6),(k2,l2,1/3),(k3,l3,1/3),(k4,l4,1/6)]:
        av+=b*k; jj+=b*l
    return a+H*av,j+H*jj


def chain(weights,features,mapping,prm,initial):
    raw=np.sum(features*weights[None,:,:],axis=2); pool=np.clip(raw,0,1)
    controls=np.zeros((6,90)); mapped=mapping>=0
    controls[:,mapped]=pool[:,mapping[mapped]]
    du=np.zeros((6,90,len(COLS)))
    for c,(p,b) in enumerate(COLS):
        du[:,mapping==p,c]=(features[:,p,b]*((raw[:,p]>0)&(raw[:,p]<1)))[:,None]
    a=initial.copy(); j=np.zeros((90,len(COLS))); acts=[a.copy()]; jac=[j.copy()]
    for t in range(6):
        for sub in range(20):
            a,j=rk(a,j,controls[t],du[t],prm)
            acts.append(a.copy()); jac.append(j.copy())
    return controls,np.asarray(acts),np.asarray(jac),raw


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('fresh output required')
    paths=[BASE/'TRAJECTORY.npz',S124/'ACTUATOR_XML_RESOLUTION.json',S124/'LOCAL_FIXED_STATE_PARTIALS.csv']
    z=np.load(paths[0],allow_pickle=False); original=z['weights']; mapping=z['actuator_pool']
    rows=json.loads(paths[1].read_text())['all_actuators']
    prm=np.asarray([[float(v) for v in r['resolved']['dynprm'].split()][:3] for r in rows])
    # XML general-muscle class has no actlimited/range, and every reconstructed
    # state is in [0,1]; standard muscle final clamp, if enabled to [0,1], is inactive.
    assert np.all(mapping[:6]==[38,38,36,-1,15,3])
    assert np.all(original>0) and np.array_equal(np.flatnonzero(mapping==38),[0,1])
    u,a,j,raw=chain(original,z['features'],mapping,prm,z['act'][0])
    hist=float(np.max(np.abs(a-z['act'])))
    ctrl=float(np.max(np.abs(u-z['applied'])))
    saved_ctrl=float(np.max(np.abs(z['ctrl'][1:]-np.repeat(u,20,axis=0))))
    # Independent fixed-state d torque / d activation; sum over all six focus
    # muscles, preserving pool38's two-muscle sharing and unmapped act3 zero.
    g=np.zeros((121,90,3))
    with paths[2].open() as f:
        for r in csv.DictReader(f):
            if r['run']=='init80ms':
                g[int(r['sample']),int(r['actuator_id']),int(r['dof'])-6]=float(r['dtorque_dactivation_fixed_state'])
    if args.candidate:
        paths.append(args.candidate); weights=np.load(args.candidate,allow_pickle=False)['weights']
        assert weights.shape==original.shape and np.isfinite(weights).all()
        assert np.all((weights>=0)&(weights<=4))
        mask=np.zeros_like(original,dtype=bool)
        for p,b in COLS: mask[p,b]=True
        assert np.array_equal(weights[~mask],original[~mask])
        assert np.array_equal(weights==0,original==0)
    else: weights=original.copy()
    cu,ca,cj,cr=chain(weights,z['features'],mapping,prm,z['act'][0])
    torque=np.einsum('sad,sa->sd',g,ca-z['act'])
    tangent=np.einsum('sad,sac->sdc',g,cj)
    child_comparison={}
    if args.candidate and (args.candidate.parent/'LOCAL_CHAIN.npz').exists():
        cp=args.candidate.parent/'LOCAL_CHAIN.npz'; paths.append(cp)
        child=np.load(cp,allow_pickle=False)
        order=[COLS.index(tuple(map(int,row))) for row in child['weight_indices']]
        baseline_tangent=np.einsum('sad,sac->sdc',g,j)
        child_comparison=dict(candidate_activation_maxabs=float(np.max(np.abs(child['candidate_act']-ca))),reference_activation_maxabs=float(np.max(np.abs(child['reference_act']-a))),activation_tangent_maxabs=float(np.max(np.abs(child['activation_derivative']-j[:,:,order]))),torque_tangent_maxabs=float(np.max(np.abs(child['torque_derivative']-baseline_tangent[:,:,order]))))
        assert max(child_comparison.values())<1e-8, child_comparison
        delta=weights-original
        assert np.max(np.abs(delta))<=.002/np.sqrt(15)+1e-14 and np.linalg.norm(delta)<=.002+1e-14
        assert np.max(np.abs(torque[:,:2]))<=.005+1e-9
        parent=Path('/home/sensen/Desktop/engine修复验收/s125-local-translation-candidate/20261024T080500Z')
        with np.load(parent/'LOCAL_CHAIN.npz') as proposal:
            proposal_delta=proposal['delta'].copy()
            assert np.array_equal(weights,proposal['reference_weights']+.99*proposal_delta)
        with (parent/'ALL_WEIGHT_DELTAS.csv').open() as f:
            for row in csv.DictReader(f):
                p,b=int(row['pool']),int(row['bin'])
                assert float(row['original'])==original[p,b]
                assert float(row['delta'])==proposal_delta[p,b]
        paths.extend([parent/'LOCAL_CHAIN.npz',parent/'ALL_WEIGHT_DELTAS.csv'])
        window=(z['time']>=.001925-1e-12)&(z['time']<.002625-1e-12)
        assert float(torque[window,2].mean()) < -1e-10
        assert np.max(np.abs(cu-u))<=.005+1e-12
        warmfile=BASE/'INITIALIZATION.npz'; paths.append(warmfile)
        warm=np.load(warmfile,allow_pickle=False)['features']
        warmraw=np.sum(warm*original[None,:,:],axis=2)
        newwarmraw=np.sum(warm*weights[None,:,:],axis=2)
        assert np.max(np.abs(np.clip(newwarmraw,0,1)-np.clip(warmraw,0,1)))<=.005+1e-12
        assert np.array_equal((newwarmraw>0)&(newwarmraw<1),(warmraw>0)&(warmraw<1))
        assert np.array_equal((cr>0)&(cr<1),(raw>0)&(raw<1))
    checks=[]
    for col,(p,b) in enumerate(COLS):
        eps=1e-6; wp=weights.copy(); wm=weights.copy(); wp[p,b]+=eps; wm[p,b]-=eps
        up,aa,_,rp=chain(wp,z['features'],mapping,prm,z['act'][0])
        um,ab,_,rm=chain(wm,z['features'],mapping,prm,z['act'][0])
        branch=bool(np.array_equal((rp>0)&(rp<1),(rm>0)&(rm<1)))
        fd=(aa-ab)/(2*eps); tf=np.einsum('sad,sa->sd',g,fd)
        err=float(np.max(np.abs(fd-cj[:,:,col]))); terr=float(np.max(np.abs(tf-tangent[:,:,col])))
        checks.append(dict(pool=p,bin=b,epsilon=eps,control_clip_branch_stable=branch,activation_fd_maxabs=err,torque_fd_maxabs=terr))
    assert hist<1e-12 and ctrl<1e-12 and saved_ctrl<1e-12
    assert np.min(ca)>=0 and np.max(ca)<=1
    assert all(c['control_clip_branch_stable'] and c['activation_fd_maxabs']<1e-7 and c['torque_fd_maxabs']<1e-5 for c in checks)
    report=dict(passed=True,child_comparison=child_comparison,scope='fixed saved neural features and body state only; not closed-loop gradient',historical_activation_maxabs=hist,historical_command_maxabs=ctrl,saved_held_control_maxabs=saved_ctrl,body_step_s=H,neural_hold_s=.0005,initial_activation_maxabs=float(np.max(np.abs(z['act'][0]))),focus_pool_map=mapping[:6].tolist(),free_columns=COLS,all_nonzero_weights=int(np.count_nonzero(original)),all_excluding_pool36=int(np.count_nonzero(original[np.arange(66)!=36])),fd_checks=checks,candidate=str(args.candidate) if args.candidate else None,local_torque_delta_maxabs=float(np.max(np.abs(torque))),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},model_load=0,forward=0,step=0,neural=0)
    peak=np.unravel_index(np.argmax(np.abs(torque[:,:2])),torque[:,:2].shape)
    report.update(peak_dof67=dict(sample=int(peak[0]),time_ms=float(z['time'][peak[0]]*1000),dof=int(peak[1]+6),signed_change=float(torque[peak]),absolute_change=float(abs(torque[peak])),raw_cap_margin=float(.005-abs(torque[peak]))),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),offline_gate_only=True,physical_admission=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
