"""S2/S4 limited offline diagnosis. No body integration, runtime or training changes."""
from pathlib import Path
import sys, json, csv, argparse, traceback
import numpy as np
import retarget_continuous_fly_motion as old
OUT=Path('/home/sensen/Desktop/engine修复验收/s4-middle-leg-retarget-diagnosis/20261102T020329Z')
SRC=old.OUT
SEG=['ThC-CTr','CTr-FTi','FTi-TiTa']
C=dict(model_compile=0,kinematics=0,com_pos=0,jacobian=0,forward=0,ik_solves=0,body_steps=0,neural_steps=0,network_inits=0,training=0)
def dump(name,x):old.dump(OUT/name,x)
def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def identities():
    paths=[Path(__file__),old.MODEL,SRC/'RETARGETED_TARGETS.npz',SRC/'LOSS_ADMISSION.npz',SRC/'TRANSFORM_AND_UNIT_LEDGER.json',SRC/'MODEL_MARKER_CONTRACT.json',old.INTERFACE/'JOINT_AND_MARKER_MAPPING.json',old.SOURCE/'NMF2_FIELDS_AND_TRANSFORMS.json']
    return {str(p):old.sha(p) for p in paths}
def angle(d,t):return np.arccos(np.clip(np.sum(d*t,axis=-1),-1,1))
def diagnose():
    from scipy.optimize import least_squares
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PREREGISTRATION.json').exists()
    a=old.load(SRC/'RETARGETED_TARGETS.npz');e=angle(a['achieved_directions'][:,:,:3],a['reference_directions'][:,:,:3])
    assert np.array_equal(e,a['direction_residual_rad'][:,:,:3])
    cases=[dict(leg='LM',frame=64,mirror=False),dict(leg='LM',frame=37,mirror=False),dict(leg='RM',frame=45,mirror=False),dict(leg='RM',frame=47,mirror=False),dict(leg='RM',frame=64,mirror=True,source_leg='LM'),dict(leg='LM',frame=45,mirror=True,source_leg='RM')]
    dump('PREREGISTRATION.json',dict(stage='S2/S4',cases=cases,seeds=['bounded springref','original previous frame','original next frame','native midpoint','deterministic rng 20261102 draw0','deterministic rng 20261102 draw1'],max_starts_per_case=6,max_nfev=300,objective='original 9 direction components + sqrt(1e-6)*(q-original_previous)/native_span',selection='LM maximum individual segment; RM maximum mean; respective minimum mean; two contralateral sagittal reflections diag(1,-1,1)',significant='geometric squared component residual reduction > 1e-4 and > 0.1*baseline',counterfactual='not planned: only if KKT proves binding; otherwise no bound relaxations',full_record_optimization=False,source_rootframe='NOT_ESTABLISHED',units='NOT_ESTABLISHED',native_joint_loss_truth=False,Claw_masked=True))
    dump('SOURCE_IDENTITY.json',identities());(OUT/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    rows=[];summary={}
    for li,leg in enumerate(old.LEGS):
        summary[leg]={}
        for split,name in enumerate(['train','validation','test']):
            sel=a['split']==split
            summary[leg][name]=dict(mean_deg=np.rad2deg(e[sel,li]).mean(0),max_deg=np.rad2deg(e[sel,li]).max(0),frames=int(sel.sum()))
        for t in range(37):
            for k in range(3):rows.append(dict(frame=int(a['source_frame'][t]),split=int(a['split'][t]),leg=leg,segment=SEG[k],direction_error_deg=float(np.rad2deg(e[t,li,k])),active_bounds=int(a['bound_saturation'][t,li].sum()),jump_from_previous=int(a['branch_jump_mask'][t-1,li].sum()) if t else 0,source_length=float(np.linalg.norm(np.diff(a['source_xyz'][t,li],axis=0)[k]))))
    table('FRAME_LEG_SEGMENT.csv',rows);dump('BASELINE_EXACT.json',dict(residual_array_exact=True,per_leg_split=summary,saturation_count=int(a['bound_saturation'].sum()),jump_count=int(a['branch_jump_mask'].sum())))
    mj,m,assets=old.compile_model();C['model_compile']+=1;dump('ASSET_IDENTITY.json',assets)
    data,thorax,bodies,jids,adr,bounds,base,p0,lengths=old.setup(mj,m);C['kinematics']+=1
    # Actual body origins and local joint pivots, with no inferred Euler convention.
    mapping=[]
    for li,leg in enumerate(old.LEGS):
        for jid in jids[li]:mapping.append(dict(leg=leg,joint=m.joint(int(jid)).name,body=m.body(int(m.jnt_bodyid[jid])).name,axis_local=m.jnt_axis[jid],pivot_local=m.jnt_pos[jid],pivot_world=data.xanchor[jid],body_origin=data.xpos[m.jnt_bodyid[jid]],body_parent=m.body(int(m.body_parentid[m.jnt_bodyid[jid]])).name,body_pos=m.body_pos[m.jnt_bodyid[jid]],body_quat=m.body_quat[m.jnt_bodyid[jid]],bounds=m.jnt_range[jid]))
    dump('AXES_PIVOTS.json',mapping)
    layout=json.loads((SRC/'TRANSFORM_AND_UNIT_LEDGER.json').read_text());sa=np.array(layout['source_anchor_mean']);sa-=sa.mean(0);sa/=np.linalg.norm(sa);ma=a['model_anchor_points'].copy();ma-=ma.mean(0);ma/=np.linalg.norm(ma)
    sd=np.linalg.norm(sa[:,None]-sa[None,:],axis=-1);md=np.linalg.norm(ma[:,None]-ma[None,:],axis=-1)
    dump('LAYOUT_NONCONFORMITY.json',dict(normalized_pair_distance_difference=sd-md,max_abs=float(np.max(np.abs(sd-md))),locked_rotation_det=np.linalg.det(a['rotation_source_to_model']),interpretation='Pair distance differences invariant under rotation: single proper rotation cannot remove morphology/measurement differences; does not prove source axis error. No per-leg refit.'))
    result=[];qs=[];tg=[];pts=[];runrows=[];kktrows=[]
    for ci,case in enumerate(cases):
        li=old.LEGS.index(case['leg']);t=int(np.flatnonzero(a['source_frame']==case['frame'])[0]);sl=old.LEGS.index(case.get('source_leg',case['leg']))
        target=a['reference_directions'][t,sl,:3].copy()
        if case['mirror']:target=target*np.array([1,-1,1])
        lo,hi=bounds[li].T;span=hi-lo;address=adr[li];dofs=m.jnt_dofadr[jids[li]];prior=a['qpos'][t-1,address] if t else base[address]
        qbase=a['qpos'][t].copy();cache={}
        def evaluate(v):
            if 'v' not in cache or not np.array_equal(v,cache['v']):
                data.qpos[:]=qbase;data.qpos[address]=v;mj.mj_kinematics(m,data);C['kinematics']+=1;mj.mj_comPos(m,data);C['com_pos']+=1
                p=data.xpos[bodies[li,:4]].copy();dv=np.diff(p,axis=0);ln=np.linalg.norm(dv,axis=1);u=dv/ln[:,None]
                jp=[]
                for bid in bodies[li,:4]:
                    j=np.zeros((3,m.nv));mj.mj_jac(m,data,j,None,data.xpos[bid],int(bid));C['jacobian']+=1;jp.append(j[:,dofs])
                jdiff=np.diff(np.array(jp),axis=0);jac=np.concatenate([(np.eye(3)-np.outer(u[k],u[k]))@jdiff[k]/ln[k] for k in range(3)])
                cache.update(v=v.copy(),p=p,u=u,j=jac)
            return cache
        def fun(v):
            z=evaluate(v);return np.r_[(z['u']-target).ravel(),np.sqrt(old.TEMPORAL)*(v-prior)/span]
        def jac(v):return np.vstack([evaluate(v)['j'],np.diag(np.sqrt(old.TEMPORAL)/span)])
        original=qbase[address];orig=evaluate(original);origerr=float(np.sum((orig['u']-target)**2));origang=np.rad2deg(angle(orig['u'],target))
        # Independent finite-difference check at midpoint; diagnostic only, never optimizer Jacobian.
        mid=(lo+hi)/2;ja=jac(mid);fd=np.column_stack([(fun(mid+np.eye(7)[j]*1e-6)-fun(mid-np.eye(7)[j]*1e-6))/2e-6 for j in range(7)]);jacerr=float(np.max(np.abs(ja-fd)));assert jacerr<1e-6
        rng=np.random.default_rng(20261102)
        seeds=[base[address],a['qpos'][max(t-1,0),address],a['qpos'][min(t+1,36),address],mid,rng.uniform(lo,hi),rng.uniform(lo,hi)]
        fits=[]
        for si,seed in enumerate(seeds):
            r=least_squares(fun,np.clip(seed,lo+1e-12,hi-1e-12),jac=jac,bounds=(lo,hi),max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9);C['ik_solves']+=1
            z=evaluate(r.x);geom=float(np.sum((z['u']-target)**2));fits.append((geom,r.x.copy(),r))
            runrows.append(dict(case=ci,seed=si,success=bool(r.success),status=int(r.status),nfev=int(r.nfev),geometry_ss=geom,total_cost=float(r.cost),optimality=float(r.optimality)))
        geom,v,r=min(fits,key=lambda x:x[0]);z=evaluate(v);p=z['p'].copy();u=z['u'].copy();j=z['j'].copy();g=j.T@(u-target).ravel();gt=jac(v).T@fun(v)
        active=np.minimum(v-lo,hi-v)<=1e-5;lower=v-lo<=1e-5;upper=hi-v<=1e-5;pg=gt.copy();pg[lower & (gt>0)]=0;pg[upper & (gt<0)]=0
        for ji in range(7):kktrows.append(dict(case=ci,joint=m.joint(int(jids[li,ji])).name,q=float(v[ji]),lower=float(lo[ji]),upper=float(hi[ji]),active=bool(active[ji]),geometry_gradient=float(g[ji]),total_gradient=float(gt[ji]),outward_binding_multiplier=float(max(gt[ji],0) if lower[ji] else max(-gt[ji],0) if upper[ji] else 0),projected_gradient=float(pg[ji])))
        sv=np.linalg.svd(j,compute_uv=False);q=qbase.copy();q[address]=v;qs.append(q);tg.append(target);pts.append(p)
        result.append(dict(case=ci,**case,row=t,original_geometry_ss=origerr,best_geometry_ss=geom,original_segment_deg=origang,best_segment_deg=np.rad2deg(angle(u,target)),significant_improvement=bool(origerr-geom>max(1e-4,.1*origerr)),jacobian_fd_maxabs=jacerr,singular_values=sv,rank=int((sv>sv[0]*1e-6).sum()),active_count=int(active.sum()),projected_gradient_max=float(np.abs(pg).max()),success=bool(r.success),scope='six diagnostic cases only; mirrors unobserved; no training admission'))
        print('case',ci,case,'original',origerr,'best',geom,flush=True)
    table('MULTISTART.csv',runrows);table('KKT_ACTIVE_BOUNDS.csv',kktrows);dump('CASES.json',result)
    np.savez_compressed(OUT/'DIAGNOSTIC_CANDIDATES.npz',qpos=np.array(qs),target_directions=np.array(tg),achieved_points=np.array(pts),leg_indices=np.array([old.LEGS.index(c['leg']) for c in cases]),source_frames=np.array([c['frame'] for c in cases]),mirror=np.array([c['mirror'] for c in cases]),native_joint_loss_mask=np.zeros((6,7),bool),root_valid=np.zeros(6,bool),Claw_valid=np.zeros(6,bool))
    dump('EXECUTION.json',dict(success=True,counters=C,executable=sys.executable,source_rootframe='NOT_ESTABLISHED',units='NOT_ESTABLISHED',counterfactual_solves=0,full_record_solves=0))
def verify():
    a=old.load(SRC/'RETARGETED_TARGETS.npz');b=old.load(OUT/'DIAGNOSTIC_CANDIDATES.npz');assert identities()==json.loads((OUT/'SOURCE_IDENTITY.json').read_text())
    mj,m,assets=old.compile_model();d=mj.MjData(m);maxerr=0.;rows=[]
    for i,q in enumerate(b['qpos']):
        li=int(b['leg_indices'][i]);names=[li*7+j for j in range(7)];contract=json.loads((SRC/'MODEL_MARKER_CONTRACT.json').read_text());joints=contract['joints'];ad=np.array([joints[k]['qpos_address'] for k in names]);lo=np.array([joints[k]['range_rad'][0] for k in names]);hi=np.array([joints[k]['range_rad'][1] for k in names]);assert np.all(q[ad]>=lo-1e-12) and np.all(q[ad]<=hi+1e-12)
        t=int(np.flatnonzero(a['source_frame']==b['source_frames'][i])[0]);other=np.ones(49,bool);other[ad]=False;assert np.array_equal(q[other],a['qpos'][t,other])
        d.qpos[:]=q;mj.mj_forward(m,d);p=np.array([d.xpos[m.body(x['model_body']).id].copy() for x in contract['legs'][li]['markers'][:4]])
        maxerr=max(maxerr,float(np.max(np.abs(p-b['achieved_points'][i]))));delta=np.diff(p,axis=0);u=delta/np.linalg.norm(delta,axis=1)[:,None];target=b['target_directions'][i];sl=(4 if li==1 else 1) if b['mirror'][i] else li;expected=a['reference_directions'][t,sl,:3].copy()
        if b['mirror'][i]:expected*=np.array([1,-1,1])
        assert np.array_equal(expected,target);rows.append(dict(case=i,geometry_ss=float(np.sum((u-target)**2)),segment_deg=np.rad2deg(angle(u,target))))
    assert maxerr<1e-10 and not b['native_joint_loss_mask'].any() and not b['root_valid'].any() and not b['Claw_valid'].any()
    cases=json.loads((OUT/'CASES.json').read_text());assert all(abs(c['best_geometry_ss']-r['geometry_ss'])<1e-12 for c,r in zip(cases,rows))
    dump('INDEPENDENT_FK_VERIFICATION.json',dict(passed=True,fresh_process=True,fk_maxabs=maxerr,cases=rows,counters=dict(model_compile=1,forward=6,ik_solves=0,body_steps=0,neural_steps=0,network_inits=0,training=0),source_hashes_exact=True,assets_exact=assets==json.loads((OUT/'ASSET_IDENTITY.json').read_text())))
def seal():
    assert not (OUT/'MANIFEST_SHA256.json').exists();assert json.loads((OUT/'PARENT_REVIEW.json').read_text())['passed'];assert json.loads((OUT/'INDEPENDENT_FK_VERIFICATION.json').read_text())['passed'];assert identities()==json.loads((OUT/'SOURCE_IDENTITY.json').read_text())
    dump('MANIFEST_SHA256.json',{str(p):old.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['diagnose','verify','seal']);mode=parser.parse_args().mode
    try:globals()[mode]()
    except Exception:
        if OUT.exists():dump('FAILED_'+mode+'.json',dict(traceback=traceback.format_exc(),counters=C))
        raise
