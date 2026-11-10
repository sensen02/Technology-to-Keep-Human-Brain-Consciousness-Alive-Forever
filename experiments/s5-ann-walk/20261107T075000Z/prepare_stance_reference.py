"""V2 FK reference: stance orientation only, cyclic joint continuity, bounded PCHIP.

No physics steps. Original failed 074700Z evidence is preserved. This asset is an
engineering teacher candidate; real contact support/slip require a later physical
screen. GAIT metadata requires periodic PCHIP, not the previous CubicSpline loader.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/s5-ann-walk/20261107T074700Z/prepare_stance_reference.py'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075000Z')
spec=importlib.util.spec_from_file_location('stance_v1_fk',OLD)
v1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
LEGS=v1.LEGS


def periodic_pchip(phase,values):
    return PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],
        np.concatenate((values[-3:-1],values,values[1:3]),axis=0),axis=0,extrapolate=False)


def hat(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def log_left_inverse(r):
    angle=np.linalg.norm(r)
    h=hat(r)
    factor=1/12+angle*angle/720 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/(angle*angle)
    return np.eye(3)-.5*h+factor*(h@h)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solver-phases',type=int,default=65)
    ap.add_argument('--export-phases',type=int,default=257)
    ap.add_argument('--global-nfev',type=int,default=200)
    args=ap.parse_args(argv)
    assert args.solver_phases>=33 and args.export_phases>=args.solver_phases
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():
        (OUT/'PREREGISTRATION.json').read_bytes()
        raise FileExistsError('preserve prior preregistration; use a new timestamp')
    sha=v1.sha
    sources=[Path(__file__),OLD,v1.SOURCE/'STANDARD_GAIT_FK.npz',v1.SOURCE/'REFERENCE_METADATA.json',v1.MODEL,v1.BASE/'incremental_actuator.py']
    hashes={str(p):sha(p) for p in sources}
    parameters=dict(duty=.65,halfstride=.15,clearance=.25,stance_orientation_weight=.5,
        swing_orientation_weight=0.,swing_transition_fraction=.15,continuity_weight=.01,
        second_difference_weight=.003,q0_weight=.001,continuation_cycles=2,**vars(args))
    thresholds=dict(position_max_native=.005,stance_orientation_max_rad=.03,
        interpolation_bound_tolerance_rad=1e-9,joint_step_max_rad=.5)
    prereg=dict(importance='核心必需',stage='S5 FK-only teacher v2; priorfailedprep preserved',parameters=parameters,
        thresholds=thresholds,source_hashes=hashes,
        reason='V1 worst errors in swing with forced fixed orientation; swing branch changes .788rad cause cubic overshoot. Stance orientation also failed on LM/LH/RM and must still pass unchanged criteria.',
        method='same q0 anatomical foot anchors and original root; perleg two-cycle continuation then sparse analytic-Jacobian cyclic whole-gait bounded least-squares',
        orientation='strict .03rad stance acceptance; no swing orientation acceptance. Weight eases toward0 over first15percent swing and returns last15percent.',
        interpolation='periodic extension PCHIP, componentwise bounded, exported257phases. Runtime MUST use periodic PCHIP (metadata), no clipping of cubic overshoot.',
        deployment='algorithm training teacher only; final deployed motion requires ANN, full physics, no root trajectory, no teacher after handover',
        counts=dict(physics_steps=0,neural_steps=0),
        source_provenance='processed real-fly LF swing-height shape only; duty/amplitudes/tripods/anchors/orientation are declared engineering retargeting',
        limitations=['FK cannot establish support, planted contacts, real slip, stability or walking.',
                     'The prior failure is preserved; no original threshold is relaxed. Swing orientation freedom is an explicitly preregistered changed design.',
                     'Tarsus5 origin is not a contact material point; constant stance orientation constrains geometry but does not prove no skid.'])
    v1.dump(OUT/'PREREGISTRATION.json',prereg)
    v1.dump(OUT/'SOURCE_SHA.json',hashes)
    shape,source_info=v1.source_profile()
    v1.dump(OUT/'SOURCE_SWING_RESAMPLING.json',source_info)
    fk=v1.FK()
    phase=np.linspace(0,2*np.pi,args.solver_phases)
    n=len(phase)-1
    goals_local,stance=v1.desired_path(phase,fk.anchor_local,shape,.65,.15,.25)
    goals_world=goals_local@fk.rotation.T+fk.origin
    u=(phase[:-1,None]/(2*np.pi)+v1.OFFSETS[None])%1.
    swing=np.clip((u-.65)/.35,0,1)
    transition=np.minimum(swing/.15,(1-swing)/.15)
    ow=np.where(stance[:-1],.5,.5*(1-v1.s5(np.clip(transition,0,1))))
    saved=np.repeat(fk.q0[None],len(phase),axis=0)
    logs=[]
    start=time.monotonic()
    for leg,name in enumerate(LEGS):
        qa=fk.leg_qa[leg]
        va=fk.leg_va[leg]
        q0=fk.q0[qa]
        lo,hi=fk.limits[leg,:,0],fk.limits[leg,:,1]
        rot0=fk.orientations[leg]
        seed=np.clip(q0,lo+1e-10,hi-1e-10)
        curve=np.repeat(seed[None],n,axis=0)
        for cycle in range(2):
            for i in range(n):
                previous=seed.copy()
                def local_res(q):
                    pos,rot=fk.pose(q,leg)
                    rv=Rotation.from_matrix(rot0.T@rot).as_rotvec()
                    return np.r_[pos-goals_world[i,leg],ow[i,leg]*rv,.001*(q-q0),.01*(q-previous)]
                fit=least_squares(local_res,seed,bounds=(lo,hi),max_nfev=90,
                    ftol=1e-9,xtol=1e-9,gtol=1e-8,x_scale='jac')
                seed=fit.x.copy()
                curve[i]=seed
        cache={}
        def evaluate(flat):
            if 'x' in cache and np.array_equal(cache['x'],flat):
                return cache['res'],cache['jac']
            q=flat.reshape(n,7)
            residual=np.empty((n,27))
            rowids=[];colids=[];values=[]
            for i in range(n):
                pos,rot=fk.pose(q[i],leg)
                rv=Rotation.from_matrix(rot0.T@rot).as_rotvec()
                residual[i,:3]=pos-goals_world[i,leg]
                residual[i,3:6]=ow[i,leg]*rv
                residual[i,6:13]=.001*(q[i]-q0)
                residual[i,13:20]=.01*(q[i]-q[(i-1)%n])
                residual[i,20:27]=.003*(q[(i-1)%n]-2*q[i]+q[(i+1)%n])
                jp=np.zeros((3,fk.m.nv));jr=np.zeros_like(jp)
                fk.mj.mj_jac(fk.m,fk.d,jp,jr,fk.d.xpos[fk.bids[leg]],fk.bids[leg])
                orient=ow[i,leg]*log_left_inverse(rv)@rot0.T@jr[:,va]
                block=np.vstack((jp[:,va],orient))
                rr,cc=np.indices((6,7))
                rowids.extend((i*27+rr.ravel()).tolist())
                colids.extend((i*7+cc.ravel()).tolist())
                values.extend(block.ravel().tolist())
                for j in range(7):
                    for r,col,val in [(6+j,i*7+j,.001),
                                      (13+j,i*7+j,.01),(13+j,((i-1)%n)*7+j,-.01),
                                      (20+j,((i-1)%n)*7+j,.003),(20+j,i*7+j,-.006),
                                      (20+j,((i+1)%n)*7+j,.003)]:
                        rowids.append(i*27+r);colids.append(col);values.append(val)
            jac=coo_matrix((values,(rowids,colids)),shape=(n*27,n*7)).tocsr()
            cache.update(x=flat.copy(),res=residual.ravel(),jac=jac)
            return cache['res'],jac
        # Finite-difference check selected columns before trusting the analytic
        # whole-cycle Jacobian. No physics integration is involved.
        x=curve.ravel().copy()
        value,jac=evaluate(x)
        jac_errors=[]
        for col in [0,7*(n//3)+2,7*(2*n//3)+5]:
            eps=1e-6
            xp=x.copy();xm=x.copy();xp[col]+=eps;xm[col]-=eps
            fd=(evaluate(xp)[0]-evaluate(xm)[0])/(2*eps)
            jac_errors.append(float(np.max(np.abs(fd-jac[:,col].toarray().ravel()))))
        assert max(jac_errors)<1e-4, (name,jac_errors)
        fit=least_squares(lambda x:evaluate(x)[0],x,jac=lambda x:evaluate(x)[1],
            bounds=(np.tile(lo,n),np.tile(hi,n)),max_nfev=args.global_nfev,
            ftol=1e-9,xtol=1e-9,gtol=1e-8,x_scale='jac',tr_solver='lsmr',
            tr_options=dict(atol=1e-8,btol=1e-8,maxiter=1000))
        curve=fit.x.reshape(n,7)
        saved[:-1,qa]=curve
        saved[-1,qa]=curve[0]
        residual=fit.fun.reshape(n,27)
        poserr=np.linalg.norm(residual[:,:3],axis=1)
        ori=[]
        for i in range(n):
            _,rr=fk.pose(curve[i],leg)
            ori.append(float(np.linalg.norm(Rotation.from_matrix(rot0.T@rr).as_rotvec())))
        ori=np.array(ori)
        log=dict(leg=name,global_success=bool(fit.success),nfev=int(fit.nfev),
            optimality=float(fit.optimality),position_max=float(poserr.max()),
            stance_orientation_max=float(ori[stance[:-1,leg]].max()),
            swing_orientation_max=float(ori[~stance[:-1,leg]].max()),
            analytic_jacobian_fd_max=max(jac_errors),forward_calls=fk.calls,
            wall_s=time.monotonic()-start)
        logs.append(log)
        print(json.dumps(log),flush=True)
    original_hinges=saved[:,fk.qa]
    pchip=periodic_pchip(phase,original_hinges)
    export_phase=np.linspace(0,2*np.pi,args.export_phases)
    hinges=pchip(export_phase)
    hinges[-1]=hinges[0]
    # Verify the EXACT exported interpolation that runtime must use.
    runtime_pchip=periodic_pchip(export_phase,hinges)
    dense=np.linspace(0,2*np.pi,2049)
    dense_q=runtime_pchip(dense)
    dense_goals,dense_stance=v1.desired_path(dense,fk.anchor_local,shape,.65,.15,.25)
    achieved=np.empty_like(dense_goals)
    oris=np.empty((len(dense),6))
    for i,q in enumerate(dense_q):
        fk.d.qpos[:]=fk.q0
        fk.d.qpos[fk.qa]=q
        fk.forward()
        achieved[i]=(fk.d.xpos[fk.bids]-fk.origin)@fk.rotation
        for leg in range(6):
            rot=fk.d.xmat[fk.bids[leg]].reshape(3,3)
            oris[i,leg]=np.linalg.norm(Rotation.from_matrix(fk.orientations[leg].T@rot).as_rotvec())
    error=np.linalg.norm(achieved-dense_goals,axis=2)
    lo=fk.m.jnt_range[fk.act_jids,0];hi=fk.m.jnt_range[fk.act_jids,1]
    excess=float(max(np.max(lo-dense_q),np.max(dense_q-hi),0.))
    # A cubic loader is diagnosed explicitly; it is never silently clipped.
    cubic=CubicSpline(export_phase,hinges,axis=0,bc_type='periodic')(dense)
    cubic_excess=float(max(np.max(lo-cubic),np.max(cubic-hi),0.))
    jump=float(np.abs(np.diff(hinges,axis=0)).max())
    stance_ori=float(oris[dense_stance].max())
    checks=dict(position_dense=float(error.max())<=.005,stance_orientation_dense=stance_ori<=.03,
        interpolated_joint_bounds=excess<=1e-9,periodic_endpoint=bool(np.array_equal(hinges[0],hinges[-1])),
        no_large_joint_branch_jump=jump<=.5,
        analytic_jacobian_validated=max(x['analytic_jacobian_fd_max'] for x in logs)<1e-4)
    metadata=dict(version='stance-only-orientation-v2',interpolation='periodic_pchip_extended',
        interpolation_recipe='Extend two knots from opposite ends by +/-2pi, PchipInterpolator(axis=0, extrapolate=False); query phase modulo2pi. NO CLIPPING.',
        cubic_compatible_at_bound_tolerance_1e9=cubic_excess<=1e-9,
        cubic_joint_bound_excess_rad=cubic_excess,legs=LEGS,parameters=parameters,
        qpos_order='native actuator order via actuator_trnid, NOT contact leg order',
        teacher_only=True,fk_feasible=bool(all(checks.values())),checks=checks,
        source_hashes=hashes,walking_success='NOT_ESTABLISHED',physics_steps=0)
    np.savez_compressed(OUT/'GAIT.npz',phase=export_phase,qpos_hinges=hinges,qpos_addresses=fk.qa,
        joint_names=np.array([fk.m.joint(int(j)).name for j in fk.act_jids]),
        metadata_json=np.array(json.dumps(metadata)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=dense_q,
        qpos_addresses=fk.qa,stance=dense_stance,target_local=dense_goals,achieved_local=achieved,
        position_error=error,orientation_error_rad=oris,anchors_local=fk.anchor_local,
        target_orientation=fk.orientations,solver_phase=phase,solver_qpos=saved,
        original_q0=fk.original_q0,anchor_q0=fk.q0,initial_projection=fk.initial_projection,
        thorax_origin=fk.origin,thorax_rotation=fk.rotation)
    v1.dump(OUT/'METADATA.json',metadata)
    report=dict(fk_feasible=bool(all(checks.values())),checks=checks,solver_logs=logs,
        physics_steps=0,neural_steps=0,forward_calls=fk.calls,
        errors=dict(position_max=float(error.max()),position_rms=float(np.sqrt(np.mean(error**2))),
            stance_orientation_max_rad=stance_ori,swing_orientation_max_rad=float(oris[~dense_stance].max()),
            pchip_joint_bound_excess=excess,cubic_joint_bound_excess=cubic_excess,
            joint_step_max=jump,per_leg_position_max=error.max(axis=0),
            per_leg_stance_orientation_max=[float(oris[dense_stance[:,j],j].max()) for j in range(6)]),
        model=dict(nq=fk.m.nq,nv=fk.m.nv,nu=fk.m.nu,nmesh=fk.m.nmesh,neq=fk.m.neq),
        source_hashes_unchanged={p:sha(p)==digest for p,digest in hashes.items()},
        limitations=prereg['limitations'],runtime_interpolation=metadata['interpolation_recipe'],
        wall_s=time.monotonic()-start)
    v1.dump(OUT/'VERIFICATION.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for j,leg in enumerate(LEGS):
        ax=axes.ravel()[j]
        ax.plot(dense/(2*np.pi),error[:,j],label='position error native')
        ax.plot(dense/(2*np.pi),oris[:,j],label='orientation error rad')
        ax.axhline(.005,color='C0',ls=':');ax.axhline(.03,color='C1',ls=':')
        ax.fill_between(dense/(2*np.pi),0,.1,where=dense_stance[:,j],alpha=.12,color='green')
        ax.set(title=leg,xlabel='Cycle fraction',ylim=(0,max(.1,float(oris[:,j].max())*1.05)))
        if j==0:ax.legend()
    fig.suptitle('V2 fixed-root FK only: stance shaded; swing orientation is free; physical walking NOT_ESTABLISHED')
    fig.savefig(OUT/'FK_RESIDUALS.png',dpi=150)
    plt.close(fig)
    text=(f"# V2 stance-oriented reference, FK only\n\nFK acceptable: **{all(checks.values())}**. Physics steps: **0**.\n\n"
        f"Dense position max: {error.max():.6g} native. Stance orientation max: {stance_ori:.6g} rad. "
        f"PCHIP bound excess: {excess:.6g}; cubic bound excess: {cubic_excess:.6g} rad. Max exported joint step: {jump:.6g} rad.\n\n"
        "Duty0.65, stance halfstride0.15 and swing clearance0.25 are unchanged. Swing orientation is free, except smooth transition weights near stance. "
        "Cyclic joint continuity and acceleration penalties are preregistered. No source or model is changed.\n\n"
        "Runtime must load the explicitly declared periodic PCHIP; do not silently retain cubic or clip overshoot. "
        "Only if FK gates pass should the next short algorithm-only physical screen assess unchanged D gates and exact sampled material contact slip. "
        "No physical success follows from this FK result; final deployment must be ANN.\n\n"
        + '\n'.join(f"- {k}: {v}" for k,v in checks.items())+'\n')
    (OUT/'REPORT.md').write_text(text)
    print(json.dumps(dict(output=str(OUT),fk_feasible=all(checks.values()),errors=report['errors']),
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':
    raise SystemExit(main())
