"""V3 FK-only reference: one bounded constant stance orientation per leg.

Original duty/stride/clearance and acceptance limits are unchanged. A constant
nearby teacher orientation is optimized jointly with the cyclic joint curve;
it is not per-phase tilt or runtime feedback. No mj_step or neural execution.
Prior failed directories are read-only inputs. Export requires periodic PCHIP.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/s5-ann-walk/20261107T075000Z/prepare_stance_reference.py'
PREVIOUS=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075000Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075536Z')
spec=importlib.util.spec_from_file_location('stance_v2_fk',OLD)
v2=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1=v2.v1
LEGS=v1.LEGS


def right_jacobian(r):
    angle=np.linalg.norm(r)
    h=v2.hat(r)
    if angle<1e-5:
        return np.eye(3)-(.5-angle*angle/24)*h+(1/6-angle*angle/120)*(h@h)
    return np.eye(3)-(1-np.cos(angle))/(angle*angle)*h+(angle-np.sin(angle))/(angle**3)*(h@h)


def exact_phase_rate_bounds(interp):
    """Analytic extrema of each cubic segment's quadratic first derivative."""
    maximum=np.zeros(interp.c.shape[2])
    for segment in range(len(interp.x)-1):
        left,right=interp.x[segment:segment+2]
        if left < -1e-12 or right > 2*np.pi+1e-12:
            continue
        a,b,c=interp.c[:3,segment]
        h=right-left
        values=np.maximum(np.abs(c),np.abs(3*a*h*h+2*b*h+c))
        interior=np.divide(-b,3*a,out=np.full_like(b,np.nan),where=np.abs(a)>1e-20)
        valid=(interior>0)&(interior<h)
        peak=np.abs(3*a*interior*interior+2*b*interior+c)
        maximum=np.maximum(maximum,np.where(valid,np.maximum(values,peak),values))
    return maximum


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solver-phases',type=int,default=129)
    ap.add_argument('--export-phases',type=int,default=257)
    ap.add_argument('--global-nfev',type=int,default=400)
    args=ap.parse_args(argv)
    assert args.solver_phases>=65 and args.export_phases>=args.solver_phases
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():
        (OUT/'PREREGISTRATION.json').read_bytes()
        raise FileExistsError('preserve prior evidence; use a new timestamp')
    sha=v1.sha
    sources=[Path(__file__),OLD,v1.ROOT/'experiments/s5-ann-walk/20261107T074700Z/prepare_stance_reference.py',
             PREVIOUS/'FK_EVIDENCE.npz',v1.SOURCE/'STANDARD_GAIT_FK.npz',
             v1.SOURCE/'REFERENCE_METADATA.json',v1.MODEL,v1.BASE/'incremental_actuator.py']
    hashes={str(p):sha(p) for p in sources}
    params=dict(duty=.65,halfstride=.15,clearance=.25,stance_orientation_weight=.5,
        swing_orientation_weight=0.,swing_transition_fraction=.15,continuity_weight=.01,
        second_difference_weight=.003,q0_weight=.001,constant_orientation_offset_weight=.002,
        constant_orientation_component_bounds_rad=[-.15,.15],**vars(args))
    prereg=dict(importance='核心必需',stage='S5 FK-only v3 constant perleg stance orientation',parameters=params,
        thresholds=dict(position_max_native=.005,stance_orientation_error_to_Rstar_max_rad=.03,
                        interpolation_bound_tolerance_rad=1e-9,joint_step_max_rad=.5),
        teacher_target='Rstar_leg = Rq0_leg @ Exp(one constant rotation vector); each vector component bounded +/-0.15rad. Actual vector and angle norm reported.',
        solve='Warmstart v2 cyclic curve resampled to128unique phases by bounded periodic PCHIP; optimize allphase joints and3constant orientation parameters jointly perleg, max400nfev; no arbitrary extra cycles.',
        fixed='Original free root/q0anchor y/z, duty.65, stancehalfstride.15, clearance.25, constant backwards stancex slope unchanged.',
        interpolation='exact two-neighbor periodic PCHIP extension; dense2049 FK verification of exported257-knot asset; no clipping',
        workspace_if_failure=dict(rearward_offsets_native=[-.15,-.14,-.125,-.10,-.075,-.05],
            seeds=['optimized late-stance pose','q0','native-bound midpoint'],max_nfev=300,
            target_orientation='hold each optimized Rstar fixed; no perpoint orientation refit',
            purpose='diagnose bounded workspace, choose next shorterstride from measured evidence; not another fullcycle'),
        counts=dict(physics_steps=0,neural_steps=0),source_hashes=hashes,
        deployment='Algorithm training teacher only; physics and final ANN deployment are separate. No root physics trajectory or posthandover teacher.',
        limitations=['FK and near-constant target orientation do not establish real support/material slip/locomotion.',
                     'Absolute orientation target changes are declared engineering parameters, not biological calibration.',
                     'Prior failures and frozen verifier/driver remain untouched.'])
    v1.dump(OUT/'PREREGISTRATION.json',prereg)
    v1.dump(OUT/'SOURCE_SHA.json',hashes)
    shape,source_info=v1.source_profile()
    v1.dump(OUT/'SOURCE_SWING_RESAMPLING.json',source_info)
    fk=v1.FK()
    with np.load(PREVIOUS/'FK_EVIDENCE.npz') as z:
        oldphase=z['solver_phase'].copy();oldq=z['solver_qpos'].copy()
    seed_curve=v2.periodic_pchip(oldphase,oldq)
    phase=np.linspace(0,2*np.pi,args.solver_phases)
    n=len(phase)-1
    goals_local,stance=v1.desired_path(phase,fk.anchor_local,shape,.65,.15,.25)
    goals_world=goals_local@fk.rotation.T+fk.origin
    u=(phase[:-1,None]/(2*np.pi)+v1.OFFSETS[None])%1.
    swing=np.clip((u-.65)/.35,0,1)
    transition=np.minimum(swing/.15,(1-swing)/.15)
    ow=np.where(stance[:-1],.5,.5*(1-v1.s5(np.clip(transition,0,1))))
    saved=np.repeat(fk.q0[None],len(phase),axis=0)
    offsets=np.zeros((6,3));target_rotations=fk.orientations.copy()
    logs=[];start=time.monotonic()
    for leg,name in enumerate(LEGS):
        qa=fk.leg_qa[leg];va=fk.leg_va[leg]
        q0=fk.q0[qa];lo,hi=fk.limits[leg,:,0],fk.limits[leg,:,1]
        rot0=fk.orientations[leg]
        curve=np.clip(seed_curve(phase[:-1])[:,qa],lo+1e-10,hi-1e-10)
        cache={}
        def evaluate(flat):
            if 'x' in cache and np.array_equal(cache['x'],flat):
                return cache['res'],cache['jac']
            q=flat[:-3].reshape(n,7);offset=flat[-3:]
            targetrot=rot0@Rotation.from_rotvec(offset).as_matrix()
            right=right_jacobian(offset)
            residual=np.empty((n,27));rowids=[];colids=[];values=[]
            for i in range(n):
                pos,rot=fk.pose(q[i],leg)
                rv=Rotation.from_matrix(targetrot.T@rot).as_rotvec()
                residual[i,:3]=pos-goals_world[i,leg]
                residual[i,3:6]=ow[i,leg]*rv
                residual[i,6:13]=.001*(q[i]-q0)
                residual[i,13:20]=.01*(q[i]-q[(i-1)%n])
                residual[i,20:27]=.003*(q[(i-1)%n]-2*q[i]+q[(i+1)%n])
                jp=np.zeros((3,fk.m.nv));jr=np.zeros_like(jp)
                fk.mj.mj_jac(fk.m,fk.d,jp,jr,fk.d.xpos[fk.bids[leg]],fk.bids[leg])
                inv=v2.log_left_inverse(rv)
                orient=ow[i,leg]*inv@targetrot.T@jr[:,va]
                block=np.vstack((jp[:,va],orient))
                rr,cc=np.indices((6,7))
                rowids.extend((i*27+rr.ravel()).tolist());colids.extend((i*7+cc.ravel()).tolist());values.extend(block.ravel().tolist())
                dr=-ow[i,leg]*inv@right
                rr,cc=np.indices((3,3))
                rowids.extend((i*27+3+rr.ravel()).tolist());colids.extend((n*7+cc.ravel()).tolist());values.extend(dr.ravel().tolist())
                for j in range(7):
                    for r,col,val in [(6+j,i*7+j,.001),(13+j,i*7+j,.01),
                        (13+j,((i-1)%n)*7+j,-.01),(20+j,((i-1)%n)*7+j,.003),
                        (20+j,i*7+j,-.006),(20+j,((i+1)%n)*7+j,.003)]:
                        rowids.append(i*27+r);colids.append(col);values.append(val)
            for j in range(3):
                rowids.append(n*27+j);colids.append(n*7+j);values.append(.002)
            jac=coo_matrix((values,(rowids,colids)),shape=(n*27+3,n*7+3)).tocsr()
            res=np.r_[residual.ravel(),.002*offset]
            cache.update(x=flat.copy(),res=res,jac=jac)
            return res,jac
        x=np.r_[curve.ravel(),np.zeros(3)]
        value,jac=evaluate(x)
        jac_errors=[]
        for col in [7*(n//3)+2,7*(2*n//3)+5,n*7,n*7+1,n*7+2]:
            eps=1e-6;xp=x.copy();xm=x.copy();xp[col]+=eps;xm[col]-=eps
            fd=(evaluate(xp)[0]-evaluate(xm)[0])/(2*eps)
            jac_errors.append(float(np.max(np.abs(fd-jac[:,col].toarray().ravel()))))
        assert max(jac_errors)<1e-4,(name,jac_errors)
        fit=least_squares(lambda x:evaluate(x)[0],x,jac=lambda x:evaluate(x)[1],
            bounds=(np.r_[np.tile(lo,n),np.full(3,-.15)],np.r_[np.tile(hi,n),np.full(3,.15)]),
            max_nfev=args.global_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-8,x_scale='jac',
            tr_solver='lsmr',tr_options=dict(atol=1e-8,btol=1e-8,maxiter=1500))
        curve=fit.x[:-3].reshape(n,7)
        saved[:-1,qa]=curve;saved[-1,qa]=curve[0]
        offsets[leg]=fit.x[-3:]
        target_rotations[leg]=rot0@Rotation.from_rotvec(offsets[leg]).as_matrix()
        pe=[];oe=[]
        for i in range(n):
            pos,rot=fk.pose(curve[i],leg)
            pe.append(np.linalg.norm(pos-goals_world[i,leg]))
            oe.append(np.linalg.norm(Rotation.from_matrix(target_rotations[leg].T@rot).as_rotvec()))
        pe=np.array(pe);oe=np.array(oe)
        log=dict(leg=name,solver_success=bool(fit.success),nfev=int(fit.nfev),optimality=float(fit.optimality),
            position_max=float(pe.max()),stance_orientation_max=float(oe[stance[:-1,leg]].max()),
            target_offset_rotvec_rad=offsets[leg],target_offset_angle_rad=float(np.linalg.norm(offsets[leg])),
            jacobian_fd_max=max(jac_errors),forward_calls=fk.calls,wall_s=time.monotonic()-start)
        logs.append(log)
        print(json.dumps(log,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    source_interp=v2.periodic_pchip(phase,saved[:,fk.qa])
    export_phase=np.linspace(0,2*np.pi,args.export_phases)
    hinges=source_interp(export_phase);hinges[-1]=hinges[0]
    runtime_interp=v2.periodic_pchip(export_phase,hinges)
    dense=np.linspace(0,2*np.pi,2049);dense_q=runtime_interp(dense)
    dense_goals,dense_stance=v1.desired_path(dense,fk.anchor_local,shape,.65,.15,.25)
    achieved=np.empty_like(dense_goals);orientation_error=np.empty((len(dense),6));actual_rotations=np.empty((len(dense),6,3,3))
    for i,q in enumerate(dense_q):
        fk.d.qpos[:]=fk.q0;fk.d.qpos[fk.qa]=q;fk.forward()
        achieved[i]=(fk.d.xpos[fk.bids]-fk.origin)@fk.rotation
        for leg in range(6):
            rot=fk.d.xmat[fk.bids[leg]].reshape(3,3)
            actual_rotations[i,leg]=rot
            orientation_error[i,leg]=np.linalg.norm(Rotation.from_matrix(target_rotations[leg].T@rot).as_rotvec())
    error=np.linalg.norm(achieved-dense_goals,axis=2)
    lo=fk.m.jnt_range[fk.act_jids,0];hi=fk.m.jnt_range[fk.act_jids,1]
    excess=float(max(np.max(lo-dense_q),np.max(dense_q-hi),0.))
    cubic=CubicSpline(export_phase,hinges,axis=0,bc_type='periodic')(dense)
    cubic_excess=float(max(np.max(lo-cubic),np.max(cubic-hi),0.))
    jump=float(np.abs(np.diff(hinges,axis=0)).max())
    seam=float(np.max(np.abs(runtime_interp(0,1)-runtime_interp(2*np.pi,1))))
    phase_rate_bounds=exact_phase_rate_bounds(runtime_interp)
    rate_bounds_1hz=2*np.pi*phase_rate_bounds
    rate_info=dict(max_rad_s_at_1hz=float(rate_bounds_1hz.max()),
        max_rad_s_at_3hz=float(3*rate_bounds_1hz.max()),
        per_joint_max_rad_s_at_1hz=rate_bounds_1hz.tolist(),
        worst_joint=fk.m.joint(int(fk.act_jids[int(rate_bounds_1hz.argmax())])).name,
        fits_30rad_s_cap_at_1hz=bool(rate_bounds_1hz.max()<=30.),
        method='Exact cubic-segment derivative extrema including interior quadratic vertices, evaluated over0..2pi; frequency conversion 2pi*f.')
    stance_error=float(orientation_error[dense_stance].max())
    checks=dict(position_dense=float(error.max())<=.005,stance_orientation_dense=stance_error<=.03,
        orientation_target_components_bounded=bool(np.abs(offsets).max()<=.15+1e-12),
        interpolated_joint_bounds=excess<=1e-9,periodic_endpoint=bool(np.array_equal(hinges[0],hinges[-1])),
        periodic_derivative_seam=seam<=1e-10,no_large_joint_branch_jump=jump<=.5,
        analytic_jacobian_validated=max(log['jacobian_fd_max'] for log in logs)<1e-4)
    metadata=dict(version='constant-stance-orientation-v3',interpolation='periodic_pchip_extended',
        interpolation_recipe='two knots extended from opposite ends by +/-2pi; PchipInterpolator axis0 extrapolateFalse; phase modulo2pi; no clipping',
        parameters=params,legs=LEGS,orientation_target_rotvec_offsets_rad=offsets.tolist(),
        orientation_target_angle_offsets_rad=np.linalg.norm(offsets,axis=1).tolist(),
        target_orientation_world=target_rotations.tolist(),teacher_only=True,
        source_hashes=hashes,checks=checks,fk_feasible=bool(all(checks.values())),
        reference_rate_bounds=rate_info,walking_success='NOT_ESTABLISHED',physics_steps=0)
    np.savez_compressed(OUT/'GAIT.npz',phase=export_phase,qpos_hinges=hinges,qpos_addresses=fk.qa,
        joint_names=np.array([fk.m.joint(int(j)).name for j in fk.act_jids]),metadata_json=np.array(json.dumps(metadata)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=dense_q,qpos_addresses=fk.qa,
        stance=dense_stance,target_local=dense_goals,achieved_local=achieved,position_error=error,
        orientation_error_rad=orientation_error,achieved_orientation_world=actual_rotations,
        q0_orientation_world=fk.orientations,target_orientation_world=target_rotations,
        constant_orientation_offsets_rad=offsets,anchors_local=fk.anchor_local,
        solver_phase=phase,solver_qpos=saved,original_q0=fk.original_q0,anchor_q0=fk.q0,
        initial_projection=fk.initial_projection,thorax_origin=fk.origin,thorax_rotation=fk.rotation)
    v1.dump(OUT/'METADATA.json',metadata)
    report=dict(fk_feasible=bool(all(checks.values())),checks=checks,solver_logs=logs,
        physics_steps=0,neural_steps=0,forward_calls=fk.calls,
        errors=dict(position_max=float(error.max()),position_rms=float(np.sqrt(np.mean(error**2))),
            stance_orientation_max_rad=stance_error,pchip_joint_bound_excess=excess,
            cubic_joint_bound_excess=cubic_excess,joint_step_max_rad=jump,derivative_seam=seam,
            per_leg_position_max=error.max(axis=0),
            per_leg_stance_orientation_max=[float(orientation_error[dense_stance[:,j],j].max()) for j in range(6)]),
        constant_orientation_offsets_rad=offsets,constant_orientation_angles_rad=np.linalg.norm(offsets,axis=1),
        reference_rate_bounds=rate_info,
        model=dict(nq=fk.m.nq,nv=fk.m.nv,nu=fk.m.nu,nmesh=fk.m.nmesh,neq=fk.m.neq,
                   gravity=fk.m.opt.gravity,timestep=fk.m.opt.timestep),
        source_hashes_unchanged={p:sha(p)==digest for p,digest in hashes.items()},
        limitations=prereg['limitations'],wall_s=time.monotonic()-start)
    v1.dump(OUT/'VERIFICATION.json',report)
    if not all(checks.values()):
        diagnostics=[]
        # Only fixed-orientation workspace points after failure; no extra cycles.
        for leg,name in enumerate(LEGS):
            if error[:,leg].max()<=.005 and orientation_error[dense_stance[:,leg],leg].max()<=.03:
                continue
            qa=fk.leg_qa[leg];q0=fk.q0[qa]
            low,high=fk.limits[leg,:,0],fk.limits[leg,:,1]
            near=np.argmin(np.linalg.norm(goals_local[:-1,leg]-(fk.anchor_local[leg]+[-.15,0,0]),axis=1))
            base_seeds=[saved[near,qa],q0,(low+high)/2]
            for dx in [-.15,-.14,-.125,-.10,-.075,-.05]:
                local=fk.anchor_local[leg]+[dx,0,0]
                point=local@fk.rotation.T+fk.origin
                candidates=[]
                for seed in base_seeds:
                    def residual(q):
                        pos,rot=fk.pose(q,leg)
                        rv=Rotation.from_matrix(target_rotations[leg].T@rot).as_rotvec()
                        return np.r_[pos-point,.5*rv,1e-5*(q-q0)]
                    fit=least_squares(residual,np.clip(seed,low+1e-10,high-1e-10),bounds=(low,high),
                        max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-9,x_scale='jac')
                    pos,rot=fk.pose(fit.x,leg)
                    pe=float(np.linalg.norm(pos-point))
                    oe=float(np.linalg.norm(Rotation.from_matrix(target_rotations[leg].T@rot).as_rotvec()))
                    candidates.append(dict(position_error=pe,orientation_error_rad=oe,
                        accepted=pe<=.005 and oe<=.03,q=fit.x,solver_success=bool(fit.success),cost=float(fit.cost)))
                acceptable=[c for c in candidates if c['accepted']]
                best=min(acceptable or candidates,key=lambda c:c['cost'])
                diagnostics.append(dict(leg=name,backward_offset=dx,**best))
        v1.dump(OUT/'STANCE_WORKSPACE.json',dict(target_orientation='each fixed optimized Rstar, unchanged across offsets',
            points=diagnostics,acceptance=dict(position=.005,orientation=.03),physics_steps=0,
            caveat='Point feasibility does not establish a continuous full gait or forward-endpoint feasibility.'))
        print(json.dumps(dict(workspace=[dict(leg=d['leg'],dx=d['backward_offset'],passed=d['accepted'],pe=d['position_error'],oe=d['orientation_error_rad']) for d in diagnostics])),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for j,leg in enumerate(LEGS):
        ax=axes.ravel()[j]
        ax.plot(dense/(2*np.pi),error[:,j],label='position error native')
        ax.plot(dense/(2*np.pi),orientation_error[:,j],label='orientation error to fixed Rstar rad')
        ax.axhline(.005,color='C0',ls=':');ax.axhline(.03,color='C1',ls=':')
        ax.fill_between(dense/(2*np.pi),0,.1,where=dense_stance[:,j],alpha=.12,color='green')
        ax.set(title=f'{leg}, constant target offset {np.linalg.norm(offsets[j]):.4f}rad',xlabel='Cycle fraction',
            ylim=(0,max(.1,float(orientation_error[:,j].max())*1.05)))
        if j==0:ax.legend(fontsize=8)
    fig.suptitle('V3 fixed-root FK only; constant nearby stance orientation; physical walking NOT_ESTABLISHED')
    fig.savefig(OUT/'FK_RESIDUALS.png',dpi=150);plt.close(fig)
    text=(f"# V3 constant stance-orientation reference, FK only\n\nFK acceptable: **{all(checks.values())}**. Physics/neural steps: **0 / 0**.\n\n"
        f"Dense position max {error.max():.6g} native; stance orientation error to constant Rstar max {stance_error:.6g}rad. "
        f"PCHIP bound excess {excess:.6g}; cubic bound excess {cubic_excess:.6g}rad.\n\n"
        "One constant rotation-vector offset per leg was optimized jointly across all phases; components remain within +/-0.15rad of q0. "
        "Actual vectors and angle norms are in METADATA.json and VERIFICATION.json. Duty0.65, stance halfstride0.15, swing clearance0.25, fixed stance y/z and backward slope are unchanged.\n\n"
        "Runtime must use the exported periodic PCHIP. No FK result establishes contact support/slip or physical walking. Final deployment must be an ANN.\n\n"
        + '\n'.join(f"- {key}: {value}" for key,value in checks.items())+'\n')
    (OUT/'REPORT.md').write_text(text)
    print(json.dumps(dict(output=str(OUT),fk_feasible=all(checks.values()),errors=report['errors'],
        constant_target_angle_offsets=np.linalg.norm(offsets,axis=1)),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':
    raise SystemExit(main())
