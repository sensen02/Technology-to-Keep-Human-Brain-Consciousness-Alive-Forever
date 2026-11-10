"""Fixed-lambda bounded smoothing of sole IK corrections; FK only, no mj_step.

Minimize ||delta-delta_raw||^2+100||periodic_D2 delta||^2 per joint.
Only when filtered GEOMETRY fails, allow one max30eval local pose reprojection
with0.01 filtered-joint prior. No parameter sweep, extra cycles or old-file edits.
"""
from pathlib import Path
import importlib.util,json,sys,time
import numpy as np
from scipy.optimize import lsq_linear,least_squares
from scipy.sparse import eye,diags,vstack
from scipy.spatial.transform import Rotation
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
RAW=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T082451Z')
WAVE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T083316Z')
UTIL=ROOT/'experiments/s5-ann-walk/20261107T075536Z/prepare_stance_reference.py'
LEGS=('LF','LM','LH','RF','RM','RH')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    sys.path.insert(0,str(BASE))
    import mujoco as mj
    from plant_probe import Plant,sha
    from incremental_actuator import DEFAULT_MODEL_PATH
    spec=importlib.util.spec_from_file_location('frozen_smooth_util',UTIL)
    util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util);dump=util.v1.dump
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():
        (OUT/'PREREGISTRATION.json').read_bytes();raise FileExistsError('preserve prior candidate')
    inputs=[Path(__file__),UTIL,RAW/'GAIT.npz',RAW/'SOLE_MEASUREMENT.json',WAVE/'GAIT.npz',DEFAULT_MODEL_PATH,BASE/'incremental_actuator.py',BASE/'plant_probe.py']
    hashes={str(p):sha(p) for p in inputs}
    config=dict(stage='S5 bounded sole-correction nullspace smoothing, FK only',source_hashes=hashes,
        smoothing=dict(lambda_fixed=100,objective='||delta-delta_raw||^2+100||D2_periodic delta||^2',
            unique_phases=512,source='081009 wave',raw='082451 sole IK',bounds='native_lo-source_q <=delta<= native_hi-source_q',RH='exact zero correction'),
        optional_reprojection=dict(trigger='filtered geometry fails position/orientation/sole-spread only',passes=1,max_nfev=30,
            pose_weights=[1,.2],filtered_joint_prior=.01,warmstart='filtered joints',no_raw_joint_prior=True),
        thresholds=dict(position=.005,orientation_rad=.03,sole_median_spread=.005,native_bound_excess=1e-9,
            periodic_derivative_seam=1e-10,rate_1hz=30,joint_step=.5),
        interpolation='513knots closed, exact two-neighbor periodic PCHIP; no clipping',
        no_search='lambda100 fixed, at most one reprojection, no extra whole cycles',physics_steps=0,neural_steps=0,
        limitation='Geometric teacher candidate only; no physical support/slip/locomotion or neural success follows.')
    dump(OUT/'PREREGISTRATION.json',config);dump(OUT/'SOURCE_SHA.json',hashes)
    source=load(WAVE/'GAIT.npz');raw=load(RAW/'GAIT.npz');meta=json.loads(str(raw['metadata_json'].item()))
    assert np.array_equal(source['phase'],raw['phase']) and len(source['phase'])==513
    phase=source['phase'];qsource=source['qpos_hinges'];qraw=raw['qpos_hinges'];n=512
    names=source['joint_names'].astype(str);ids=[np.flatnonzero(np.char.startswith(names,'joint_'+leg)) for leg in LEGS]
    assert all(len(i)==7 for i in ids)
    p=Plant();m=p.m;data=mj.MjData(m);assert np.array_equal(p.qa,source['qpos_addresses'])
    lo=m.jnt_range[p.jids,0];hi=m.jnt_range[p.jids,1];q0=p.q0.copy();q0[p.qa]=np.clip(q0[p.qa],lo,hi)
    d_raw=qraw[:n]-qsource[:n];assert np.abs(d_raw[:,ids[5]]).max()<1e-12
    D=diags([np.ones(n-1),-2*np.ones(n),np.ones(n-1)],[-1,0,1],shape=(n,n),format='lil');D[0,n-1]=1;D[n-1,0]=1;D=D.tocsr()
    A=vstack([eye(n,format='csr'),10*D],format='csr');filtered=np.zeros_like(d_raw);logs=[]
    begin=time.monotonic()
    for j in range(42):
        if j in ids[5]:
            logs.append(dict(joint=str(names[j]),bypass_zero=True));continue
        lower=lo[j]-qsource[:n,j];upper=hi[j]-qsource[:n,j]
        result=lsq_linear(A,np.r_[d_raw[:,j],np.zeros(n)],bounds=(lower,upper),
            tol=1e-9,lsq_solver='lsmr',lsmr_tol=1e-9,max_iter=100,lsmr_maxiter=2000)
        filtered[:,j]=result.x
        logs.append(dict(joint=str(names[j]),success=bool(result.success),status=int(result.status),
            iterations=int(result.nit),optimality=float(result.optimality),cost=float(result.cost),
            active_bounds=int(np.sum(result.active_mask!=0)),max_bound_violation=float(max(np.max(lower-result.x),np.max(result.x-upper),0.))))
    filtered=np.vstack((filtered,filtered[0]));qfiltered=qsource+filtered;qfiltered[-1]=qfiltered[0]
    dump(OUT/'SMOOTH_SOLVERS.json',logs)
    calls=0
    def fk(q):
        nonlocal calls
        data.qpos[:]=q0;data.qpos[p.qa]=q;mj.mj_forward(m,data);calls+=1
        assert data.time==0 and np.array_equal(data.qpos[:7],p.q0[:7])
    plane=int(np.flatnonzero(p.planes)[0]);gg=[np.flatnonzero(p.feet==j) for j in range(6)];ft=np.zeros(6)
    def soles():return np.array([min(mj.mj_geomDistance(m,data,int(g),plane,1000.,ft) for g in gg[j]) for j in range(6)])
    dense=np.linspace(0,2*np.pi,4097);oldinterp=util.v2.periodic_pchip(phase,qsource)
    old_dense=oldinterp(dense);pos=np.empty((4097,6,3));rot=np.empty((4097,6,3,3));sole_before=np.empty((4097,6))
    for i,q in enumerate(old_dense):
        fk(q);pos[i]=data.xpos[p.bids];rot[i]=data.xmat[p.bids].reshape(6,3,3);sole_before[i]=soles()
    measurement=json.loads((RAW/'SOLE_MEASUREMENT.json').read_text());offset=np.asarray(measurement['body_Z_offsets']);axis=np.asarray(measurement['body_Z_world_direction'])
    targets=pos+offset[None,:,None]*axis[None,None]
    stance=((dense[:,None]/(2*np.pi)+np.asarray(meta['phase_offsets']))%1)<meta['parameters']['duty']
    def evaluate(qknots):
        curve=util.v2.periodic_pchip(phase,qknots);dq=curve(dense)
        actual=np.empty_like(pos);actual_rot=np.empty_like(rot);sd=np.empty_like(sole_before)
        for i,q in enumerate(dq):
            fk(q);actual[i]=data.xpos[p.bids];actual_rot[i]=data.xmat[p.bids].reshape(6,3,3);sd[i]=soles()
        pe=np.linalg.norm(actual-targets,axis=2);oe=np.empty((4097,6))
        for j in range(6):oe[:,j]=np.linalg.norm(Rotation.from_matrix(np.swapaxes(rot[:,j],1,2)@actual_rot[:,j]).as_rotvec(),axis=1)
        med=np.array([np.median(sd[stance[:,j],j]) for j in range(6)])
        rate=2*np.pi*util.exact_phase_rate_bounds(curve);seam=float(abs(curve(0,1)-curve(2*np.pi,1)).max())
        excess=float(max(np.max(lo-dq),np.max(dq-hi),0.));jump=float(abs(np.diff(qknots,axis=0)).max())
        checks=dict(position=float(pe.max())<=.005,orientation=float(oe.max())<=.03,sole_spread=float(np.ptp(med))<=.005,
            bounds=excess<=1e-9,seam=seam<=1e-10,rate=float(rate.max())<=30.,joint_step=jump<.5,
            endpoint=bool(np.array_equal(qknots[0],qknots[-1])))
        info=dict(passed=bool(all(checks.values())),checks=checks,position_max=float(pe.max()),orientation_max=float(oe.max()),
            sole_medians=med,sole_spread=float(np.ptp(med)),bounds_excess=excess,seam=seam,max_rate_1hz=float(rate.max()),
            rates_1hz=rate,worst_rate_joint=str(names[int(rate.argmax())]),joint_step=jump,
            sole_min=[float(sd[stance[:,j],j].min()) for j in range(6)],sole_max=[float(sd[stance[:,j],j].max()) for j in range(6)],
            sole_within_leg_range=[float(np.ptp(sd[stance[:,j],j])) for j in range(6)])
        return info,dict(q=dq,actual=actual,rotation=actual_rot,sole=sd,position_error=pe,orientation_error=oe)
    first,first_data=evaluate(qfiltered);dump(OUT/'FILTERED_VERIFICATION.json',first)
    print(json.dumps(dict(filtered=first),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    need=not all(first['checks'][key] for key in ('position','orientation','sole_spread'))
    final=qfiltered.copy();projection=[]
    if need:
        # Exactly ONE pass, each seed and regularizer anchored to the filtered q.
        for j in range(6):
            if j==5:continue
            ii=ids[j];qa=p.qa[ii];va=p.va[ii]
            successes=0;max_evals=0
            for i in range(n):
                reference=qfiltered[i,ii];target=targets[i*8,j];Rtarget=rot[i*8,j];cache={}
                def residual(q):
                    if 'q' in cache and np.array_equal(cache['q'],q):return cache['r'],cache['J']
                    full=qfiltered[i].copy();full[ii]=q;fk(full)
                    point=data.xpos[p.bids[j]];rvec=Rotation.from_matrix(Rtarget.T@data.xmat[p.bids[j]].reshape(3,3)).as_rotvec()
                    r=np.r_[point-target,.2*rvec,.01*(q-reference)]
                    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,data,jp,jr,point,p.bids[j])
                    J=np.vstack((jp[:,va],.2*util.v2.log_left_inverse(rvec)@Rtarget.T@jr[:,va],.01*np.eye(7)))
                    cache.update(q=q.copy(),r=r,J=J);return r,J
                fit=least_squares(lambda q:residual(q)[0],np.clip(reference,lo[ii]+1e-12,hi[ii]-1e-12),
                    jac=lambda q:residual(q)[1],bounds=(lo[ii],hi[ii]),max_nfev=30,ftol=1e-9,xtol=1e-9,gtol=1e-8,x_scale='jac')
                final[i,ii]=fit.x;successes+=bool(fit.success);max_evals=max(max_evals,int(fit.nfev))
            final[-1,ii]=final[0,ii];projection.append(dict(leg=LEGS[j],solves=n,successes=successes,max_nfev=max_evals))
        result,evidence=evaluate(final)
    else:result,evidence=first,first_data
    # Precise phase of the exact piecewise-cubic rate extremum.
    curve=util.v2.periodic_pchip(phase,final);peak=0.;location=None
    for segment in range(len(curve.x)-1):
        left,right=curve.x[segment:segment+2]
        if left<-1e-12 or right>2*np.pi+1e-12:continue
        a,b,c=curve.c[:3,segment];h=right-left
        for j in range(42):
            candidates=[0.,h]
            if abs(a[j])>1e-20:
                v=-b[j]/(3*a[j])
                if 0<v<h:candidates.append(v)
            for x in candidates:
                speed=abs(3*a[j]*x*x+2*b[j]*x+c[j])*2*np.pi
                if speed>peak:peak=float(speed);location=dict(joint=str(names[j]),phase_rad=float(left+x),global_cycle=float((left+x)/(2*np.pi)))
    meta.update(version='sole-correction-smoothed-v1',source=str(WAVE/'GAIT.npz'),raw_correction_source=str(RAW/'GAIT.npz'),
        smoothing_lambda=100,optional_reprojection_performed=need,fk_feasible=result['passed'],checks=result['checks'],
        max_rate_at_1hz=result['max_rate_1hz'],rates_at_1hz=result['rates_1hz'].tolist(),
        dense_sole_medians_after=result['sole_medians'].tolist(),sole_spread_after=result['sole_spread'],
        teacher_only=True,physics_steps=0)
    np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=final,qpos_addresses=p.qa,joint_names=names,metadata_json=np.array(json.dumps(meta)))
    np.savez_compressed(OUT/'CORRECTIONS.npz',phase=phase,delta_raw=np.vstack((d_raw,d_raw[0])),delta_smoothed=filtered,
        delta_final=final-qsource,projection_delta=final-qfiltered,source_q=qsource,filtered_q=qfiltered,
        delta_lower=lo-qsource,delta_upper=hi-qsource)
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=evidence['q'],source_qpos_hinges=old_dense,
        nominal_stance=stance,target_world=targets,source_position_world=pos,achieved_world=evidence['actual'],
        source_orientation_world=rot,achieved_orientation_world=evidence['rotation'],position_error=evidence['position_error'],
        orientation_error_rad=evidence['orientation_error'],sole_before=sole_before,sole_after=evidence['sole'],
        body_Z_offsets=offset,body_Z_world_direction=axis,original_q0=p.q0,anchor_q0=q0)
    result.update(optional_reprojection_performed=need,reprojection_solver_status=projection,exact_rate_peak=location,
        FK_calls=calls,physics_steps=0,neural_steps=0,source_hashes_unchanged={p:sha(p)==h for p,h in hashes.items()},
        wall_s=time.monotonic()-begin,limitation=config['limitation'])
    dump(OUT/'VERIFICATION.json',result);dump(OUT/'METADATA.json',meta)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True,constrained_layout=True)
    for j,leg in enumerate(LEGS):
        ax=axes.ravel()[j];mask=stance[:,j]
        ax.plot(dense[mask]/(2*np.pi),sole_before[mask,j],'.',ms=1,label='source wave')
        ax.plot(dense[mask]/(2*np.pi),evidence['sole'][mask,j],'.',ms=1,label='smoothed sole correction')
        ax.axhline(measurement['commonplane'],color='k',ls=':');ax.axhline(0,color='r',lw=.5)
        ax.set(title=leg,ylabel='True wholefoot sole-plane distance',xlabel='Cycle fraction')
        if j==0:ax.legend(fontsize=8)
    fig.suptitle('Fixed lambda100 source-relative smoothing; FK only; physical walking NOT_ESTABLISHED')
    fig.savefig(OUT/'SOLE_PLANES.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(final=result),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
