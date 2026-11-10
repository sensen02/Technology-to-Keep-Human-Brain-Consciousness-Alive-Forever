"""Equalize wave-foot true geometric sole planes by small fixed per-leg raises.

Only FK/IK and mj_geomDistance are used; no mj_step or neural execution. Preserve
source foot x/y and FULL per-phase orientation, including swing. The fixed offsets
are measured from whole-foot tarsal-geometry distance, not Tarsus5 origin height.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T082451Z')
UTIL=ROOT/'experiments/s5-ann-walk/20261107T075536Z/prepare_stance_reference.py'
LEGS=('LF','LM','LH','RF','RM','RH')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-nfev',type=int,default=100)
    args=ap.parse_args(argv)
    sys.path.insert(0,str(BASE))
    import mujoco as mj
    from plant_probe import Plant,sha
    from incremental_actuator import DEFAULT_MODEL_PATH
    spec=importlib.util.spec_from_file_location('frozen_sole_utils',UTIL)
    util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():
        (OUT/'PREREGISTRATION.json').read_bytes();raise FileExistsError('preserve prior sole-preparation evidence')
    files=[Path(__file__),UTIL,SOURCE/'GAIT.npz',SOURCE/'METADATA.json',DEFAULT_MODEL_PATH,BASE/'incremental_actuator.py',BASE/'plant_probe.py']
    hashes={str(f):sha(f) for f in files}
    prereg=dict(importance='核心必需',stage='S5 whole-foot sole-plane equalization, FK/IK only',
        source=str(SOURCE/'GAIT.npz'),source_hashes=hashes,
        measurement='At original q0 free root, perphase minimum signed mj_geomDistance between every leg Tarsal geom (Plant.feet==leg) and ground plane; median over nominalstance.',
        target='commonplane=max(six stance medians); each leg raised by fixed delta=commonplane-median in q0Thorax body-Z. Source x/y and complete perphase foot orientation preserved.',
        bounds=dict(offset_nonnegative=True,maximum_offset_native=.02),
        IK=dict(by_name_7DOF=True,position_weight=1.,orientation_weight=.2,source_joint_regularizer=.0001,max_nfev=args.max_nfev,
                warmstart='same-phase source7jointvector',zero_offset_leg='exact source bypass',extra_cycles=0),
        interpolation='same513knots, exact two-neighbor periodic PCHIP; no reference clipping',
        unchanged='wave duty5/6, sixphaseoffsets, x/y paths, source geometry orientations, fullmodel71mesh and original free-root initialization',
        thresholds=dict(dense_position_translation_error=.005,dense_orientation_relative_source_rad=.03,
            joint_bound_excess_rad=1e-9,derivative_seam=1e-10,max_rate_1hz=30.,max_joint_step_rad=.5,
            interleg_stance_median_sole_spread=.005),
        reporting='Recompute actual whole-foot stance sole-plane spread after correction; never assume offsets achieve exact alignment.',
        counts=dict(physics_steps=0,neural_steps=0),
        limitations=['Pure FK cannot establish support, material slip, actual unloading, stability or walking.',
                     'Fixed-root FK is an offline geometric diagnostic; no root overwrite or auxiliary force is proposed for physical deployment.',
                     'Sole distances are whole-foot collision geometry measurements, not biological foot calibration. Final deployment requires ANN.'])
    dump=util.v1.dump
    dump(OUT/'PREREGISTRATION.json',prereg);dump(OUT/'SOURCE_SHA.json',hashes)
    source=load(SOURCE/'GAIT.npz');metadata=json.loads(str(source['metadata_json'].item()))
    phase=source['phase'];oldq=source['qpos_hinges'];names=source['joint_names'].astype(str)
    assert len(phase)==513 and np.array_equal(oldq[0],oldq[-1])
    oldinterp=util.v2.periodic_pchip(phase,oldq)
    p=Plant();m=p.m;data=mj.MjData(m)
    assert np.array_equal(p.qa,source['qpos_addresses'])
    assert list(names)==[m.joint(int(j)).name for j in p.jids]
    q0=p.q0.copy();q0[p.qa]=np.clip(q0[p.qa],m.jnt_range[p.jids,0],m.jnt_range[p.jids,1])
    calls=0
    def forward(pose):
        nonlocal calls
        data.qpos[:]=pose;mj.mj_forward(m,data);calls+=1
        assert data.time==0 and np.array_equal(data.qpos[:7],p.q0[:7])
    forward(q0)
    body_z=data.xmat[p.thorax].reshape(3,3)[:,2].copy()
    planes=np.flatnonzero(p.planes);assert len(planes)==1
    floor=int(planes[0]);geoms=[np.flatnonzero(p.feet==j) for j in range(6)]
    assert all(len(g)>0 for g in geoms)
    fromto=np.zeros(6)
    def sole_distances():
        out=np.empty(6)
        for j in range(6):
            distances=[mj.mj_geomDistance(m,data,int(g),floor,1000.,fromto) for g in geoms[j]]
            assert all(np.isfinite(x) and abs(x)<500 for x in distances)
            out[j]=min(distances)
        return out
    stance=((phase[:,None]/(2*np.pi)+np.asarray(metadata['phase_offsets'])[None])%1)<metadata['parameters']['duty']
    source_position=np.empty((513,6,3));source_rotation=np.empty((513,6,3,3));sole_before=np.empty((513,6))
    for i,q in enumerate(oldq):
        pose=q0.copy();pose[p.qa]=q;forward(pose)
        source_position[i]=data.xpos[p.bids]
        source_rotation[i]=data.xmat[p.bids].reshape(6,3,3)
        sole_before[i]=sole_distances()
    medians=np.array([np.median(sole_before[stance[:,j],j]) for j in range(6)])
    common=float(medians.max());offset=common-medians
    assert np.all(offset>=0) and np.max(offset)<.02
    measurements=dict(legs=LEGS,stance_medians=medians,commonplane=common,body_Z_offsets=offset,
        body_Z_world_direction=body_z,spread_before=float(np.ptp(medians)),
        tarsal_geometry_names=[[m.geom(int(g)).name for g in gg] for gg in geoms],
        floor=m.geom(floor).name,physics_steps=0)
    dump(OUT/'SOLE_MEASUREMENT.json',measurements)
    print(json.dumps(measurements,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    ids=[np.array([i for i,name in enumerate(names) if name.startswith('joint_'+leg)],int) for leg in LEGS]
    assert all(len(j)==7 for j in ids)
    corrected=oldq.copy();solver_logs=[];start=time.monotonic()
    for j,leg in enumerate(LEGS):
        if offset[j]<=1e-12:
            solver_logs.append(dict(leg=leg,bypass=True,offset=float(offset[j]),source_columns_exact=True));continue
        ii=ids[j];qa=p.qa[ii];va=p.va[ii]
        low=m.jnt_range[p.jids[ii],0];high=m.jnt_range[p.jids[ii],1]
        successes=0;max_evals=0;max_pe=0.;max_oe=0.;max_correction=0.
        for i in range(512):
            original=oldq[i,ii];target=source_position[i,j]+offset[j]*body_z;target_R=source_rotation[i,j]
            full=q0.copy();full[p.qa]=oldq[i];cache={}
            def evaluate(q):
                if 'q' in cache and np.array_equal(cache['q'],q):return cache['r'],cache['J']
                pose=full.copy();pose[qa]=q;forward(pose)
                pos=data.xpos[p.bids[j]];rot=data.xmat[p.bids[j]].reshape(3,3)
                rv=Rotation.from_matrix(target_R.T@rot).as_rotvec()
                residual=np.r_[pos-target,.2*rv,.0001*(q-original)]
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
                mj.mj_jac(m,data,jp,jr,pos,p.bids[j])
                jac=np.vstack((jp[:,va],.2*util.v2.log_left_inverse(rv)@target_R.T@jr[:,va],.0001*np.eye(7)))
                cache.update(q=q.copy(),r=residual,J=jac)
                return residual,jac
            initial=np.clip(original,low+1e-12,high-1e-12)
            fit=least_squares(lambda q:evaluate(q)[0],initial,jac=lambda q:evaluate(q)[1],bounds=(low,high),
                max_nfev=args.max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-9,x_scale='jac')
            r,_=evaluate(fit.x);corrected[i,ii]=fit.x
            successes+=bool(fit.success);max_evals=max(max_evals,int(fit.nfev))
            max_pe=max(max_pe,float(np.linalg.norm(r[:3])));max_oe=max(max_oe,float(np.linalg.norm(r[3:6])/.2))
            max_correction=max(max_correction,float(np.abs(fit.x-original).max()))
        corrected[-1,ii]=corrected[0,ii]
        log=dict(leg=leg,bypass=False,offset=float(offset[j]),solver_successes=successes,solves=512,
            max_nfev=max_evals,position_error=max_pe,orientation_error=max_oe,joint_correction_max=max_correction,
            FK_calls=calls,wall_s=time.monotonic()-start)
        solver_logs.append(log);print(json.dumps(log),flush=True)
    runtime=util.v2.periodic_pchip(phase,corrected)
    dense=np.linspace(0,2*np.pi,4097);before_q=oldinterp(dense);after_q=runtime(dense)
    dense_stance=((dense[:,None]/(2*np.pi)+np.asarray(metadata['phase_offsets'])[None])%1)<metadata['parameters']['duty']
    before_pos=np.empty((4097,6,3));after_pos=np.empty_like(before_pos)
    before_rot=np.empty((4097,6,3,3));after_rot=np.empty_like(before_rot)
    dense_sole_before=np.empty((4097,6));dense_sole_after=np.empty_like(dense_sole_before)
    for i in range(4097):
        pose=q0.copy();pose[p.qa]=before_q[i];forward(pose)
        before_pos[i]=data.xpos[p.bids];before_rot[i]=data.xmat[p.bids].reshape(6,3,3);dense_sole_before[i]=sole_distances()
        pose[p.qa]=after_q[i];forward(pose)
        after_pos[i]=data.xpos[p.bids];after_rot[i]=data.xmat[p.bids].reshape(6,3,3);dense_sole_after[i]=sole_distances()
    targets=before_pos+offset[None,:,None]*body_z[None,None]
    error=np.linalg.norm(after_pos-targets,axis=2);orientation_error=np.empty((4097,6))
    for j in range(6):orientation_error[:,j]=np.linalg.norm(Rotation.from_matrix(np.swapaxes(before_rot[:,j],1,2)@after_rot[:,j]).as_rotvec(),axis=1)
    med_before=np.array([np.median(dense_sole_before[dense_stance[:,j],j]) for j in range(6)])
    med_after=np.array([np.median(dense_sole_after[dense_stance[:,j],j]) for j in range(6)])
    low=m.jnt_range[p.jids,0];high=m.jnt_range[p.jids,1]
    bound_excess=float(max(np.max(low-after_q),np.max(after_q-high),0.))
    seam=float(np.abs(runtime(0,1)-runtime(2*np.pi,1)).max())
    rates=2*np.pi*util.exact_phase_rate_bounds(runtime)
    jump=float(np.abs(np.diff(corrected,axis=0)).max())
    checks=dict(translation_error=float(error.max())<=.005,orientation_relative_source=float(orientation_error.max())<=.03,
        joint_bounds=bound_excess<=1e-9,periodic_endpoint=bool(np.array_equal(corrected[0],corrected[-1])),
        derivative_seam=seam<=1e-10,rate_1hz=float(rates.max())<=30.,joint_jump=jump<.5,
        nominal_phase_and_duty_unchanged=bool(np.array_equal(phase,source['phase'])),
        interleg_stance_median_sole_spread=float(np.ptp(med_after))<=.005)
    newmeta=dict(metadata)
    newmeta.update(version='sole-aligned-wave-v1',source=str(SOURCE/'GAIT.npz'),source_sha256=sha(SOURCE/'GAIT.npz'),
        transformation='Fixed perleg bodyZ raise from true wholefoot stance medians, perphase sourceorientation and x/y preserved by bounded IK',
        sole_measurement=measurements,source_stance_Rstar_preserved=True,
        dense_sole_medians_before=med_before.tolist(),dense_sole_medians_after=med_after.tolist(),
        sole_spread_before=float(np.ptp(med_before)),sole_spread_after=float(np.ptp(med_after)),
        checks=checks,fk_feasible=bool(all(checks.values())),max_rate_at_1hz=float(rates.max()),rates_at_1hz=rates.tolist(),
        teacher_only=True,physics_steps=0,limitations='FK geometry only; no physical support/slip or neural walking claim.')
    # JSON-safe metadata also embeds small measurement arrays as lists.
    newmeta=json.loads(json.dumps(newmeta,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
    np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=corrected,qpos_addresses=p.qa,
        joint_names=names,metadata_json=np.array(json.dumps(newmeta)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=after_q,source_qpos_hinges=before_q,
        qpos_addresses=p.qa,nominal_stance=dense_stance,target_world=targets,source_position_world=before_pos,
        achieved_world=after_pos,source_orientation_world=before_rot,achieved_orientation_world=after_rot,
        position_error=error,orientation_error_rad=orientation_error,sole_before=dense_sole_before,sole_after=dense_sole_after,
        body_Z_offsets=offset,body_Z_world_direction=body_z,original_q0=p.q0,anchor_q0=q0)
    dump(OUT/'METADATA.json',newmeta)
    report=dict(fk_accepted=bool(all(checks.values())),checks=checks,physics_steps=0,neural_steps=0,FK_calls=calls,
        errors=dict(position_max=float(error.max()),orientation_max=float(orientation_error.max()),
            per_leg_position_max=error.max(axis=0),per_leg_orientation_max=orientation_error.max(axis=0),
            bounds_excess=bound_excess,seam_derivative=seam,joint_jump=jump,max_rate_1hz=float(rates.max()),
            worst_rate_joint=names[int(rates.argmax())]),
        sole=dict(offsets=offset,commonplane=common,before_stance_medians=med_before,after_stance_medians=med_after,
            spread_before=float(np.ptp(med_before)),spread_after=float(np.ptp(med_after)),
            within_leg_stance_variation_before=[float(np.ptp(dense_sole_before[dense_stance[:,j],j])) for j in range(6)],
            within_leg_stance_variation_after=[float(np.ptp(dense_sole_after[dense_stance[:,j],j])) for j in range(6)],
            actual_stance_min_before=[float(dense_sole_before[dense_stance[:,j],j].min()) for j in range(6)],
            actual_stance_max_before=[float(dense_sole_before[dense_stance[:,j],j].max()) for j in range(6)],
            actual_stance_min_after=[float(dense_sole_after[dense_stance[:,j],j].min()) for j in range(6)],
            actual_stance_max_after=[float(dense_sole_after[dense_stance[:,j],j].max()) for j in range(6)]),
        solver_logs=solver_logs,source_hashes_unchanged={p:sha(p)==h for p,h in hashes.items()},
        model=dict(nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,neq=m.neq,gravity=m.opt.gravity),
        wall_s=time.monotonic()-start,limitations=prereg['limitations'])
    dump(OUT/'VERIFICATION.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True,constrained_layout=True)
    for j,leg in enumerate(LEGS):
        ax=axes.ravel()[j];selected=dense_stance[:,j]
        ax.plot(dense[selected]/(2*np.pi),dense_sole_before[selected,j],'.',ms=1.2,label='source nominal stance')
        ax.plot(dense[selected]/(2*np.pi),dense_sole_after[selected,j],'.',ms=1.2,label='corrected nominal stance')
        ax.axhline(common,color='k',ls=':',label='chosen common median plane')
        ax.axhline(0,color='r',lw=.6);ax.set(title=leg+' fixed raise '+format(offset[j],'.6f'),ylabel='True minimum tarsal geom signed distance',xlabel='Cycle fraction')
        if j==0:ax.legend(fontsize=8)
    fig.suptitle('Whole-foot sole planes at original free-root pose; FK only, physical walking NOT_ESTABLISHED')
    fig.savefig(OUT/'SOLE_PLANES.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(output=str(OUT),fk_accepted=all(checks.values()),checks=checks,errors=report['errors'],sole=report['sole']),
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
