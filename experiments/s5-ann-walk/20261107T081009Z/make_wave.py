"""Rephase accepted per-leg FK paths to a five-support-leg wave gait.
Engineering timing change only, not raw biological data. No physics steps.
The phase warp is continuous but has velocity corners before PCHIP smoothing;
all exported geometry and target-rate bounds are independently evaluated below.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T080221Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z')
UTIL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T075536Z/prepare_stance_reference.py')
spec=importlib.util.spec_from_file_location('stance_geometry_utils',UTIL)
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
v1=u.v1
LEGS=v1.LEGS
OLD_DUTY=.65
DUTY=5/6
# u=(global_cycle+offset) modulo1. Chronological liftoffs reverse this list.
ORDER=('LF','RM','LH','RF','LM','RH')
OFFSETS=np.array([ORDER.index(leg)/6 for leg in LEGS])


def original_phase(phase,leg):
    new_u=(np.asarray(phase)/(2*np.pi)+OFFSETS[leg])%1.
    old_u=np.where(new_u<DUTY,new_u/DUTY*OLD_DUTY,
                   OLD_DUTY+(new_u-DUTY)/(1-DUTY)*(1-OLD_DUTY))
    return ((old_u-v1.OFFSETS[leg])%1.)*2*np.pi


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prereg=OUT/'PREREGISTRATION.json'
    if prereg.exists():raise FileExistsError('preserve previous wave candidate')
    v1.dump(prereg,dict(importance='核心必需',stage='S5 load-sharing wave teacher reference, FK only',
        reason='Physical tripod screens at scale2 retained almost all mid/hind feet on ground; scale8 still left LH loaded. Test five-support-leg timing without increasing torque caps.',
        source=str(SOURCE/'GAIT.npz'),source_sha256=v1.sha(SOURCE/'GAIT.npz'),
        source_fk='independently accepted per-leg geometry, same joint ranges and constant stance orientations',
        changes=dict(old_duty=OLD_DUTY,new_duty=DUTY,offset_leg_order=ORDER,
                     offsets_by_LF_LM_LH_RF_RM_RH=OFFSETS,stance_halfstride=.125,swing_clearance=.25),
        method='per-leg phase reparameterization of accepted V4 followed by periodic2-neighbor PCHIP on513knots; no root motion/forces',
        caveat='phase warp has velocity corners before interpolation; explicitly verify exported geometry and exact piecewise-cubic derivative caps. This is an engineering wave gait, source height shape remains real-data-derived.',
        thresholds=dict(position_max=.005,stance_orientation_max_rad=.03,joint_bound_excess=1e-9,
                        seam_derivative=1e-10,max_rate_at_1hz=30.),
        counts=dict(physics_steps=0,neural_steps=0),
        sources={str(p):v1.sha(p) for p in [Path(__file__),UTIL,SOURCE/'GAIT.npz',SOURCE/'METADATA.json',v1.MODEL]},
        success='FK pass only; physical support, actual foot slip and final ANN after teacher withdrawal remain unestablished'))
    with np.load(SOURCE/'GAIT.npz') as z:
        oldphase=z['phase'].copy();oldq=z['qpos_hinges'].copy();qa=z['qpos_addresses'].copy()
        names=z['joint_names'].astype(str)
        oldmeta=json.loads(str(z['metadata_json'].item()))
    interp=u.v2.periodic_pchip(oldphase,oldq)
    fk=v1.FK();assert np.array_equal(qa,fk.qa)
    shape,_=v1.source_profile()
    phase=np.linspace(0,2*np.pi,513)
    q=np.empty((len(phase),42));covered=[]
    for j,leg in enumerate(LEGS):
        ids=np.array([i for i,name in enumerate(names) if name.startswith('joint_'+leg)])
        assert len(ids)==7
        q[:,ids]=interp(original_phase(phase,j))[:,ids]
        covered.extend(ids.tolist())
    assert sorted(covered)==list(range(42))
    q[-1]=q[0]
    runtime=u.v2.periodic_pchip(phase,q)
    dense=np.linspace(0,2*np.pi,4097);dq=runtime(dense)
    stance=np.stack([((dense/(2*np.pi)+offset)%1.)<DUTY for offset in OFFSETS],axis=1)
    goals=np.empty((len(dense),6,3))
    for j in range(6):
        goal,_=v1.desired_path(original_phase(dense,j),fk.anchor_local,shape,OLD_DUTY,.125,.25)
        goals[:,j]=goal[:,j]
    actual=np.empty_like(goals);rot_error=np.empty((len(dense),6))
    target_rot=np.asarray(oldmeta['target_orientation_world'])
    for i,hinges in enumerate(dq):
        fk.d.qpos[:]=fk.q0;fk.d.qpos[fk.qa]=hinges;fk.forward()
        actual[i]=(fk.d.xpos[fk.bids]-fk.origin)@fk.rotation
        for j,bid in enumerate(fk.bids):
            rot=fk.d.xmat[bid].reshape(3,3)
            rot_error[i,j]=np.linalg.norm(Rotation.from_matrix(target_rot[j].T@rot).as_rotvec())
    err=np.linalg.norm(actual-goals,axis=2)
    lo=fk.m.jnt_range[fk.act_jids,0];hi=fk.m.jnt_range[fk.act_jids,1]
    excess=float(max(np.max(lo-dq),np.max(dq-hi),0.))
    seam=float(np.abs(runtime(0,1)-runtime(2*np.pi,1)).max())
    rates=2*np.pi*u.exact_phase_rate_bounds(runtime)
    checks=dict(position=float(err.max())<=.005,stance_orientation=float(rot_error[stance].max())<=.03,
                native_bounds=excess<=1e-9,seam=seam<=1e-10,rate_1hz=float(rates.max())<=30.)
    meta=dict(version='stance-wave-v1',parameters=dict(duty=DUTY,halfstride=.125,clearance=.25),
        interpolation='periodic_pchip_extended',interpolation_recipe='two opposite-end knots extended by +/-2pi, modulo phase, no clipping',
        source=str(SOURCE/'GAIT.npz'),source_sha256=v1.sha(SOURCE/'GAIT.npz'),
        legs=LEGS,phase_offsets=OFFSETS.tolist(),offset_order=ORDER,target_orientation_world=target_rot.tolist(),
        checks=checks,fk_feasible=bool(all(checks.values())),teacher_only=True,physics_steps=0,
        max_rate_at_1hz=float(rates.max()),rates_at_1hz=rates.tolist(),
        limitations='Reparameterized engineering wave timing and smoothed velocity corners; FK is not walking or evidence of original-brain function.')
    np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=q,qpos_addresses=qa,joint_names=names,metadata_json=np.array(json.dumps(meta)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=dq,stance=stance,
                        target_local=goals,achieved_local=actual,position_error=err,orientation_error_rad=rot_error)
    report=dict(fk_feasible=bool(all(checks.values())),checks=checks,
                position_max=float(err.max()),stance_orientation_max=float(rot_error[stance].max()),
                joint_bound_excess=excess,seam_derivative=seam,max_rate_at_1hz=float(rates.max()),
                forward_calls=fk.calls,physics_steps=0,source_sha256=v1.sha(SOURCE/'GAIT.npz'))
    v1.dump(OUT/'VERIFICATION.json',report)
    v1.dump(OUT/'METADATA.json',meta)
    print(json.dumps(report),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
