"""Build an engineering stance-consistent reference by fixed-root FK/IK only.

No mj_step, neural execution or physical walking occurs here. Stance has constant
body-frame backwards speed, fixed y/z, and q0 Tarsus5 orientation. Source LF swing
height is detrended, normalized and smoothly resampled; all timing/amplitudes and
tripod phasing are declared engineering choices. Deployment must be a learned ANN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T074700Z')
SOURCE = Path('/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z')
MODEL = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
LEGS = ('LF','LM','LH','RF','RM','RH')
DOFS = ('Coxa_yaw','Coxa_pitch','Coxa_roll','Trochanter_yaw',
        'Trochanter_pitch','Trochanter_roll','Tibia_pitch')
OFFSETS = np.array([0., .5, 0., .5, 0., .5])  # LF/LH/RM versus RF/RH/LM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    if path.exists():
        path.read_bytes()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')


def s5(x):
    return 10*x**3-15*x**4+6*x**5


def source_profile():
    with np.load(SOURCE/'STANDARD_GAIT_FK.npz', allow_pickle=False) as z:
        phase = z['phase'].copy()
        height = z['foot_body'][:, 0, 2].copy()
    # LF elevated lobe in the source is bounded by phase0 and the first local
    # minimum after its peak. This selection is saved verbatim for review.
    peak = int(np.argmax(height[:-1]))
    right = next((i for i in range(peak+1, len(height)-1)
                  if height[i]<=height[i-1] and height[i]<=height[i+1]), len(height)-1)
    # If no clear minimum occurs soon after the peak, use the first return to the
    # takeoff baseline. This avoids including the long stance baseline wander.
    returns = np.flatnonzero(height[peak+1:] <= height[0])
    if len(returns):
        right = min(right, peak+1+int(returns[0]))
    assert 0 < peak < right
    x = (phase[:right+1]-phase[0])/(phase[right]-phase[0])
    y = height[:right+1]-np.linspace(height[0],height[right],right+1)
    y = np.maximum(y,0.)
    y[0] = y[-1] = 0.
    y /= y.max()
    interp = PchipInterpolator(x,y)
    sample = np.linspace(0,1,4097)
    norm = float(interp(s5(sample)).max())
    def shape(v):
        return np.maximum(0.,interp(s5(np.clip(v,0,1))))/norm
    return shape, dict(source_lf_indices=list(range(right+1)), source_peak_index=peak,
        source_phase=phase[:right+1], source_lf_z=height[:right+1],
        detrended_normalized_z=y, resampling='PCHIP of detrended source LF lobe, evaluated at quintic smoothstep(swing_fraction), renormalized to unit peak',
        caveat='Height shape only. Duty, stride, clearance, anchors, tripod offsets and foot orientation are engineering choices, not raw real gait.')


def desired_path(phase, anchors, shape, duty, halfstride, clearance):
    u = (phase[:,None]/(2*np.pi)+OFFSETS[None])%1.
    stance = u<duty
    points = np.repeat(anchors[None],len(phase),axis=0)
    points[:,:,0] += halfstride-2*halfstride*u/duty
    v = np.clip((u-duty)/(1-duty),0,1)
    # Match stance x velocity and zero acceleration at both swing boundaries.
    # A small swing overshoot is mathematically necessary to reverse direction
    # while entering/leaving a constant-speed backwards stance continuously.
    slope = -2*halfstride*(1-duty)/duty
    swing_x = -halfstride+slope*v+(2*halfstride-slope)*s5(v)
    points[:,:,0] = np.where(stance,points[:,:,0],anchors[None,:,0]+swing_x)
    points[:,:,2] += np.where(stance,0.,clearance*shape(v))
    return points, stance


class FK:
    def __init__(self):
        sys.path.insert(0,str(BASE))
        import mujoco as mj
        from incremental_actuator import load_full_model, validate_full_model
        self.mj = mj
        self.m = load_full_model(MODEL)
        validate_full_model(self.m)
        self.d = mj.MjData(self.m)
        self.original_q0 = self.m.key_qpos[0].copy()
        self.act_jids = self.m.actuator_trnid[:,0].copy()
        self.qa = self.m.jnt_qposadr[self.act_jids].copy()
        self.q0 = self.original_q0.copy()
        self.q0[self.qa] = np.clip(self.q0[self.qa], self.m.jnt_range[self.act_jids,0], self.m.jnt_range[self.act_jids,1])
        self.initial_projection = self.q0[self.qa]-self.original_q0[self.qa]
        self.d.qpos[:] = self.q0  # FK initialization only; original free root preserved.
        self.calls = 0
        self.forward()
        self.va = self.m.jnt_dofadr[self.act_jids].copy()
        self.jids = np.array([[self.m.joint('joint_'+leg+dof).id for dof in DOFS] for leg in LEGS])
        self.leg_qa = self.m.jnt_qposadr[self.jids]
        self.leg_va = self.m.jnt_dofadr[self.jids]
        self.bids = [self.m.body(leg+'Tarsus5').id for leg in LEGS]
        self.thorax = self.m.body('Thorax').id
        self.origin = self.d.xpos[self.thorax].copy()
        self.rotation = self.d.xmat[self.thorax].reshape(3,3).copy()
        self.anchor_world = self.d.xpos[self.bids].copy()
        self.anchor_local = (self.anchor_world-self.origin)@self.rotation
        self.orientations = self.d.xmat[self.bids].reshape(6,3,3).copy()
        self.limits = self.m.jnt_range[self.jids].copy()
        assert len(np.unique(self.leg_qa)) == 42 and set(self.leg_qa.ravel()) == set(self.qa)
        assert np.all(self.leg_qa >= 7)
        self.planes = self.m.geom_type == mj.mjtGeom.mjGEOM_PLANE

    def forward(self):
        self.mj.mj_forward(self.m,self.d)
        self.calls += 1
        assert self.d.time == 0., 'FK preparation must not advance physics time'
        assert np.array_equal(self.d.qpos[:7],self.q0[:7])

    def pose(self,joint_values,leg):
        self.d.qpos[self.leg_qa[leg]] = joint_values
        self.forward()
        bid = self.bids[leg]
        return self.d.xpos[bid].copy(),self.d.xmat[bid].reshape(3,3).copy()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phases',type=int,default=65,choices=[65,129])
    ap.add_argument('--duty',type=float,default=.65)
    ap.add_argument('--halfstride',type=float,default=.15)
    ap.add_argument('--clearance',type=float,default=.25)
    ap.add_argument('--orientation-weight',type=float,default=.2,help='native length per radian')
    ap.add_argument('--regularizer',type=float,default=1e-4)
    ap.add_argument('--max-nfev',type=int,default=160)
    args=ap.parse_args(argv)
    assert .5<args.duty<1 and args.halfstride>0 and args.clearance>0
    OUT.mkdir(parents=True,exist_ok=True)
    prereg=OUT/'PREREGISTRATION.json'
    if prereg.exists():
        old=json.loads(prereg.read_text())
        raise FileExistsError('Preserve preregistration and prior outputs; use a new timestamp rather than overwrite '+str(prereg))
    hashes={str(p):sha(p) for p in [Path(__file__),SOURCE/'STANDARD_GAIT_FK.npz',SOURCE/'REFERENCE_METADATA.json',MODEL,BASE/'incremental_actuator.py']}
    config=dict(importance='核心必需',stage='S5 stance reference engineering FK, no physical gait claim',
        parameters=vars(args),legs=LEGS,tripods=[['LF','LH','RM'],['RF','RH','LM']],
        phase_offsets_cycles=OFFSETS,stance='body-frame constant backwards x speed; q0 anatomical y/z and q0 Tarsus5 orientation',
        swing='constant y; smooth x return matching stance velocity; real-source LF height shape resampled to engineering clearance',
        inverse_kinematics='bounded by-name 7 actuated hinges per leg; position3 + orientation rotation-vector3 + small q0 regularizer',
        frame='q0 Thorax body frame; original keyframe free root fixed only for FK; no physical root trajectory',
        anchor_projection='Project only keyframe42 hinge angles into native bounds before anchor FK; record projection; original root preserved exactly',
        orientation='q0 Tarsus5 body orientation kept near constant in stance AND swing',
        thresholds=dict(position_max_native=.005,orientation_max_rad=.03,periodic_endpoint=1e-12,
                        joint_jump_max_rad=.5,interpolated_joint_bound_tolerance=1e-9),
        sampling='65 or129 equally spaced phases including identical closed endpoint; densely verify periodic cubic interpolation',
        provenance='reference height shape is derived from processed real-source FK; phasing, amplitudes and orientation are engineering retargeting',
        counts=dict(physics_step=0,neural_step=0),source_hashes=hashes,
        deployment='Algorithm teacher/training asset only. Final deployment requires exported ANN after teacher withdrawal.',
        limitations=['FK feasibility cannot establish support, contact material slip, locomotion or stability.',
                     'Tarsus5 origin differs from sole/contact point. Constant rigid foot orientation and constant stance height constrain geometry but do not establish ground contact.',
                     'All source/model files preserved. No simulator root writes are proposed. Absolute SI force and original brain necessity NOT_ESTABLISHED.'])
    dump(prereg,config)  # Must precede model compilation, forward calls and IK.
    dump(OUT/'SOURCE_SHA.json',hashes)
    shape,source_info=source_profile()
    dump(OUT/'SOURCE_SWING_RESAMPLING.json',source_info)
    fk=FK()
    phase=np.linspace(0,2*np.pi,args.phases)
    goals_local,stance=desired_path(phase,fk.anchor_local,shape,args.duty,args.halfstride,args.clearance)
    goals_world=goals_local@fk.rotation.T+fk.origin
    qposes=np.repeat(fk.q0[None],len(phase),axis=0)
    rows=[]
    start=time.monotonic()
    for leg,name in enumerate(LEGS):
        qa=fk.leg_qa[leg]
        q0=fk.q0[qa]
        lo,hi=fk.limits[leg,:,0],fk.limits[leg,:,1]
        scale=np.maximum(hi-lo,.1)
        seed=np.clip(q0,lo+1e-10,hi-1e-10)
        for i in range(len(phase)-1):
            target=goals_world[i,leg]
            desired_R=fk.orientations[leg]
            def residual(q):
                point,rot=fk.pose(q,leg)
                rvec=Rotation.from_matrix(desired_R.T@rot).as_rotvec()
                return np.r_[point-target,args.orientation_weight*rvec,args.regularizer*(q-q0)/scale]
            fit=least_squares(residual,seed,bounds=(lo,hi),max_nfev=args.max_nfev,
                              ftol=1e-10,xtol=1e-10,gtol=1e-9,x_scale='jac')
            err=residual(fit.x)
            # A q0 restart reduces branch lock-in; retain whichever has smaller
            # declared residual, rather than silently changing the objective.
            if np.linalg.norm(err[:3])>.005 or np.linalg.norm(err[3:6])/args.orientation_weight>.03:
                retry=least_squares(residual,np.clip(q0,lo+1e-10,hi-1e-10),bounds=(lo,hi),
                    max_nfev=args.max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-9,x_scale='jac')
                if np.linalg.norm(retry.fun)<np.linalg.norm(fit.fun):
                    fit=retry
                    err=residual(fit.x)
            seed=fit.x.copy()
            qposes[i,qa]=seed
            rows.append(dict(leg=name,phase_index=i,position_error=float(np.linalg.norm(err[:3])),
                orientation_error_rad=float(np.linalg.norm(err[3:6])/args.orientation_weight),
                solver_success=bool(fit.success),nfev=int(fit.nfev),cost=float(fit.cost),
                active_bounds=int(np.sum(fit.active_mask!=0))))
        qposes[-1,qa]=qposes[0,qa]
        selected=[r for r in rows if r['leg']==name]
        print(json.dumps(dict(leg=name,position_max=max(r['position_error'] for r in selected),
            orientation_max=max(r['orientation_error_rad'] for r in selected),forward_calls=fk.calls,
            wall_s=round(time.monotonic()-start,2))),flush=True)
    # Independent full-body verification at all knots, with root unchanged.
    achieved=np.empty_like(goals_world)
    orientations=np.empty((len(phase),6,3,3))
    knot_ground_penetration=[]
    for i,q in enumerate(qposes):
        fk.d.qpos[:]=q
        fk.forward()
        achieved[i]=fk.d.xpos[fk.bids]
        orientations[i]=fk.d.xmat[fk.bids].reshape(6,3,3)
        for contact in fk.d.contact[:fk.d.ncon]:
            ga,gb=int(contact.geom1),int(contact.geom2)
            if fk.planes[ga] or fk.planes[gb]:
                geom=gb if fk.planes[ga] else ga
                knot_ground_penetration.append(dict(phase_index=i,geom=fk.m.geom(geom).name,
                                                    distance=float(contact.dist)))
    poserr=np.linalg.norm(achieved-goals_world,axis=2)
    orierr=np.empty((len(phase),6))
    for leg in range(6):
        orierr[:,leg]=np.linalg.norm(Rotation.from_matrix(fk.orientations[leg].T@orientations[:,leg]).as_rotvec(),axis=1)
    hinges=qposes[:,fk.qa]
    lo_act=fk.m.jnt_range[fk.act_jids,0]
    hi_act=fk.m.jnt_range[fk.act_jids,1]
    dense_phase=np.linspace(0,2*np.pi,2049)
    spline=CubicSpline(phase,hinges,axis=0,bc_type='periodic')
    dense_hinges=spline(dense_phase)
    dense_goals,dense_stance=desired_path(dense_phase,fk.anchor_local,shape,args.duty,args.halfstride,args.clearance)
    dense_errors=[]
    for i in range(0,len(dense_phase),8):
        fk.d.qpos[:]=fk.q0
        fk.d.qpos[fk.qa]=dense_hinges[i]
        fk.forward()
        actual=(fk.d.xpos[fk.bids]-fk.origin)@fk.rotation
        dense_errors.append(np.linalg.norm(actual-dense_goals[i],axis=1))
    dense_errors=np.asarray(dense_errors)
    bound_excess=float(max(np.max(lo_act-dense_hinges),np.max(dense_hinges-hi_act),0.))
    jump=float(np.abs(np.diff(hinges,axis=0)).max())
    checks=dict(position_knots=float(poserr.max())<=.005,orientation_knots=float(orierr.max())<=.03,
        knot_joint_bounds=bool(np.all(hinges>=lo_act-1e-12) and np.all(hinges<=hi_act+1e-12)),
        interpolated_joint_bounds=bound_excess<=1e-9,
        periodic_joint_endpoint=bool(np.array_equal(hinges[0],hinges[-1])),
        no_large_joint_branch_jump=jump<=.5,
        interpolated_position_error=float(dense_errors.max())<=.005)
    metadata=dict(version='stance-orientation-engineering-v1',legs=LEGS,parameters=vars(args),
        frame='q0 thorax frame, q0 full free root fixed for FK only',
        joint_names=[fk.m.joint(int(j)).name for j in fk.act_jids],
        reference='processed real-source height shape with engineering stance/phase/amplitudes, NOT raw real motion',
        stance_speed_per_cycle_native=2*args.halfstride/args.duty,
        expected_body_speed_at_3hz_native=3*2*args.halfstride/args.duty,
        source_hashes=hashes,checks=checks,fk_feasible=bool(all(checks.values())),
        teacher_only=True,walking_success='NOT_ESTABLISHED',physics_steps=0)
    np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=hinges,qpos_addresses=fk.qa,
        joint_names=np.array(metadata['joint_names']),metadata_json=np.array(json.dumps(metadata,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=phase,stance=stance,qpos=qposes,
        q0=fk.q0,original_keyframe_q0=fk.original_q0,initial_projection=fk.initial_projection,
        thorax_origin=fk.origin,thorax_rotation=fk.rotation,
        anchors_local=fk.anchor_local,target_local=goals_local,target_world=goals_world,
        achieved_world=achieved,target_orientation=fk.orientations,achieved_orientation=orientations,
        position_error=poserr,orientation_error_rad=orierr,
        leg_qpos_addresses=fk.leg_qa,leg_joint_names=np.array([['joint_'+leg+dof for dof in DOFS] for leg in LEGS]),
        dense_phase=dense_phase,dense_qpos_hinges=dense_hinges,dense_position_error_samples=dense_errors)
    report=dict(stage='FK engineering reference, NO physical simulation',checks=checks,
        fk_feasible=all(checks.values()),physics_steps=0,neural_steps=0,forward_calls=fk.calls,
        anchor_projection=dict(max_abs_rad=float(np.abs(fk.initial_projection).max()),
            actuator_deltas=fk.initial_projection,original_free_root_preserved=bool(np.array_equal(fk.q0[:7],fk.original_q0[:7]))),
        model=dict(nq=fk.m.nq,nv=fk.m.nv,nu=fk.m.nu,nmesh=fk.m.nmesh,neq=fk.m.neq,
                   gravity=fk.m.opt.gravity,timestep=fk.m.opt.timestep),
        errors=dict(position_max=float(poserr.max()),position_rms=float(np.sqrt((poserr**2).mean())),
            orientation_max_rad=float(orierr.max()),orientation_rms_rad=float(np.sqrt((orierr**2).mean())),
            dense_position_max=float(dense_errors.max()),interpolated_bound_excess_rad=bound_excess,
            joint_step_max_rad=jump,per_leg_position_max=poserr.max(axis=0),per_leg_orientation_max_rad=orierr.max(axis=0)),
        source_profile=source_info,solver_rows=rows,ground_contact_geometry_FK_only=knot_ground_penetration,
        source_hashes_unchanged={str(p):sha(p)==digest for p,digest in hashes.items()},
        outputs={str(OUT/name):sha(OUT/name) for name in ['GAIT.npz','FK_EVIDENCE.npz','PREREGISTRATION.json','SOURCE_SHA.json','SOURCE_SWING_RESAMPLING.json']},
        limitations=config['limitations'],wall_s=time.monotonic()-start)
    dump(OUT/'VERIFICATION.json',report)
    dump(OUT/'METADATA.json',metadata)
    text=(f"# Stance-consistent reference: FK/IK only\n\n"
        f"FK constraints passed: **{all(checks.values())}**. Physics/neural steps: **0 / 0**. Full model: 49 qpos, 48 qvel, 42 motors, 71 meshes, no equalities.\n\n"
        f"Position max/RMS: {poserr.max():.6g} / {np.sqrt((poserr**2).mean()):.6g} native. "
        f"Orientation max/RMS: {orierr.max():.6g} / {np.sqrt((orierr**2).mean()):.6g} rad. "
        f"Dense-interpolation position max: {dense_errors.max():.6g}. Joint-bound excess: {bound_excess:.6g} rad; max adjacent joint step: {jump:.6g} rad.\n\n"
        "Stance: fixed q0 Tarsus5 y/z and orientation; backwards x at constant speed. Engineering duty0.65, stance halfstride0.15, swing clearance0.25; tripod LF/LH/RM versus RF/RH/LM. "
        "Swing x has disclosed small overshoot to match backwards stance velocity continuously; LF source height lobe is detrended, normalized, and smoothly reparameterized.\n\n"
        "This is an algorithm teacher/training asset. FK does not establish contact support, no-slip motion, stability or walking. "
        "The physical screen must use finite bounded motors, free root and unchanged gravity/contact, followed by the full D gates and exact sampled material-contact slip. "
        "Final deployment requires the ANN; no root trajectory or reference controller may persist after handover.\n\n"
        + '\n'.join(f"- {key}: {value}" for key,value in checks.items())+'\n')
    (OUT/'REPORT.md').write_text(text)
    print(json.dumps(dict(output=str(OUT),fk_feasible=all(checks.values()),errors=report['errors'],
                         forward_calls=fk.calls,physics_steps=0),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':
    raise SystemExit(main())
