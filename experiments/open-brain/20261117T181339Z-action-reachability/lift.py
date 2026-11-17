"""Bounded single-leg IK and free-root physical lift; no anatomical-limit claim."""
import numpy as np
from scipy.optimize import least_squares
from context import *


def run():
    config=dict(target_lift_native=.20,delta_bounds_rad=[.015,.15,.5],legs=['lf','lm','lh','rf','rm','rh'],
        ik_max_nfev=100,ik_error_tolerance_native=.005,physical_steps_per_case=2000,dt=.0001,
        ramp_s=.03,hold_end_s=.12,return_end_s=.15,total_s=.20,
        physical_lift_pass_native=.15,min_upright=.8,
        statement='Engineering local angle windows, not biological joint ranges. Root fixed only during FK/IK, free during dynamics.',
        planned_physics_steps=36000,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(HERE/'context.py'):sha(HERE/'context.py')})
    write(HERE/'LIFT_CONFIG.json',config)
    results=[]
    for leg in config['legs']:
        for bound in config['delta_bounds_rad']:
            b=body_run.Body();b.at_start(119)
            selected=[i for i,dof in enumerate(b.dofs) if dof.child.pos==leg]
            assert len(selected)==7
            aid=b.actuators[selected];jid=b.m.actuator_trnid[aid,0];qadr=b.m.jnt_qposadr[jid]
            foot=b.sim._internal_bodyids_by_fly[b.name][b.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5'))]
            initial_state=np.empty(mujoco.mj_stateSize(b.m,body_run.FLAGS));mujoco.mj_getState(b.m,b.d,initial_state,body_run.FLAGS)
            original_qpos=b.d.qpos.copy();startfoot=b.d.xpos[foot].copy();startthorax=b.d.xpos[b.thorax].copy()
            target=startfoot+np.array([0.,0.,.20])
            reference=b.neutral[selected].copy()
            actual_reference=original_qpos[qadr].copy()
            calls=0
            def objective(delta):
                nonlocal calls
                calls+=1
                b.d.qpos[:]=original_qpos;b.d.qpos[qadr]=actual_reference+delta
                mujoco.mj_fwdPosition(b.m,b.d)
                return np.r_[b.d.xpos[foot]-target,.0001*delta]
            opt=least_squares(objective,np.zeros(7),bounds=(-bound*np.ones(7),bound*np.ones(7)),max_nfev=100,
                              ftol=1e-10,xtol=1e-10,gtol=1e-10)
            objective(opt.x);error=float(np.linalg.norm(b.d.xpos[foot]-target))
            mujoco.mj_setState(b.m,b.d,initial_state,body_run.FLAGS);mujoco.mj_forward(b.m,b.d)
            trace=[];poses=[]
            for tick in range(2000):
                t=tick*.0001
                if t<.03:scale=.5-.5*np.cos(np.pi*t/.03)
                elif t<.12:scale=1.
                elif t<.15:scale=.5+.5*np.cos(np.pi*(t-.12)/.03)
                else:scale=0.
                control=b.neutral.copy();control[selected]+=scale*opt.x
                b.apply(control)
                # Relative height removes apparent lift caused by whole-body motion.
                dz=(b.d.xpos[foot,2]-startfoot[2])-(b.d.xpos[b.thorax,2]-startthorax[2])
                trace.append([b.d.time,dz,b.d.xpos[foot,2],b.d.xpos[b.thorax,2],b.d.xmat[b.thorax,8],
                              np.max(np.abs(b.d.actuator_force[b.actuators])),np.max(np.abs(b.d.xfrc_applied))])
                if tick%10==0:poses.append(b.d.qpos.copy())
            a=np.array(trace);name=f'{leg}_bound{bound:g}'
            np.savez_compressed(HERE/f'LIFT_{name}.npz',trace=a,qpos=np.array(poses),delta=opt.x,control_reference=b.neutral)
            result=dict(leg=leg,bound_rad=bound,ik_error_native=error,ik_reachable=error<=.005,
                 delta_rad=opt.x.tolist(),actual_lift_peak_native=float(a[:,1].max()),
                 actual_lift_hold_mean_native=float(a[600:1200,1].mean()),min_upright=float(a[:,4].min()),
                 force_peak_native=float(a[:,5].max()),force_saturated=bool(a[:,5].max()>=.99*65),
                 outside_force_applied_max=float(a[:,6].max()),configured_joint_limits=int(b.m.jnt_limited[jid].sum()),
                 physical_lift_pass=bool(a[600:1200,1].mean()>=.15 and a[:,4].min()>=.8),
                 physical_steps=2000,position_forward_calls=calls,warning=b.d.warning.number.tolist())
            assert np.isfinite(a).all() and not b.d.warning.number.any()
            results.append(result);write(HERE/'LIFT_PROGRESS.json',dict(results=results))
            print('lift',leg,bound,'IK',round(error,5),'actual',round(result['actual_lift_hold_mean_native'],4),flush=True)
    write(HERE/'LIFT_RESULT.json',dict(results=results,physical_steps=36000,
       anatomical_compliance='UNKNOWN_NO_JOINT_LIMITS_IN_BASE_MODEL',neural_ticks=0,training_updates=0))


if __name__=='__main__':run()
