"""Nominal-preserving joint-space realization of contact-point velocity damping.
Teacher diagnostic only. Actual finitePD/nativecaps unchanged; no surface/bodyforce
assignment, friction/solver changes, or deployment-neural claim.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T120826Z/support_feedforward.py')
spec=importlib.util.spec_from_file_location('bounded_original_teacher_runtime',SOURCE)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
b=f.b;n=f.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T122002Z')
b.OUT=OUT
DAMPING=5.


class DampingTeacher(f.SupportTeacher):
    def desired(self,d,t):
        before=self.p.snapshot(d);start=time.monotonic();a=f.ACTUATORS[-1]
        stance=(self.phase(t)+self.offset)%1<5/6
        entry=stance&~self.previous_stance;self.previous_stance=stance.copy()
        loads=np.zeros(6);vel=np.zeros((6,3));J=np.zeros((6,3,self.m.nv));force=np.zeros(6)
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp);check=0.
        for ci in range(d.ncon):
            co=d.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(self.p.planes[ga] or self.p.planes[gb]):continue
            geom=gb if self.p.planes[ga] else ga;leg=int(self.p.feet[geom])
            if leg<0 or not stance[leg]:continue
            mj.mj_contactForce(self.m,d,ci,force)
            normal=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if self.p.planes[ga] else -1)
            if normal<=0:continue
            body=int(self.m.geom_bodyid[geom]);mj.mj_jac(self.m,d,jp,jr,co.pos,body)
            vp=jp@d.qvel;twist=np.zeros(6)
            mj.mj_objectVelocity(self.m,d,mj.mjtObj.mjOBJ_BODY,body,twist,0)
            vr=twist[3:]+np.cross(twist[:3],co.pos-d.xipos[body])
            check=max(check,float(abs(vp-vr).max()))
            loads[leg]+=normal;vel[leg]+=normal*vp;J[leg]+=normal*jp
        active=loads>0
        vel[active]/=loads[active,None];J[active]/=loads[active,None,None]
        raw=-DAMPING*np.einsum('lkj,lk->j',J[:,:2,self.p.va],vel[:,:2])
        limited=np.clip(raw,-a.torque_limit,a.torque_limit)
        alpha=float(np.clip((t-1.)/.1,0.,1.))
        correction=alpha*limited/a.kp
        nominal=self.ref.warm(t);unbounded=nominal+correction
        desired=np.clip(unbounded,self.ref.lo,self.ref.hi)
        assert check<1e-8 and np.isfinite(desired).all()
        assert np.array_equal(before,self.p.snapshot(d)),'Damping planner wrote actual physics'
        self.rows.append(dict(time=t,stance=stance,active=active,actual_normal_load=loads,
            contact_velocity=vel,contact_J=J,velocity_crosscheck=check,
            nominal_qref=nominal.copy(),tau_damping_raw=raw,tau_damping_clipped=limited,
            alpha=alpha,requested_correction=correction,target_correction=desired-nominal,
            commanded_target=desired.copy(),target_native_clip_mask=desired!=unbounded,
            PD_target_before=a.target_rad.copy(),actual_q=d.qpos[self.p.qa].copy(),
            actual_qvel=d.qvel[self.p.va].copy(),actual_root=d.qpos[:7].copy(),
            actual_ctrl=d.ctrl.copy(),planner_wall_s=time.monotonic()-start))
        return desired,dict(stance=stance,stance_entry=entry,virtual_root=d.qpos[:7].copy(),
            reference_body_position=self.reference_position.copy(),anchor_position=d.xpos[self.p.bids].copy(),
            anchor_rotation=d.xmat[self.p.bids].reshape(6,3,3).copy(),foot_goal_position=d.xpos[self.p.bids].copy(),
            foot_goal_rotation=d.xmat[self.p.bids].reshape(6,3,3).copy(),
            ik_residual=np.c_[np.linalg.norm(vel[:,:2],axis=1),loads/f.W],body_feedback_velocity=np.zeros(2))


b.AnchorTeacher=DampingTeacher


def prereg(folder,args):
    f.prereg(folder,args)
    path=folder/'PREREGISTRATION.json';value=json.loads(path.read_text())
    value.update(stage='S5 nominal-preserving contact-velocity damping teacher diagnostic',
        parameters=dict(damping_native_force_per_velocity=DAMPING,torque_bias_bound='originalnativecap perjoint',
            target_bias_bound='nativecap/kp=.0015625rad atgain128',activation_s=1.,ramp_s=.1),
        previous='Staticnormalpreloadphysicallystablebutdidnotreducehindslip. No staticforceallocationusedinthisarm.',
        planner='Atactualpositive-normalnominalstancecontacts, averageworldpointJ/v byFnperleg. tau_damp=-5*sum(Jxy_act.T*vpoint_xy), clippednativecaps.',
        conversion='desired=clip(fullnominalqref+alpha*tau_damp/kp,nativeanglebounds); original1msrateintegration30rad/s+finitePD128/nativeτcaps unchanged.',
        assumptions='Jointmotortorquesrealizefeedback; no directground/bodyforceassignment orcontactlawchange. BodyfreeDOFsareunactuated. Thisisactivejointcontrol, notmaterialfrictionincrease.',
        previous_static_force_allocation='NOT_USED; sourceplan()nevercalled',
        actuator='OriginalfinitePD128, no targetvelocityterm, originalnativeforcebounds and30rad/s targetrate',
        physics_contact='Originalgravity/friction/solref/solimp/cone/Newton/noslip0 untouched',
        student='Algorithmteacheronly; afterteacherwithdrawalsuchcalculationisforbiddenin finalarm. LearnedANNmustindependentlypass withdeclaredsensoryfeatures.',
        diagnostic='CONTACT_DAMPING.npz every5ms keepsactualJ/v/load/mask/raw/clippedcorrection/nominal/actualstates andPDtargets. No actualτsetoutsideoriginalactuator.',
        additional_sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE]},
        unused_inherited_metadata='Originalanchor/fullposeIK/bodypositionreference/staticFF fields overridden above; none used.')
    for key in ('anchors','ik','swing','body_reference'):
        value[key]='No anchor/IK/bodyfeedback/staticforceplan; originalnominalreferenceplusdeclaredcontactvelocityjointdampingonly'
    n.dump(path,value)


b.prereg=prereg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=4.)
    ap.add_argument('--tag',default='contact_damping_screen');ap.add_argument('--feedback-off',action='store_true')
    args=ap.parse_args();result=None
    try:result=b.simulate(args)
    finally:
        rows=[row for instance in f.INSTANCES for row in instance.rows]
        if rows:
            folder=OUT/args.tag
            fields={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
            with np.load(folder/'COMMANDS.npz') as z:
                commands=z['rows'];ix=np.rint(fields['time']/.001).astype(int)+4;a=f.ACTUATORS[-1]
                fields['PD_target_after_5ms']=np.clip(commands[ix,1:43]+.001*np.clip(commands[ix,43:85]+commands[ix,85:127],-30,30),a.target_bounds_rad[:,0],a.target_bounds_rad[:,1])
            trace=folder/'CONTACT_DAMPING.npz';np.savez_compressed(trace,**fields)
            if result is not None:
                result.update(contact_damping_ticks=len(rows),contact_damping_trace_sha256=n.sha(trace),static_feedforward_used=False)
                n.dump(folder/'RESULT.json',result)


if __name__=='__main__':main()
