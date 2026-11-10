"""Tiny nominal-preserving support preload teacher; no extra torque authority.

Static normal-force allocation is a planning approximation, not physical force
assignment. Full nominal reference remains, with at most.003rad feedforward bias.
Parent owns actual physics launch and independent native support/material-slip audit.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
import mujoco as mj

SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T091243Z/anchor_teacher.py')
spec=importlib.util.spec_from_file_location('support_original_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T120826Z')
n.GAIT=OUT.parent/'20261107T101114Z/GAIT.npz';n.FREQ=1.;b.OUT=OUT
W=n.dsm.WEIGHT
PARAM=dict(gain_scale=128.,RH_Trochanter_yaw_offset=.02,offset_ramp_s=.5,
    planning_start=1.,feedforward_ramp_s=.1,correction_bound_rad=.003,
    active_normal_force_threshold=1e-9,force_upper_W=2.,allocation_maxiter=30,
    allocation_ftol=1e-12,allocation_equality_tolerance_W=1e-6)
INSTANCES=[];ACTUATORS=[]
OriginalReference=n.Reference
OriginalActuator=b.IncrementalActuator


def s5(v):
    x=np.clip(v,0.,1.);return x*x*x*(10-15*x+6*x*x)


class Reference(OriginalReference):
    def __init__(self,p):
        super().__init__(p)
        self.RH_yaw_index=[p.m.joint(int(j)).name for j in p.jids].index('joint_RHTrochanter_yaw')

    def warm(self,t):
        q=super().warm(t).copy()
        q[self.RH_yaw_index]+=s5(t/PARAM['offset_ramp_s'])*PARAM['RH_Trochanter_yaw_offset']
        return np.clip(q,self.lo,self.hi)


class Actuator(OriginalActuator):
    def __init__(self,*args,**kwargs):
        # Anchor standalone runtime supplies scale32, not a mutated gait motor.
        kwargs['kp']=np.asarray(kwargs['kp'])*4.
        kwargs['kd']=np.asarray(kwargs['kd'])*2.
        super().__init__(*args,**kwargs);ACTUATORS.append(self)


class SupportTeacher(b.AnchorTeacher):
    def __init__(self,p,ref):
        super().__init__(p,ref);self.rows=[];self.previous_allocation=np.zeros(6)
        INSTANCES.append(self)

    def plan(self,d,stance):
        load=np.zeros(6);points=np.zeros((6,3));Jz=np.zeros((6,self.m.nv));mu=[]
        force=np.zeros(6);jp=np.zeros((3,self.m.nv))
        for ci in range(d.ncon):
            co=d.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(self.p.planes[ga] or self.p.planes[gb]):continue
            geom=gb if self.p.planes[ga] else ga;li=int(self.p.feet[geom])
            if li<0:continue
            mj.mj_contactForce(self.m,d,ci,force)
            normal=float((co.frame.reshape(3,3).T@force[:3])[2])*(1 if self.p.planes[ga] else -1)
            if normal<=0:continue
            mj.mj_jac(self.m,d,jp,None,co.pos,int(self.m.geom_bodyid[geom]))
            load[li]+=normal;points[li]+=normal*co.pos;Jz[li]+=normal*jp[2]
            mu.extend(np.asarray(co.friction[:2]).tolist())
        positive=load>0;points[positive]/=load[positive,None];Jz[positive]/=load[positive,None]
        active=stance&(load>PARAM['active_normal_force_threshold']);ids=np.flatnonzero(active)
        fz=np.zeros(6);eqres=np.array([-W,0.,0.]);fallback=True;status='INSUFFICIENT_CONTACTS';nit=0
        com=d.subtree_com[self.p.thorax].copy()
        if len(ids)>=3:
            A=np.vstack([np.ones(len(ids)),points[ids,0]-com[0],points[ids,1]-com[1]])
            rhs=np.array([1.,0.,0.]);preferred=np.full(len(ids),1/len(ids))
            initial=self.previous_allocation[ids]/W
            if initial.sum()<=0:initial=preferred.copy()
            fit=minimize(lambda x:.5*np.sum((x-preferred)**2),np.clip(initial,0.,2.),
                jac=lambda x:x-preferred,method='SLSQP',bounds=[(0.,PARAM['force_upper_W'])]*len(ids),
                constraints=[dict(type='eq',fun=lambda x:A@x-rhs,jac=lambda x:A)],
                options=dict(maxiter=PARAM['allocation_maxiter'],ftol=PARAM['allocation_ftol'],disp=False))
            status=str(fit.message);nit=int(fit.nit)
            if np.isfinite(fit.x).all():
                fz[ids]=fit.x*W;eqres=(A@fit.x-rhs)*W
                fallback=not(fit.success and abs(eqres).max()<=PARAM['allocation_equality_tolerance_W']*W
                    and fz.min()>=-1e-9 and fz.max()<=PARAM['force_upper_W']*W+1e-9)
        if not fallback:self.previous_allocation=fz.copy()
        return dict(fz=fz,active=active,Jz=Jz*active[:,None],contact_point=points,actual_normal_load=load,
            equilibrium_residual=eqres,fallback=fallback,status=status,solver_iterations=nit,
            actual_com=com,actual_mu_range=np.array([min(mu),max(mu)]) if mu else np.zeros(2),
            unmodeled_swing_normal_load=load*(~stance))

    def desired(self,d,t):
        before=self.p.snapshot(d);start=time.monotonic();a=ACTUATORS[-1]
        stance=(self.phase(t)+self.offset)%1<5/6;entry=stance&~self.previous_stance;self.previous_stance=stance.copy()
        plan=self.plan(d,stance);nominal=self.ref.warm(t)
        raw=(d.qfrc_bias-d.qfrc_passive-plan['Jz'].T@plan['fz'])[self.p.va]
        clipped=np.clip(raw,-a.torque_limit,a.torque_limit)
        limit=np.minimum(a.torque_limit/a.kp,PARAM['correction_bound_rad'])
        alpha=float(np.clip((t-PARAM['planning_start'])/PARAM['feedforward_ramp_s'],0.,1.))
        correction=np.zeros(42) if plan['fallback'] else alpha*np.clip(clipped/a.kp,-limit,limit)
        unbounded=nominal+correction;desired=np.clip(unbounded,self.ref.lo,self.ref.hi)
        assert np.isfinite(desired).all() and abs(correction).max()<=.003+1e-12
        assert np.array_equal(before,self.p.snapshot(d)),'Planner changed actual physics'
        self.rows.append(dict(time=t,nominal_qref=nominal.copy(),tau_feedforward_raw=raw.copy(),
            tau_feedforward_clipped=clipped.copy(),target_correction=desired-nominal,
            requested_correction=correction.copy(),correction_bound=limit.copy(),target_native_clip_mask=desired!=unbounded,
            alpha=alpha,commanded_target=desired.copy(),PD_target_before=a.target_rad.copy(),
            actual_q=d.qpos[self.p.qa].copy(),actual_qvel=d.qvel[self.p.va].copy(),actual_root=d.qpos[:7].copy(),
            actual_ctrl=d.ctrl.copy(),plan_wall_s=time.monotonic()-start,**plan))
        return desired,dict(stance=stance,stance_entry=entry,virtual_root=d.qpos[:7].copy(),
            reference_body_position=self.reference_position.copy(),anchor_position=d.xpos[self.p.bids].copy(),
            anchor_rotation=d.xmat[self.p.bids].reshape(6,3,3).copy(),foot_goal_position=d.xpos[self.p.bids].copy(),
            foot_goal_rotation=d.xmat[self.p.bids].reshape(6,3,3).copy(),
            ik_residual=np.c_[plan['actual_normal_load']/W,plan['fz']/W],body_feedback_velocity=np.zeros(2))


n.Reference=Reference;b.IncrementalActuator=Actuator;b.AnchorTeacher=SupportTeacher
original_prereg=b.prereg


def prereg(folder,args):
    original_prereg(folder,args);path=folder/'PREREGISTRATION.json';value=json.loads(path.read_text())
    value.update(stage='S5 tiny nominal-preserving static support feedforward teacher test',parameters=PARAM,
        reference='101114broadlift1Hz +.02RHtrochanteryawS5warmramp, same measuredselectedteacher order',
        planner='Onlynormalfz allocation min.5||fz/W-1/N||², loadedactualnominalstance. Hardsumfz=W, xyCOMmoment0;0<=fz<=2W; SLSQPmax30. Infeasiblesupportpolygon/status/residual failure=>unchangednominal.',
        conversion='tauFF=(actualbias-actualpassive-Jz.T*fz)_hinges. ClipFFnative, target=nominal+alpha*clip(tauFF/kp,+/-min(nativecap/kp,.003)), nativejointclip. Alpha0→1 over1.0–1.1s.',
        actuator='OriginalfinitePD128zeroDtarget,1msincrementintegration30rad/scap/nativeforcecaps. Actualtau is tracking plusbias; feedforwardaddsNOforceauthority.',
        assumptions='Staticnormalplanning ignoresFx/Fy,inertia/qacc/contactmoments andsoftcontactlaw. Allocatedfz areNEVERwrittenasphysicalforces; actualcontactsdetermineallforces.',
        diagnostic='FEEDFORWARD.npz ALL5msticks retainedafterphysicalrun: nominal,raw/clippedFFtau,fz,active,Jz,eqres,status/fallback,alpha,correction,actualq/v/root,targetbefore+actualintegratedtargetafter5ms. TEACHER.ik_residual=[actualFn/W,plannedfz/W],NOTIK.',
        previous_anchor_algorithm='NOTUSED: no anchors/fullposeIK or torqueinversionaboutactualq; nominalshapealwaysretained.',
        student='Teacher-onlyphysicsfeasibility, noANNaction/brainclaim; laterstudentrequiresdeclaredsensorylearningifteacherpasses.',
        additional_sources={str(p):n.sha(p) for p in [Path(__file__),SOURCE,Path(__file__).parents[1]/'20261107T101753Z/run_gain128.py']})
    value['physics'].update(kp='limit/.2*128',kd='limit*.005*sqrt128')
    value.update(ik='NOT_RUN',anchors='NOT_USED; actualcontactpoints allocatedonlyforstaticpreload',
        swing='Fullnominal101114reference retained, no swingIK',body_reference='Capturedroot onlyforaudit; notusedforbodypositionfeedback')
    n.dump(path,value)


b.prereg=prereg


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seconds',type=float,default=4.)
    ap.add_argument('--tag',default='support_feedforward_screen');ap.add_argument('--feedback-off',action='store_true')
    args=ap.parse_args();result=None
    try:result=b.simulate(args)
    finally:
        rows=[r for teacher in INSTANCES for r in teacher.rows]
        if rows:
            folder=OUT/args.tag;fields={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            commands=folder/'COMMANDS.npz'
            if commands.exists():
                with np.load(commands) as z:
                    c=z['rows'];indices=np.rint(fields['time']/.001).astype(int)+4
                    a=ACTUATORS[-1]
                    fields['PD_target_after_5ms']=np.clip(c[indices,1:43]+.001*c[indices,43:85],a.target_bounds_rad[:,0],a.target_bounds_rad[:,1])
            np.savez_compressed(folder/'FEEDFORWARD.npz',**fields)
            if result is not None:
                result.update(feedforward_ticks=len(rows),feedforward_fallback_count=int(fields['fallback'].sum()),
                    feedforward_trace_sha256=n.sha(folder/'FEEDFORWARD.npz'))
                n.dump(folder/'RESULT.json',result)


if __name__=='__main__':main()
