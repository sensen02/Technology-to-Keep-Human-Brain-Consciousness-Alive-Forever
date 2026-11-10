"""Engineering fixtures for measured-reference teacher; not behavioral acceptance."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import numpy as np
import mujoco as mj
from physics import Core
from measured_teacher import MeasuredTeacher,TeacherConfig,foot_jacobians,bounded_rate
from identity import HERE,atomic_json,sha256


def measured(c,leg):
    m=c.measure()
    return np.r_[m['foot'][leg,:2],m['padz'][leg]]


def minimum_pad(c,leg):
    pads=c.pads[__import__('physics').LEGS[leg]]
    return min(pads,key=lambda g:float(c.d.geom_xpos[g,2]-c.plant.geom_size[g,0]))


def main():
    ref=json.loads((HERE/'REFERENCE_CHECKS.json').read_text())
    assert ref['status']=='PASS' and all(x['passed'] for x in ref['checks'])
    for path,digest in ref['source_files_sha256'].items():assert sha256(path)==digest
    rng=np.random.default_rng(20261110)
    c=Core(adhesion=False)
    base=c.d.qpos.copy()
    # Static differentiation fixtures only: no behavioral rollout uses these poses.
    v=np.zeros(c.plant.nv);v[3:6]=[.13,-.09,.21]
    mj.mj_integratePos(c.plant,base,v,1.)
    base[c.a_qpos]=np.clip(base[c.a_qpos]+rng.uniform(-.08,.08,42),
                           c.actuator.target_bounds_rad[:,0],c.actuator.target_bounds_rad[:,1])
    c.d.qpos[:]=base;mj.mj_forward(c.plant,c.d)
    errors=[];skipped=0;epsilon=1e-6
    for leg in range(6):
        rel,world,_=foot_jacobians(c,leg)
        analytic=np.vstack([rel[:2],world[2]])
        assert np.max(np.abs(rel[:2,:6]))<1e-12
        for dof in range(c.plant.nv):
            velocity=np.zeros(c.plant.nv);velocity[dof]=1.
            points=[];active=[]
            for sign in [1.,-1.]:
                q=base.copy();mj.mj_integratePos(c.plant,q,velocity,sign*epsilon)
                c.d.qpos[:]=q;mj.mj_forward(c.plant,c.d)
                points.append(measured(c,leg));active.append(minimum_pad(c,leg))
            numeric=(points[0]-points[1])/(2*epsilon)
            errors.extend(np.abs(numeric[:2]-analytic[:2,dof]).tolist())
            if active[0]==active[1]:errors.append(abs(numeric[2]-analytic[2,dof]))
            else:skipped+=1
            c.d.qpos[:]=base;mj.mj_forward(c.plant,c.d)
    assert max(errors)<2e-6,max(errors)
    solver=[]
    for scale in [.01,1.,100.]:
        A=rng.normal(size=(3,7));b=rng.normal(size=3)*scale;bias=rng.normal(size=7)*.01
        lower=np.full(7,-2.);upper=np.full(7,2.);lower[0]=0.;upper[1]=0.
        q,raw,fast=bounded_rate(A,b,bias,lower,upper,.01)
        assert np.all(q>=lower-1e-10) and np.all(q<=upper+1e-10)
        grad=A.T@(A@q-b)+.01*(q-bias)
        interior=(q>lower+1e-6)&(q<upper-1e-6)
        assert np.max(np.abs(grad[interior]),initial=0)<1e-6
        assert np.all(grad[q<=lower+1e-6]>=-1e-6)
        assert np.all(grad[q>=upper-1e-6]<=1e-6)
        solver.append(dict(scale=scale,fast=fast,bounds_hit=int((~interior).sum())))
    c=Core(adhesion=False);other=Core(adhesion=False)
    action=np.full(42,.1)
    expected=np.clip(c.actuator.target_rad+action*2*.005,
                     c.actuator.target_bounds_rad[:,0],c.actuator.target_bounds_rad[:,1])
    c.tick_policy(action,5)
    for _ in range(5):other.tick(action)
    assert np.max(np.abs(c.actuator.target_rad-expected))<1e-13
    assert np.array_equal(c.d.qpos,other.d.qpos)
    # Actual common-height feedback response: paired same-state 500ms settle,
    # then hold phase0 while regulating a small +/-height target for100ms.
    response=[]
    for shift in [-.02,.02]:
        plant=Core(adhesion=False)
        for _ in range(500):plant.tick(np.zeros(42))
        controller=MeasuredTeacher(plant,TeacherConfig())
        h0=float(plant.d.subtree_com[plant.th,2]);controller.height+=shift
        a,info=controller.action(plant,0.)
        predicted=[]
        for leg,ids in enumerate(controller.ids):
            J=foot_jacobians(plant,leg)[1][2,plant.a_dof[ids]]
            predicted.append(float(J@(a[ids]*2)))
        for _ in range(20):
            a,_=controller.action(plant,0.)
            plant.tick_policy(a,5)
        response.append(dict(target_shift=shift,start_height=h0,end_height=float(plant.d.subtree_com[plant.th,2]),
             predicted_mean_stance_joint_vertical_speed=float(np.mean(predicted)),nonfoot=plant.measure()['nonfoot']))
    assert response[1]['predicted_mean_stance_joint_vertical_speed']<response[0]['predicted_mean_stance_joint_vertical_speed']
    assert response[1]['end_height']>response[0]['end_height']
    paths=[HERE/name for name in ['measured_teacher.py','check_teacher.py','physics.py','reference.py']]
    report=dict(all_passed=True,status='PASS',jacobian_max_abs_error=max(errors),jacobian_coordinates=len(errors),
                pad_switch_nonsmooth_coordinates_skipped=skipped,solver_cases=solver,height_responses=response,
                tests=['actual mixed metric FD','relative root columns cancel','BVLS/interior KKT and bounds',
                       '5ms target integration vsfive1ms','actual paired100ms height target response'],
                source_hashes={str(p):sha256(p) for p in paths},
                limitation='engineering fixtures and short feedback response, not walking or real-reference tracking proof')
    atomic_json(HERE/'TEACHER_CHECKS.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
