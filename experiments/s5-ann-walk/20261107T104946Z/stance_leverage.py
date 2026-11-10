"""Nine-pose RH load-leverage diagnostic; scratch FK only, no mj_step.

Measured phase-local force is held fixed. RH foot6D pose/root/other joints and
native bounds stay fixed; optimize redundancy to reduce quasistatic joint demand.
This is not a static-standing gate, contact-dynamics model or walking controller.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path.insert(0,str(BASE))
import mujoco as mj
from plant_probe import Plant
from incremental_actuator import DEFAULT_MODEL_PATH
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T104946Z')
GAIT=OUT.parent/'20261107T101114Z/GAIT.npz'
RUN=OUT.parent/'20261107T101753Z/teacher_gain128_4s'
WAVE=OUT.parent/'20261107T081009Z/METADATA.json'
WEIGHT=24.446354755878005


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def dump(p,value):
    p.write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def log_inverse(r):
    x,y,z=r;H=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    angle=np.linalg.norm(r)
    factor=1/12+angle*angle/720 if angle<1e-4 else (1-.5*angle/np.tan(.5*angle))/(angle*angle)
    return np.eye(3)-.5*H+factor*(H@H)


def main():
    def forbidden(*a,**kw):raise AssertionError('No physics in stance leverage diagnosis')
    mj.mj_step=forbidden
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior diagnosis '+str(OUT))
    files=[Path(__file__),GAIT,WAVE,RUN/'BODY.npz',RUN/'NATIVE.npz',RUN/'COMMANDS.npz',RUN/'RESULT.json',
        DEFAULT_MODEL_PATH,BASE/'plant_probe.py',BASE/'incremental_actuator.py',
        ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py']
    hashes={str(f):sha(f) for f in files}
    local_phases=np.linspace(.05,.78,9)
    prereg=dict(stage='S5 nine RH fixed-foot pose redundancy/load-leverage diagnostic ONLY',source_hashes=hashes,
        phases=local_phases,actual_force=dict(folder=str(RUN),nominal_clock='local=(BODY.time-.25+RHoffset)%1',
            window_halfwidth_cycle=.02,release='BODYframe begins>=1s',averaging='unweighted time mean over selected5msframes, including zero-load frames',
            force='Sum actual RH tarsal groundcontact forces fromscratchmj_forward+contactForce; actualworld -> actualThoraxbody -> sourceq0world. Validate worldFz vs same-timeNATIVE RH force, tolerance1e-6.'),
        point='Source RH tarsal geom withsmallest mj_geomDistance tofloor; sourcegeom materialpoint from fromto[:3]. Attachedpoint followsgeom during optimization; foot6D equality keepswholefootgeometryfixed.',
        prediction='tau_RH=(qfrc_bias-qfrc_passive-Jpoint.T*Ffixed)_RH7 at qvel0; neitherforce nor predictedtorque assigned to physics.',
        optimize=dict(method='SLSQP',variables='RH7hingesbyname plus epigraph t',maxiter=100,ftol=1e-9,
            objective='t+1e-4*||q-qsource||²',constraints='Footposition3 + rotationlog3 equalitytosource, nativejointbounds,t>=+/−tau/nativecaps,t>=0',
            start='sourcejointpose (only roundoff-bound projection ifneeded), t=maxnativeutil+1e-9',extra_seeds=0),
        reporting='All nine baseline/optimizedjointutilizations,qdelta,status,poseerrors and unchangedload/root. Poseacceptanceposition.005/orientation.03; solverfailuresretained.',
        limitations='Quasistaticonelegloadcase omits actual qacc/inertia,distributedcontactmoments andotherlegloadadaptation. Fixedroot isofflineFKonly, no fixture. Lowerpredictedtau isnotwalkingfeasibility or proofactualdynamiccapimproves. No gaitasset/controller/ANN.',
        physics_steps=0,neural_steps=0)
    dump(OUT/'PREREGISTRATION.json',prereg)
    begin=time.monotonic();p=Plant();m=p.m;G=load(GAIT);B=load(RUN/'BODY.npz');N=load(RUN/'NATIVE.npz')['rows']
    wave=json.loads(WAVE.read_text());offset=float(wave['phase_offsets'][wave['legs'].index('RH')])
    phase=G['phase'];qg=G['qpos_hinges'];curve=PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],
        np.concatenate([qg[-3:-1],qg,qg[1:3]]),axis=0)
    names=[m.joint(int(j)).name for j in p.jids];ix=np.array([i for i,name in enumerate(names) if name.startswith('joint_RH')])
    assert len(ix)==7 and np.array_equal(p.qa,G['qpos_addresses'])
    qa,va=p.qa[ix],p.va[ix];lo,hi=m.jnt_range[p.jids[ix]].T;limits=p.limits[ix].copy()
    actual=mj.MjData(m);data=mj.MjData(m);data.qpos[:]=p.q0;mj.mj_forward(m,data)
    source_rotation=data.xmat[p.thorax].reshape(3,3).copy()
    times=B['time'];local=(times-.25+offset)%1;release=times-.005>=1.-1e-9
    masks=[release&(abs((local-u+.5)%1-.5)<=.02) for u in local_phases]
    selected=np.any(np.array(masks),axis=0);force_world=np.zeros((len(times),3));force_body=np.zeros_like(force_world)
    force_source=np.zeros_like(force_world);errors=[];force6=np.zeros(6);fk_calls=1
    for i in np.flatnonzero(selected):
        actual.qpos[:]=B['qpos'][i];actual.qvel[:]=B['qvel'][i];actual.ctrl[:]=B['ctrl'][i]
        actual.time=float(times[i]);actual.qacc_warmstart[:]=0.;mj.mj_forward(m,actual);fk_calls+=1
        F=np.zeros(3)
        for ci in range(actual.ncon):
            contact=actual.contact[ci];ga,gb=int(contact.geom1),int(contact.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga
            if p.feet[geom]!=5:continue
            mj.mj_contactForce(m,actual,ci,force6)
            F+=(1 if p.planes[ga] else -1)*(contact.frame.reshape(3,3).T@force6[:3])
        force_world[i]=F;force_body[i]=actual.xmat[p.thorax].reshape(3,3).T@F
        force_source[i]=source_rotation@force_body[i]
        ni=int(round(times[i]/.0000125))-1;errors.append(abs(F[2]-N[ni,6]))
    force_error=float(max(errors,default=0.))
    if force_error>1e-6:
        dump(OUT/'STATUS.json',dict(status='FORCE_RECONSTRUCTION_FAILED',max_error=force_error,physics_steps=0));return 2
    geoms=np.flatnonzero(p.feet==5);floor=int(np.flatnonzero(p.planes)[0]);foot=int(p.bids[5]);fromto=np.zeros(6)
    rows=[]
    for u,mask in zip(local_phases,masks):
        assert mask.any()
        ph=((u-offset)%1)*2*np.pi;full=curve(ph);sourceq=full[ix].copy()
        F=force_source[mask].mean(0)
        data.qpos[:]=p.q0;data.qpos[p.qa]=full;data.qvel[:]=0.;mj.mj_forward(m,data);fk_calls+=1
        desired_pos=data.xpos[foot].copy();desired_rot=data.xmat[foot].reshape(3,3).copy()
        candidates=[]
        for geom in geoms:
            distance=mj.mj_geomDistance(m,data,int(geom),floor,1000.,fromto)
            candidates.append((float(distance),int(geom),fromto[:3].copy()))
        distance,geom,point=min(candidates,key=lambda x:x[0]);bid=int(m.geom_bodyid[geom])
        attached=data.xmat[bid].reshape(3,3).T@(point-data.xpos[bid])
        geom_positions=data.geom_xpos[geoms].copy();cache={}
        def evaluate(q):
            nonlocal fk_calls
            if 'q' in cache and np.array_equal(cache['q'],q):return cache
            data.qpos[:]=p.q0;data.qpos[p.qa]=full;data.qpos[qa]=q;data.qvel[:]=0.
            mj.mj_forward(m,data);fk_calls+=1
            current_point=data.xpos[bid]+data.xmat[bid].reshape(3,3)@attached
            Jp=np.zeros((3,m.nv));mj.mj_jac(m,data,Jp,None,current_point,bid)
            tau=(data.qfrc_bias-data.qfrc_passive-Jp.T@F)[va]
            rv=Rotation.from_matrix(desired_rot.T@data.xmat[foot].reshape(3,3)).as_rotvec()
            Jf=np.zeros((3,m.nv));Jr=np.zeros_like(Jf);mj.mj_jac(m,data,Jf,Jr,data.xpos[foot],foot)
            eq=np.r_[data.xpos[foot]-desired_pos,rv]
            eqjac=np.vstack([Jf[:,va],log_inverse(rv)@desired_rot.T@Jr[:,va]])
            cache.clear();cache.update(q=q.copy(),tau=tau.copy(),eq=eq,eqjac=eqjac,
                point=current_point.copy(),tarsal_geom_position_error=float(abs(data.geom_xpos[geoms]-geom_positions).max()))
            assert np.array_equal(data.qpos[:7],p.q0[:7])
            assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
            return cache
        baseline=evaluate(sourceq);tau0=baseline['tau'].copy()
        qstart=np.clip(sourceq,lo,hi);x0=np.r_[qstart,float(abs(tau0/limits).max())+1e-9]
        def objective(x):return x[7]+1e-4*np.sum((x[:7]-sourceq)**2)
        def objective_jac(x):return np.r_[2e-4*(x[:7]-sourceq),1.]
        def eqjac(x):return np.c_[evaluate(x[:7])['eqjac'],np.zeros(6)]
        constraints=[dict(type='eq',fun=lambda x:evaluate(x[:7])['eq'],jac=eqjac),
            dict(type='ineq',fun=lambda x:np.r_[x[7]-evaluate(x[:7])['tau']/limits,x[7]+evaluate(x[:7])['tau']/limits])]
        fit=minimize(objective,x0,method='SLSQP',jac=objective_jac,bounds=list(zip(lo,hi))+[(0.,None)],
            constraints=constraints,options=dict(maxiter=100,ftol=1e-9,disp=False))
        result=evaluate(fit.x[:7]);tau1=result['tau'].copy();pe=float(np.linalg.norm(result['eq'][:3]));oe=float(np.linalg.norm(result['eq'][3:]))
        row=dict(local_phase=float(u),global_phase_rad=float(ph),samples=int(mask.sum()),
            actual_world_force_mean=force_world[mask].mean(0),actual_body_force_mean=force_body[mask].mean(0),source_world_fixed_force=F,
            actual_world_normal_mean=float(force_world[mask,2].mean()),actual_world_tangential_mean_vector=force_world[mask,:2].mean(0),
            actual_world_tangential_mean_magnitude=float(np.linalg.norm(force_world[mask,:2],axis=1).mean()),
            native_mean_foot_load_W=float(force_world[mask,2].mean()/WEIGHT),
            point_geom=m.geom(geom).name,point_source_world=point,point_optimized_world=result['point'],source_sole_distance=distance,
            joint_names=[names[j] for j in ix],native_limits=limits,source_q=sourceq,optimized_q=fit.x[:7],q_delta=fit.x[:7]-sourceq,
            initial_roundoff_bound_projection=float(abs(qstart-sourceq).max()),baseline_tau=tau0,optimized_tau=tau1,
            baseline_signed_utilization=tau0/limits,optimized_signed_utilization=tau1/limits,
            baseline_max_utilization=float(abs(tau0/limits).max()),optimized_max_utilization=float(abs(tau1/limits).max()),
            baseline_position_error=0.,baseline_orientation_error=0.,position_error=pe,orientation_error=oe,
            epigraph_violation=float(max(abs(tau1/limits).max()-fit.x[7],0.)),
            equality_max_component=float(abs(result['eq']).max()),contact_point_error=float(np.linalg.norm(result['point']-point)),
            tarsal_geom_position_error=result['tarsal_geom_position_error'],solver_success=bool(fit.success),solver_status=int(fit.status),
            solver_message=str(fit.message),iterations=int(fit.nit),function_evaluations=int(fit.nfev),epigraph=float(fit.x[7]),
            native_bound_excess=float(max(np.max(lo-fit.x[:7]),np.max(fit.x[:7]-hi),0.)),
            pose_within_thresholds=bool(pe<=.005 and oe<=.03))
        rows.append(row);dump(OUT/'STANCE_LEVERAGE.json',dict(stage=prereg['stage'],force_reconstruction_max_error=force_error,rows=rows,physics_steps=0))
        print(json.dumps(dict(u=float(u),baseline=row['baseline_max_utilization'],optimized=row['optimized_max_utilization'],success=row['solver_success'],position=pe,orientation=oe,status=row['solver_status'])),flush=True)
    improved=sum(r['solver_success'] and r['pose_within_thresholds'] and r['optimized_max_utilization']<r['baseline_max_utilization']-1e-4 for r in rows)
    final=dict(status='NINE_CASE_DIAGNOSIS_COMPLETE',rows=rows,successful_improved_cases=int(improved),
        force_reconstruction_max_error=force_error,FK_calls=fk_calls,physics_steps=0,neural_steps=0,wall_s=time.monotonic()-begin,
        source_hashes_unchanged={f:sha(f)==h for f,h in hashes.items()},limitation=prereg['limitations'])
    dump(OUT/'STANCE_LEVERAGE.json',final);dump(OUT/'STATUS.json',dict(status=final['status'],successful_improved_cases=improved,physics_steps=0))
    np.savez_compressed(OUT/'LOAD_CASES.npz',local_phases=local_phases,selected_times=times[selected],
        force_world=force_world[selected],force_body=force_body[selected],force_source=force_source[selected],
        baseline_tau=[r['baseline_tau'] for r in rows],optimized_tau=[r['optimized_tau'] for r in rows],
        source_q=[r['source_q'] for r in rows],optimized_q=[r['optimized_q'] for r in rows],native_limits=limits)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(local_phases,[r['baseline_max_utilization'] for r in rows],'o-',label='source posture')
    ax.plot(local_phases,[r['optimized_max_utilization'] for r in rows],'s-',label='bounded same-foot-pose redundancy')
    ax.axhline(1,color='r',ls='--',label='native torque cap');ax.set(xlabel='RH nominal stance cycle phase',ylabel='predicted quasistatic max |tau/nativecap|',title='Measured-force leverage hypothesis only; actual dynamic torque differs; 0 physics')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'LEVERAGE.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status=final['status'],successful_improved_cases=improved,FK_calls=fk_calls,wall_s=final['wall_s'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
