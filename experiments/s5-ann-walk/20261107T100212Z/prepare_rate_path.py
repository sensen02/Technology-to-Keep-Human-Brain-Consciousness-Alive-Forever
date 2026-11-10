"""Offline position-only trajectory IK with hard joint/rate constraints.

Only swing poses are variables; stance and periodic endpoints stay source exact.
No filter/reprojection loop, physics integration, or export of failed gait assets.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T095519Z/prepare_swing_free.py'
DEPS=ROOT/'experiments/s5-ann-walk/20261107T092817Z/deps'
sys.path.insert(0,str(DEPS))
import osqp
spec=importlib.util.spec_from_file_location('frozen_free_swing_geometry',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
mj=b.mj
INPUT=b.OUT
OUT=INPUT.parent/'20261107T100212Z'
PARAM=dict(source_prior=.001,D2_residual_weight=.01,trust_step_rad=.05,
    knot_speed_rad_s=20.,knot_dt_s=1/512,maximum_GN_iterations=30,
    backtracking=[1.,.5,.25],solver_eps=1e-9,solver_max_iter=20000)


class LegPath:
    def __init__(self,prep,leg,desired,seed):
        self.p=prep;self.leg=leg;self.ix=prep.ids[leg]
        self.free=np.flatnonzero(~prep.stance[:512,leg]);self.n=len(self.free)*7
        self.source=prep.old[:512,self.ix].copy();self.goals=desired[self.free,leg]
        self.seed=seed[self.free][:,self.ix]-self.source[self.free]
        rows=(self.free[:,None]*7+np.arange(7)).ravel()
        self.B=sparse.csc_matrix((np.ones(self.n),(rows,np.arange(self.n))),shape=(512*7,self.n))
        D1=sparse.lil_matrix((512,512));D2=sparse.lil_matrix((512,512))
        for i in range(512):
            D1[i,i]=-1;D1[i,(i+1)%512]=1
            D2[i,(i-1)%512]=1;D2[i,i]=-2;D2[i,(i+1)%512]=1
        self.D1=sparse.kron(D1.tocsr(),sparse.eye(7),format='csc')
        self.D2=sparse.kron(D2.tocsr(),sparse.eye(7),format='csc')
        self.V=self.D1@self.B;self.R=self.D2@self.B
        self.baseD1=np.asarray(self.D1@self.source.ravel()).ravel()
        self.baseD2=np.asarray(self.D2@self.source.ravel()).ravel()
        step=PARAM['knot_speed_rad_s']*PARAM['knot_dt_s']
        self.lower=np.r_[(prep.lo[self.ix]-self.source[self.free]).ravel(),-step-self.baseD1]
        self.upper=np.r_[(prep.hi[self.ix]-self.source[self.free]).ravel(),step-self.baseD1]
        self.A=sparse.vstack([sparse.eye(self.n),self.V],format='csc')
        fixed=np.asarray(abs(self.V).sum(axis=1)).ravel()==0
        self.fixed_violation=float(max(np.max(self.lower[self.n:][fixed]),np.max(-self.upper[self.n:][fixed]),0.))
        self.fixed_stance_knot_speed_max=float(abs(self.baseD1[fixed]).max()/PARAM['knot_dt_s'])
        blocks=sparse.block_diag([np.ones((7,7)) for _ in self.free],format='csc')
        pattern=sparse.triu(blocks+(self.R.T@self.R).astype(bool),format='csc')
        self.pc=np.repeat(np.arange(self.n),np.diff(pattern.indptr));self.pr=pattern.indices.copy()
        self.pattern=pattern;self.solver=None;self.logs=[]

    def matrix_values(self,H):
        return np.asarray(H.tocsr()[self.pr,self.pc]).ravel()

    def solver_options(self):
        return dict(verbose=False,eps_abs=PARAM['solver_eps'],eps_rel=PARAM['solver_eps'],
            max_iter=PARAM['solver_max_iter'],polishing=True,warm_starting=True)

    def metrics(self,x):
        q=self.source.copy();q[self.free]+=x.reshape(-1,7)
        excess=float(max(np.max(self.p.lo[self.ix]-q),np.max(q-self.p.hi[self.ix]),0.))
        speed=float(abs(self.D1@q.ravel()).max()/PARAM['knot_dt_s'])
        violation=float(max(np.max(self.lower-self.A@x),np.max(self.A@x-self.upper),0.))
        return dict(native_bound_excess=excess,knot_speed_max=speed,constraint_violation=violation)

    def initialize(self):
        if self.fixed_violation>1e-10:
            return None,dict(status='FIXED_STANCE_RATE_INFEASIBLE',fixed_violation=self.fixed_violation,
                fixed_stance_knot_speed_max=self.fixed_stance_knot_speed_max)
        solver=osqp.OSQP()
        solver.setup(P=sparse.eye(self.n,format='csc'),q=-self.seed.ravel(),A=self.A,
            l=self.lower,u=self.upper,**self.solver_options())
        result=solver.solve(raise_error=False)
        x=result.x
        info=dict(status=result.info.status,iterations=int(result.info.iter),
            fixed_stance_knot_speed_max=self.fixed_stance_knot_speed_max)
        if x is None or not np.isfinite(x).all():return None,info
        info.update(self.metrics(x))
        valid=result.info.status_val in (1,2) and info['constraint_violation']<=1e-8
        return x.copy() if valid else None,info

    def evaluate(self,x):
        errors=[];blocks=[]
        for k,i in enumerate(self.free):
            q=self.p.old[i].copy();q[self.ix]=self.source[i]+x[k*7:k*7+7]
            self.p.fk(q);bid=self.p.p.bids[self.leg];point=self.p.d.xpos[bid]
            errors.append(point-self.goals[k])
            J=np.zeros((3,self.p.m.nv));mj.mj_jac(self.p.m,self.p.d,J,None,point,bid)
            blocks.append(J[:,self.p.p.va[self.ix]])
        e=np.asarray(errors);J=sparse.block_diag(blocks,format='csc')
        d2=self.baseD2+self.R@x
        objective=float(np.sum(e*e)+PARAM['source_prior']**2*(x@x)+PARAM['D2_residual_weight']**2*(d2@d2))
        return objective,e,J

    def step(self,x,error,J):
        w=PARAM['D2_residual_weight']**2
        H=J.T@J+PARAM['source_prior']**2*sparse.eye(self.n)+w*(self.R.T@self.R)
        linear=-J.T@(J@x-error.ravel())+w*(self.R.T@self.baseD2)
        lower=self.lower.copy();upper=self.upper.copy()
        lower[:self.n]=np.maximum(lower[:self.n],x-PARAM['trust_step_rad'])
        upper[:self.n]=np.minimum(upper[:self.n],x+PARAM['trust_step_rad'])
        if self.solver is None:
            P=self.pattern.copy();P.data=self.matrix_values(H)
            self.solver=osqp.OSQP();self.solver.setup(P=P,q=np.asarray(linear).ravel(),A=self.A,
                l=lower,u=upper,**self.solver_options());self.solver.warm_start(x=x)
        else:self.solver.update(Px=self.matrix_values(H),q=np.asarray(linear).ravel(),l=lower,u=upper)
        result=self.solver.solve(raise_error=False)
        return result

    def optimize(self,x):
        current=self.evaluate(x)
        for iteration in range(PARAM['maximum_GN_iterations']):
            objective,error,J=current
            result=self.step(x,error,J)
            log=dict(iteration=iteration,status=result.info.status,solver_iterations=int(result.info.iter),
                objective_before=objective,position_max_before=float(np.linalg.norm(error,axis=1).max()))
            if result.x is None or not np.isfinite(result.x).all() or result.info.status_val not in (1,2):
                log['termination']='QP_FAILED';self.logs.append(log);break
            candidate=result.x;check=self.metrics(candidate)
            if check['constraint_violation']>1e-8:
                log.update(termination='QP_BOUND_VIOLATION',**check);self.logs.append(log);break
            accepted=False
            for fraction in PARAM['backtracking']:
                trial=x+fraction*(candidate-x);values=self.evaluate(trial)
                if values[0]<objective-1e-14:
                    accepted=True;break
            if not accepted:
                log.update(termination='NO_OBJECTIVE_DECREASE',**self.metrics(x));self.logs.append(log);break
            stepmax=float(abs(trial-x).max());x=trial;current=values
            log.update(accepted_fraction=fraction,step_max=stepmax,objective_after=values[0],
                position_max_after=float(np.linalg.norm(values[1],axis=1).max()),**self.metrics(x))
            self.logs.append(log)
            if iteration%5==0:
                print(json.dumps(dict(leg=b.LEGS[self.leg],GN=iteration,position=log['position_max_after'],knot_speed=log['knot_speed_max'],status=result.info.status)),flush=True)
            if stepmax<1e-7:break
        return x,current


def plot(prep,qknots,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=b.b.util.v2.periodic_pchip(prep.phase,qknots)
    phase=np.linspace(0,2*np.pi,4097);velocity=curve(phase,1)*2*np.pi
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,constrained_layout=True)
    for li,ax in enumerate(axes.flat):
        for j in prep.ids[li]:ax.plot(phase/(2*np.pi),velocity[:,j],lw=.8)
        ax.axhline(30,color='r',ls='--');ax.axhline(-30,color='r',ls='--')
        ax.set(title=b.LEGS[li],xlabel='cycle',ylabel='joint rad/s');ax.grid(alpha=.2)
    fig.suptitle('Hard knot-rate constrained whole-swing IK; exact PCHIP max '+format(result['max_rate_1hz'],'.5f')+'rad/s\nFK only, 0 physics; geometry acceptance '+str(result['passed']))
    fig.savefig(OUT/'RATE_PATH.png',dpi=150);plt.close(fig)


def main():
    def forbidden(*a,**k):raise AssertionError('No mj_step in rate-constrained trajectory preparation')
    mj.mj_step=forbidden
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior candidate '+str(OUT))
    files=[Path(__file__),SOURCE,b.OLD,b.b.SOURCE/'GAIT.npz',b.b.WAVE,
        INPUT/'CORRECTIONS.npz',INPUT/'FK_EVIDENCE.npz',INPUT/'PREREGISTRATION.json',
        b.b.UTIL,Path(b.b.util.v2.__file__),Path(b.b.util.v1.__file__),
        b.b.BASE/'plant_probe.py',b.b.BASE/'incremental_actuator.py',b.b.DEFAULT_MODEL_PATH,
        ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',ROOT/'tools/dynamic_support_metrics.py',
        ROOT/'experiments/s5-ann-walk/20261107T092817Z/DEPENDENCIES.json',
        Path(osqp.__file__),DEPS/'osqp/interface.py',Path(osqp.ext_builtin.__file__)]
    hashes={str(p):b.b.sha(p) for p in files}
    prereg=dict(stage='S5 explicit-rate constrained whole-swing position IK',source_hashes=hashes,parameters=PARAM,
        reason='095519130.91rad/s failure was position-nullspace boundary-return motion; measured Cartesian velocity task did not require30rad/s.',
        unchanged='095519fixedrootCartesianXYZtargets,duty5/6,.25height,XYwarp,airorientationfree;083316stanceknotsbitwise includingperiodicendpoint',
        variables='Only7hingesbyname foreachleg swingknots. Allstance rows eliminated/fixed. Sparse GN positionJacobians with actual mj_jac.',
        initialization='Unique nearest constrained projection of095519finaljointcurve; raw curve retained only as inputdiagnostic. Nativebounds andadjacentallphaseknotdifferences≤20/512rad includingfixedstanceboundaries.',
        objective='sum||Cartesianpositionerror||² + .001²||source-relativejointdelta||² + .01²||periodicD2 fulljointtrajectory||²',
        solver=dict(name='OSQP',version=osqp.__version__,warm_start=True,sparse_fixed_Hessian_pattern=True),
        iterations='Atmost30GN perleg, truststep.05rad, actualnonlinearobjectivebacktracking[1,.5,.25], stopifnoimprovementorstep<1e-7; noextracycles',
        acceptance='Dense4097position.005,actualwholefootplateauclearance.1,stancesolemedian/pointdeviation.005,nativebounds0(tol1e-9),periodicPCHIPseam1e-10,exactrate≤30at1Hz',
        rate_bound='20rad/s finiteknotspeed intended to allow1.5uniformPCHIPfactor; exact cubicderivative extrema independently checked, never assumed.',
        no_postprocessing='No filtering, pointwise reproject, targetclipping, lowerlift or changedfrequency.',
        export='GAIT onlyifALLfinalchecks pass; retainedfailedFKnotforphysics',physics_steps=0,neural_steps=0)
    b.b.dump(OUT/'PREREGISTRATION.json',prereg);b.b.dump(OUT/'SOURCE_SHA.json',hashes)
    start=time.monotonic();prep=b.Preparation()
    C=b.b.load(INPUT/'CORRECTIONS.npz');E=b.b.load(INPUT/'FK_EVIDENCE.npz')
    assert np.array_equal(prep.phase,E['phase'][::8])
    target=E['target_world'][::8];seed=C['source']+C['final'];qknots=prep.old.copy()
    logs=[];initial=[];solutions=[]
    for li in range(6):
        problem=LegPath(prep,li,target,seed);x,info=problem.initialize();info['leg']=b.LEGS[li]
        initial.append(info);b.b.dump(OUT/'INITIALIZATION.json',initial)
        print(json.dumps(dict(initialization=info)),flush=True)
        if x is None:
            status=dict(status='RATE_INITIALIZATION_FAILED',failed_leg=b.LEGS[li],initialization=initial,
                physics_steps=0,neural_steps=0,FK_calls=prep.calls,wall_s=time.monotonic()-start)
            b.b.dump(OUT/'STATUS.json',status);return 2
        x,values=problem.optimize(x)
        qknots[problem.free[:,None],problem.ix]=problem.source[problem.free]+x.reshape(-1,7)
        logs.append(dict(leg=b.LEGS[li],iterations=problem.logs,final_objective=values[0],
            final_position_max=float(np.linalg.norm(values[1],axis=1).max()),**problem.metrics(x)))
        solutions.append(dict(leg=b.LEGS[li],free_knots=problem.free,delta=x.reshape(-1,7)))
        b.b.dump(OUT/'GN_LOG.json',logs)
    qknots[-1]=qknots[0]
    result,evidence=prep.evaluate(qknots)
    result['checks']['all_knot_rate_constraints']=bool(all(l['knot_speed_max']<=20+1e-7 for l in logs))
    result['passed']=all(result['checks'].values())
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',**evidence)
    np.savez_compressed(OUT/'TRAJECTORY.npz',phase=prep.phase,qpos_hinges=qknots,source=prep.old,
        delta=qknots-prep.old,target_world=target,qpos_addresses=prep.p.qa,joint_names=prep.names)
    meta=json.loads(str(prep.source['metadata_json'].item()))
    meta.update(version='lift-first-rate-constrained-v3',source=str(b.b.SOURCE/'GAIT.npz'),
        fk_feasible=bool(result['passed']),checks=result['checks'],air_orientation_free=True,
        max_rate_at_1hz=result['max_rate_1hz'],teacher_only=True,physics_steps=0,
        preparation=str(Path(__file__).resolve()),preparation_sha256=hashes[str(Path(__file__))])
    if result['passed']:
        np.savez_compressed(OUT/'GAIT.npz',phase=prep.phase,qpos_hinges=qknots,qpos_addresses=prep.p.qa,
            joint_names=prep.names,metadata_json=np.array(json.dumps(meta)))
    result.update(status='FK_ACCEPTED' if result['passed'] else 'DENSE_GEOMETRY_OR_RATE_FAILED',
        gait_exported=bool(result['passed']),physics_steps=0,neural_steps=0,FK_calls=prep.calls,
        GN_legs=logs,wall_s=time.monotonic()-start,source_hashes_unchanged={p:b.b.sha(p)==h for p,h in hashes.items()})
    b.b.dump(OUT/'VERIFICATION.json',result);b.b.dump(OUT/'STATUS.json',result);b.b.dump(OUT/'METADATA.json',meta)
    plot(prep,qknots,result)
    print(json.dumps(dict(status=result['status'],checks=result['checks'],position_max=result['position_max'],max_rate=result['max_rate_1hz'],plateau_clearance=result['plateau_min_clearance'],FK_calls=prep.calls,wall_s=result['wall_s'])),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
