"""Bounded physical teacher-reference performance search; parent launch required.

Thirteen declared trials only: baseline and six single-joint +/-0.02rad offsets.
Actual original free-body physics generates every score; no root/state writes
beyond fresh keyframe initialization. Search scores are NOT acceptance gates.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import mujoco as mj

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T083316Z/run_aligned.py'
spec=importlib.util.spec_from_file_location('bounded_search_reference_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
GAIT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z/GAIT.npz')
n.GAIT=GAIT;n.FREQ=1.
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
spec=importlib.util.spec_from_file_location('search_original_actuator',BASE/'incremental_actuator.py')
a_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(a_module)
Actuator=a_module.IncrementalActuator
OUT=GAIT.parent.parent/'20261107T112003Z'
DT,CDT,PDT,SECONDS,HANDOVER=.0000125,.001,.005,3.,1.
W=n.dsm.WEIGHT
JOINTS=('joint_RHCoxa_pitch','joint_RHTrochanter_yaw','joint_LHCoxa_pitch',
        'joint_LMCoxa_yaw','joint_RMCoxa_yaw','joint_RHTrochanter_roll')
LEGS=('LF','LM','LH','RF','RM','RH')
PARAM=dict(parameter_bounds_rad=[-.03,.03],tested_offset_rad=.02,maximum_trials=13,
    seconds=SECONDS,warmup_seconds=HANDOVER,scored_seconds=2.,gain_scale=128.,
    target_rate_cap=30.,native_dt=DT,command_dt=CDT,policy_dt=PDT,
    terminate_COM_below=.8,terminate_nonfoot_force_W=.1,body_endpoint_mean_seconds=.1)


def candidates():
    values=[('baseline',np.zeros(6))]
    for i,name in enumerate(JOINTS):
        for sign,label in [(-1,'minus'),(1,'plus')]:
            x=np.zeros(6);x[i]=sign*PARAM['tested_offset_rad']
            values.append((name+'_'+label,x))
    assert len(values)==13 and all(abs(x).max()<=.03 for _,x in values)
    return values


def smooth5(v):
    x=np.clip(v,0.,1.);return x**3*(10.-15*x+6*x*x)


class Search:
    def __init__(self):
        self.p=n.Plant();self.m=self.p.m;self.ref=n.Reference(self.p)
        a_module.validate_full_model(self.m)
        assert self.m.opt.timestep==DT
        names=[self.m.joint(int(j)).name for j in self.p.jids]
        self.indices=np.array([names.index(j) for j in JOINTS])
        assert len(set(self.indices))==6
        self.lo,self.hi=self.m.jnt_range[self.p.jids].T
        self.physics_steps=0;self.trials_started=0
        self.force=np.zeros(6);self.jp=np.zeros((3,self.m.nv));self.jr=np.zeros_like(self.jp)

    def target(self,t,parameters):
        raw=self.ref.warm(t).copy()
        raw[self.indices]+=smooth5(t/.5)*parameters
        desired=np.clip(raw,self.lo,self.hi)
        return desired,raw!=desired

    def contact(self,d):
        loads=np.zeros(6);numerator=np.zeros(6);nonfoot=0.;velocity_check=0.
        velocities={}
        for ci in range(d.ncon):
            co=d.contact[ci];ga,gb=int(co.geom1),int(co.geom2)
            if not(self.p.planes[ga] or self.p.planes[gb]):continue
            geom=gb if self.p.planes[ga] else ga;li=int(self.p.feet[geom])
            mj.mj_contactForce(self.m,d,ci,self.force)
            normal=float((co.frame.reshape(3,3).T@self.force[:3])[2])*(1 if self.p.planes[ga] else -1)
            if li<0:nonfoot+=max(0.,normal);continue
            if normal<=0:continue
            bid=int(self.m.geom_bodyid[geom])
            mj.mj_jac(self.m,d,self.jp,self.jr,co.pos,bid)
            v=self.jp@d.qvel
            # BODY objectVelocity translation is centered at inertial xipos.
            if bid not in velocities:
                twist=np.zeros(6);mj.mj_objectVelocity(self.m,d,mj.mjtObj.mjOBJ_BODY,bid,twist,0)
                velocities[bid]=twist
            twist=velocities[bid]
            rigid=twist[3:]+np.cross(twist[:3],co.pos-d.xipos[bid])
            velocity_check=max(velocity_check,float(abs(v-rigid).max()))
            loads[li]+=normal;numerator[li]+=normal*np.linalg.norm(v[:2])
        return loads,numerator,nonfoot,velocity_check

    def trial(self,tag,parameters):
        if self.trials_started>=PARAM['maximum_trials']:raise RuntimeError('Hard13-trial budget exhausted')
        if parameters.shape!=(6,) or not np.isfinite(parameters).all() or abs(parameters).max()>.03:
            raise ValueError('Six finite parameters within+/-0.03rad required')
        self.trials_started+=1
        folder=OUT/tag;folder.mkdir(exist_ok=False)
        n.dump(folder/'TRIAL_PREREGISTRATION.json',dict(index=self.trials_started,tag=tag,parameters_rad=parameters,
            joints=JOINTS,source_sha256=n.sha(__file__),seconds=SECONDS,maximum_steps=240000,
            actual_step_count_before=self.physics_steps,physics='original noauxforces/free root/native torque bounds'))
        d=self.p.fresh()  # Sole permitted physical state initialization.
        a=Actuator(self.m,d,kp=self.p.limits/.2*128,kd=self.p.limits*.005*np.sqrt(128),max_rate_rad_s=30.)
        rows=[];body=[];command=[];rate=np.zeros(42);clips=np.zeros(42,int);ticks=0
        reason=None;exception=None;trialsteps=0;start=time.monotonic()
        try:
            for k in range(int(round(SECONDS/CDT))):
                if k%5==0:
                    desired,clipped=self.target(float(d.time),parameters)
                    clips+=clipped;ticks+=1
                    rate=np.clip((desired-a.target_rad)/PDT,-30.,30.)
                command.append(np.r_[d.time,a.target_rad,rate])
                a.update(rate,CDT)
                for _ in range(80):
                    d.ctrl[:]=a.torque(d)
                    assert not d.qfrc_applied.any() and not d.xfrc_applied.any(),'Auxiliary force forbidden'
                    mj.mj_step(self.m,d);self.physics_steps+=1;trialsteps+=1
                    mj.mj_forward(self.m,d)  # Match original every-native-step runtime.
                    if not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                        raise FloatingPointError('Nonfinite physical state')
                    if d.warning.number.any():raise RuntimeError('MuJoCo warning')
                if k%5==4:
                    loads,numerator,nonfoot,check=self.contact(d)
                    R=d.xmat[self.p.thorax].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
                    tilt=float(np.arccos(np.clip(R[2,2],-1.,1.)))
                    rows.append(np.r_[d.time,loads,numerator,nonfoot,d.subtree_com[self.p.thorax],yaw,tilt,
                        np.mean(np.isclose(abs(d.ctrl),a.torque_limit,rtol=1e-7,atol=1e-9)),check])
                    body.append(self.p.bodyrow(d))
                    if check>1e-8:raise RuntimeError('Contact material velocity cross-check failed')
                    if d.time>=HANDOVER-1e-9:
                        if d.subtree_com[self.p.thorax,2]<PARAM['terminate_COM_below']:reason='COM_BELOW_0p8';break
                        if nonfoot>PARAM['terminate_nonfoot_force_W']*W:reason='NONFOOT_ABOVE_0p1W';break
        except (AssertionError,ValueError,RuntimeError,FloatingPointError) as exc:
            reason='EXCEPTION';exception=type(exc).__name__+': '+str(exc)
        data=np.asarray(rows);score=score_trial(data,reason)
        np.savez_compressed(folder/'SEARCH_TRACE.npz',rows=data,commands=np.asarray(command),
            kp=a.kp,kd=a.kd,limits=a.torque_limit,parameters=parameters,
            **{key:np.asarray([r[key] for r in body]) for key in body[0]} if body else {})
        outcome=dict(tag=tag,parameters_rad=parameters,joints=JOINTS,trial_index=self.trials_started,
            actual_physics_steps=trialsteps,actual_physics_step_calls_total=self.physics_steps,
            actual_seconds=float(d.time),termination=reason,exception=exception,finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            warnings=int(d.warning.number.sum()),target_native_angle_clip_fraction=clips/max(ticks,1),
            wall_s=time.monotonic()-start,score=score,diagnostic_only=True,strict_acceptance='NOT_RUN')
        n.dump(folder/'RESULT.json',outcome);print(json.dumps(outcome,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),flush=True)
        return outcome


def score_trial(data,reason):
    # Trace columns: t,Fn6,FweightedSpeed6,nonfoot,COM3,yaw,tilt,capfraction,Jvelocitycheck.
    failed=reason is not None
    if data.ndim!=2 or len(data)==0:return dict(objective=1e6,failed=True,reason='NO_TRACE')
    release=data[data[:,0]>=1.-1e-9]
    if len(release)<40:return dict(objective=1e6+1e4*failed,failed=True,reason='INSUFFICIENT_RELEASE')
    width=min(20,len(release)//4)
    initial=release[:width,14:17].mean(0);final=release[-width:,14:17].mean(0)
    initial_time=release[:width,0].mean();final_time=release[-width:,0].mean()
    yaw=np.unwrap(release[:,17]);heading=float(yaw[:width].mean())
    forward=np.array([np.cos(heading),np.sin(heading)])
    displacement=final[:2]-initial[:2];duration=max(float(final_time-initial_time),1e-12)
    speed=float(displacement@forward/duration);lateral=float(abs(displacement@np.array([-forward[1],forward[0]]))/duration)
    force=release[:,1:7].sum(0);numerator=release[:,7:13].sum(0)
    slips=np.divide(numerator,force,out=np.zeros(6),where=force>0)
    ratios=slips/max(speed,.02)
    support=release[:,1:7].sum(1)/W;nf=release[:,13]/W
    heightdrop=max(float(release[:width,16].mean()-release[:,16].min())-.1,0.)
    headingchange=float(yaw[-width:].mean()-yaw[:width].mean())
    penalty=1000.*failed+100.*float(nf.mean())+100.*float(nf.max())
    penalty+=100.*max(.05-speed,0.)+20.*max(.8-float(support.mean()),0.)
    penalty+=20.*heightdrop+2.*lateral/max(speed,.02)+abs(headingchange)
    penalty+=10.*int((force<=0).any())
    return dict(objective=float(ratios.mean()+penalty),failed=bool(failed),termination=reason,
        foot_normal_weighted_material_speed=slips,slip_to_progress_ratios=ratios,
        forward_speed=speed,lateral_speed=lateral,heading_change=headingchange,
        foot_support_mean=float(support.mean()),nonfoot_force_mean_W=float(nf.mean()),nonfoot_force_max_W=float(nf.max()),
        minimum_COM=float(release[:,16].min()),maximum_tilt_rad=float(release[:,18].max()),
        sampled_torque_cap_fraction=float(release[:,19].mean()),material_velocity_crosscheck_max=float(release[:,20].max()),
        score_window=[float(release[0,0]),float(release[-1,0])],completed_full_duration=bool(data[-1,0]>=SECONDS-1e-8))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',help='Explicit parent launch of preregistered13 actual-physics trials')
    args=ap.parse_args()
    if not args.run:ap.error('No physics by default; parent must launch with --run after source review')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior search '+str(OUT))
    files=[Path(__file__),SOURCE,Path(n.__file__),GAIT,BASE/'incremental_actuator.py',BASE/'plant_probe.py',
        ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',a_module.DEFAULT_MODEL_PATH]
    seen=set()
    def sources(module):
        if id(module) in seen:return
        seen.add(id(module));path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and '/site-packages/' not in str(path):
            files.append(Path(path))
            for value in vars(module).values():
                if isinstance(value,types.ModuleType):sources(value)
    sources(b);sources(a_module)
    hashes={str(f):n.sha(f) for f in set(files)}
    n.dump(OUT/'PREREGISTRATION.json',dict(stage='S5 bounded actualphysics teacher-reference singlecoordinate search',
        parameters=PARAM,joint_order=JOINTS,candidates=[dict(tag=tag,parameters=x) for tag,x in candidates()],source_hashes=hashes,
        references='101114broadlift exactperiodictwo-neighborPCHIP,1Hz, originalwavelegorder. Constantjointoffsets aftersmooth.5sinitialramp; nophase/loadgate.',
        objective='mean6forceweightedmaterialspeed/max(smoothedforwardprogressspeed,.02) +1000*failure +100*meanNonfootW+100*maxNonfootW +100*max(.05-speed,0)+20*max(.8-meanfootsupport,0)+20*max(COMdrop-.1,0)+2*lateralspeed/max(speed,.02)+abs(yawchange)+10*anyunloadedleg',
        measurements='Materialcontactpoint J*qvel every5ms crosscheckedagainstbodytwist+leverarm. PositiveFnweights fromactualgroundcontact; smoothedCOMendpointmeans100ms forprogress. No nativecontactseries insearch.',
        selection='Minimumobjective amongfullycompletedfinitewarningfree/nonfailed trials only; singlebestmeasuredcandidate, no combiningcoordinateeffects or unvalidatedheldoutuse. All13attemptedwithinbudget evenfailedcandidates.',
        early_failure='After1s terminateifactualCOM<.8 oractualnonfootnormalforce>.1W; retainallfailures/integrationcounts. Earlyfailurecannotbechosen.',
        integrity='Original49/48/42/71/free root/neq0 gravity[0,0,-9801],dt12.5us/contact/friction/mass/nativecaps unchanged; originalPD128 andmj_forward everyphysicsstep,1msincrementintegration/5mspolicy,30rad/scap. Actualqpos/qvel onlyfresh initialization.',
        teacher='Engineeringparametersearch only; neithersearchscoresnoralgorithmteachercountasneuralgoal.',
        acceptance='SelectedparameterMUSTlaternewstrictnative3srelease/replay/physicalclearance/steps/heading/materialslip<=.5; 5msscorescannotreplaceoriginalgates. No holdoutorfinalverificationperformedhere.'))
    n.dump(OUT/'SOURCE_SHA.json',hashes)
    search=Search();results=[]
    for tag,parameters in candidates():
        result=search.trial(tag,parameters);results.append(result)
        n.dump(OUT/'SEARCH_RESULTS.json',dict(results=results,trials_started=search.trials_started,
            actual_physics_steps=search.physics_steps,maximum_trials=13))
    valid=[r for r in results if not r['score']['failed'] and r['score'].get('completed_full_duration') and r['warnings']==0 and r['finite']]
    best=min(valid,key=lambda r:r['score']['objective']) if valid else None
    n.dump(OUT/'BEST_PARAMETERS.json',dict(selected=best,baseline=results[0],trials_attempted=search.trials_started,
        actual_physics_steps=search.physics_steps,source_hashes_unchanged={f:n.sha(f)==h for f,h in hashes.items()},
        status='PENDING_STRICT_HELDOUT_NATIVE_REPLAY' if best is not None else 'NO_VALID_TRIAL',neural_success=False))


if __name__=='__main__':main()
