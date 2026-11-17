"""Native limited-body teacher rollouts with declared cervical-command replay."""
import argparse
import types
import time
from body_model import *

COMMAND_PATH=BASE/'20261117T170820Z-body-readaptation/TEACHER/tr_cruise_0/TRACE.npz'


def terminal_mask(self,obs):
    projection=np.dot(obs.stumbling_contact_forces,obs.fly_heading)
    old=(projection[:,:3]<self.stumbling_force_threshold).any(axis=1)
    phases=self.cpg_network.curr_phases%(2*np.pi)
    swing=np.array([self.preprogrammed_steps.swing_period[leg][0]<phases[i]<self.preprogrammed_steps.swing_period[leg][1]+self.swing_extension
                    for i,leg in enumerate(self.legs)])
    return old|((projection[:,3]<self.stumbling_force_threshold)&swing)


def setup(b,kind):
    if kind.startswith('terminal'):b.motor._get_stumbling_mask=types.MethodType(terminal_mask,b.motor)


def run(height,seed,kind,out):
    started=time.monotonic();b=Body(height=height,adhesion=True,seed=seed);b.restore(seed);setup(b,kind)
    commands=np.load(COMMAND_PATH,allow_pickle=False)['trace'][:,6:8].copy()
    out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__).resolve(),HERE/'body_model.py',HERE/'JOINT_LIMITS.json',HERE/'CANONICAL_INIT.npz',COMMAND_PATH]
    write(out/'INPUTS.json',dict(source_sha256={str(p):sha(p) for p in paths},kind=kind,height=height,seed=seed,steps=17000))
    mujoco.mj_saveModel(b.m,str(out/'BODY.mjb'),None)
    initial=b.d.xpos[b.thorax].copy();rows={k:[] for k in ('qpos','qvel','control','actuator_force','pads','pad_force','position','upright','nonfoot','block_force','load','command')}
    warning=None
    for tick,command in enumerate(commands):
        links=('tibia','tarsus1','tarsus2','tarsus5') if kind.startswith('terminal') else ('tibia','tarsus1','tarsus2')
        obs=HybridControllerObservation.from_sim(b.sim,b.name,legs=b.motor.legs,stumbling_links=links)
        action=b.motor.step(command,obs)
        if tick%10==0:
            control=action.joint_angles.copy()
            if kind.endswith('_ff'):
                required=b.d.qfrc_bias[b.vadr]-b.d.qfrc_passive[b.vadr]
                control+=required/b.m.actuator_gainprm[b.actuators,0]
            control=np.clip(control,b.minimum,b.maximum);pads=action.adhesion_onoff.copy()
        b.apply(control,pads)
        load,force,nf,bf=b.contact()
        vals=dict(qpos=b.d.qpos.copy(),qvel=b.d.qvel.copy(),control=control.copy(),actuator_force=b.d.actuator_force[b.actuators].copy(),
             pads=pads.copy(),pad_force=b.d.actuator_force[b.adhesion_ids].copy(),position=b.d.xpos[b.thorax].copy(),upright=b.d.xmat[b.thorax,8],
             nonfoot=nf,block_force=bf,load=load.copy(),command=command.copy())
        for k in rows:rows[k].append(vals[k])
        if b.d.warning.number.any() or not np.isfinite(b.d.qpos).all():warning=b.d.warning.number.tolist();break
    a={k:np.asarray(v) for k,v in rows.items()};a['initial_position']=initial
    np.savez_compressed(out/'TRACE.npz',**a)
    limited=b.m.jnt_limited.astype(bool);jids=np.where(limited)[0];q=a['qpos'][:,b.m.jnt_qposadr[jids]]
    excess=np.maximum(np.maximum(b.m.jnt_range[jids,0]-q,q-b.m.jnt_range[jids,1]),0)
    pos=a['position'];rms=float(np.sqrt(np.mean(np.sum((np.diff(pos[-2000:,:2],axis=0)/DT)**2,axis=1))))
    gates=dict(completed=len(pos)==17000,finite=bool(all(np.isfinite(v).all() for v in a.values())),no_warning=warning is None,
       upright=bool(a['upright'].min()>=.8),height=bool(pos[:,2].min()>=.7*initial[2]),
       nonfoot=bool(a['nonfoot'].max()/b.weight<=.2),joint_limit=bool(excess.max()<=.02),
       target_limits=bool((a['control']>=b.minimum-1e-12).all() and (a['control']<=b.maximum+1e-12).all()),stand=rms<=.30)
    if height:gates.update(block_contact=bool(a['block_force'].max()>1e-6),cross=bool(pos[-1,0]>=3.15))
    else:gates['advance']=bool(pos[-1,0]-initial[0]>=2.)
    result=dict(kind=kind,height=height,seed=seed,pass_=all(gates.values()),gates=gates,dx=float(pos[-1,0]-initial[0]),final_x=float(pos[-1,0]),
       min_height=float(pos[:,2].min()),min_upright=float(a['upright'].min()),stand_rms=rms,
       joint_excess=float(excess.max()),nonfoot_peak_bodyweight=float(a['nonfoot'].max()/b.weight),
       block_contact_ticks=int(np.sum(a['block_force']>1e-6)),steps=len(pos),wall_seconds=time.monotonic()-started,
       warning=warning,trace_sha256=sha(out/'TRACE.npz'),body_sha256=sha(out/'BODY.mjb'),source='teacher_not_ANN')
    assert all(sha(p)==v for p,v in json.loads((out/'INPUTS.json').read_text())['source_sha256'].items())
    write(out/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('native','terminal','terminal_ff'));p.add_argument('height',type=float)
    p.add_argument('seed',type=int);p.add_argument('out',type=Path);a=p.parse_args();run(a.height,a.seed,a.kind,a.out)
