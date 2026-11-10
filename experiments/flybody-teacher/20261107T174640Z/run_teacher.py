"""Official saved walking policy with equivalent float32/canonical wrappers.
Reference-conditioned ENGINEERING teacher only. No current42-body acceptance claim.
"""
from pathlib import Path
import argparse,json,hashlib,time,subprocess,importlib.metadata
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mujoco
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
ROOT=Path(__file__).resolve().parent
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T174640Z')
LEGS=['T1_left','T2_left','T3_left','T1_right','T2_right','T3_right']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2)

def make_env(seconds=None):
    env=walk_imitation(random_state=np.random.RandomState(41))
    if seconds is not None:
        count=round(seconds/.002)
        qp,qv=constant_speed_trajectory(n_steps=count+65,speed=2,init_pos=(0,0,.1278),control_timestep=.002)
        env.task._traj_generator.set_next_trajectory(qp,qv)
    return env

def observation(ts):return {k:np.asarray(v,dtype=np.float32).copy() for k,v in ts.observation.items()}
def action(policy,obs):
    d=policy({k:tf.convert_to_tensor(v[None]) for k,v in obs.items()})
    result=d.mean()[0].numpy()
    assert result.shape==(59,) and np.isfinite(result).all()
    return result

def real_action(canonical,spec):
    # Match SinglePrecisionWrapper THEN CanonicalSpecWrapper exactly, including in-place float32 arithmetic.
    a=np.clip(canonical,-1.,1.).copy()
    a=.5*(a+1.)
    minimum=spec.minimum.astype(np.float32);maximum=spec.maximum.astype(np.float32)
    a*=maximum-minimum;a+=minimum
    return a

class Recorder:
    def __init__(self,physics):
        self.p=physics;self.m=physics.model;self.d=physics.data
        self.thorax=self.m.name2id('walker/thorax','body')
        self.weight=float(self.m.body_subtreemass[self.thorax]*np.linalg.norm(self.m.opt.gravity))
        self.geom_names=[self.m.id2name(i,'geom') or '' for i in range(self.m.ngeom)]
        self.body_names=[self.m.id2name(i,'body') or '' for i in range(self.m.nbody)]
        self.scratch=mujoco.MjData(self.m.ptr)
        self.rows=[];self.contacts=[];self.ctrls=[];self.states=[]
    def native(self,physics,random_state):
        # dm_control split-step leaves updated contact geometry with preceding solve forces.
        # Recompute consistent diagnostic contact forces on a separate scratch state only.
        # This is explicitly not the saved runtime solver force; live sensors/state untouched.
        d=self.scratch;m=self.m
        d.time=self.d.time
        for key in ['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata']:
            getattr(d,key)[:]=getattr(self.d,key)
        mujoco.mj_forward(m.ptr,d)
        loads=np.zeros(6);nonfoot=0.;slip_num=np.zeros(6)
        for i,c in enumerate(d.contact):
            b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
            name1=self.body_names[b1];name2=self.body_names[b2]
            if b1==0 and name2.startswith('walker/'):
                b=b2;sign=1
            elif b2==0 and name1.startswith('walker/'):
                b=b1;sign=-1
            else:continue
            force=np.zeros(6);mujoco.mj_contactForce(m.ptr,d,i,force)
            world=sign*np.asarray(c.frame).reshape(3,3).T@force[:3]
            name=self.body_names[b]
            leg=next((k for k,s in enumerate(LEGS) if s in name and ('tarsus' in name or 'claw' in name)), -1)
            twist=d.cvel[b];velocity=twist[3:]+np.cross(twist[:3],c.pos-d.xipos[b])
            load=max(0.,float(world[2]))
            if leg>=0:loads[leg]+=load;slip_num[leg]+=load*np.linalg.norm(velocity[:2])
            else:nonfoot+=load
            self.contacts.append(np.r_[d.time,c.geom1,c.geom2,leg,c.dist,c.pos,world,velocity,force[0]])
        self.rows.append(np.r_[d.time,loads,nonfoot,d.subtree_com[self.thorax],d.qpos[:7],slip_num,
            np.max(np.abs(d.qfrc_applied)),np.max(np.abs(d.xfrc_applied)),d.ncon])
    def state(self):
        d=self.d
        self.states.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[self.thorax].copy()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float);ap.add_argument('--tag',required=True);args=ap.parse_args()
    folder=OUT/args.tag
    folder.mkdir(parents=True,exist_ok=False)
    versions={k:importlib.metadata.version(k) for k in ['tensorflow-cpu','tensorflow-probability','numpy','dm-control','mujoco']}
    dump(folder/'PROTOCOL.json',dict(seconds=args.seconds,seed=41,reference='official default300' if args.seconds is None else 'declared constant speed2 root reference extended only',
        identity='official reference-conditioned teacher NOT brain',versions=versions,source=str(Path(__file__)),source_sha256=sha(__file__),policy_sha256=sha(ROOT/'policies/walking/saved_model.pb'),
        contact_force_phase='scratch-forward on copied post-native-step state; not live split-step solver force',
        no_weight_updates=True,real_root_writes='official initial reset only; perstep ghost/reference root writes explicitly separate',native_contact_dt=.0002,
        wrappers='equivalent numpy float32 + canonical clipping/scaling; independently inspect equivalence to Acme source',
        current42_gates='NOT_APPLICABLE different model; record physical metrics, no substituted acceptance'))
    policy=tf.saved_model.load(str(ROOT/'policies/walking'))
    env=make_env(args.seconds);start=time.monotonic()
    try:
        ts=env.reset();physics=env.physics
        assert physics.model.na==59 and physics.model.nu==59
        rec=Recorder(physics);rec.state()
        physics.model.save_binary(str(folder/'MODEL.mjb'))
        dump(folder/'MODEL_IDENTITY.json',dict(nq=physics.model.nq,nv=physics.model.nv,nu=physics.model.nu,na=physics.model.na,neq=physics.model.neq,
            nmesh=physics.model.nmesh,weight=rec.weight,gravity=physics.model.opt.gravity.tolist(),geom_names=rec.geom_names,body_names=rec.body_names,
            action_names=env.action_spec().name.split('\t'),action_min=env.action_spec().minimum.tolist(),action_max=env.action_spec().maximum.tolist(),
            legs=LEGS,control_dt=env.control_timestep(),physics_dt=physics.timestep(),expected_controls=env.task._episode_steps,
            archive_source_sha256=sha(ROOT/'official-flybody/flybody/tasks/walk_imitation.py')))
        original=env.task.after_substep
        def after(phys,random_state):
            original(phys,random_state);rec.native(phys,random_state)
        env.task.after_substep=after
        obsrows={k:[] for k in ts.observation};actions=[];scaled=[];rewards=[];discounts=[]
        limit=env.task._episode_steps
        for step in range(limit):
            obs=observation(ts)
            for k,v in obs.items():obsrows[k].append(v)
            a=action(policy,obs);r=real_action(a,env.action_spec())
            actions.append(a.copy());scaled.append(r.copy())
            before=float(physics.data.time)
            ts=env.step(r)
            assert physics.data.time>before, 'Unexpected automatic reset'
            rewards.append(float(ts.reward));discounts.append(float(ts.discount));rec.state()
            if (step+1)%500==0:print(json.dumps(dict(control=step+1,time=float(physics.data.time),wall_s=time.monotonic()-start)),flush=True)
            if ts.last():break
        np.savez_compressed(folder/'BODY.npz',**{k:np.asarray([s[k] for s in rec.states]) for k in rec.states[0]})
        np.savez_compressed(folder/'OBS.npz',**{k:np.asarray(v) for k,v in obsrows.items()})
        np.savez_compressed(folder/'POLICY.npz',canonical=np.asarray(actions),scaled=np.asarray(scaled),reward=np.asarray(rewards),discount=np.asarray(discounts))
        np.savez_compressed(folder/'NATIVE.npz',rows=np.asarray(rec.rows),contacts=np.asarray(rec.contacts))
        native=np.asarray(rec.rows);last=rec.states[-1];first=rec.states[0]
        displacement=last['qpos'][:3]-first['qpos'][:3]
        result=dict(status='RECORDED_NOT_INDEPENDENTLY_VERIFIED',identity='reference-conditioned pretrained ANN teacher',controls=len(actions),native_steps=len(native),sim_s=float(physics.data.time),
            terminal=bool(ts.last()),reference_end=bool(env.task._reached_traj_end),discount=float(ts.discount),
            root_displacement=displacement.tolist(),com_z_min=float(native[:,10].min()),foot_force_mean_over_weight=float(native[:,1:7].sum(1).mean()/rec.weight),
            nonfoot_max=float(native[:,7].max()),external_force_max=float(np.abs(native[:,24:26]).max()),warnings=physics.data.warning.number.tolist(),
            wall_s=time.monotonic()-start,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
            caveats=['Official adhesive actuators enabled; contact loads alone do not prove muscle-only support.','No current42 support/slip gates or biological validation claimed.','Inference reward constant1 is not measured task success.'])
        dump(folder/'RESULT.json',result);print(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
