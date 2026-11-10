"""Fixed-budget PPO physical residual ANN; parent launches --train explicitly.

Frozen learned40-frame motor base +477-input residual actor/critic. No phase/time,
world position, root height or teacher after1s warmup. Original finite PD/native
plant preserved. Training5ms rewards/traces never replace strict native acceptance.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import mujoco as mj

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T111149Z/run_velocity.py'
spec=importlib.util.spec_from_file_location('residual_original_motor_runtime',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
SENSORY_SOURCE=ROOT/'experiments/s5-ann-walk/20261107T091243Z/proprioception.py'
spec=importlib.util.spec_from_file_location('residual_fixed_sensory',SENSORY_SOURCE)
sensory=importlib.util.module_from_spec(spec);spec.loader.exec_module(sensory)
BASE_MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T111149Z/MODEL_VELOCITY.npz')
OUT=BASE_MODEL.parent.parent/'20261107T123634Z'
JOINTS=('joint_RHCoxa_pitch','joint_RHTrochanter_yaw','joint_LHCoxa_pitch',
        'joint_LMCoxa_yaw','joint_RMCoxa_yaw','joint_RHTrochanter_roll')
CFG=dict(seed=123634,updates=32,rollout=256,decisions=8192,epochs=6,minibatch=64,
    lr=3e-4,clip=.2,gamma=.995,gae=.95,value_coef=.5,entropy_coef=0.,grad_norm=.5,target_kl=.03,
    residual_rad=.003,initial_logstd=-3.,logstd_bounds=[-5.,-1.5],episode_total_s=8.,warmup_s=1.,
    terminate_COM=.95,terminate_nonfoot_W=.05,forward_filter_tau=.1,policy_dt=.005,
    maximum_warmups=16,maximum_native_steps=6000000)
W=n.dsm.WEIGHT


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0=nn.Linear(477,128);self.fc1=nn.Linear(128,128)
        self.actor=nn.Linear(128,6);self.critic=nn.Linear(128,1)
        self.logstd=nn.Parameter(torch.full((6,),CFG['initial_logstd']))
        nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)

    def forward(self,obs):
        h=torch.tanh(self.fc1(torch.tanh(self.fc0(obs))))
        return self.actor(h),self.critic(h).squeeze(-1)

    def evaluate(self,obs,latent):
        mean,value=self(obs)
        logstd=self.logstd.clamp(*CFG['logstd_bounds'])
        normal=torch.distributions.Normal(mean,logstd.exp())
        # Stable log(1-tanh(u)^2); squash Jacobian is included for every action.
        logjac=2*(math.log(2.)-latent-F.softplus(-2*latent))
        logprob=(normal.log_prob(latent)-logjac).sum(-1)
        return logprob,value

    def sample(self,obs):
        mean,_=self(obs)
        latent=mean+self.logstd.clamp(*CFG['logstd_bounds']).exp()*torch.randn_like(mean)
        logprob,value=self.evaluate(obs,latent)
        return torch.tanh(latent),latent,logprob,value,mean


class NumpyActor:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:self.w={k:z[k].copy() for k in z.files}

    def __call__(self,obs):
        w=self.w;x=np.asarray(obs,dtype=np.float32)
        h=np.tanh(x@w['actor_w0'].T+w['actor_b0']);h=np.tanh(h@w['actor_w1'].T+w['actor_b1'])
        return np.tanh(h@w['actor_w2'].T+w['actor_b2']), (h@w['critic_w'].T+w['critic_b']).squeeze(-1)


def residual_rate(base,target,action,indices):
    w=base.w;pred=base.predict_normalized(base.buf[None])[0]
    qnext=w['center']+w['scale']*pred
    delta=np.zeros(42);delta[indices]=CFG['residual_rad']*np.asarray(action,dtype=float)
    return np.clip((qnext+delta-target)/n.PDT,-n.CAP,n.CAP),qnext.copy(),delta


def gae(rewards,values,next_values,terminated,done):
    advantages=np.zeros_like(rewards,dtype=np.float64);last=0.
    for i in range(len(rewards)-1,-1,-1):
        td=rewards[i]+CFG['gamma']*next_values[i]*(not terminated[i])-values[i]
        last=td+CFG['gamma']*CFG['gae']*(not done[i])*last;advantages[i]=last
    return advantages.astype(np.float32),(advantages+values).astype(np.float32)


def ppo_update(net,optimizer,batch,rng):
    obs=torch.from_numpy(batch['obs']);latent=torch.from_numpy(batch['latent'])
    oldlog=torch.from_numpy(batch['logprob']);returns=torch.from_numpy(batch['returns'])
    advantage=torch.from_numpy(batch['advantage']);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    logs=[];stop=False
    for epoch in range(CFG['epochs']):
        for ids in rng.permutation(len(obs)).reshape(-1,CFG['minibatch']):
            lp,value=net.evaluate(obs[ids],latent[ids]);logratio=lp-oldlog[ids];ratio=logratio.exp()
            policy=torch.maximum(-advantage[ids]*ratio,-advantage[ids]*ratio.clamp(1-CFG['clip'],1+CFG['clip'])).mean()
            value_loss=((value-returns[ids])**2).mean()
            loss=policy+CFG['value_coef']*value_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
            kl=float(((ratio-1)-logratio).mean().detach())
            if kl>CFG['target_kl']:
                logs.append(dict(epoch=epoch,KL=kl,stopped_before_update=True));stop=True;break
            optimizer.zero_grad();loss.backward();gradient=float(nn.utils.clip_grad_norm_(net.parameters(),CFG['grad_norm']))
            if not np.isfinite(gradient):raise FloatingPointError('Nonfinite PPO gradient')
            optimizer.step()
            with torch.no_grad():net.logstd.clamp_(*CFG['logstd_bounds'])
            logs.append(dict(epoch=epoch,loss=float(loss.detach()),policy=float(policy.detach()),value=float(value_loss.detach()),
                KL=kl,gradient_norm_before_clip=gradient,clip_fraction=float(((ratio-1).abs()>CFG['clip']).float().mean().detach())))
        if stop:break
    return dict(minibatches=logs,early_KL_stop=stop,logstd=net.logstd.detach().numpy().copy())


class PhysicalFailure(RuntimeError):pass


class Environment:
    def __init__(self):
        self.p=n.Plant();self.m=self.p.m;self.ref=n.Reference(self.p)
        assert (self.m.nq,self.m.nv,self.m.nu,self.m.nmesh,self.m.neq)==(49,48,42,71,0)
        assert self.m.opt.timestep==n.DT and np.array_equal(self.m.opt.gravity,[0.,0.,-9801.])
        names=[self.m.joint(int(j)).name for j in self.p.jids]
        self.indices=np.array([names.index(j) for j in JOINTS])
        self.total=dict(physics_steps=0,command_ticks=0,warmup_policy_ticks=0,neural_policy_decisions=0,
            warmups_started=0,resets=0,episodes_saved=0,failures=0)
        self.episode=-1;self.closed=True;self.failures=[]

    def contacts(self):
        d=self.d;loads=np.zeros(6);slipsq=0.;nonfoot=0.;check=0.;force=np.zeros(6)
        for ci in range(d.ncon):
            c=d.contact[ci];ga,gb=int(c.geom1),int(c.geom2)
            if not(self.p.planes[ga] or self.p.planes[gb]):continue
            geom=gb if self.p.planes[ga] else ga;leg=int(self.p.feet[geom])
            mj.mj_contactForce(self.m,d,ci,force)
            normal=float((c.frame.reshape(3,3).T@force[:3])[2])*(1 if self.p.planes[ga] else -1)
            if leg<0:nonfoot+=max(0.,normal);continue
            if normal<=0:continue
            J=np.zeros((3,self.m.nv));bid=int(self.m.geom_bodyid[geom]);mj.mj_jac(self.m,d,J,None,c.pos,bid)
            v=J@d.qvel;twist=np.zeros(6);mj.mj_objectVelocity(self.m,d,mj.mjtObj.mjOBJ_BODY,bid,twist,0)
            rigid=twist[3:]+np.cross(twist[:3],c.pos-d.xipos[bid]);check=max(check,float(abs(v-rigid).max()))
            loads[leg]+=normal;slipsq+=normal*np.sum(v[:2]**2)
        if check>1e-8:raise PhysicalFailure('Contact Jacobian/twist mismatch')
        return loads,float(slipsq),float(nonfoot),check

    def prepare(self):
        if self.prepared:return self.obs
        self.base.observe(self.a.target_rad)  # Exactlyonce/tick, includingallwarmup.
        loads,_,_,_=self.contacts()
        proprio=sensory.features(self.p,self.d,self.a.target_rad,loads)
        self.obs=np.r_[proprio,self.base.buf[self.base.lags].ravel()].astype(np.float32)
        assert self.obs.shape==(477,) and np.isfinite(self.obs).all()
        self.prepared=True;return self.obs

    def advance(self,rate):
        for _ in range(5):
            self.commands.append(np.r_[self.d.time,self.a.target_rad,rate,np.zeros(42)])
            self.a.update(rate,n.CDT);self.total['command_ticks']+=1
            for _ in range(80):
                self.d.ctrl[:]=self.a.torque(self.d)
                if self.d.qfrc_applied.any() or self.d.xfrc_applied.any():raise PhysicalFailure('Auxiliary force')
                if self.total['physics_steps']>=CFG['maximum_native_steps']:raise RuntimeError('Native-step resource guard reached')
                mj.mj_step(self.m,self.d);self.total['physics_steps']+=1;mj.mj_forward(self.m,self.d)
                if not(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all()):raise PhysicalFailure('Nonfinite actual state')
                if self.d.warning.number.any():raise PhysicalFailure('MuJoCo warning')

    def measure(self,action):
        loads,slipsq,nonfoot,check=self.contacts();R=self.d.xmat[self.p.thorax].reshape(3,3)
        velocity=R.T@self.d.qvel[:3];omega=self.d.qvel[3:6];gravity=R.T@np.array([0.,0.,-1.])
        beta=-np.expm1(-n.PDT/CFG['forward_filter_tau'])
        self.filtered_forward+=beta*(velocity[0]-self.filtered_forward)
        force=loads.sum()/W
        reward=float(np.clip(self.filtered_forward/.3,-1.,1.5)-2*slipsq/(W*.3**2)
            -.05*np.sum(omega**2)-.01*np.sum(gravity[:2]**2)-.01*np.mean(action**2)
            -5*(force-1)**2-50*nonfoot/W)
        if not np.isfinite(reward):raise PhysicalFailure('Nonfinite training reward')
        fail=None
        if self.d.subtree_com[self.p.thorax,2]<CFG['terminate_COM']:fail='COM_BELOW_0p95'
        if nonfoot>CFG['terminate_nonfoot_W']*W:fail='NONFOOT_ABOVE_0p05W'
        return reward,fail,dict(loads=loads,Fn_weighted_slip_squared=slipsq,nonfoot_force=nonfoot,
            filtered_forward_velocity=self.filtered_forward,body_linear_velocity=velocity,body_omega=omega,
            gravity_body=gravity,velocity_crosscheck=check)

    def record(self,obs,rate,action,latent,logprob,value,reward,done,terminal,control,extra,mean=None,base_next=None,delta=None):
        self.policy.append(dict(time=self.policy_time,observation=obs.copy(),target_before=self.target_before.copy(),
            commanded_rate=rate.copy(),residual_action=np.asarray(action).copy(),latent_action=np.asarray(latent).copy(),
            actor_mean_raw=np.zeros(6) if mean is None else np.asarray(mean).copy(),
            base_qnext=np.zeros(42) if base_next is None else base_next.copy(),
            residual_delta_q=np.zeros(42) if delta is None else delta.copy(),
            logprob=float(logprob),critic_value=float(value),reward=float(reward),done=done,terminated=terminal,
            control=control,**extra))
        self.body.append(self.p.bodyrow(self.d))

    def reset(self):
        assert self.closed,'Mustsaveoldphysicalepisodebeforereset'
        if self.total['warmups_started']>=CFG['maximum_warmups']:raise RuntimeError('Physical-warmup resource guard reached')
        self.episode+=1;self.total['resets']+=1;self.total['warmups_started']+=1
        self.d=self.p.fresh()  # Onlyactualstateinitialization: neverassignactualqpos/qvelafterthis.
        self.a=n.IncrementalActuator(self.m,self.d,kp=self.p.limits/.2*2.,kd=self.p.limits*.005*np.sqrt(2.),max_rate_rad_s=30.)
        assert np.allclose(self.a.kp,self.p.limits/.2*128) and np.allclose(self.a.kd,self.p.limits*.005*np.sqrt(128))
        self.base=n.Policy(BASE_MODEL);self.commands=[];self.policy=[];self.body=[];self.closed=False
        self.filtered_forward=0.;self.prepared=False;self.initial_snapshot=self.p.snapshot(self.d)
        self.handover=None;self.ann_steps=0
        for _ in range(200):
            obs=self.prepare().copy();self.policy_time=float(self.d.time);self.target_before=self.a.target_rad.copy()
            rate=np.clip((self.ref.warm(self.d.time)-self.a.target_rad)/n.PDT,-30.,30.)
            self.advance(rate);self.prepared=False;self.total['warmup_policy_ticks']+=1
            _,failure,metrics=self.measure(np.zeros(6))
            self.record(obs,rate,np.zeros(6),np.zeros(6),0.,0.,0.,False,False,'teacher_warmup',metrics)
            if failure:raise PhysicalFailure('Warmup failure: '+failure)
        self.handover=dict(physics_integration=self.p.snapshot(self.d),target=self.a.target_rad.copy())
        obs=self.prepare();self.handover['history']=self.base.buf.copy()
        assert np.array_equal(self.handover['physics_integration'],self.p.snapshot(self.d))
        return obs

    def step(self,action,latent,logprob,value,mean):
        obs=self.prepare().copy();self.policy_time=float(self.d.time);self.target_before=self.a.target_rad.copy()
        rate,baseqnext,delta=residual_rate(self.base,self.a.target_rad,action,self.indices)
        self.total['neural_policy_decisions']+=1;self.ann_steps+=1;failure=None;metrics={}
        try:
            self.advance(rate);self.prepared=False
            reward,failure,metrics=self.measure(action)
        except (PhysicalFailure,ValueError,FloatingPointError) as exc:
            reward=0.;failure=type(exc).__name__+': '+str(exc);self.prepared=False
            metrics=dict(loads=np.zeros(6),Fn_weighted_slip_squared=0.,nonfoot_force=0.,filtered_forward_velocity=self.filtered_forward,
                body_linear_velocity=np.zeros(3),body_omega=np.zeros(3),gravity_body=np.zeros(3),velocity_crosscheck=0.)
        terminal=failure is not None
        if terminal:
            reward-=25.
            self.total['failures']+=1;self.failures.append(dict(episode=self.episode,time=float(self.d.time),reason=failure))
        truncated=self.ann_steps>=1400;done=terminal or truncated
        self.record(obs,rate,action,latent,logprob,value,reward,done,terminal,'residual_ANN',metrics,mean=mean,base_next=baseqnext,delta=delta)
        self.last_base_next=baseqnext;self.last_delta=delta
        nextobs=np.zeros(477,np.float32) if terminal else self.prepare().copy()
        return nextobs,reward,terminal,done,failure

    def save(self,reason):
        if self.closed:return
        folder=OUT/'episodes';folder.mkdir(exist_ok=True);path=folder/('episode_'+format(self.episode,'03d')+'.npz')
        if path.exists():raise FileExistsError('Preservephysicaltrace '+str(path))
        fields={key:np.asarray([r[key] for r in self.policy]) for key in self.policy[0]} if self.policy else {}
        fields.update({'BODY_'+key:np.asarray([r[key] for r in self.body]) for key in self.body[0]} if self.body else {})
        fields.update(COMMANDS=np.asarray(self.commands),initial_physics_integration=self.initial_snapshot,
            FINAL_qpos=self.d.qpos.copy(),FINAL_qvel=self.d.qvel.copy(),FINAL_ctrl=self.d.ctrl.copy(),FINAL_time=np.array(self.d.time),
            kp=self.a.kp,kd=self.a.kd,limits=self.a.torque_limit,end_reason=np.array(reason))
        if self.handover:fields.update({'HANDOVER_'+k:v for k,v in self.handover.items()})
        np.savez_compressed(path,**fields);self.closed=True;self.total['episodes_saved']+=1


def export_arrays(net,metadata):
    with np.load(BASE_MODEL,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    for i,name in enumerate(('fc0','fc1','actor')):
        layer=getattr(net,name);arrays['actor_w'+str(i)]=layer.weight.detach().numpy().copy();arrays['actor_b'+str(i)]=layer.bias.detach().numpy().copy()
    arrays.update(critic_w=net.critic.weight.detach().numpy().copy(),critic_b=net.critic.bias.detach().numpy().copy(),
        logstd=net.logstd.detach().numpy().copy(),delta_scale=np.array(CFG['residual_rad']),
        indices=np.asarray(metadata['action']['indices'],dtype=np.int64),proprio_names=np.asarray(sensory.FEATURE_NAMES),
        base_sha256=np.array(n.sha(BASE_MODEL)),network_input_order=np.array('proprio183 THEN basebuf[lags].reshape294'),
        metadata_json=np.array(json.dumps(metadata)))
    return arrays


def export(net,path,metadata):
    if path.exists():raise FileExistsError(path)
    arrays=export_arrays(net,metadata)
    np.savez_compressed(path,**arrays)
    rng=np.random.default_rng(CFG['seed']);test=rng.normal(0,.3,(16,477)).astype(np.float32)
    with torch.no_grad():mean,value=net(torch.from_numpy(test));expected=torch.tanh(mean).numpy();value=value.numpy()
    actual,actualvalue=NumpyActor(path)(test);parity=float(max(abs(actual-expected).max(),abs(actualvalue-value).max()))
    assert parity<1e-5
    return dict(path=str(path),sha256=n.sha(path),Torch_NumPy_parity=parity)


def preregister():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('PreservepriorPPOexperiment '+str(OUT))
    files={Path(__file__),SOURCE,SENSORY_SOURCE,BASE_MODEL,n.GAIT}
    seen=set()
    def visit(module):
        if id(module) in seen:return
        seen.add(id(module));path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(ROOT)) and '/site-packages/' not in str(path):
            files.add(Path(path))
            for item in vars(module).values():
                if isinstance(item,types.ModuleType):visit(item)
    visit(b)
    from incremental_actuator import DEFAULT_MODEL_PATH
    files.add(DEFAULT_MODEL_PATH)
    hashes={str(p):n.sha(p) for p in sorted(files)}
    with np.load(n.GAIT,allow_pickle=False) as z:joint_names=z['joint_names'].astype(str).tolist()
    indices=[joint_names.index(name) for name in JOINTS]
    metadata=dict(stage='S5 realphysics PPO replacementANN smallresidual training',config=CFG,
        source_hashes=hashes,base_model=str(BASE_MODEL),base_sha256=n.sha(BASE_MODEL),reference=str(n.GAIT),
        network='477->128tanh->128tanh shared, actor6Normaltanh/critic1; actorheadzeros; logstd−3clamped[−5,−1.5]. No policytime/phase/worldposition/rootheight/anchors.',
        observations=dict(dimensions=477,proprio=sensory.contract(),history='Frozenbasecausal40framehistorynormalized7lagsflatten294. Observeexactlyonce5mst globallyincludingactual1swarmup.'),
        action=dict(joints=JOINTS,indices=indices,max_delta_rad=.003,formula='rate=clip((frozenbaselearnednextq+sixjointANNdelta-currenttarget)/.005,±30), original1msnativeangle/rateintegrator'),
        reward='clip(lowpassactualbodyforwardvelocity/.3,-1,1.5)−2*sumFn*worldXYmaterialspeed²/(W*.3²)−.05*|bodyomega|²−.01*|gravitybodyxy|²−.01mean(action²)−5*(sumFn/W−1)²−50*nonfoot/W; terminaladditional−25',
        reward_filter='Exact1-exp(-.005/.1), zeroatfreshreset thenphysicalwarmupupdates; rewardonlynotactorinput.',
        PPO='32x256decisions,6epochs,minibatch64,Adam3e-4,clip.2,gamma.995,GAE.95,valuecoef.5,entropy0,gradclip.5,KL.03earlyminibatchstop; no sweep. Timeoutbootstrapownterminalstatevalue, GAEresetacrossepisodeboundary.',
        episodes='Originalphysicalfreshkeyframe→1srealreferencewarmup(no+.02offset)→atmost7sbase+residualANN. EarlyCOM<.95/nonfoot>.05W/nonfinite/warnings=-25failurethenactualfreshwarmreset. Warmupfailureabortretained.',
        physics='Original49/48/42/71neq0free root/gravity[0,0,-9801]/12.5us, originalPD128zeroDtarget/nativecaps, mj_forwardeveryphysicsstep; noactualstatewritesexceptfreshinit orauxforces.',
        traces='Every5mspolicyOBS/action/reward/done+fullBODY+1msCOMMANDS compressedperepisode, warmhandoverintegration/historytarget. No nativeDseriesduringtraining; countsretainactualallsteps/warmups/failures.',
        exports='4compositeNPZcheckpointsupdates8/16/24/32: allbasekeysbitwise+actor_w0/b0,w1/b1,w2/b2+critic_w/b,logstd,delta_scale,indices,proprio_names,base_sha256. TorchNumPyparity; CHECKPOINT_LAST.ptoptimizer/rng/counterskept, alsoonabort. Physicsepisodeboundaryresumeprotocolmustbedeclaredseparately.',
        resource_guard='Stopandretainpartialcheckpoint/traces if16physicalwarmupsor6millionnative-steps exhausted; no indefinitefailure-resetloop or extratrainingupdates.',
        acceptance='TrainingrewardNOTphysicalacceptance. ParentseparateoriginalnativeD/realsteps/heading/materialslip<=.5+hold/cut+video. Frozenbase/teacherstillfailsmaterialslip; no originalbrainclaim.',
        versions=dict(torch=torch.__version__,numpy=np.__version__,mujoco=mj.__version__),environment={k:__import__('os').environ.get(k) for k in ('PYTHONDONTWRITEBYTECODE','MUJOCO_GL','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
    n.dump(OUT/'PREREGISTRATION.json',metadata);return metadata


def train(steps=8192):
    if steps!=CFG['decisions']:raise ValueError('Fixed protocol requires exactly8192neuraldecisions')
    metadata=preregister()  # BeforePlant,physicalwarmuporANNsetup.
    torch.set_num_threads(4);torch.manual_seed(CFG['seed']);rng=np.random.default_rng(CFG['seed'])
    net=ActorCritic();optimizer=torch.optim.Adam(net.parameters(),lr=CFG['lr']);env=None;logs=[];exports=[];start=time.monotonic()
    try:
        env=Environment();obs=env.reset()
        for update in range(1,CFG['updates']+1):
            rollout={k:[] for k in ('obs','latent','logprob','values','rewards','next_values','terminated','done')}
            for _ in range(CFG['rollout']):
                with torch.no_grad():action,latent,logprob,value,mean=net.sample(torch.from_numpy(obs)[None])
                action=action[0].numpy();latent=latent[0].numpy();lp=float(logprob[0]);v=float(value[0]);mean=mean[0].numpy()
                nextobs,reward,terminal,done,reason=env.step(action,latent,lp,v,mean)
                with torch.no_grad():nextvalue=0. if terminal else float(net(torch.from_numpy(nextobs)[None])[1][0])
                for key,item in dict(obs=obs.copy(),latent=latent,logprob=lp,values=v,rewards=reward,next_values=nextvalue,terminated=terminal,done=done).items():rollout[key].append(item)
                obs=nextobs
                if done:
                    env.save(reason or 'EPISODE_TIME_LIMIT')
                    if env.total['neural_policy_decisions']<CFG['decisions']:obs=env.reset()
            arrays={k:np.asarray(v,dtype=np.float32 if k not in ('terminated','done') else bool) for k,v in rollout.items()}
            advantage,returns=gae(arrays['rewards'],arrays['values'],arrays['next_values'],arrays['terminated'],arrays['done'])
            arrays.update(advantage=advantage,returns=returns)
            result=ppo_update(net,optimizer,arrays,rng)
            item=dict(update=update,counters=dict(env.total),reward_mean=float(arrays['rewards'].mean()),reward_min=float(arrays['rewards'].min()),
                reward_max=float(arrays['rewards'].max()),advantage_mean=float(advantage.mean()),returns_mean=float(returns.mean()),advantage_length=len(advantage),
                episode_boundaries=int(arrays['done'].sum()),failures=env.failures.copy(),PPO=result,wall_s=time.monotonic()-start)
            logs.append(item);n.dump(OUT/'TRAIN_LOG.json',logs)
            if update%8==0:exports.append(export(net,OUT/('MODEL_RESIDUAL_u'+format(update,'02d')+'.npz'),metadata))
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),update=update,config=CFG,
                torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,counters=env.total,metadata=metadata),OUT/'CHECKPOINT_LAST.pt')
            n.dump(OUT/'STATUS.json',dict(status='TRAINING' if update<CFG['updates'] else 'TRAINED_NOT_PHYSICALLY_ACCEPTED',
                update=update,counters=env.total,exports=exports,wall_s=time.monotonic()-start,failures=env.failures,
                source_hashes_unchanged={f:n.sha(f)==h for f,h in metadata['source_hashes'].items()}))
            print(json.dumps(dict(update=update,reward=item['reward_mean'],failures=env.total['failures'],decisions=env.total['neural_policy_decisions'],physics_steps=env.total['physics_steps'],wall_s=item['wall_s'])),flush=True)
        assert env.total['neural_policy_decisions']==CFG['decisions']
    except Exception as exc:
        if env is not None and not env.closed and env.ann_steps==0:
            env.total['failures']+=1
            env.failures.append(dict(episode=env.episode,time=float(env.d.time),reason='WARMUP_ABORT: '+str(exc)))
        n.dump(OUT/'FAILURE.json',dict(exception=type(exc).__name__+': '+str(exc),counters=env.total if env else None,wall_s=time.monotonic()-start))
        torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),config=CFG,metadata=metadata,
            torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,counters=env.total if env else None,
            completed_updates=len(logs),partial_update_aborted=True),OUT/'CHECKPOINT_LAST.pt')
        raise
    finally:
        if env is not None:
            env.save('TRAINING_BUDGET_END_OR_ABORT')
            n.dump(OUT/'FINAL_STATUS.json',dict(status='TRAINED_NOT_PHYSICALLY_ACCEPTED' if len(logs)==CFG['updates'] else 'ABORTED_PARTIAL_TRAINING',
                counters=env.total,failures=env.failures,completed_updates=len(logs),exports=exports,episode_files_saved_before_final_status=True))


def self_test():
    def forbidden(*a,**kw):raise AssertionError('No mj_step in residualANNselftest')
    mj.mj_step=forbidden;torch.manual_seed(CFG['seed']);net=ActorCritic()
    rng=np.random.default_rng(CFG['seed']);obs=torch.from_numpy(rng.normal(0,.2,(64,477)).astype(np.float32))
    with torch.no_grad():mean,_=net(obs)
    assert torch.equal(mean,torch.zeros_like(mean))
    base=n.Policy(BASE_MODEL)
    for _ in range(40):base.observe(base.w['center']+base.w['scale']*rng.normal(0,.1,42))
    target=base.w['center'].astype(float);indices=np.array([0,1,2,3,4,5])
    rate,_,delta=residual_rate(base,target,np.zeros(6),indices)
    assert np.array_equal(rate,base.action(target)) and not delta.any()
    action,latent,lp,value,_=net.sample(obs);lp2,_=net.evaluate(obs,latent)
    assert torch.allclose(lp,lp2) and torch.isfinite(lp).all() and np.max(abs(action.detach().numpy()))<=1
    adv,returns=gae(np.array([1.,2.,3.]),np.zeros(3),np.zeros(3),np.array([False,True,False]),np.array([False,True,False]))
    assert abs(adv[1]-2)<1e-7 and abs(adv[0]-(1+CFG['gamma']*CFG['gae']*2))<1e-6 and adv[2]==3
    timeout,_=gae(np.array([1.]),np.array([0.]),np.array([2.]),np.array([False]),np.array([True]))
    assert abs(timeout[0]-(1+CFG['gamma']*2))<1e-6
    optimizer=torch.optim.Adam(net.parameters(),lr=CFG['lr'])
    batch=dict(obs=obs.numpy(),latent=latent.detach().numpy(),logprob=lp.detach().numpy(),
        returns=value.detach().numpy()+.1,advantage=np.linspace(-1,1,64,dtype=np.float32))
    result=ppo_update(net,optimizer,batch,rng)
    assert all(torch.isfinite(p).all() for p in net.parameters()) and len(result['minibatches'])>0
    metadata=dict(action=dict(indices=indices.tolist()))
    arrays=export_arrays(net,metadata)
    with np.load(BASE_MODEL,allow_pickle=False) as frozen:
        assert all(np.array_equal(arrays[k],frozen[k]) for k in frozen.files)
    assert arrays['actor_w0'].shape==(128,477) and arrays['actor_w1'].shape==(128,128) and arrays['actor_w2'].shape==(6,128)
    assert len(arrays['proprio_names'])==183 and arrays['delta_scale']==.003
    numpy=NumpyActor.__new__(NumpyActor);numpy.w=arrays
    with torch.no_grad():mu,v=net(obs);expected=torch.tanh(mu).numpy();v=v.numpy()
    a2,v2=numpy(obs.numpy());parity=float(max(abs(a2-expected).max(),abs(v2-v).max()));assert parity<1e-5
    result=dict(zero_physics_steps=True,zero_actor_base_action_bitwise=True,squashed_logprob='PASS',GAE_terminal_reset_and_timeout_bootstrap='PASS',
        PPO_finite_gradients='PASS',Torch_NumPy_parity=parity,composite_base_keys_bitwise=True,actor_shapes='128x477/128x128/6x128',source_sha256=n.sha(__file__))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'SELF_TEST.json').exists():raise FileExistsError('Preserve prior selftest')
    n.dump(OUT/'SELF_TEST.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--train',action='store_true');group.add_argument('--self-test',action='store_true')
    ap.add_argument('--steps',type=int,default=8192)
    args=ap.parse_args()
    if args.steps!=8192:ap.error('Fixed preregistered protocol requires --steps8192')
    train(args.steps) if args.train else self_test()
