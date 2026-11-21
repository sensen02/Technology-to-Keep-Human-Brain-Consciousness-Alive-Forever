"""Continue the frozen-brain actor with real-eye calibration and a bounded motor residual."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from controller import HERE, PAIR, terrain, proposal, decode_levels
from forage import Course, MotorTable, HORIZON, GAMMA, MEAL, OBS
from balanced import MirrorActor, MIRROR
import learn


class MeasuredMotor(MotorTable):
    def outputs(self,advance,turn):
        pairs=np.array([[np.interp(turn,self.turns,self.table[i,:,j])
                         for i in range(len(self.paces))] for j in range(2)])
        # Preserve exactly zero input as an available action.
        v,w=[np.interp(advance,np.r_[0.,self.paces],np.r_[0.,p]) for p in pairs]
        speed,yaw=decode_levels(v,w)
        return float(min(speed,.86-.25*abs(yaw))),float(yaw)


class PhysicalDomain(Course):
    def __init__(self,motor,vision,profile="small"):
        super().__init__(motor)
        self.profile=profile
        f=vision['train_x']
        y=vision['train_y']
        x=np.c_[f,np.sqrt(np.maximum(f,0.)),np.ones(len(f))]
        calibrated=np.clip(x@vision['weights'],0.,1.)
        valid=np.max(f[:,:30].reshape(-1,6,5),axis=2)>0.
        calibrated*=valid
        bins=np.minimum((y.ravel()*8).astype(int),7)
        self.samples=[calibrated.ravel()[bins==i] for i in range(8)]
        assert all(len(group) for group in self.samples)

    def reset(self,seed,stage,side,food=True,noise=True):
        self.rng=np.random.default_rng(seed+87761)
        self.noise=noise
        super().reset(seed,0,side=0,food=food)
        self.terrain=terrain(seed,stage,side)
        self.cost=self.terrain.distance(self.position)
        self.last_target=None
        self.stage=stage
        self.risk_memory=np.zeros(6)
        return self.observe()

    def observe(self):
        x=super().observe()
        if x[0]:
            self.last_target=x[:4].copy()
        elif getattr(self,'last_target',None) is not None and self.steps%5!=0:
            x[1:4]=self.last_target[1:4]
        if not hasattr(self,'risk_memory'):
            return x
        # Both training and held-out courses use the same image noise family.
        # Calibration uses RGB training seeds only, never held-out images.
        levels=np.minimum((x[4:10]*8).astype(int),7)
        measured=np.array([self.rng.choice(self.samples[idx]) for idx in levels]) if self.noise else x[4:10]
        if self.profile=='strong' and self.stage in ('wide_bridge','bridge'):
            measured=.55*measured+.45*x[4:10]
        self.risk_memory=.45*self.risk_memory+.55*measured
        x[4:10]=self.risk_memory
        noise=self.rng.normal(0.,.04,2) if self.noise else 0.
        x[10:12]=np.maximum(x[10:12],.27+noise)
        return x

    def advance_actor(self,obs,action):
        advance,turn=proposal(obs,action,self.profile)
        v,w=self.motor.outputs(advance,turn)
        # Model uncertainty comes from actual gaps between continuous live and
        # stationary calibrated readouts. Apply low-frequency variation.
        bias=.19*np.sin(self.steps/7.+self.terrain.goal_y)
        strength=1.-.18*(np.sin(self.steps/9.)**2) if self.noise else 1.
        distance=obs[3]
        near=np.interp(distance,(.135,.32),(.41,1.)) if obs[0] else 1.
        original=self.motor
        self.motor=FixedDrive(v*strength*near,np.clip(w+bias if advance else 0.,-.76,.76))
        state,r,done,end,info=super().step(action)
        self.motor=original
        # Further cost for bumping a lip instead of letting graze costs vanish.
        r-=.18*(info['collision_steps']>getattr(self,'last_hit_count',0))
        self.last_hit_count=info['collision_steps']
        return state,r,done,end,info


class FixedDrive:
    def __init__(self,v,w):
        self.value=(v,w)

    def drive(self,_):
        return self.value


def rollouts(agent,env):
    rows={}
    for stage in (('flat','bridge','long_bridge') if env.profile=='small' else ('flat','wide_bridge','bridge','long_bridge')):
        tests=[]
        for seed in (range(16811,16835) if env.profile=='small' else range(18911,18935)):
            side=1 if seed%2 else -1
            x=env.reset(seed,stage,side,food=True,noise=True)
            env.last_hit_count=0
            total=0.
            for _ in range(HORIZON):
                x,r,t,e,info=env.advance_actor(x,agent.action(x,False))
                total+=r
                if t or e:
                    break
            tests.append(info | dict(seed=seed,side=side,total_reward=total,stage=stage))
        rows[stage]=tests
    return rows


def summarize(rows):
    return {k:dict(n=len(v),success=sum(r['success'] for r in v),fall=sum(r['fall'] for r in v),
                   food_avg=float(np.mean([r['ingested'] for r in v]))) for k,v in rows.items()}


def export(model,output,brain_hash):
    data={'identity':json.dumps(dict(brain_sha256=brain_hash,arm='frozen')),
          'mirror':np.array(MIRROR)}
    for layer in (0,2,4):
        linear=model.actor.net[layer]
        data[f'w{layer}']=linear.weight.detach().numpy()
        data[f'b{layer}']=linear.bias.detach().numpy()
    np.savez_compressed(output,**data)


def train(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    learn.Actor=MirrorActor
    agent=learn.SAC(4307)
    previous=torch.load(args.start,map_location='cpu',weights_only=True)
    agent.actor.load_state_dict(previous['actor'])
    initial=dict(arm='frozen',brain_sha256=previous['identity']['brain_sha256'],
                 source=str(args.start))
    cal=json.loads((PAIR/'NEURAL_CAL/CALIBRATION.json').read_text())
    motor=MeasuredMotor(cal['trials']['1.0']['rows'])
    with np.load(args.vision,allow_pickle=False) as v:
        env=PhysicalDomain(motor,{key:v[key] for key in ('train_x','train_y','weights')},args.profile)
    args.out.mkdir()
    rng=np.random.default_rng(891129)
    start=rollouts(agent,env)
    replay=learn.Replay(args.steps)
    rounds=0
    x=env.reset(71450,'flat',1)
    env.last_hit_count=0
    history=[dict(step=0,result=summarize(start))]
    print('initial',history[0],flush=True)
    with (args.out/'TRAIN.jsonl').open('x') as log:
        for n in range(1,args.steps+1):
            chosen=agent.action(x,True)
            if n<=700:
                chosen=np.clip(.60*agent.action(x,False)+.40*rng.uniform(-1.,1.,2),-1.,1.)
            new,r,term,cut,_=env.advance_actor(x,chosen)
            replay.add(x,chosen,r,new,term)
            x=new
            if term or cut:
                rounds+=1
                stage='flat' if rounds%5==0 else ('long_bridge' if rounds%5==3 and n>3000 else 'bridge')
                if args.profile=='strong' and rounds%5 in (1,2):
                    stage='wide_bridge'
                side=1 if rounds%2 else -1
                x=env.reset(71450+rounds,stage,side,food=rounds%11!=0)
                env.last_hit_count=0
            if n>=700 and n%2==0:
                loss=agent.update(replay.batch(rng))
                assert np.isfinite(loss).all()
            if n%1500==0 or n==args.steps:
                summary=summarize(rollouts(agent,PhysicalDomain(motor,vision_data(args.vision),args.profile)))
                path=args.out/f'{n:05d}.pt'
                agent.save(path,n,initial,rng)
                row=dict(step=n,updates=agent.updates,episodes=rounds,result=summary,checkpoint=str(path))
                history.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(row,flush=True)
    export(agent,args.out/'POLICY.npz',initial['brain_sha256'])
    final=rollouts(agent,PhysicalDomain(motor,vision_data(args.vision),args.profile))
    (args.out/'EVAL.json').write_text(json.dumps(dict(start=start,final=final,history=history),
                                                indent=2,allow_nan=False)+'\n')


def vision_data(path):
    with np.load(path,allow_pickle=False) as f:
        return {key:f[key] for key in ('train_x','train_y','weights')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('vision',type=Path)
    p.add_argument('start',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--steps',type=int,default=15000)
    p.add_argument('--profile',choices=('small','strong'),default='small')
    train(p.parse_args())
