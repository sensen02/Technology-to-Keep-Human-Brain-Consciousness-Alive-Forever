"""Small continuous SAC for matched frozen and locally tuned brain response arms."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from forage import Course, GAMMA, HORIZON, MotorTable, OBS


def mlp(num_in,num_out):
    return nn.Sequential(nn.Linear(num_in,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,num_out))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=mlp(OBS,4)

    def forward(self,obs,sample=True):
        mu,log_std=self.net(obs).chunk(2,-1)
        log_std=log_std.clamp(-5.,1.)
        dist=torch.distributions.Normal(mu,log_std.exp())
        raw=dist.rsample() if sample else mu
        act=torch.tanh(raw)
        log_prob=(dist.log_prob(raw)-2.*(np.log(2.)-raw-F.softplus(-2.*raw))).sum(-1)
        return act,log_prob


class SAC:
    def __init__(self,seed):
        torch.manual_seed(seed)
        self.actor=Actor()
        self.q1,self.q2=mlp(OBS+2,1),mlp(OBS+2,1)
        self.t1,self.t2=(deepcopy(q).requires_grad_(False) for q in (self.q1,self.q2))
        self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.qopt=torch.optim.Adam([*self.q1.parameters(),*self.q2.parameters()],lr=3e-4)
        self.log_alpha=torch.tensor(np.log(.10),dtype=torch.float32,requires_grad=True)
        self.alopt=torch.optim.Adam([self.log_alpha],lr=2e-4)
        self.updates=0

    @staticmethod
    def q(q1,q2,obs,action):
        x=torch.cat((obs,action),-1)
        return q1(x).squeeze(-1),q2(x).squeeze(-1)

    def action(self,obs,explore):
        with torch.no_grad():
            return self.actor(torch.as_tensor(obs),sample=explore)[0].numpy()

    def update(self, batch):
        o,a,r,n,term=map(torch.as_tensor,batch)
        alpha=self.log_alpha.exp().detach()
        with torch.no_grad():
            ap,lp=self.actor(n)
            future=torch.minimum(*self.q(self.t1,self.t2,n,ap))-alpha*lp
            target=r+GAMMA*(1.-term)*future
            assert torch.equal(target[term==1.],r[term==1.])
        q1,q2=self.q(self.q1,self.q2,o,a)
        value=F.smooth_l1_loss(q1,target)+F.smooth_l1_loss(q2,target)
        self.qopt.zero_grad(set_to_none=True)
        value.backward()
        self.qopt.step()
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        proposed,lp=self.actor(o)
        policy=(alpha*lp-torch.minimum(*self.q(self.q1,self.q2,o,proposed))).mean()
        self.aopt.zero_grad(set_to_none=True)
        policy.backward()
        self.aopt.step()
        self.q1.requires_grad_(True)
        self.q2.requires_grad_(True)
        self.alopt.zero_grad(set_to_none=True)
        (-self.log_alpha*(lp.detach()-2.).mean()).backward()
        self.alopt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-7.,0.)
            for q,t in ((self.q1,self.t1),(self.q2,self.t2)):
                for p,s in zip(q.parameters(),t.parameters(),strict=True):
                    s.lerp_(p,.012)
        self.updates+=1
        return float(value.detach()),float(policy.detach()),float(alpha)

    def save(self,path,step,identity,rng):
        contents=dict(actor=self.actor.state_dict(),q1=self.q1.state_dict(),q2=self.q2.state_dict(),
                     t1=self.t1.state_dict(),t2=self.t2.state_dict(),log_alpha=self.log_alpha.detach(),
                     actor_optimizer=self.aopt.state_dict(),critic_optimizer=self.qopt.state_dict(),
                     alpha_optimizer=self.alopt.state_dict(),updates=self.updates,step=step,identity=identity,
                     rng=torch.get_rng_state(),numpy_rng=json.dumps(rng.bit_generator.state))
        tmp=path.with_suffix('.tmp')
        torch.save(contents,tmp)
        tmp.replace(path)


class Replay:
    def __init__(self,steps):
        self.values=[np.zeros(shape,dtype=np.float32) for shape in
                     ((steps,OBS),(steps,2),(steps,),(steps,OBS),(steps,))]
        self.size=0

    def add(self,*row):
        for arr,value in zip(self.values,row,strict=True):
            arr[self.size]=value
        self.size+=1

    def batch(self,rng):
        who=rng.integers(0,self.size,96)
        return [arr[who] for arr in self.values]


def evaluate(actor, motor, level, food=True):
    env=Course(motor)
    rows=[]
    for seed in range(83291,83315):
        side=1 if seed%2 else -1
        x=env.reset(seed,level,side=side,food=food)
        total=0.
        path=[env.position.tolist()]
        for _ in range(HORIZON):
            with torch.no_grad():
                a=actor(torch.from_numpy(x),sample=False)[0].numpy()
            x,r,done,end,info=env.step(a)
            path.append(env.position.tolist())
            total+=r
            if done or end:
                break
        rows.append(info | dict(seed=seed,total=total,side=side,steps=env.steps,path=path,
                               terrain=asdict(env.terrain)))
    return rows


def digest_summary(rows):
    return dict(success=sum(r['success'] for r in rows),fall=sum(r['fall'] for r in rows),
                intake_mean=float(np.mean([r['ingested'] for r in rows])),
                correct_side=sum(r['crossed_side']==r['side'] for r in rows),
                reward_mean=float(np.mean([r['total'] for r in rows])),count=len(rows))


def train(cal,arm,seed,total,out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir()
    model=SAC(seed)
    rng=np.random.default_rng(seed+167)
    stored=Replay(total)
    gain=cal['frozen_gain'] if arm=='frozen' else cal['tuned_gain']
    meta=cal['trials'][str(gain)]
    identity=dict(arm=arm,brain_sha256=meta['weights_sha256'],gain=gain)
    motor=MotorTable(meta['rows'])
    env=Course(motor)
    x=env.reset(seed*100,0)
    episodes=0
    initial=evaluate(model.actor,motor,2)
    history=[dict(step=0,**digest_summary(initial))]
    model.save(out/'00000.pt',0,identity,rng)
    with (out/'TRAIN.jsonl').open('x') as diary:
        for step in range(1,total+1):
            action=rng.uniform(-1.,1.,size=2) if step<=600 else model.action(x,True)
            new,r,done,end,info=env.step(action)
            stored.add(x,action,r,new,done)
            x=new
            if done or end:
                episodes+=1
                level=0 if step<1800 or episodes%8==0 else (1 if step<4500 else 2)
                x=env.reset(seed*100+episodes,level)
            loss=None
            if step>=600 and step%2==0:
                loss=model.update(stored.batch(rng))
                if not np.isfinite(loss).all():
                    raise ValueError('Nonfinite SAC update')
            if step%1500==0 or step==total:
                held=evaluate(model.actor,motor,2)
                checkpoint=out/f'{step:05d}.pt'
                model.save(checkpoint,step,identity,rng)
                row=dict(step=step,updates=model.updates,episodes=episodes,loss=loss,**digest_summary(held))
                diary.write(json.dumps(row)+'\n')
                diary.flush()
                history.append(row)
                print(arm,row,flush=True)
    scored=digest_summary(held)
    flat=evaluate(model.actor,motor,0)
    control=evaluate(model.actor,motor,2,food=False)
    result=dict(identity=identity,initial=digest_summary(initial),heldout=scored,flat=digest_summary(flat),
                no_food=digest_summary(control),steps=total,updates=model.updates,progress=history,
                heldout_cases=held,flat_cases=flat,no_food_cases=control)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cal',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--arm',choices=('frozen','tuned'),required=True)
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=307)
    args=parser.parse_args()
    train(json.loads(args.cal.read_text()),args.arm,args.seed,args.steps,args.out)
