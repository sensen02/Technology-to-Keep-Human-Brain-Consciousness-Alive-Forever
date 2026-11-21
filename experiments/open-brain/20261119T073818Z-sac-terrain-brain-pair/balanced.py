"""Pair the mirrored visual actor with a measured physical-intent inverse."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import learn
from forage import MotorTable

RAW_ACTOR=learn.Actor
MIRROR=(0,1,2,3,9,8,7,6,5,4,11,10,12,13)
N_GRID=37


class MirrorActor(RAW_ACTOR):
    def forward(self,obs,sample=True):
        shadow=obs[...,list(MIRROR)].clone()
        shadow[...,1]*=-1.
        direction=self.net(obs)
        other=self.net(shadow)
        other[...,1]*=-1.
        stats=(direction+other)*.5
        mean,log_std=stats.chunk(2,-1)
        log_std=log_std.clamp(-5.,1.)
        dist=torch.distributions.Normal(mean,log_std.exp())
        z=dist.rsample() if sample else mean
        lp=(dist.log_prob(z)-2.*(np.log(2.)-z-torch.nn.functional.softplus(-2.*z))).sum(-1)
        return torch.tanh(z),lp


class BalancedMotor(MotorTable):
    def __init__(self,rows):
        super().__init__(rows)
        grid=np.linspace(-1.,1.,N_GRID)
        commands=np.array([(v,w) for v in grid for w in grid])
        achievable=np.asarray([super(BalancedMotor,self).drive(a) for a in commands])
        target=commands.copy()
        target[:,0]=(target[:,0]+1.)*.44
        target[:,1]*=1.50
        indices=[]
        for t in target:
            loss=((achievable-t)/np.array([.8,2.]))**2
            indices.append(int(loss.sum(axis=1).argmin()))
        table_indices=np.asarray(indices).reshape(N_GRID,N_GRID)
        self.intent=commands[table_indices]
        self.achievable=achievable[table_indices]

    def drive(self,action):
        return self.achievable[tuple(np.clip(np.rint((action+1.)*.5*(N_GRID-1)),0,N_GRID-1).astype(int))]


def main(cal,arm,out,steps,seed):
    learn.Actor=MirrorActor
    learn.MotorTable=BalancedMotor
    learn.train(cal,arm,seed,steps,out)
    gain=cal['frozen_gain'] if arm=='frozen' else cal['tuned_gain']
    m=BalancedMotor(cal['trials'][str(gain)]['rows'])
    np.save(out/'INTENT_GRID.npy',m.intent,allow_pickle=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cal',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--arm',choices=('frozen','tuned'),required=True)
    p.add_argument('--steps',type=int,default=24000)
    p.add_argument('--seed',type=int,default=642)
    args=p.parse_args()
    main(json.loads(args.cal.read_text()),args.arm,args.out,args.steps,args.seed)
