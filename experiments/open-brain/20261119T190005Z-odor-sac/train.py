"""Train two continuous head intents with the existing P1 SAC implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from odor_env import HERE, HORIZON, Proxy, course, odor_field, odor_steering, sample

PAIR = HERE.parent/'20261119T073818Z-sac-terrain-brain-pair'
sys.path.insert(0,str(PAIR))
from learn import SAC
from policy_runtime import actor_mean

CAL = PAIR/'NEURAL_CAL/CALIBRATION.json'
DEV = tuple(range(70151,70157))
HOLDOUT = tuple(range(79101,79113))
SEED = 77511


def dump(path, item):
    path.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')


class Motor:
    def __init__(self, calibration):
        rows = calibration['trials']['1.0']['rows']
        self.brain = calibration['original_weights_sha256']
        ordered = sorted(rows,key=lambda r:r['intent'])
        self.raw = np.asarray([r['drive'] for r in ordered]).reshape(3,3,2)

    def __call__(self, pace, turn):
        if not pace:
            return 0.,0.
        j = int(turn >= 0.)
        y = (turn+.22*(1-j))/.22
        turns = self.raw[:,j]*(1-y)+self.raw[:,j+1]*y
        l,r = [np.interp(pace,(0.,.28,.55,.76),np.r_[0.,turns[:,k]]) for k in range(2)]
        yaw = float(np.clip(1.12*(l-r),-.76,.76))
        return min(float(.92*np.clip((l+r)/.96,0.,1.)),.86-.25*abs(yaw)), yaw


class Replay:
    def __init__(self, capacity=90000):
        self.rows = [np.empty((capacity,*dims),dtype=np.float32) for dims in ((14,),(2,),(),(14,),())]
        self.count = 0

    def add(self,*row):
        slot = self.count%len(self.rows[0])
        for buffer,value in zip(self.rows,row,strict=True):
            buffer[slot] = value
        self.count += 1

    def batch(self,rng):
        idx = rng.integers(0,min(self.count,len(self.rows[0])),96)
        return [a[idx] for a in self.rows]


def assess(actor,motor,seeds, *, active=True,stage='wall'):
    rows = []
    env = Proxy(motor)
    for seed in seeds:
        for side in (1,-1):
            cfg = course(seed,stage,side)
            obs = env.reset(cfg,seed+887,active,fixed_pose=True)
            for _ in range(HORIZON):
                obs,reward,end,timeout,info = env.step(actor(obs))
                if end or timeout:
                    break
            rows.append(info | dict(seed=seed,stage=stage,side=side,path=env.path))
    return rows


def summarize(rows):
    return dict(meals=sum(r['meal'] for r in rows), falls=sum(r['fall'] for r in rows),
                contact_steps=sum(r['hits'] for r in rows), count=len(rows),
                mean_intake=float(np.mean([r['intake'] for r in rows])),
                mean_x=float(np.mean([r['end_xy'][0] for r in rows])))


def export(actor, path):
    weights = {}
    for ix in (0,2,4):
        layer=actor.net[ix]
        weights[f'w{ix}']=layer.weight.detach().numpy().copy()
        weights[f'b{ix}']=layer.bias.detach().numpy().copy()
    np.savez_compressed(path,**weights)


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir()
    rng = np.random.default_rng(SEED)
    motor = Motor(json.loads(CAL.read_text()))
    agent = SAC(SEED)
    proxy, replay = Proxy(motor), Replay()
    obstacle = getattr(args,'course','wall')
    guided = getattr(args,'guided',False)
    stages = ('open',obstacle)
    if guided:
        from reachability import route_action
    identity = dict(brain_sha256=motor.brain,seed=SEED,trained=['head odor translation actor'],
                    calibration_path=str(CAL),calibration_sha=hashlib.sha256(CAL.read_bytes()).hexdigest())
    dump(args.out/'PLAN.json',dict(identity=identity,development=DEV,heldout=HOLDOUT,
                                  steps=args.steps,interval=args.interval,paired_stages=stages,
                                  guided_replay=guided))
    # Freeze the geometry pools before optimization and never sample validation.
    pool = {stage:[course(s,stage,d) for s in range(60000,60018) for d in (1,-1)]
            for stage in stages}
    for layouts in pool.values():
        for cfg in layouts:
            f=odor_field(cfg)
            assert min(sample(f,((.103,.03),(.103,-.03))))>1e-4
    def reset(step,episode):
        stage = 'open' if (not guided and step < 12000) or episode%5 == 0 else obstacle
        cfg = pool[stage][int(rng.integers(len(pool[stage])))]
        return proxy.reset(cfg,SEED*3+episode,active=episode%16!=15)
    prior = {s:summarize(assess(odor_steering,motor,DEV,stage=s)) for s in stages}
    print('development gradient baseline',prior,flush=True)
    initial = {s:summarize(assess(lambda x:agent.action(x,False),motor,DEV,stage=s))
               for s in stages}
    print('development random actor',initial,flush=True)
    save_log = open(args.out/'TRAIN.jsonl','w')
    obs = reset(0,0)
    episode = 0
    rank = (-1,)
    best = None
    resolved = time.monotonic()
    losses = None
    guided_steps = 0
    for step in range(1,args.steps+1):
        from_demo = guided and ((step<=7000 and episode%6!=5) or
                                (step<=65000 and episode%7==6))
        if from_demo:
            action = np.clip(route_action(proxy)+rng.normal(0.,.09,2),-1.,1.)
            guided_steps += 1
        else:
            action = rng.uniform(-1.,1.,2) if step<=1500 else agent.action(obs,True)
        nxt,reward,done,timeout,info = proxy.step(action)
        replay.add(obs,action,reward,nxt,float(done))
        obs=nxt
        if step>=1500 and step%2==0:
            losses=agent.update(replay.batch(rng))
            assert np.isfinite(losses).all()
        if done or timeout:
            episode+=1
            obs=reset(step,episode)
        if step%args.interval!=0 and step!=args.steps:
            continue
        scores={s:summarize(assess(lambda x:agent.action(x,False),motor,DEV,stage=s))
                for s in stages}
        file=args.out/f'{step:07d}.pt'
        agent.save(file,step,identity,rng)
        row=dict(step=step,updates=agent.updates,elapsed_s=time.monotonic()-resolved,
                 checkpoint=file.name,scores=scores,losses=losses)
        print(json.dumps(row),flush=True)
        save_log.write(json.dumps(row)+'\n')
        save_log.flush()
        w,o=scores[obstacle],scores['open']
        candidate=(w['meals']+o['meals'],w['meals'],o['meals'],-w['falls'],-w['contact_steps'])
        if candidate>rank:
            rank, best=candidate,row
    save_log.close()
    selected=torch.load(args.out/best['checkpoint'],weights_only=True,map_location='cpu')
    agent.actor.load_state_dict(selected['actor'])
    assert selected['actor_optimizer']['state'] and selected['critic_optimizer']['state']
    export(agent.actor,args.out/'POLICY.npz')
    with np.load(args.out/'POLICY.npz',allow_pickle=False) as policy:
        rng2=np.random.default_rng(581)
        x=rng2.uniform(-1.,1.,(512,14)).astype(np.float32)
        parity=float(np.max(np.abs(actor_mean(policy,x)-agent.action(x,False))))
        assert parity<1.5e-5
        actor=lambda sense:actor_mean(policy,sense)
        held={}
        for name,policy_fn in (('SAC',actor),('BILATERAL',odor_steering)):
            held[name]={s:assess(policy_fn,motor,HOLDOUT,stage=s) for s in stages}
        held['SAC_OFF']={obstacle:assess(actor,motor,HOLDOUT,active=False,stage=obstacle)}
    dump(args.out/'HELDOUT.json',held)
    dump(args.out/'FINAL.json',dict(selected=best,identity=identity,
         run_steps=args.steps,updates=agent.updates,elapsed_s=time.monotonic()-resolved,
         guided_steps=guided_steps,
         dev_gradient=prior,dev_initial=initial,export_max_error=parity,
         heldout={k:{s:summarize(rows) for s,rows in v.items()} for k,v in held.items()}))
    print((args.out/'FINAL.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--steps',type=int,default=110000)
    parser.add_argument('--interval',type=int,default=10000)
    main(parser.parse_args())
