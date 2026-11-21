"""Approximately 30 wall-clock minutes of compound-course SAC."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from terrain_spec import HERE, CLOSED, REWARD, STAGES, make_terrain, dry_check
sys.path.insert(0,str(REWARD))
from bridge_reward import CFG, Snapshot, cost, clearance, smoke
import train_long as old
import learn
from balanced import MirrorActor

START=REWARD/'TRAIN/060000.pt'
VISION=CLOSED/'VISION/VISION.npz'
SEED=571505
DEVELOPMENT=tuple(range(28200,28206))
HELDOUT=tuple(range(29300,29320))
HORIZON=105
REPLAY_COUNTS=dict(flat=12,wide_bridge=34,slope_bridge=22,staggered=20,staggered_slope=8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Compound(old.prior.PhysicalDomain):
    def reset(self,seed,stage,side,food=True):
        self.risk_memory=None
        self.gate_index=0
        self.min_margin=None
        self.entry_margin=None
        self.cross_count=0
        self.steady=0
        self.invalid_crosses=0
        self.last_hit_count=0
        self.actual_stage=stage
        super().reset(seed,'flat',side,food=food,noise=True)
        self.terrain=make_terrain(seed,stage,side)
        self.stage='wide_bridge' if stage!='flat' else 'flat'
        self.risk_memory=np.zeros(6)
        self.last_target=None
        self.cost=self.terrain.distance(self.position)
        return self.observe()

    def observe(self):
        if self.risk_memory is None:
            return np.zeros(14,dtype=np.float32)
        return super().observe()

    def active_gate(self):
        return self.terrain.gates[self.gate_index] if self.gate_index<len(self.terrain.gates) else None

    def validated_cross(self,a,b,gate):
        if gate is None:
            return False
        # Only the swept part of this motion inside the unsupported interval
        # affects an attempt. Backing out gives the next attempt a fresh margin.
        if b.x<gate.near and a.x<gate.near:
            self.min_margin=None
        if b.x>a.x and a.x<gate.far and b.x>=gate.near:
            u0=max(0.,(gate.near-a.x)/(b.x-a.x))
            u1=min(1.,(gate.far-a.x)/(b.x-a.x))
            y0=a.y+u0*(b.y-a.y)
            y1=a.y+u1*(b.y-a.y)
            gap=min(clearance(gate,y0),clearance(gate,y1))
            self.min_margin=gap if self.min_margin is None else min(gap,self.min_margin)
            if self.entry_margin is None:
                self.entry_margin=float(gap)
            if b.x>=gate.far and not b.fall:
                safe=self.min_margin>=CFG.foot_margin_cm
                self.gate_index+=1
                self.cross_count+=int(safe)
                self.invalid_crosses+=int(not safe)
                self.min_margin=None
                return bool(safe)
        return False

    def advance_actor(self,obs,action):
        t=self.terrain
        current=self.active_gate()
        before=Snapshot(*self.position,self.ingested,has_food=self.food)
        hit0=self.hits
        incline=max(0.,float(t.slope(self.position[0])))
        # Retain the same measured neural table and model the modest uphill load
        # as a local actuator speed reduction, with no new online input fields.
        original=self.motor
        self.motor=SlopeDrive(original,incline)
        try:
            state,_,_,_,info=super().advance_actor(obs,action)
        finally:
            self.motor=original
        after=Snapshot(*self.position,self.ingested,self.hits>hit0,bool(info['fall']),bool(self.food))
        bonus=self.validated_cross(before,after,current)
        arrived=bool(info['contact'] and self.speed<=.08 and not after.hit)
        self.steady=self.steady+1 if arrived else 0
        fed=bool(self.ingested>=CFG.meal_volume and self.steady>=2 and not after.fall)
        all_safe=self.cross_count==len(t.gates)
        success=fed and all_safe
        done=after.fall or fed
        timeout=self.steps>=HORIZON and not done
        if current is not None:
            target=type('Target',(),dict(side=current.side,edge=current.edge,near=current.near,
                                        far=current.far,goal_x=t.goal_x,goal_y=t.goal_y))
        else:
            target=VirtualTarget(t.goal_x,t.goal_y)
        pieces=cost(target,before,after,crossed_before=False,crossed_now=bonus,
                    success=success,terminal=done,truncated=timeout,
                    speed=self.speed,turn=self.turn)
        # Reward can transfer a small amount before every support gate is
        # proved, but an unsupported shortcut cannot collect a full meal.
        reward=pieces.value
        if not all_safe:
            reward-=.65*pieces.food
        if fed and not all_safe:
            reward-=2.5
        return state,float(reward),done,timeout,dict(
            success=success,fall=after.fall,fed=fed,intake=float(self.ingested),
            supported=self.cross_count,invalid=self.invalid_crosses,gates=len(t.gates),
            entry_margin=self.entry_margin,position=self.position.tolist())


class VirtualTarget:
    side=0
    def __init__(self,x,y):
        self.goal_x=x
        self.goal_y=y


class SlopeDrive:
    def __init__(self,parent,incline):
        self.parent,self.incline=parent,incline

    def outputs(self,advance,turn):
        speed,yaw=self.parent.outputs(advance,turn)
        return speed*(1.-min(.35,1.7*self.incline)),yaw


class Ring:
    """Stage-specific replay, with bounded storage and replacement."""
    def __init__(self,capacity=96000):
        self.arr=[np.empty(shape,dtype=np.float32) for shape in (
                 (capacity,14),(capacity,2),(capacity,),(capacity,14),(capacity,))]
        self.capacity=capacity
        self.count=0

    def add(self,*row):
        index=self.count%self.capacity
        for buf,field in zip(self.arr,row,strict=True):
            buf[index]=field
        self.count+=1

    def sample(self,size,rng):
        indexes=rng.integers(0,min(self.count,self.capacity),size)
        return [a[indexes] for a in self.arr]


def batch(bank,rng):
    slices=[]
    for stage,count in REPLAY_COUNTS.items():
        ring=bank[stage] if bank[stage].count>=24 else bank['wide_bridge']
        if ring.count==0:
            ring=max(bank.values(),key=lambda r:r.count)
        slices.append(ring.sample(count,rng))
    return [np.concatenate(group) for group in zip(*slices,strict=True)]


def schedule(episode,elapsed):
    pool=('wide_bridge','wide_bridge','flat','slope_bridge','wide_bridge','slope_bridge')
    if elapsed>=360.:
        pool=('wide_bridge','slope_bridge','staggered','wide_bridge','flat',
              'slope_bridge','wide_bridge','staggered')
    if elapsed>=840.:
        pool=('wide_bridge','staggered','slope_bridge','wide_bridge','flat',
              'staggered_slope','wide_bridge','slope_bridge','staggered')
    return pool[(episode//2)%len(pool)]


def assess(agent,env,seeds):
    cases={}
    for group in STAGES:
        rows=[]
        for seed in seeds:
            for s in (-1,1):
                obs=env.reset(seed,group,s)
                total=0.
                for k in range(HORIZON):
                    obs,r,done,cut,info=env.advance_actor(obs,agent.action(obs,False))
                    total+=r
                    if done or cut:
                        break
                rows.append(info|dict(seed=seed,mirror=s,ticks=k+1,reward=total))
        cases[group]=rows
    return cases


def digest(cases):
    return {k:dict(n=len(v),meal=sum(r['success'] for r in v),fall=sum(r['fall'] for r in v),
                   ate=sum(r['fed'] for r in v),full_support=sum(r['supported']==r['gates'] for r in v),
                   by_mirror={str(s):sum(r['success'] and r['mirror']==s for r in v) for s in (-1,1)})
            for k,v in cases.items()}


def rank(row,wide_floor):
    s=row['scores']
    regression=s['wide_bridge']['meal']>=wide_floor and s['flat']['meal']==s['flat']['n']
    complex_groups=[s[n] for n in ('slope_bridge','staggered','staggered_slope')]
    per_mirror=[sum(g['by_mirror'][m] for g in complex_groups) for m in ('-1','1')]
    return (regression,min(per_mirror),sum(per_mirror),-sum(g['fall'] for g in complex_groups),
            s['wide_bridge']['meal'],-row['step'])


def new_env():
    c=json.loads((old.prior.PAIR/'NEURAL_CAL/CALIBRATION.json').read_text())
    return Compound(old.prior.MeasuredMotor(c['trials']['1.0']['rows']),old.prior.vision_data(VISION),'strong')


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dry_check()
    make=old.prior
    learn.Actor=MirrorActor
    saved=torch.load(START,weights_only=True,map_location='cpu')
    ident=dict(arm='frozen',brain_sha256=saved['identity']['brain_sha256'],
               baseline=str(START),baseline_sha=sha(START),task='compound_terrain')
    args.out.mkdir()
    (args.out/'PLAN.json').write_text(json.dumps(dict(
        requested_seconds=args.seconds,seed=SEED,stages=STAGES,development=DEVELOPMENT,
        heldout=HELDOUT,identity=ident,replay_per_batch=REPLAY_COUNTS,
        reward_checks=smoke()),indent=2,allow_nan=False)+'\n')
    rng=np.random.default_rng(SEED)
    agent=learn.SAC(SEED)
    agent.actor.load_state_dict(saved['actor'])
    env,judge=new_env(),new_env()
    dev=digest(assess(agent,judge,DEVELOPMENT))
    baseline=dict(step=0,updates=0,elapsed=0.,scores=dev,checkpoint=str(START),sha256=sha(START))
    floor=max(0,dev['wide_bridge']['meal']-1)
    print('before training',json.dumps(baseline,allow_nan=False),flush=True)
    bank={s:Ring() for s in STAGES}
    stage='wide_bridge'
    obs=env.reset(176000,stage,1)
    steps=0
    episodes=0
    last_log=0.
    frames=[baseline]
    by_stage=dict.fromkeys(STAGES,0)
    start=time.monotonic()
    aux=None
    gradient_seconds=0.
    with (args.out/'CHECKPOINTS.jsonl').open('x') as trace:
        while time.monotonic()-start<args.seconds:
            elapsed=time.monotonic()-start
            steps+=1
            a=agent.action(obs,True)
            nxt,reward,done,truncated,_=env.advance_actor(obs,a)
            assert np.isfinite(nxt).all() and np.isfinite(reward)
            bank[stage].add(obs,a,reward,nxt,done)
            by_stage[stage]+=1
            obs=nxt
            if done or truncated:
                episodes+=1
                stage=schedule(episodes,elapsed)
                obs=env.reset(176000+episodes//2,stage,1 if episodes%2==0 else -1,
                              food=(episodes//2)%21!=0)
            if steps>=1600 and steps%2==0:
                sampled=batch(bank,rng)
                if aux is None:
                    state=torch.get_rng_state()
                    check=learn.SAC(SEED+1)
                    check.actor.load_state_dict(saved['actor'])
                    old_weight=check.actor.net[0].weight.detach().clone()
                    aux=check.update(sampled)
                    assert np.isfinite(aux).all() and not torch.equal(old_weight,check.actor.net[0].weight)
                    torch.set_rng_state(state)
                    (args.out/'FIRST_UPDATE.json').write_text(json.dumps(aux)+'\n')
                begin=time.monotonic()
                loss=agent.update(sampled)
                gradient_seconds+=time.monotonic()-begin
                assert np.isfinite(loss).all()
            if elapsed-last_log>=args.interval:
                last_log=elapsed
                target=args.out/f'{steps:08d}.pt'
                agent.save(target,steps,ident,rng)
                row=dict(step=steps,updates=agent.updates,elapsed=elapsed,episodes=episodes,
                         scores=digest(assess(agent,judge,DEVELOPMENT)),checkpoint=str(target),
                         sha256=sha(target),stage_decisions=by_stage.copy())
                frames.append(row)
                trace.write(json.dumps(row,allow_nan=False)+'\n')
                trace.flush()
                print('checkpoint',json.dumps(row,allow_nan=False),flush=True)
        elapsed=time.monotonic()-start
        if frames[-1]['step']!=steps:
            target=args.out/f'{steps:08d}.pt'
            agent.save(target,steps,ident,rng)
            end=dict(step=steps,updates=agent.updates,elapsed=elapsed,episodes=episodes,
                     scores=digest(assess(agent,judge,DEVELOPMENT)),checkpoint=str(target),
                     sha256=sha(target),stage_decisions=by_stage.copy())
            frames.append(end)
            trace.write(json.dumps(end,allow_nan=False)+'\n')
    best=max(frames,key=lambda e:rank(e,floor))
    candidate=torch.load(Path(best['checkpoint']),weights_only=True,map_location='cpu')
    assert sha(Path(best['checkpoint']))==best['sha256']
    agent.actor.load_state_dict(candidate['actor'])
    make.export(agent,args.out/'POLICY.npz',ident['brain_sha256'])
    after=assess(agent,judge,HELDOUT)
    agent.actor.load_state_dict(saved['actor'])
    before=assess(agent,judge,HELDOUT)
    report=dict(elapsed=elapsed,gradient_seconds=gradient_seconds,wide_floor=floor,initial=baseline,
                selected=best,terminal=frames[-1],heldout_before=digest(before),
                heldout_after=digest(after),cases_before=before,cases_after=after)
    temp=args.out/'FINAL.tmp'
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temp.replace(args.out/'FINAL.json')
    print('completed',elapsed,steps,agent.updates,'selected',best['step'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--seconds',type=float,default=1800.)
    p.add_argument('--interval',type=float,default=120.)
    args=p.parse_args()
    if args.seconds<2. or args.interval<1.:
        p.error('Durations too short')
    run(args)
