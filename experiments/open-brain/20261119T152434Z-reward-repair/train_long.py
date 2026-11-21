"""Continue a frozen-Eon SAC actor with the corrected bridge reward."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'20261119T084744Z-closed-loop-forage'
sys.path.insert(0,str(PREV))
import refine_sac as prior
from balanced import MirrorActor
from forage import HORIZON, MEAL
import learn

from bridge_reward import CFG, Snapshot, clearance, cost, smoke

STAGES=('flat','wide_bridge','bridge','long_bridge')
PREVIOUS=PREV/'STRONG/27000.pt'
VISION=PREV/'VISION/VISION.npz'
TRAIN_SEED=61933
DEV_SEEDS=tuple(range(23560,23572))
HOLD_SEEDS=tuple(range(25412,25436))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SafetyCourse(prior.PhysicalDomain):
    def reset(self,*args,**kwargs):
        self.risk_memory=None
        self.crossed=False
        self.stable_frames=0
        self.true_cross=False
        self.last_hit_count=0
        self.safe_pass=True
        self.first_margin=None
        self.parts={}
        return super().reset(*args,**kwargs)

    def observe(self):
        # Older reset internally observes once on a flat initialization.
        if self.risk_memory is None:
            return np.zeros(14,dtype=np.float32)
        return super().observe()

    def advance_actor(self,obs,action):
        before=Snapshot(*self.position,self.ingested,has_food=self.food)
        previous_hits=self.hits
        state,_old,_,_,info=super().advance_actor(obs,action)
        after=Snapshot(*self.position,self.ingested,self.hits>previous_hits,bool(info['fall']),self.food)
        t=self.terrain
        formerly=self.crossed
        if t.side:
            lane=(after.x>=t.near-.04 and before.x<=t.far+.045)
            if lane:
                side_margin=min(clearance(t,before.y),clearance(t,after.y))
                if self.first_margin is None:
                    self.first_margin=float(side_margin)
                self.safe_pass &= side_margin>=CFG.foot_margin_cm
            if before.x<t.far<=after.x and not after.fall:
                self.true_cross=True
                if self.safe_pass and clearance(t,after.y)>=CFG.foot_margin_cm:
                    self.crossed=True
        near_food=bool(info['contact'] and self.speed<=.08 and not after.hit)
        self.stable_frames=self.stable_frames+1 if near_food else 0
        stable_food=bool(self.ingested>=MEAL and self.stable_frames>=2 and not after.fall)
        task=bool(stable_food and (self.crossed or not t.side))
        term=bool(after.fall or stable_food)
        cut=self.steps>=HORIZON and not term
        reward=cost(t,before,after,crossed_before=formerly,crossed_now=self.crossed,success=task,
                    terminal=term,truncated=cut,speed=self.speed,turn=self.turn)
        for key,value in asdict(reward).items():
            self.parts[key]=self.parts.get(key,0.)+value
        return state,reward.value,term,cut,dict(
            success=task,intake=float(self.ingested),ate=stable_food,fall=after.fall,
            cross=self.crossed,entered=self.true_cross,margin=self.first_margin,
            position=self.position.tolist(),hits=self.hits)


def evaluate(agent,env,seeds):
    result={}
    for stage in STAGES:
        rows=[]
        for seed in seeds:
            for side in (-1,1):
                x=env.reset(seed,stage,side,noise=True)
                for _ in range(HORIZON):
                    x,r,t,e,info=env.advance_actor(x,agent.action(x,False))
                    if t or e:
                        break
                rows.append(info|dict(seed=seed,side=side,steps=env.steps,parts=dict(env.parts)))
        result[stage]=rows
    return result


def summary(result):
    return {k:dict(n=len(v),success=sum(r['success'] for r in v),fall=sum(r['fall'] for r in v),
                   crossing=sum(r['cross'] for r in v),ingestion=sum(r['ate'] for r in v),
                   by_side={str(s):sum(r['success'] and r['side']==s for r in v) for s in (-1,1)})
            for k,v in result.items()}


def rating(stats):
    w=stats['wide_bridge']
    b=stats['bridge']
    f=stats['flat']
    # Main task is safe foraging on the wide bridge. Reward totals are never
    # used to promote a checkpoint with a higher failure count.
    return (min(w['by_side'].values()),w['success'],-w['fall'],
            min(b['by_side'].values()),b['success'],-b['fall'],f['success'])


def new_env():
    cal=json.loads((prior.PAIR/'NEURAL_CAL/CALIBRATION.json').read_text())
    return SafetyCourse(prior.MeasuredMotor(cal['trials']['1.0']['rows']),prior.vision_data(VISION),'strong')


def checkpoint_test(agent, replay, rng):
    # Check actual masked Bellman target, update, and exact reload prior to the
    # full budget. No probe weights are retained for the real training.
    before=agent.actor.net[0].weight.detach().clone()
    v=agent.update(replay.batch(rng))
    assert all(np.isfinite(v)) and agent.updates==1
    assert not torch.equal(before,agent.actor.net[0].weight)
    return v


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    learn.Actor=MirrorActor
    initial=torch.load(PREVIOUS,map_location='cpu',weights_only=True)
    identity=dict(arm='frozen',brain_sha256=initial['identity']['brain_sha256'],
                  content_sha256=sha(PREVIOUS),source=str(PREVIOUS),reward='supported_route_v1')
    args.out.mkdir()
    config=dict(reward=asdict(CFG),steps=args.steps,seed=TRAIN_SEED,development=DEV_SEEDS,
                holdout=HOLD_SEEDS,baseline=str(PREVIOUS),baseline_sha=sha(PREVIOUS),
                save_period=6000,warmup=1500)
    (args.out/'SCORING.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.out/'REWARD_RECEIPTS.json').write_text(json.dumps(smoke(),indent=2)+'\n')
    rng=np.random.default_rng(TRAIN_SEED)
    replay=learn.Replay(args.steps)
    agent=learn.SAC(TRAIN_SEED)
    agent.actor.load_state_dict(initial['actor'])
    env, evaluator=new_env(),new_env()
    baseline=evaluate(agent,evaluator,DEV_SEEDS)
    (args.out/'BEFORE_DEVELOPMENT.json').write_text(json.dumps(baseline,allow_nan=False)+'\n')
    base_stats=summary(baseline)
    print('baseline under new reward and supported scoring',base_stats,flush=True)
    paths=[]
    rounds=0
    x=env.reset(62140,'wide_bridge',1)
    probe=None
    reward_total=0.
    with (args.out/'UPDATES.jsonl').open('x') as log:
        for n in range(1,args.steps+1):
            if n<=1500:
                a=np.clip(.60*agent.action(x,False)+.40*rng.uniform(-1.,1.,2),-1.,1.)
            else:
                a=agent.action(x,True)
            new,r,term,cut,info=env.advance_actor(x,a)
            assert np.isfinite(r)
            reward_total+=r
            replay.add(x,a,r,new,term)
            x=new
            if term or cut:
                rounds+=1
                curriculum=(
                    ('wide_bridge','wide_bridge','wide_bridge','flat','wide_bridge') if n<=45000 else
                    ('wide_bridge','bridge','wide_bridge','flat','bridge') if n<=100000 else
                    ('wide_bridge','bridge','long_bridge','wide_bridge','flat','bridge'))
                stage=curriculum[(rounds//2)%len(curriculum)]
                x=env.reset(62140+rounds//2,stage,1 if rounds%2 else -1,food=(rounds//2)%15!=0)
            if n>=1500 and n%2==0:
                if probe is None:
                    probe_agent=learn.SAC(TRAIN_SEED+1)
                    probe_agent.actor.load_state_dict(initial['actor'])
                    rng_state=rng.bit_generator.state
                    torch_state=torch.get_rng_state()
                    probe=checkpoint_test(probe_agent,replay,rng)
                    rng.bit_generator.state=rng_state
                    torch.set_rng_state(torch_state)
                    (args.out/'UPDATE_SMOKE.json').write_text(json.dumps({'one_real_update_loss':probe})+'\n')
                losses=agent.update(replay.batch(rng))
                assert np.isfinite(losses).all()
            if n%6000==0 or n==args.steps:
                episodes=evaluate(agent,evaluator,DEV_SEEDS)
                stats=summary(episodes)
                target=args.out/f'{n:06d}.pt'
                agent.save(target,n,identity,rng)
                row=dict(step=n,updates=agent.updates,episodes=rounds,stats=stats,
                         checkpoint=str(target),sha256=sha(target),mean_online_reward=reward_total/n)
                paths.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                log.flush()
                print(row,flush=True)
    # Fixed held-out environment sets are first used only after selection.
    best=max(paths,key=lambda x:(rating(x['stats']),-x['step']))
    selected=torch.load(Path(best['checkpoint']),map_location='cpu',weights_only=True)
    agent.actor.load_state_dict(selected['actor'])
    prior.export(agent,args.out/'POLICY.npz',identity['brain_sha256'])
    hold_after=evaluate(agent,evaluator,HOLD_SEEDS)
    agent.actor.load_state_dict(initial['actor'])
    hold_before=evaluate(agent,evaluator,HOLD_SEEDS)
    report=dict(selected=best,last=paths[-1],holdout_before=summary(hold_before),
                holdout_after=summary(hold_after),cases_before=hold_before,cases_after=hold_after)
    (args.out/'HELDOUT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    compact={key:report[key] for key in ('selected','holdout_before','holdout_after')}
    print('independent',json.dumps(compact),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--steps',type=int,default=180000)
    run(parser.parse_args())
