"""Post-meal ground-only comparison using the saved orchard pose and a trained gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261119T042050Z-orchard-survival'
PREVIOUS=HERE.parent/'20261119T021500Z-orchard-behavior'
sys.path[:0]=[str(PRIOR),str(PREVIOUS)]
from run_case import (NEURAL, RESOURCES, SEED, Contacts, OrchardEyes,
                      build_orchard, fold_until_launch, land, view_candidates, write_line)
from steering_recheck import ReversedYawWalk
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState

PREMEAL=PRIOR/'GROUND_EXTENSION/FRESH'
TRIALS={'HUNGRY':(.24,.15,4.8), 'FULL':(.55,.55,2.), 'FADE':(.27,.32,7.6)}


def xy_frame(eye, items):
    origin, direction, _=eye.head_pose()
    ahead=direction[:2]/np.linalg.norm(direction[:2])
    right=np.array([ahead[1],-ahead[0]])
    options=view_candidates(eye,items)
    for ray in options:
        angle=np.deg2rad(ray['bearing'])
        xy=origin[:2]+ray['horiz']*(ahead*np.cos(angle)+right*np.sin(angle))
        ray['local_odometry_xy']=xy.tolist()
    return origin[:2].copy(),ahead,options


class FruitSearch:
    """Track the previously tasted target and hold a visual cue over brief occlusion."""

    def __init__(self, eye):
        _,_,items,_=eye.sample()
        _,_,options=xy_frame(eye,items)
        closest=min((r for r in options if not r['high']),key=lambda item:item['horiz'])
        self.empty=np.array(closest['local_odometry_xy'])
        self.track=None

    def select(self,eye,items,now):
        origin,ahead,options=xy_frame(eye,items)
        low=[]
        for item in options:
            estimated=np.asarray(item['local_odometry_xy'])
            if (item['high'] or item['delta_z']<-.53
                    or np.linalg.norm(estimated-self.empty)<.91):
                continue
            if item['clipped'] and self.track is None:
                continue
            low.append(item)
        if low:
            prior=self.track
            def score(ray):
                base=ray['horiz']
                if prior is not None and now-prior[1]<.36:
                    base+=.7*np.linalg.norm(np.asarray(ray['local_odometry_xy'])-prior[0])
                return base
            chosen=min(low,key=score)
            if not chosen['clipped']:
                point=np.array(chosen['local_odometry_xy'])
                self.track=(point,now)
            elif prior is not None and now-prior[1]<2. and chosen['horiz']<1.35:
                ray=prior[0]-origin
                bearing=float(np.rad2deg(np.arctan2(np.dot(ray,[ahead[1],-ahead[0]]),np.dot(ray,ahead))))
                chosen={**chosen,'bearing':bearing}
            return chosen,options
        if self.track is not None and now-self.track[1]<.28:
            ray=self.track[0]-origin
            return dict(bearing=float(np.rad2deg(np.arctan2(np.dot(ray,[ahead[1],-ahead[0]]),
                              np.dot(ray,ahead)))),horiz=float(np.linalg.norm(ray)),box=None),options
        return None,options


def run(case,out):
    reserve,circ,duration=TRIALS[case]
    out.mkdir(parents=True)
    env,bank=build_orchard()
    env.random_state.seed(SEED)
    env._time_limit=duration+.01
    task=env.task
    task.check_termination=lambda _:False
    env.reset()
    phy=env.physics
    dt,native=env.control_timestep(),phy.timestep()
    steps,block=round(duration/dt),round(.02/dt)
    assert steps%block==0
    task._ref_qpos=np.pad(task._ref_qpos,((0,steps+task._future_steps),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,steps+task._future_steps),(0,0)),mode='edge')
    eye=OrchardEyes(phy)
    walk=ReversedYawWalk(env,duration)
    _,stow=fold_until_launch(task,phy)
    start=np.load(PREMEAL/'QPOS.npy',allow_pickle=False)[-1].copy()
    phy.data.qpos[:]=start
    phy.data.qvel[:]=0.
    phy.forward()
    updater=env._observation_updater
    updater.reset(env._physics_proxy,env.random_state)
    observation=updater.get_observation()
    walk.position[:]=phy.bind(task._root_joint).qpos[:2]
    walk.heading=land.angles(start[3:7])[0]
    bank.stocks[RESOURCES[0].name]=EMPTY
    initial=np.r_[phy.data.qpos,phy.data.qvel]
    before={name:food.volume for name,food in bank.stocks.items()}
    trace,meals,notes,poses=[],[],[],[]
    hunt=FruitSearch(eye)
    contacts=Contacts(phy)
    feet=set()
    touched=None
    nonfoot=0
    update=task.after_substep
    mouth_idx={r.name:i for i,r in enumerate(RESOURCES)}
    body,state=FeedingBody(),FeedingState(EMPTY,reserve=reserve,circulating=circ)
    intake=dict.fromkeys(before,0.)
    speed=turn=0.
    action=np.zeros(env.action_spec().shape)
    parked=last_taste=park_action=None
    taste_run=0.

    def sample(physics,rng):
        nonlocal feet,touched,nonfoot
        update(physics,rng)
        floor,fruit,feet,nonfoot,_=contacts.sample()
        oral=physics.data.geom_xpos[contacts.mouth].mean(axis=0)
        supported=bool(feet) and not nonfoot
        touched=bank.contact(oral,supported)
        com=physics.named.data.subtree_com['walker/']
        extra=max(np.abs(physics.data.qfrc_applied).max(),np.abs(physics.data.xfrc_applied).max())
        trace.append((physics.time(),*com,len(floor),max(map(len,fruit.values()),default=0),
                      nonfoot,mouth_idx.get(touched,-1),extra,
                      np.max(np.abs(physics.bind(task._wing_joints).qpos-stow))))

    task.after_substep=sample
    worker=HERE/'brain_worker.py'
    with (out/'NEURAL.log').open('w') as log,subprocess.Popen(
        [str(NEURAL),'-uB',str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=log,text=True,bufsize=1) as neural:
        write_line(neural.stdin,dict(ticks=2*steps,adapter=False,cut=False,seed=SEED,
                                    stream=str(out/'NEURAL.npz')))
        assert json.loads(neural.stdout.readline())['ready']
        for tick in range(steps):
            now=phy.time()
            k=tick%block
            if k==0:
                _,_,items,_=eye.sample()
                chosen,options=hunt.select(eye,items,now)
                center=float(np.clip(chosen['bearing']/35.,-1.,1.)) if chosen else 0.
                row=dict(height_probability=0.,center=center,seen=chosen is not None,
                         front=tuple(.08 if ('T1',side) in feet else 0. for side in ('left','right')),
                         contact=touched is not None,landing_eye=False,circulating=state.circulating)
                write_line(neural.stdin,row)
                line=neural.stdout.readline()
                if not line:
                    raise RuntimeError('neural worker exited')
                packet=json.loads(line)
                ground=np.asarray(packet['walk'])
                gate=packet['hunger']
                notes.append(dict(time_s=now,selection=chosen,candidates=options,gate=gate,**row))
            wanted=float(2.05*np.clip(ground[k].mean()/.27,0.,1.))
            yaw=float(np.clip(5.*(ground[k,1]-ground[k,0]),-2.5,2.5))
            if chosen and chosen['box']:
                x0,_,x1,_=chosen['box']
                width=x1-x0
                wanted*=min(np.interp(width,(160,360),(1.,.24)),
                            np.interp(chosen['horiz'],(.73,1.5),(.28,1.)))
                wanted*=np.interp(abs(chosen['bearing']),(17,42),(1.,.47))
                if width>=280:
                    yaw*=.65
            if touched is not None:
                wanted=yaw=0.
                last_taste=now
                taste_run+=dt
                if parked is None and taste_run>=.025:
                    parked,park_action=now,action.copy()
            else:
                taste_run=0.
            if parked is not None and now-last_taste>.16:
                parked=None
            speed+=-np.expm1(-dt/.032)*(wanted-speed)
            turn+=-np.expm1(-dt/.055)*(yaw-turn)
            if parked is not None:
                action=park_action*(1.-land.smooth((now-parked)/.045))
            elif tick%10==0 and (abs(speed)>.03 or abs(turn)>.03):
                action=walk.action(phy,speed,turn,observation)
            step=env.step(action.copy())
            observation=step.observation
            assert not step.last()
            if (tick+1)%25==0:
                poses.append(phy.data.qpos.copy())
            if (tick+1)%5==0:
                receipt=bank.advance(body,state,5*dt,feed=packet['feed'][k],contact=touched,
                                     activity_use=.012 if speed>.25 else 0.)
                state=receipt.state
                if receipt.ingested.volume:
                    intake[touched]+=receipt.ingested.volume
                meals.append((phy.time(),state.reserve,state.circulating,state.crop.volume,
                              mouth_idx.get(touched,-1),receipt.ingested.volume,receipt.absorbed,
                              receipt.metabolized,receipt.excreted.nutrient,
                              state.crop.nutrient+state.midgut.nutrient))
            if (tick+1)%round(.5/dt)==0:
                print(round(phy.time(),3),np.round(trace[-1][1:4],3).tolist(),
                      'search',gate[2],round(state.circulating,3),'intake',intake,flush=True)
        assert neural.wait()==0

    moves=np.asarray(trace)
    ledger=np.asarray(meals)
    after={name:food.volume for name,food in bank.stocks.items()}
    assert len(moves)==round(duration/native) and np.isfinite(ledger).all()
    assert max(abs(before[name]-after[name]-intake[name]) for name in before)<1e-12
    assert not moves[:,8].any()
    assert abs(reserve+.9*ledger[:,5].sum()-ledger[-1,1]-ledger[:,7:9].sum()-ledger[-1,9])<1e-11
    for name,values in [('TRACE',moves),('MEAL',ledger),('QPOS',poses),('INITIAL',initial)]:
        np.save(out/(name+'.npy'),np.asarray(values),allow_pickle=False)
    (out/'EYES.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in notes))
    first=lambda indexes: float(ledger[indexes[0],0]) if len(indexes) else None
    with np.load(out/'NEURAL.npz',allow_pickle=False) as neural:
        ground_peak=float(neural['ground'].max())
        first_dn=next((i*.0001 for i,x in enumerate(neural['ground']) if x.mean()>.05),None)
    details=dict(case=case,duration=duration,initial_reserve=reserve,initial_circulating=circ,
                 first_search_s=next((row['time_s'] for row in notes if row['gate'][2]),None),
                 first_walk_request_s=next((row['time_s'] for row in notes if row['gate'][3]),None),
                 first_post_neck_walk_s=first_dn,ground_peak=ground_peak,
                 first_intake_s=first(np.flatnonzero(ledger[:,5]>0)),intake=intake,stocks_after=after,
                 reserve_final=state.reserve,circulating_final=state.circulating,
                 max_moved_mm=float(10*np.linalg.norm(moves[:,1:3]-moves[0,1:3],axis=1).max()),
                 min_horizontal_to_second_mm=float(10*np.hypot(moves[:,1]-RESOURCES[1].x,
                                                                moves[:,2]-RESOURCES[1].y).min()),
                 native_steps=len(moves),neural_steps=2*steps,max_external=float(moves[:,8].max()))
    (out/'RESULT.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
    print(details,flush=True)
    walk.env.close()
    env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=TRIALS)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    run(args.case,args.output.resolve())
