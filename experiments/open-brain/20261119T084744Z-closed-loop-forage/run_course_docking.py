"""Use actual P1 eyes and Eon→BANC→FlyBody dynamics to measure full meals."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import (HERE, PAIR, BRAIN_SHA, CHALLENGES, MEAL, init_course, start_pose,
                    observation, OrchardEyes)
from controller import decode_levels, infer
from live_course import Contacts
from run_case import NEURAL, fold_until_launch, land, write_line
from steering_recheck import ReversedYawWalk
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState


def bounded_reference(walk,phy):
    q=phy.data.qpos
    displacement=walk.position-q[:2]
    limit=.052
    length=np.linalg.norm(displacement)
    if length>limit:
        walk.position[:]=q[:2]+limit*displacement/length
    actual=land.angles(q[3:7])[0]
    error=np.arctan2(np.sin(walk.heading-actual),np.cos(walk.heading-actual))
    walk.heading=actual+float(np.clip(error,-.20,.20))


def run(args):
    args.out.mkdir()
    with np.load(args.vision,allow_pickle=False) as v:
        calibration=v['weights']
    policy=None
    if args.policy:
        with np.load(args.policy,allow_pickle=False) as f:
            policy={k:f[k] for k in f.files}
    env,cfg,bank=init_course(args.seed,args.challenge,args.side,food=not args.no_food)
    task=env.task
    env._time_limit=args.duration+.01
    task.check_termination=lambda _:False
    env.reset()
    start=start_pose(env)
    phy=env.physics
    native=phy.timestep()
    dt=env.control_timestep()
    steps=round(args.duration/dt)
    block=round(.02/dt)
    assert steps%block==0
    task._ref_qpos=np.pad(task._ref_qpos,((0,steps+task._future_steps),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,steps+task._future_steps),(0,0)),mode='edge')
    fold_until_launch(task,phy)
    updater=env._observation_updater
    updater.reset(env._physics_proxy,env.random_state)
    obs=updater.get_observation()
    walk=ReversedYawWalk(env,args.duration)
    walk.position[:]=start[:2]
    scan=Contacts(phy)
    eye=OrchardEyes(phy)
    feeder=FeedingBody()
    state=FeedingState(EMPTY,reserve=.26,circulating=.23)
    command=(0.,0.)
    pace=turn=0.
    an=np.zeros(2)
    motor=np.zeros(env.action_spec().shape)
    poses,measurements,eyes,ledger,commands,trace=[],[],[],[],[],[]
    taste_start=None
    decaying=None
    stable_start=None
    meal_at=None
    meal_motor=None
    memory=None
    last_seen=-1.
    total=0.
    first=None
    first_cross=None
    first_fall=None
    max_force=0.
    pose_origin=phy.data.qpos[:3].copy()
    th=phy.model.name2id('walker/thorax','body')
    near_factor=1.
    front=np.zeros(2)
    window=round(.08/dt)
    anchor=PAIR/'NEURAL_CAL/CALIBRATION.json'
    stream=args.out/'NECK.npz'
    state_names=tuple(bank.stocks)
    controls=0
    actual_steps=0
    steady_seconds=0.

    worker=HERE/'early_neural_worker.py' if args.early else PAIR/'neural_worker.py'
    with (args.out/'NEURAL.log').open('w') as log,subprocess.Popen(
         [str(NEURAL),'-uB',str(worker)],stdin=subprocess.PIPE,
         stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1) as neural:
        write_line(neural.stdin,dict(ticks=2*steps,adapter=False,cut=False,seed=args.neural_seed,
                                    arm='frozen',calibration=str(anchor),stream=str(stream)))
        line=neural.stdout.readline()
        if not line:
            raise RuntimeError('Neural process did not initialize')
        ready=json.loads(line)
        assert ready['weights']==BRAIN_SHA
        for tick in range(steps):
            now=tick*dt
            k=tick%block
            feet,nonfoot,oral=scan.scan(phy)
            bite=bank.contact(oral,len(feet)>=3 and phy.data.xmat[th,8]>=.88)
            if first_fall is None and now>.24 and len(feet)<3 and nonfoot and phy.data.xmat[th,8]<.55:
                first_fall=now
            if (args.early and now>.24 and k==0 and
                    (first_fall is not None or (total>=MEAL and steady_seconds>=.16))):
                write_line(neural.stdin,dict(finished=True))
                break
            if k==0:
                if now>=.16 and tick%window==0:
                    sense,image,recall=observation(eye,calibration,an,pace,state.circulating,
                        memory if now-last_seen<=.40 else None)
                    if sense[0]:
                        memory=recall
                        last_seen=now
                    action,command=infer(policy,sense,profile=args.profile)
                    if tick%(5*window)==0 or len(eyes)<3:
                        eyes.append(image.copy())
                    measurements.append((now,*sense,*action,*command))
                    # A fixed local approach regulator reduces gait speed near the oral radius.
                    near_factor=float(np.interp(sense[3],(.135,.32),(.41,1.))) if sense[0] else 1.
                front=np.array([.08 if ('T1',side) in feet else 0. for side in ('left','right')])
                write_line(neural.stdin,dict(intent=command,height_probability=0.,center=0.,seen=False,
                                       front=front.tolist(),contact=bite is not None,
                                       landing_eye=False,circulating=state.circulating))
                line=neural.stdout.readline()
                if not line:
                    raise RuntimeError('Neural process stopped early')
                packet=json.loads(line)
                drive=np.asarray(packet['walk'])
                an=.52*an+.48*np.asarray(packet['arrived_an'])/8.
                commands.append((now,*command,*drive.mean(axis=0),*an))
            speed,world_yaw=decode_levels(*drive[k])
            speed=min(speed, .86-.25*abs(world_yaw))*near_factor
            if bite:
                speed=world_yaw=0.
                taste_start=now if taste_start is None else taste_start
                if decaying is None:
                    decaying=motor.copy()
            elif taste_start is not None and now-taste_start>.06:
                taste_start=decaying=None
            if meal_at is not None:
                speed=world_yaw=0.
            pace+=(speed-pace)*(-np.expm1(-dt/.046))
            turn+=(world_yaw-turn)*(-np.expm1(-dt/.082))
            if tick%10==0:
                if meal_at is not None:
                    motor=meal_motor*(1.-land.smooth((now-meal_at)/.060))
                elif decaying is not None:
                    motor=decaying*(1.-land.smooth((now-taste_start)/.05))
                elif abs(pace)+abs(turn)>.021:
                    if stable_start is None:
                        stable_start=now
                        walk.position[:]=phy.data.qpos[:2]
                        walk.heading=land.angles(phy.data.qpos[3:7])[0]
                    ratio=land.smooth((now-stable_start)/.15)
                    bounded_reference(walk,phy)
                    motor=ratio*walk.action(phy,pace*ratio,-turn*ratio,obs)
                    controls+=1
                else:
                    motor[:]=0.
                    walk.filtered[:]=0.
                    stable_start=None

            if np.linalg.norm(phy.data.qvel[:2])<=.08 and len(feet)>=4 and phy.data.xmat[th,8]>.94:
                steady_seconds+=dt
            else:
                steady_seconds=0.
            result=env.step(motor.copy())
            actual_steps+=1
            assert not result.last()
            obs=result.observation
            if (tick+1)%5==0:
                receipt=bank.advance(feeder,state,5*dt,feed=packet['feed'][k],contact=bite,activity_use=.012*pace)
                state=receipt.state
                total+=receipt.ingested.volume
                if meal_at is None and total>=MEAL:
                    meal_at=now
                    meal_motor=motor.copy()
                if first is None and receipt.ingested.volume>0.:
                    first=now
                ledger.append((now,total,state.circulating,float(bite is not None),
                               receipt.ingested.volume))
            if (tick+1)%25==0:
                poses.append(phy.data.qpos.copy())
                q=phy.data.qpos[:3]
                trace.append((now,*q,len(feet),nonfoot,phy.data.xmat[th,8],*phy.data.qvel[:3]))
                if first_cross is None and q[0]>=cfg.far and cfg.side:
                    first_cross=(now,float(q[1]),float(q[2]))
            max_force=max(max_force,float(np.abs(phy.data.qfrc_applied).max()),
                          float(np.abs(phy.data.xfrc_applied).max()))
            if (tick+1)%round(.8/dt)==0:
                print(round(now,2),np.round(phy.data.qpos[:3],3).tolist(),'intake',round(total,4),
                      'command',np.round(command,3).tolist(),'fall',first_fall,flush=True)
        assert neural.wait()==0

    stock_loss=sum(.9-bank.stocks[key].volume for key in state_names)
    assert abs(stock_loss-total)<1e-10 and max_force==0.
    data=np.asarray(trace)
    actual_duration=actual_steps*dt
    final=data[data[:,0]>=actual_duration-.14]
    stop=bool(np.all((final[:,4]>=4)&(final[:,6]>.94)&
                     (np.linalg.norm(final[:,7:9],axis=1)<=.08)))
    np.savez_compressed(args.out/'BODY.npz',poses=poses,trace=trace,visual=eyes,
                        meal=ledger,observation=measurements,commands=commands,initial=start)
    result=dict(challenge=args.challenge,seed=args.seed,side=args.side,neural_seed=args.neural_seed,
                geometry=asdict(cfg),profile=args.profile,model_sha=BRAIN_SHA,policy=str(args.policy),food=not args.no_food,
                intake=total,stock_loss=stock_loss,first_intake_s=first,meal_complete_s=meal_at,
                meal_pass=bool(total>=MEAL and first_fall is None and stop),
                intake_target_pass=total>=MEAL,ended_standing=stop,
                max_x_mm=float(10*(data[:,1].max()-pose_origin[0])),first_far_cross=first_cross,
                first_fall_s=first_fall,native_steps=round(actual_duration/native),
                native_extra_force=max_force,duration_s=actual_duration,
                max_duration_s=args.duration,early=args.early,body_updates=controls)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result,flush=True)
    walk.env.close()
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('vision',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--policy',type=Path)
    p.add_argument('--challenge',choices=CHALLENGES,default='flat')
    p.add_argument('--seed',type=int,default=11530)
    p.add_argument('--side',type=int,choices=(-1,1),default=1)
    p.add_argument('--neural-seed',type=int,default=29)
    p.add_argument('--duration',type=float,default=4.8)
    p.add_argument('--no-food',action='store_true')
    p.add_argument('--early',action='store_true')
    p.add_argument('--profile',choices=('small','strong'),default='small')
    run(p.parse_args())
