"""Run a saved SAC policy on the P1 FlyBody with live Eon/BANC and stocked fruit."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from forage import DT, MEAL, Terrain, HERE


sys.path.insert(0,str(HERE.parent/'20261119T021500Z-orchard-behavior'))
from run_case import NEURAL, OrchardEyes, fold_until_launch, land, write_line
from steering_recheck import ReversedYawWalk
from orchard import FoodBank, Resource, solid
from legged_fixture import build_env
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState


def build(seed,level,side,visible=True):
    ground=Terrain.from_seed(seed,level,side)
    env=build_env()
    arena=env.task._arena.mjcf_model
    world=arena.worldbody
    plane=arena.find('geom','groundplane')
    plane.pos=(0.,0.,0. if not level else -.23)
    plane.material=None
    plane.rgba=(.57,.57,.45,1.) if not level else (.10,.015,.58,1.)
    world.add('light',name='sky',directional=True,pos=(0,0,5),dir=(.3,.2,-1.),
              ambient=(.55,.55,.55),diffuse=(.50,.50,.50),castshadow=False)
    if level:
        for name,lo,hi,ylo,yhi in (
                ('before',-2.,ground.near,-1.7,1.7),
                ('after',ground.far,4.,-1.7,1.7),
                ('bridge',ground.near,ground.far,
                 ground.edge if ground.side==1 else -1.7,1.7 if ground.side==1 else -ground.edge)):
            solid(world,name,'box',pos=((lo+hi)/2.,(ylo+yhi)/2.,-.10),
                  size=((hi-lo)/2.,(yhi-ylo)/2.,.10),rgba=(.55,.53,.37,1.))
    for which in ('left','right'):
        world.add('camera', name='orchard_eye_'+which, fovy=56.)
    fruit=Resource('trial_fruit','ground',ground.goal_x,ground.goal_y,.023,.27,.9)
    if visible:
        solid(world,fruit.geom,'cylinder',pos=(fruit.x,fruit.y,-.0065),
              size=(fruit.radius,.0295),rgba=(.88,.07,.025,1.),friction=(1.13,.005,.0001))
    return env,ground,FoodBank([fruit] if visible else [])


class Contacts:
    def __init__(self,phy):
        m=phy.model.ptr
        self.geom_foot={}
        self.walker=[]
        for i in range(m.ngeom):
            name=m.body(m.geom_bodyid[i]).name
            self.walker.append(name.startswith('walker/'))
            if 'tarsus_' in name or 'claw_' in name:
                self.geom_foot[i]=tuple(name.rsplit('_',2)[-2:])
        self.mouth=[m.geom('walker/labrum_'+side+'_lower_collision').id for side in ('left','right')]

    def scan(self,phy):
        data=phy.data
        foot=set()
        nonfoot=0
        for c in data.contact[:data.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            if self.walker[b]:
                a,b=b,a
            if not self.walker[a] or self.walker[b] or c.efc_address<0:
                continue
            if a in self.geom_foot:
                foot.add(self.geom_foot[a])
            else:
                nonfoot+=1
        oral=data.geom_xpos[self.mouth].mean(axis=0)
        return foot,nonfoot,oral


def visual(eye, ascending, speed, circulating):
    left,right,items,_=eye.sample()
    item=max(items,key=lambda v:v.area_px,default=None)
    distance=1.
    if item:
        a=np.deg2rad(-item.bearing_deg)
        # Estimate range from stereo when available, otherwise apparent width.
        x0,_,x1,_=item.box
        depth=item.depth_cm or (.54*203./max(x1-x0,1))
        distance=min(1.,depth/2.6)
    hazards=[]
    for rgb in (left,right):
        low=rgb[rgb.shape[0]//2:].astype(np.int16)
        obstacle=(low[:,:,2]-low[:,:,0]>18)&(low[:,:,2]-low[:,:,1]>14)
        # A nearby floor gap occupies a larger part of the lower visual field.
        hazards.append(np.array([min(1.,np.mean(c)*5.) for c in np.array_split(obstacle,6,axis=1)]))
    hazard=np.maximum(*hazards)[::-1]
    state=np.array([float(item is not None),np.sin(a) if item else 0.,np.cos(a) if item else 0.,
               distance,*hazard,*np.asarray(ascending).clip(0.,1.),speed,.30-circulating],dtype=np.float32)
    return state,left


def run(args):
    args.out.mkdir()
    with np.load(args.checkpoint,allow_pickle=False) as weights:
        saved={k:weights[k] for k in weights.files}
    identity=json.loads(saved['identity'].item())
    arm=identity['arm']
    env,ground,bank=build(args.seed,args.level,args.side,visible=not args.no_food)
    env.random_state.seed(29)
    env._time_limit=args.duration+.01
    task=env.task
    task.check_termination=lambda _:False
    env.reset()
    phy=env.physics
    dt=env.control_timestep()
    steps=round(args.duration/dt)
    block=round(.02/dt)
    assert steps%block==0
    extra=steps+task._future_steps
    task._ref_qpos=np.pad(task._ref_qpos,((0,extra),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,extra),(0,0)),mode='edge')
    source=HERE.parent/'20261119T052620Z-hunger-foraging/VERIFIED/FADE/INITIAL.npy'
    start=np.load(source)[:phy.model.nq].copy()
    start[:3]=(0.,0.,.1378)
    start[3:7]=(1.,0.,0.,0.)
    with phy.reset_context():
        phy.data.qpos[:]=start
        phy.data.qvel[:]=0.
        phy.data.ctrl[:]=0.
        phy.data.act[:]=0.
    _,stow=fold_until_launch(task,phy)
    walker=ReversedYawWalk(env,args.duration)
    walker.position[:]=start[:2]
    upstream=env._observation_updater
    upstream.reset(env._physics_proxy,env.random_state)
    obs=upstream.get_observation()
    eye=OrchardEyes(phy)
    contacts=Contacts(phy)
    feeder=FeedingBody()
    state=FeedingState(EMPTY,reserve=.26,circulating=.23)
    pace=turn=0.
    feedback=np.zeros(2)
    proposed=(0.,0.)
    motor=np.zeros(env.action_spec().shape)
    poses,meals,observations,frames=[],[],[],[]
    start_move=False
    walk_begin=None
    last_taste=0.
    total=0.
    fall=0
    zeros=0.
    fields=phy.model.name2id('walker/thorax','body')

    with (args.out/'NEURAL.log').open('w') as log,subprocess.Popen(
        [str(NEURAL),'-uB',str(HERE/'neural_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        text=True,bufsize=1,stderr=log) as neural:
        write_line(neural.stdin,dict(ticks=2*steps,adapter=False,cut=False,seed=29,arm=arm,
                   calibration=str(args.cal.resolve()),stream=str(args.out/'NECK.npz')))
        line=neural.stdout.readline()
        if not line:
            raise RuntimeError('Neural worker did not initialize')
        ready=json.loads(line)
        assert ready['weights']==identity['brain_sha256']
        for tick in range(steps):
            now=tick*dt
            k=tick%block
            feet,nonfoot,oral=contacts.scan(phy)
            bite=bank.contact(oral,bool(feet) and not nonfoot)
            if now>.25 and len(feet)<=2 and nonfoot and phy.data.xmat[fields,8]<.53:
                fall+=1
            if k==0:
                if start_move and tick%round(DT/dt)==0:
                    sense,left=visual(eye,feedback,pace,state.circulating)
                    actor_output=sense
                    for layer in (0,2,4):
                        actor_output=actor_output@saved[f'w{layer}'].T+saved[f'b{layer}']
                        if layer!=4:
                            actor_output=np.maximum(actor_output,0.)
                    actor_output=np.tanh(actor_output[:2])
                    proposed=(.76*(actor_output[0]+1.)/2.,.22*actor_output[1])
                    if len(frames)<4 or tick%round(.8/dt)==0:
                        frames.append(left)
                    observations.append((now,*sense,*actor_output))
                write_line(neural.stdin,dict(intent=list(map(float,proposed)),
                    height_probability=0.,center=0.,seen=False,
                    front=[.08 if ('T1',side) in feet else 0. for side in ('left','right')],
                    contact=bite is not None,landing_eye=False,circulating=state.circulating))
                line=neural.stdout.readline()
                if not line:
                    raise RuntimeError('Neural worker ended before final time')
                packet=json.loads(line)
                feedback=.6*feedback+.4*np.asarray(packet['arrived_an'])/8.
                drive=np.array(packet['walk'])
            start_move=now>=.24
            desired=.98*np.clip(drive[k].sum()/1.4,0.,1.)
            desired_turn=3.25*(drive[k,1]-drive[k,0])
            if bite is not None:
                desired=desired_turn=0.
                last_taste=now
            pace+=(desired-pace)*(-np.expm1(-dt/.032))
            turn+=(desired_turn-turn)*(-np.expm1(-dt/.055))
            if tick%10==0:
                if now-last_taste<.035 and last_taste>.1:
                    motor*=.91
                elif abs(pace)+abs(turn)>.03:
                    if walk_begin is None:
                        walk_begin=now
                        walker.position[:]=phy.data.qpos[:2]
                        walker.heading=land.angles(phy.data.qpos[3:7])[0]
                    ramp=land.smooth((now-walk_begin)/.14)
                    motor=ramp*walker.action(phy,pace*ramp,turn*ramp,obs)
                else:
                    motor[:]=0.
                    walker.filtered[:]=0.
                    walk_begin=None
            step=env.step(motor.copy())
            assert not step.last()
            obs=step.observation
            if (tick+1)%5==0:
                row=bank.advance(feeder,state,5*dt,feed=packet['feed'][k],contact=bite,
                                 activity_use=.012*pace)
                state=row.state
                total+=row.ingested.volume
                meals.append((now,total,state.circulating,float(bite is not None)))
            if (tick+1)%25==0:
                poses.append(phy.data.qpos.copy())
            zeros=max(zeros,float(np.abs(phy.data.qfrc_applied).max()),
                      float(np.abs(phy.data.xfrc_applied).max()))
            if (tick+1)%round(.80/dt)==0:
                print(arm,round(now,2),np.round(phy.data.qpos[:3],3).tolist(),round(total,4),flush=True)
        assert neural.wait()==0
    assert zeros==0.
    assert abs(sum(.9-v.volume for v in bank.stocks.values())-total)<1e-10
    np.savez_compressed(args.out/'LIVE.npz',trace=poses,meals=meals,visual=frames,observation=observations)
    result=dict(arm=arm,level=args.level,terrain=asdict(ground),checkpoint=str(args.checkpoint),
                weights=ready['weights'],seed=args.seed,food_hidden=args.no_food,
                intake=total,passed=total>=MEAL and fall==0,fall_control_samples=fall,
                final_position=poses[-1][:3].tolist(),extra_force=zeros)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    walker.env.close()
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cal',type=Path)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=83291)
    p.add_argument('--level',type=int,choices=(0,1),default=1)
    p.add_argument('--side',type=int,choices=(-1,1),default=1)
    p.add_argument('--duration',type=float,default=4.8)
    p.add_argument('--no-food',action='store_true')
    run(p.parse_args())
