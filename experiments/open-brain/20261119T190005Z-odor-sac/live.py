"""Run trained odor navigation through frozen Eon, BANC and FlyBody."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np

from odor_env import HERE, course, intent, odor_field, odor_steering, sample
from proximity import DETOUR_RANGE, RANGE, FittedNose, ray_hazard

CLOSED = HERE.parent/'20261119T084744Z-closed-loop-forage'
sys.path.insert(0,str(CLOSED))
import run_course_docking as docking
from common import FoodBank, Resource, build_env, detail, solid
from controller import actor_mean, infer as vision_infer

SENSES = ('odor','blank','vision','both')
WALL = 'odor_barrier'


def build(seed,stage,side,*,food=True):
    cfg = course(seed,stage,side)
    env = build_env()
    env.random_state.seed(29)
    arena = env.task._arena.mjcf_model
    world = arena.worldbody
    plane = arena.find('geom','groundplane')
    plane.material = None
    plane.pos = (0.,0.,0.)
    plane.rgba = (.53,.48,.33,1.)
    world.add('light',name='odor_sky',directional=True,pos=(0,0,5),dir=(.16,.32,-1),
              ambient=(.47,.46,.45),diffuse=(.49,.47,.43))
    if stage != 'open':
        solid(world,WALL,'box',pos=((cfg.near+cfg.far)*.5,(cfg.low+cfg.high)*.5,.25),
              size=((cfg.far-cfg.near)*.5,(cfg.high-cfg.low)*.5,.25),
              rgba=(.40,.45,.39,1.))
    for name in ('left','right'):
        world.add('camera',name='orchard_eye_'+name,fovy=56)
    resources = []
    if food:
        fruit = Resource('trial_fruit','ground',cfg.goal_x,cfg.goal_y,.004,.49,.9)
        detail(world,fruit.geom,'cylinder',pos=(fruit.x,fruit.y,.002),
               size=(fruit.radius,.002),rgba=(.85,.065,.025,1.))
        resources.append(fruit)
    return env,cfg,FoodBank(resources)


class Sensor:
    def __init__(self,args,cfg):
        self.args = args
        self.field = odor_field(cfg)
        self.nose = FittedNose()
        self.rows = []
        self.last_visual = None
        self.yaw_rate = 0.
        self.ant = None
        self.wall_id = -1
        self.wall_group = np.array([1,0,0,0,0,0],dtype=np.uint8)

    def read(self,eye,cal,an,pace,circulating,_last):
        phy = eye.phy
        if self.ant is None:
            self.ant = [phy.model.name2id('walker/antenna_'+side,'body') for side in ('left','right')]
            assert min(self.ant)>=0
            if self.args.stage != 'open':
                self.wall_id = phy.model.name2id(WALL,'geom')
                assert self.wall_id>=0
        antenna = phy.data.xpos[self.ant].copy()
        concentrations = sample(self.field,antenna[:,:2])
        _, heading, _ = eye.head_pose()
        yaw = math.atan2(heading[1],heading[0])
        threat = (0.,0.)
        if self.wall_id>=0 and self.args.assisted:
            p = antenna.mean(axis=0)
            geom = np.empty(1,dtype=np.int32)

            def cast(dx,dy):
                distance = mujoco.mj_ray(phy.model.ptr,phy.data.ptr,p,np.array((dx,dy,0.)),
                                        self.wall_group,1,-1,geom)
                return distance if geom[0]==self.wall_id else -1.

            limit = DETOUR_RANGE if self.args.stage=='detour' else RANGE
            threat = ray_hazard(yaw,cast,limit)
        self.yaw_rate = .67*self.yaw_rate+.33*float(phy.data.qvel[5])
        readings = concentrations if self.args.sense in ('odor','both') else (0.,0.)
        obs = self.nose.observe(readings,pace,float(np.clip(self.yaw_rate,-.9,.9)),threat)
        vision,left,_ = self.visual(eye,cal,an,pace,circulating,None)
        self.last_visual = vision
        self.rows.append([float(phy.time()),*antenna.ravel(),*concentrations,*readings,
                          *threat,float(vision[0]),yaw])
        # Keep the sensing and action timing identical to the original runner.
        return obs,left,None

    def infer(self,policy,obs,profile):
        if self.args.sense in ('vision','both') and self.last_visual[0]>0.:
            return vision_infer(None,self.last_visual,profile=profile)
        action = odor_steering(obs) if policy is None else actor_mean(policy,obs)
        return action,intent(obs,action)


class WallContacts(docking.Contacts):
    def __init__(self,phy):
        super().__init__(phy)
        self.wall = mujoco.mj_name2id(phy.model.ptr,mujoco.mjtObj.mjOBJ_GEOM,WALL)
        self.contacts = []
        self.last = None

    def scan(self,phy):
        feet,nonfoot,oral = super().scan(phy)
        if self.wall < 0:
            return feet,nonfoot,oral
        touched = tuple(sorted(int(c.geom1 if c.geom2==self.wall else c.geom2)
                               for c in phy.data.contact[:phy.data.ncon]
                               if self.wall in (c.geom1,c.geom2) and c.efc_address>=0))
        if touched!=self.last:
            self.contacts.append((float(phy.time()),touched))
            self.last=touched
        return feet,nonfoot,oral


def run(args):
    cfg = course(args.seed,args.stage,args.side)
    sensor = Sensor(args,cfg)
    sensor.visual = docking.observation
    contact_rows = []

    class Capture(WallContacts):
        def __init__(self,phy):
            super().__init__(phy)
            self.contacts = contact_rows

    docking.init_course=build
    docking.observation=sensor.read
    docking.infer=sensor.infer
    docking.Contacts=Capture
    docking.run(SimpleNamespace(out=args.out,vision=CLOSED/'VISION/VISION.npz',policy=args.policy,
        seed=args.seed,challenge=args.stage,side=args.side,no_food=False,duration=args.duration,
        neural_seed=29,early=True,profile='strong'))
    np.savez_compressed(args.out/'SENSE.npz',samples=sensor.rows,
        columns=['t','left_x','left_y','left_z','right_x','right_y','right_z',
                 'field_l','field_r','used_l','used_r','obstacle_l','obstacle_r','visual_fruit','head_yaw'])
    (args.out/'ARM.json').write_text(json.dumps(dict(sense=args.sense,assisted=args.assisted,
        wall_contact_changes=contact_rows),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--policy',type=Path)
    p.add_argument('--seed',type=int,default=88207)
    p.add_argument('--side',type=int,choices=(-1,1),default=1)
    p.add_argument('--stage',choices=('open','wall','detour'),default='wall')
    p.add_argument('--sense',choices=SENSES,default='odor')
    p.add_argument('--duration',type=float,default=7.2)
    p.add_argument('--assisted',action='store_true')
    run(p.parse_args())
