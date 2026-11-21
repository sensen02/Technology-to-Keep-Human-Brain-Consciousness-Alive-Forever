"""Shared P1 course, optical features and fixed experiment identities."""
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PAIR=HERE.parent/'20261119T073818Z-sac-terrain-brain-pair'
ORCHARD=HERE.parent/'20261119T012530Z-survival-orchard'
sys.path.insert(0,str(PAIR))
sys.path.insert(0,str(ORCHARD))
from forage import Terrain, DT, OBS, MEAL
from policy_runtime import actor_mean
from orchard import FoodBank, Resource, detail, solid
from orchard_eye import OrchardEyes
sys.path.insert(0,str(HERE.parent/"20261118T072349Z-flight-descending"))
from legged_fixture import build_env

CHALLENGES=('flat','wide_bridge','bridge','long_bridge')
TRAIN_SEEDS=tuple(range(11530,11538))
VISION_HOLDOUT=(11610,11611,11612,11613)
PHYSICAL_HOLDOUT=(15103,15107,15121,15131)
BRAIN_SHA='71904e7a9765e5b0fb58afbe9c0ce27efa591f84997b51f0996f58a6eafbeaf3'
START=HERE.parent/'20261119T052620Z-hunger-foraging/VERIFIED/FADE/INITIAL.npy'
FRUIT_RADIUS=.49


from controller import terrain

def init_course(seed,challenge,side,*,food=True):
    cfg=terrain(seed,challenge,side)
    env=build_env()
    env.random_state.seed(29)
    arena=env.task._arena.mjcf_model
    w=arena.worldbody
    plane=arena.find('geom','groundplane')
    plane.material=None
    plane.pos=(0.,0.,0. if not cfg.side else -.24)
    plane.rgba=(.51,.48,.34,1.) if not cfg.side else (.115,.025,.50,1.)
    w.add('light',name='sky',directional=True,pos=(0,0,5),dir=(.15,.25,-1),
          ambient=(.49,.49,.48),diffuse=(.45,.46,.47),castshadow=False)
    if cfg.side:
        for name,lo,hi,ylo,yhi in (
                ('start',-2.,cfg.near,-1.40,1.40),
                ('end',cfg.far,4.,-1.40,1.40),
                ('bridge',cfg.near,cfg.far,cfg.edge if cfg.side==1 else -1.40,
                 1.40 if cfg.side==1 else -cfg.edge)):
            solid(w,name,'box',pos=((lo+hi)/2.,(ylo+yhi)/2.,-.12),
                  size=((hi-lo)/2.,(yhi-ylo)/2.,.12),rgba=(.49,.45,.32,1.))
        lo,hi=(-1.40,cfg.edge) if cfg.side==1 else (-cfg.edge,1.40)
        # Engineering contrast marking in the introductory course. It has no collision.
        for idx,x in enumerate((cfg.near-.024,cfg.far+.024)):
            detail(w,f'lip_{idx}','box',pos=(x,(lo+hi)*.5,.002),
                   size=(.023,(hi-lo)*.5,.002),rgba=(.095,.015,.75,1.))
    for side in ('left','right'):
        w.add('camera',name='orchard_eye_'+side,fovy=56)
    if food:
        r=Resource('trial_fruit','ground',cfg.goal_x,cfg.goal_y,.004,FRUIT_RADIUS,.9)
        detail(w,r.geom,'cylinder',pos=(r.x,r.y,.002),size=(r.radius,.002),
               rgba=(.84,.065,.024,1.))
        bank=FoodBank([r])
    else:
        bank=FoodBank([])
    return env,cfg,bank


def start_pose(env):
    phy=env.physics
    start=np.load(START,allow_pickle=False)[:phy.model.nq].copy()
    start[:3]=(0.,0.,.1378)
    start[3:7]=(1.,0.,0.,0.)
    with phy.reset_context():
        phy.data.qpos[:]=start
        phy.data.qvel[:]=0.
        phy.data.ctrl[:]=0.
        phy.data.act[:]=0.
    return start


def optic_features(left,right):
    def features(rgb):
        lower=rgb[rgb.shape[0]//2:].astype(np.int16)
        mask=(lower[:,:,2]-lower[:,:,0]>18)&(lower[:,:,2]-lower[:,:,1]>14)
        # Order columns in left-turn coordinate order, rows in image order.
        people=[]
        for segment in np.array_split(mask,6,axis=1)[::-1]:
            densities=[block.mean() for block in np.array_split(segment,4,axis=0)]
            rows=np.flatnonzero(segment.sum(axis=1)>=2)
            people.extend(densities+[float(rows[-1]+1)/len(segment) if len(rows) else 0.])
        return np.asarray(people)
    a,b=features(left),features(right)
    return np.r_[np.maximum(a,b),np.minimum(a,b)].astype(np.float32)


def desired_hazard(cfg,xy,yaw):
    directions=yaw+np.linspace(-.88,.88,6)
    rays=np.column_stack((np.cos(directions),np.sin(directions)))
    ranges=np.linspace(.055,.86,24)
    sensor=np.asarray(xy)+rays[:,None,:]*ranges[None,:,None]
    return (cfg.blocked(sensor)*(1.-ranges[None,:]/.94)).max(axis=1)


def decode_visual(cal,left,right):
    f=optic_features(left,right)
    h=np.r_[f,np.sqrt(np.maximum(f,0.)),1.]@cal
    # A neutral field has no contrast marker. Suppress extrapolated risk.
    local=np.max(f[:30].reshape(6,5),axis=1)>0.
    return np.where(local,np.clip(h,0.,1.),0.).astype(np.float32)


def observation(eye,cal,an,pace,circulating,last=None):
    left,right,items,_=eye.sample()
    item=max(items,key=lambda x:x.area_px,default=None)
    if item:
        angle=np.deg2rad(-item.bearing_deg)
        box=item.box
        depth=item.depth_cm or (2.*FRUIT_RADIUS*203./max(1,box[2]-box[0]))
        r=min(1.,depth/2.6)
        visual=np.array([1.,np.sin(angle),np.cos(angle),r])
    elif last is not None:
        visual=last.copy()
        visual[0]=0.
    else:
        visual=np.array([0.,0.,0.,1.])
    risk=decode_visual(cal,left,right)
    obs=np.r_[visual,risk,np.asarray(an).clip(0.,1.),pace,.30-circulating].astype(np.float32)
    assert obs.shape==(OBS,)
    return obs,left,visual if item else last
