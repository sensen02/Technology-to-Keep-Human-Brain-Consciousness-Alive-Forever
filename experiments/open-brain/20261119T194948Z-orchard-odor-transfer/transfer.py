"""Evaluate the frozen P1 odor policy on sourced crops of the user's orchard."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
SAC=HERE.parent/'20261119T190005Z-odor-sac'
sys.path.insert(0,str(SAC))
from live import CLOSED, docking, FoodBank, Resource, build_env, solid, actor_mean
from odor_env import GRID, ORIGIN, XX, YY, intent, lattice, sample
from proximity import DETOUR_RANGE, FittedNose, ray_hazard

POLICY=SAC/'TRAIN_DETOUR/POLICY.npz'
SETTLED=SAC/'LIVE_DETOUR_BLANK_L/BODY.npz'
FRUIT_MASK=np.array([0,0,0,1,0,0],dtype=np.uint8)
SCENE_MASK=np.array([1,0,0,1,0,0],dtype=np.uint8)
SKIN_REACH=.055
SKIN_PATCH_RADIUS=.49
THICKNESS=.006
FACES=(0,1,2,3,5,4,0,4,1,0,3,4,1,5,2,1,4,5,2,3,0,2,5,3)
COLORS={'grass':(.31,.45,.13,1.),'litter':(.45,.30,.16,1.),
        'stone':(.40,.38,.33,1.),'twig':(.33,.20,.10,1.)}


@dataclass(frozen=True)
class Experiment:
    site: str
    fruit: str
    goal_x: float
    goal_y: float
    goal_z: float
    side: int=0
    far: float=1.76


class Crop:
    def __init__(self,site,support='mesh'):
        self.site=site
        folder=HERE/site
        self.source=json.loads((folder/'SOURCE.json').read_text())
        with np.load(folder/'CROP.npz',allow_pickle=False) as z:
            for key in z.files:
                setattr(self,key,z[key])
        self.floor=RegularGridInterpolator((self.y,self.x),self.terrain,
                                           bounds_error=False,fill_value=None)
        self.support=support
        self.cfg=Experiment(site,self.source['apple'],*self.goal.tolist())
        x,y=np.meshgrid(np.linspace(-.36,2.12,25),np.linspace(-.48,.48,13))
        design=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
        relief=self.floor(design[:,[1,0]])
        self.plane=np.linalg.lstsq(design,relief,rcond=None)[0]
        self.plane_corridor_error_mm=10*float(np.max(np.abs(relief-design@self.plane)))

    def ground(self,x,y):
        return float(self.floor((y,x)))


def slice_line(tri,heights):
    points=[]
    for i,j in ((0,1),(1,2),(2,0)):
        a,b=heights[i],heights[j]
        if a*b<0.:
            p=tri[i]+(tri[j]-tri[i])*(-a/(b-a))
            points.append(p[:2])
    return np.asarray(points) if len(points)==2 else None


def odor(crop):
    """Engineering ground-level diffusion past the same extracted obstruction."""
    blocked=np.zeros_like(XX,dtype=bool)
    for tri in crop.debris:
        heights=tri[:,2]-crop.floor(tri[:,[1,0]])-.127
        line=slice_line(tri,heights)
        if line is None:
            continue
        a,b=line
        v=b-a
        fraction=np.clip(((XX-a[0])*v[0]+(YY-a[1])*v[1])/(v@v),0.,1.)
        blocked|=(XX-a[0]-fraction*v[0])**2+(YY-a[1]-fraction*v[1])**2<=(.60*GRID)**2
    row_y=YY[0]
    edge=np.full(len(row_y),np.inf)
    for tri in crop.skin:
        heights=tri[:,2]-crop.floor(tri[:,[1,0]])-.127
        line=slice_line(tri,heights)
        if line is None:
            continue
        a,b=line[np.argsort(line[:,1])]
        inside=(row_y>=a[1])&(row_y<=b[1])
        if b[1]>a[1]:
            pos=a[0]+(b[0]-a[0])*(row_y[inside]-a[1])/(b[1]-a[1])
            edge[inside]=np.minimum(edge[inside],pos)
    blocked|=XX>=edge[None,:]
    nearest=(XX-crop.goal[0])**2+(YY-crop.goal[1])**2
    nearest[blocked]=np.inf
    ix,iy=np.unravel_index(np.argmin(nearest),blocked.shape)
    source_xy=[float(XX[ix,iy]),float(YY[ix,iy])]
    offset=float(np.sqrt(nearest[ix,iy]))
    assert offset<.16
    ids,graph=lattice(blocked)
    degree=np.asarray(graph.sum(axis=1)).ravel()
    diffusion=.036/GRID**2
    decay=.16
    operator=diags(degree*diffusion+decay)-diffusion*graph
    strength=.7752049896163535
    rhs=np.zeros(len(degree))
    rhs[ids[ix,iy]]=strength/GRID**2
    values=spsolve(operator,rhs)
    assert np.min(values)>=0 and abs(decay*values.sum()*GRID**2-strength)<1e-8
    out=np.full_like(XX,np.nan)
    out[~blocked]=values/(np.max(values)/strength)
    return out,dict(source_xy_cm=source_xy,source_offset_mm=offset*10.,
                    obstacle_cells=int(blocked.sum()),odor_strength=strength)


def prism(arena,name,triangle,color,*,skin=False,soil=False):
    v=np.asarray(triangle,dtype=float)
    normal=np.cross(v[1]-v[0],v[2]-v[0])
    length=np.linalg.norm(normal)
    if length<1e-8:
        return False
    normal/=length
    offset=normal*THICKNESS
    if soil:
        if normal[2]<0.:
            v=v[[0,2,1]]
        points=np.vstack((v,v-(0.,0.,.25)))
    else:
        points=np.vstack((v if skin else v+.5*offset,v-offset if skin else v-.5*offset))
    mesh=arena.asset.add('mesh',name=name,vertex=points.ravel(),face=FACES)
    geom=solid(arena.worldbody,name,'mesh',mesh=mesh,rgba=color)
    if skin:
        geom.group=3
    return True


def cast(phy,point,vector,mask):
    geom=np.full(1,-1,dtype=np.int32)
    d=mujoco.mj_ray(phy.model.ptr,phy.data.ptr,np.asarray(point,dtype=float),
                    np.asarray(vector,dtype=float),mask,1,-1,geom)
    return float(d),int(geom[0])


class SkinFood(FoodBank):
    """Feed at a small patch on the imported skin when the mouth can reach it."""
    def __init__(self,crop):
        super().__init__([Resource('orchard_skin','skin',*crop.goal,SKIN_PATCH_RADIUS,.9)])
        self.crop=crop
        self.phy=None
        self.best_gap=None
        self.near_min=None
        self.consumable_checks=0
        self.skin_contact_checks=0

    def contact(self,mouth_cm,supported):
        delta=np.asarray(mouth_cm)-self.crop.goal
        point_distance=float(np.linalg.norm(delta))
        if self.near_min is None or point_distance<self.near_min:
            self.near_min=point_distance
        if not supported or point_distance>SKIN_PATCH_RADIUS or self.stocks['orchard_skin'].volume<=0:
            return None
        self.consumable_checks+=1
        # Test both sides for the thin shell when the oral center lies inside it.
        for direction in ((1.,0.,0.),(-1.,0.,0.)):
            gap,geom=cast(self.phy,mouth_cm,direction,FRUIT_MASK)
            if gap<0. or geom not in self.skin_ids:
                continue
            if self.best_gap is None or gap<self.best_gap:
                self.best_gap=gap
            if gap<=SKIN_REACH:
                self.skin_contact_checks+=1
                return 'orchard_skin'
        return None


def build(crop):
    env=build_env()
    env.random_state.seed(29)
    arena=env.task._arena.mjcf_model
    plane=arena.find('geom','groundplane')
    low=float(crop.terrain.min())
    plane.material=None
    plane.pos=(0,0,low-.35)
    plane.rgba=(.35,.32,.21,1.)
    world=arena.worldbody
    if crop.support=='mesh':
        for i,tri in enumerate(crop.soil):
            prism(arena,f'crop_soil_{i:03d}',tri,(.45,.38,.23,1.),soil=True)
    elif crop.support=='heightfield':
        span=float(np.ptp(crop.terrain))
        heightfile=HERE/crop.site/'HEIGHTFIELD.png'
        values=np.round((crop.terrain[::-1]-low)/span*65535).astype(np.uint16)
        Image.fromarray(values).save(heightfile)
        hf=arena.asset.add('hfield',name='extracted_soil',file=str(heightfile),
            size=(float(np.ptp(crop.x))/2.,float(np.ptp(crop.y))/2.,span,.20))
        solid(world,'crop_soil_field','hfield',hfield=hf,pos=(np.mean(crop.x),np.mean(crop.y),low),
              rgba=(.45,.38,.23,1.))
    else:
        a,b,z=crop.plane
        plane.name='crop_soil_plane'
        plane.pos=(0.,0.,z)
        rot=plane_frame(a,b)
        plane.quat=rot.as_quat()[[3,0,1,2]]
    world.add('light',name='crop_light',directional=True,pos=(0.,0.,3.),dir=(.28,.25,-1.),
              ambient=(.49,.48,.45),diffuse=(.58,.56,.53))
    for side in ('left','right'):
        world.add('camera',name='orchard_eye_'+side,fovy=56.)
    for i,tri in enumerate(crop.skin):
        prism(arena,f'crop_skin_{i:03d}',tri,(.56,.14,.045,1.),skin=True)
    for i,(tri,label) in enumerate(zip(crop.debris,crop.labels,strict=True)):
        category=crop.source['objects'][int(label)][1]
        prism(arena,f'crop_{category}_{i:03d}',tri,COLORS[category])
    return env, SkinFood(crop)


def bind_physics(env,bank):
    phy=env.physics
    bank.phy=phy
    surfaces=scene_ids(phy)
    bank.skin_ids={i for i,name in surfaces.items() if name.startswith('crop_skin_')}
    soil=[i for i,name in surfaces.items() if name.startswith('crop_soil_')]
    if bank.crop.support!='plane':
        floor=phy.model.name2id('groundplane','geom')
        for attr in ('geom_solref','geom_solimp','geom_friction','geom_solmix',
                     'geom_margin','geom_gap','geom_condim','geom_priority'):
            values=getattr(phy.model,attr)
            values[soil]=values[floor]
    # Reserve the two sensing/render groups for the imported surfaces only.
    groups=phy.model.geom_group
    for i in range(phy.model.ngeom):
        if i not in surfaces and groups[i] in (0,3):
            groups[i]=4
    return phy


def scene_ids(phy):
    return {i:phy.model.id2name(i,'geom') for i in range(phy.model.ngeom)
            if (phy.model.id2name(i,'geom') or '').startswith('crop_')}


def plane_frame(a,b):
    up=np.array([-a,-b,1.])
    up/=np.linalg.norm(up)
    forward=np.array([1.,0.,a])
    forward/=np.linalg.norm(forward)
    return Rotation.from_matrix(np.column_stack((forward,np.cross(up,forward),up)))


def fit_spawn(crop,env,base_start):
    q=base_start(env)
    with np.load(SETTLED,allow_pickle=False) as stored:
        resting=stored['poses'][-1]
    assert q.shape==resting.shape
    q[:]=resting
    q[:2]=0.
    dx,dy=np.meshgrid(np.linspace(-.27,.27,9),np.linspace(-.27,.27,9))
    design=np.column_stack((dx.ravel(),dy.ravel(),np.ones(dx.size)))
    relief=crop.floor(design[:,[1,0]])
    a,b,z=(crop.plane if crop.support=='plane' else np.linalg.lstsq(design,relief,rcond=None)[0])
    residual=float(max(0.,np.max(relief-design@[a,b,z]))) if crop.support!='plane' else 0.
    change=plane_frame(a,b)
    up=change.apply([0.,0.,1.])
    q[2]=z+q[2]/up[2]+residual+.001
    flat=Rotation.from_quat(q[[4,5,6,3]])
    prior_heading=(flat.apply([1.,0.,0.]))
    flat=Rotation.from_rotvec([0.,0.,-math.atan2(prior_heading[1],prior_heading[0])])*flat
    q[3:7]=(change*flat).as_quat()[[3,0,1,2]]
    phy=env.physics
    with phy.reset_context():
        phy.data.qpos[:]=q
        phy.data.qvel[:]=0.
        phy.data.ctrl[:]=0.
        phy.data.act[:]=0.
    return q


class Sensor:
    def __init__(self,field,active,visual):
        self.field=field
        self.active=active
        self.nose=FittedNose()
        self.visual=visual
        self.ant=None
        self.rows=[]
        self.yaw_rate=0.

    def read(self,eye,cal,an,pace,circulating,_last):
        phy=eye.phy
        if self.ant is None:
            self.ant=[phy.model.name2id('walker/antenna_'+side,'body') for side in ('left','right')]
            assert min(self.ant)>=0
            self.surfaces=scene_ids(phy)
            eye.options.geomgroup[3]=1
        antenna=phy.data.xpos[self.ant].copy()
        raw=sample(self.field,antenna[:,:2])
        readings=raw if self.active else (0.,0.)
        p=antenna.mean(axis=0)
        _, heading, _=eye.head_pose()
        yaw=math.atan2(heading[1],heading[0])

        def query(dx,dy):
            d,geom=cast(phy,p,(dx,dy,0.),SCENE_MASK)
            return d if geom in self.surfaces else -1.

        threat=ray_hazard(yaw,query,DETOUR_RANGE)
        self.yaw_rate=.67*self.yaw_rate+.33*float(phy.data.qvel[5])
        obs=self.nose.observe(readings,pace,float(np.clip(self.yaw_rate,-.9,.9)),threat)
        vision,left,_=self.visual(eye,cal,an,pace,circulating,None)
        self.rows.append([float(phy.time()),*antenna.ravel(),*raw,*readings,
                          *threat,float(vision[0]),yaw])
        return obs,left,None


def infer(policy,obs,profile):
    action=actor_mean(policy,obs)
    return action,intent(obs,action)


def run(args):
    crop=Crop(args.site,args.support)
    field,odor_info=odor(crop)
    sensor=Sensor(field,args.odor,docking.observation)
    env,bank=build(crop)
    base_start=docking.start_pose
    physical=[]

    class Capture(docking.Contacts):
        def __init__(self,phy):
            super().__init__(phy)
            self.scene=scene_ids(phy)
            self.detail={i:name for i,name in self.scene.items() if not name.startswith('crop_soil_')}
            self.last=None

        def scan(self,phy):
            feet,nonfoot,mouth=super().scan(phy)
            touched=tuple(sorted({self.detail[i] for c in phy.data.contact[:phy.data.ncon] if c.efc_address>=0
                                  for i in (c.geom1,c.geom2) if i in self.detail}))
            if self.last!=touched:
                physical.append((float(phy.time()),touched))
                self.last=touched
            return feet,nonfoot,mouth

    def start(_):
        bind_physics(env,bank)
        return fit_spawn(crop,env,base_start)

    docking.init_course=lambda *a,**k:(env,crop.cfg,bank)
    docking.start_pose=start
    docking.observation=sensor.read
    docking.infer=infer
    docking.Contacts=Capture
    docking.run(SimpleNamespace(out=args.out,vision=CLOSED/'VISION/VISION.npz',policy=POLICY,
        seed=99217,challenge=args.site,side=0,no_food=False,duration=args.duration,
        neural_seed=29,early=True,profile='strong'))
    np.savez_compressed(args.out/'SENSE.npz',samples=sensor.rows,field=field,
        columns=['t','left_x','left_y','left_z','right_x','right_y','right_z',
                 'field_l','field_r','used_l','used_r','risk_l','risk_r','rgb_diagnostic','head_yaw'])
    (args.out/'TRANSFER.json').write_text(json.dumps(dict(odor_enabled=args.odor,odor=odor_info,
        support=args.support,plane_corridor_error_mm=crop.plane_corridor_error_mm,
        contact_changes=physical,min_supported_skin_gap_mm=None if bank.best_gap is None else 10*bank.best_gap,
        min_oral_to_patch_mm=10*bank.near_min,mouth_gate_samples=bank.skin_contact_checks,
        mouth_near_samples=bank.consumable_checks,skin_reach_mm=10*SKIN_REACH),
        indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site',choices=('CLEAR','DEBRIS'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--odor',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--duration',type=float,default=6.4)
    parser.add_argument('--support',choices=('mesh','heightfield','plane'),default='mesh')
    run(parser.parse_args())
