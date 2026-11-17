"""A static rectangular floor recess for the full frozen plant."""
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / '20261116T000536Z'
sys.path.insert(0,str(BASE))
from walking_bench import stand

LEG = 'RF'
TERRAIN='gap_terrain_'
LIMIT=10.


def box_layout(case):
    xmin,xmax,ymin,ymax = map(float,case['opening'])
    depth=float(case['depth'])
    if not (-LIMIT<xmin<xmax<LIMIT and -LIMIT<ymin<ymax<LIMIT and .02<=depth<.4):
        raise ValueError('Invalid floor rectangle/recess depth')
    if case.get('axis_first','y')=='x':
        bands=(('west',-LIMIT,xmin,-LIMIT,LIMIT,0.),
               ('east',xmax,LIMIT,-LIMIT,LIMIT,0.),
               ('south',xmin,xmax,-LIMIT,ymin,0.),
               ('north',xmin,xmax,ymax,LIMIT,0.))
    else:
        bands=(('south',-LIMIT,LIMIT,-LIMIT,ymin,0.),
               ('north',-LIMIT,LIMIT,ymax,LIMIT,0.),
               ('west',-LIMIT,xmin,ymin,ymax,0.),
               ('east',xmax,LIMIT,ymin,ymax,0.))
    parts=(*bands,('plug' if case.get('filled') else 'bottom',
                  xmin,xmax,ymin,ymax,0. if case.get('filled') else -depth))
    for name,x0,x1,y0,y1,top in parts:
        yield TERRAIN+name,((x0+x1)/2.,(y0+y1)/2.,top-.4),((x1-x0)/2.,(y1-y0)/2.,.4)


def make_gap_model(case):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    original=stand.physics.act_mod.load_full_model(stand.physics.PAD_XML)
    if case is None:
        return original
    floor=original.geom('floor')
    spec=_load_mjcf(stand.physics.PAD_XML)
    disabled=spec.geom('floor')
    disabled.contype=0
    disabled.conaffinity=0
    disabled.rgba=[0.,0.,0.,0.]
    disabled.material=''
    for name,pos,size in box_layout(case):
        color=[.30,.24,.18,1.] if name.endswith('bottom') else [.20,.24,.26,1.]
        spec.worldbody.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,size=size,pos=pos,
                 rgba=color,contype=int(floor.contype[0]),conaffinity=int(floor.conaffinity[0]),
                 condim=int(floor.condim[0]), friction=floor.friction,solref=floor.solref,
                 solimp=floor.solimp,margin=float(floor.margin[0]),gap=float(floor.gap[0]),
                 solmix=float(floor.solmix[0]),priority=int(floor.priority[0]))
    model=spec.compile()
    stand.physics.act_mod.validate_full_model(model)
    model.mat_rgba[:original.nmat]=original.mat_rgba
    for i in range(original.ngeom):
        old=original.geom(i)
        new=model.geom(old.name)
        if old.name=='floor':
            assert int(new.contype[0])==int(new.conaffinity[0])==0
            continue
        model.geom_rgba[new.id]=original.geom_rgba[old.id]
        for attr in ('type','contype','conaffinity','condim','size','pos','quat',
                     'friction','solref','solimp','margin','gap','priority'):
            np.testing.assert_array_equal(getattr(new,attr),getattr(old,attr),err_msg=f'{old.name} {attr}')
    for attr in ('jnt_type','jnt_range','jnt_qposadr','jnt_dofadr','body_mass','body_inertia',
                 'body_ipos','body_iquat','actuator_trnid','actuator_gear','dof_damping','dof_armature','jnt_stiffness'):
        np.testing.assert_array_equal(getattr(model,attr),getattr(original,attr),err_msg=attr)
    assert all(model.geom(name).bodyid[0]==0 for name,_,_ in box_layout(case))
    return model


def terrain_masks(model):
    roles=np.zeros(model.ngeom,dtype=np.int32)
    for i in range(model.ngeom):
        name=model.geom(i).name
        if name=='floor' and model.geom_contype[i]:
            roles[i]=1
        elif name.startswith(TERRAIN):
            roles[i]=2 if name.endswith('bottom') else 1
    return roles
