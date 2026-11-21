from orchard import build_orchard,RESOURCES,ground_height,BED
from validate import ray,support
from legged_fixture import set_ground_pose
import numpy as np
env,_=build_orchard()
world=env.task._arena.mjcf_model.worldbody
r=RESOURCES[4]
probe=world.add('body',name='drop',pos=(r.x+.1,r.y,r.top+.12))
probe.add('freejoint')
probe.add('geom',name='probe',type='sphere',mass=.000001,size=(.03,),group=3,contype=2,conaffinity=1,solref=(.002,1.))
env.reset()
phy=env.physics
set_ground_pose(env)
m=phy.model
foot=next(i for i in range(m.ngeom) if 'walker/' in m.id2name(i,'geom') and 'claw' in m.id2name(i,'geom') and m.geom_contype[i])
for g in ('groundplane','orchard_terrain',r.geom,'table_top','probe',m.id2name(foot,'geom')):
 i=m.name2id(g,'geom')
 print(g,'params',m.geom_contype[i],m.geom_conaffinity[i],m.geom_pos[i],phy.data.geom_xpos[i],m.geom_solref[i],m.geom_margin[i],m.geom_size[i],m.geom_priority[i])
print('grav',m.opt.gravity,'contact',m.npair,'extent',m.stat.extent)
print('height',ground_height()[128,55],m.hfield_data.reshape(257,257)[128,55],m.hfield_size,ray(phy,0,0),ray(phy,.02,.02),ray(phy,r.x+.1,r.y))
changes=None
for i in range(4000):
 phy.step()
 h=support(phy,['probe','orchard_terrain','groundplane'])
 if h!=changes or i%1000==0:
  print(i,phy.named.data.geom_xpos['probe'],h)
  changes=h
env.close()
