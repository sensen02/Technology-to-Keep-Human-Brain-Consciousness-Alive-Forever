from pathlib import Path
import sys
root=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain')
sys.path.insert(0,str(root/'20261117T184709Z-body-terrain-repair'))
from body_model import Body, mujoco as mj,np
b=Body()
b.restore(1151)
print(mj.__version__,b.weight,b.m.opt.gravity,b.m.nbody,b.m.ngeom,b.m.npair,b.d.qpos[:7])
print(b.sim.__dict__.keys())
print(b.sim.world.__dict__.keys() if hasattr(b.sim,'world') else '')
print([(b.m.geom(int(g)).name,int(b.m.geom_contype[g]),int(b.m.geom_conaffinity[g]),b.m.geom_friction[g]) for g in b.ground_geoms])
print([(b.m.geom(i).name,b.m.geom_contype[i],b.m.geom_conaffinity[i]) for i in list(b.geom_leg)[:5]])
print('pairs',b.m.pair_geom1[:8],b.m.pair_geom2[:8],b.m.pair_friction[:1])
print('legs',b.legs,'feet',b.feet,b.m.geom(b.m.body_geomadr[b.feet[0]]).name)
s=b.sim.world.mjcf_root
print(type(s),len(s.keys),s.option.timestep,s.option.integrator,s.option.jacobian,b.m.pair_dim[0],b.m.pair_solref[0],b.m.pair_solimp[0])
print('feet geoms',[(b.m.geom(i).name,b.m.geom_size[i],b.m.geom_type[i]) for i in list(b.geom_leg)[-8:]])
print('legs full',[(i,b.m.actuator(i).name) for i in b.adhesion_ids])
