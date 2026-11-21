from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T194948Z-orchard-odor-transfer'))
import transfer as tr
crop=tr.Crop('CLEAR','mesh')
env,bank=tr.build(crop)
env.reset()
phy=tr.bind_physics(env,bank)
tr.fit_spawn(crop,env,tr.docking.start_pose)
m=phy.model.ptr
for n in ('groundplane','crop_soil_000','crop_skin_000'):
 i=m.geom(n).id
 print(n,m.geom_contype[i],m.geom_conaffinity[i],m.geom_margin[i])
for c in phy.data.contact[:phy.data.ncon]:
 a,b=(m.geom(int(i)) for i in (c.geom1,c.geom2))
 if c.efc_address>=0:
  print(a.name,b.name,m.body(a.bodyid.item()).name,m.body(b.bodyid.item()).name,float(c.dist))
