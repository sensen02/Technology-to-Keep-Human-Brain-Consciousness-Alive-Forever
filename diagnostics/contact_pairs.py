import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco, collections
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model
gname=lambda i: str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i))
bname=lambda i: str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]))
dt=cfg.timestep_s
for k in range(int(0.12/dt)):
    be.step()
kinds=collections.Counter(); detailed=[]
for c in range(be.data.ncon):
    con=be.data.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,be.data,c,f)
    g1,g2=con.geom1,con.geom2; b1,b2=bname(g1),bname(g2)
    fly1 = b1.startswith('nmf'); fly2 = b2.startswith('nmf')
    kind = ('fly-fly (SELF)' if (fly1 and fly2) else
            ('fly-world' if (fly1 or fly2) else 'world-world'))
    kinds[kind]+=1
    detailed.append((abs(float(f[0])), b1, gname(g1), b2, gname(g2), kind))
print("contact kinds at t=0.12s:", dict(kinds))
detailed.sort(reverse=True)
print(f"\n{'|Fn|':>9}  geom1 (body)                geom2 (body)                kind")
for f,g1,b1,g2,b2,kind in detailed[:12]:
    print(f"{f:9.1f}  {g1:<26}{b1:<26} {g2:<26}{b2:<22} {kind}")
tot=sum(d[0] for d in detailed)
self_tot=sum(d[0] for d in detailed if d[5].startswith('fly-fly'))
print(f"\ntotal |Fn| {tot:.1f}; of which SELF (fly-fly) {self_tot:.1f} = {100*self_tot/tot:.0f}%")
