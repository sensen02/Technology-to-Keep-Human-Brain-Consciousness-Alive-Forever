import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
GZ = float(sys.argv[1])
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model; m.opt.gravity[:] = [0.0, 0.0, GZ]
BW=float(np.sum(np.asarray(m.body_mass,dtype=float)))*9810.0
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i)) for i in range(m.ngeom)]
tars=[i for i,n in enumerate(names) if n and 'tarsus' in n]
dt=cfg.timestep_s; rows=[]
for k in range(int(0.3/dt)):
    be.step()
    if k % max(1,int(0.01/dt)): continue
    s=0.0
    for c in range(be.data.ncon):
        con=be.data.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,be.data,c,f)
        if con.geom1 in tars or con.geom2 in tars: s+=abs(float(f[0]))
    obs=be.observe(); L=int((np.asarray(obs.contact_found_raw)>0).sum())
    rows.append((obs.time_s,L,s/BW,float(be.data.xpos[1][2])))
r=np.array(rows)
print(f"gravity_z={GZ:8.1f} (m*g={BW:.3f}):  legs_down {r[:,1].mean():.2f}  "
      f"sum|Fn|/m*g: mean {r[:,2].mean():6.2f} median {np.median(r[:,2]):6.2f}  "
      f"thorax_z {r[:,3].mean():.3f}")
