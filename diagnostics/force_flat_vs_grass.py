import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
PRESET = None if sys.argv[1]=="flat" else "meadow_grass"
cfg = BodyConfig(seed=0, scene_preset=PRESET, add_tracking_camera=False,
                 add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model
BW=float(np.sum(np.asarray(m.body_mass,dtype=float)))*9810.0
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i)) for i in range(m.ngeom)]
tars=[i for i,n in enumerate(names) if n and 'tarsus' in n]
dt=cfg.timestep_s; rows=[]
for k in range(int(0.5/dt)):
    be.step()
    if k % max(1,int(0.01/dt)): continue
    tot=0.0; nc=0
    for c in range(be.data.ncon):
        con=be.data.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,be.data,c,f)
        if con.geom1 in tars or con.geom2 in tars: tot+=abs(float(f[0])); nc+=1
    obs=be.observe(); L=int((np.asarray(obs.contact_found_raw)>0).sum())
    rows.append((obs.time_s, L, nc, tot/BW, float(be.data.xpos[1][2])))
r=np.array(rows)
print(f"scene={'flat' if PRESET is None else PRESET}: body weight = {BW:.4f} model units")
print(f"  t<0.45s: legs_down {r[r[:,0]<0.45,1].mean():.2f}  F/BW mean {r[r[:,0]<0.45,3].mean():6.2f} "
      f"median {np.median(r[r[:,0]<0.45,3]):6.2f}  thorax_z {r[r[:,0]<0.45,4].mean():.3f}")
print(f"  whole 0.5s: F/BW mean {r[:,3].mean():.2f}  first sample {r[0,3]:.2f}  last {r[-1,3]:.2f}")
