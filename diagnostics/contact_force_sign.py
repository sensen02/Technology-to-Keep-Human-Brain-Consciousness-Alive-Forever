import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model; BW=float(np.sum(np.asarray(m.body_mass,dtype=float)))*9810.0
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i)) for i in range(m.ngeom)]
tars=[i for i,n in enumerate(names) if n and 'tarsus' in n]
print(f"m*g = {BW:.4f} model units.  gravity={np.asarray(m.opt.gravity).tolist()}")
dt=cfg.timestep_s
for k in range(int(0.3/dt)):
    be.step()
    if k % max(1,int(0.02/dt)): continue
    s_abs=s_signed=0.0; pos=neg=0; zsum=0.0
    for c in range(be.data.ncon):
        con=be.data.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,be.data,c,f)
        if not (con.geom1 in tars or con.geom2 in tars): continue
        # the normal acts along the contact frame's first axis; project onto WORLD z to get the
        # vertical component, which is what must balance the weight
        fr = np.asarray(con.frame, dtype=float).reshape(3,3)
        world_n = fr[:,0]*float(f[0])
        zsum += float(world_n[2])
        s_abs += abs(float(f[0])); s_signed += float(f[0])
        if f[0] > 0: pos += 1
        else: neg += 1
    obs=be.observe(); L=int((np.asarray(obs.contact_found_raw)>0).sum())
    print(f"  t={obs.time_s:.2f} legs={L} contacts={pos+neg} (+{pos}/-{neg})  "
          f"sum|Fn|={s_abs:9.2f}  sum_Zn={zsum:9.2f}  (m*g={BW:.2f})")
