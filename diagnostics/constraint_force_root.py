import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model; d=be.data; mass=float(np.sum(np.asarray(m.body_mass))); W=mass*9810
root=1
adr=int(m.body_dofadr[root])          # free joint: x,y,z then rotations
print(f"root body {root}: dofadr={adr}  m*g={W:.3f}")
dt=cfg.timestep_s
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i)) for i in range(m.ngeom)]
tars=[i for i,n in enumerate(names) if n and 'tarsus' in n]
print("  t      qfrc_constraint[0:3] (root free joint)   sum|Fn| tarsal   thorax_z")
for k in range(int(0.3/dt)):
    be.step()
    if k % max(1,int(0.02/dt)): continue
    qc = np.asarray(d.qfrc_constraint[adr:adr+3], dtype=float)
    s=0.0
    for c in range(d.ncon):
        con=d.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,d,c,f)
        if con.geom1 in tars or con.geom2 in tars: s+=abs(float(f[0]))
    print(f"  {d.time:.3f}   qc=[{qc[0]:7.3f} {qc[1]:7.3f} {qc[2]:7.3f}]  z/mg={qc[2]/W:6.2f}   "
          f"sum|Fn|={s:8.2f}  z={float(d.xpos[root][2]):.4f}")
print("\nNOTE: the free joint's constraint force IS the net external (contact) force on the fly,")
print("      in the joint's own coordinates -- no convention guessing involved.")
