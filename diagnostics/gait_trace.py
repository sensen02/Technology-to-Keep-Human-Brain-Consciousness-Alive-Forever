import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset="meadow_grass", add_tracking_camera=False,
                 add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
dt = cfg.timestep_s
print("  t(s)   legs  thorax_z  x_mm   y_mm")
rows=[]
for k in range(int(1.0/dt)):
    be.step()
    if k % max(1,int(0.025/dt))==0:
        obs=be.observe(); L=int((np.asarray(obs.contact_found_raw)>0).sum())
        x=be.data.xpos[1]; rows.append((obs.time_s,L,float(x[2]),float(x[0]),float(x[1])))
for i,(t,L,z,x,y) in enumerate(rows):
    if i%2==0: print(f"  {t:.3f}  {L:>3}   {z:>7.4f}  {x:>6.2f} {y:>6.2f}")
r=np.array(rows)
print(f"\nlegs down: first 0.3s mean {r[r[:,0]<0.3,1].mean():.2f}; last 0.3s mean {r[r[:,0]>0.7,1].mean():.2f}")
print(f"zeroleg fraction: first 0.3s {(r[r[:,0]<0.3,1]==0).mean():.2f}; last 0.3s {(r[r[:,0]>0.7,1]==0).mean():.2f}")
print(f"thorax z: first 0.3s mean {r[r[:,0]<0.3,2].mean():.4f}; last 0.3s mean {r[r[:,0]>0.7,2].mean():.4f}")
print(f"total travel: x {r[:,3].ptp():.2f} mm, y {r[:,4].ptp():.2f} mm over 1 s")
