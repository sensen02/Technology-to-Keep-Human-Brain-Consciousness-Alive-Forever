import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset="meadow_grass", add_tracking_camera=False,
                 add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
dt = cfg.timestep_s
tid = 1
def up_dot():
    # world-Z component of the thorax body's own up axis (its local +z), from the rotation matrix
    R = np.asarray(be.data.xmat[tid], dtype=float).reshape(3,3)
    return float(R[2,2]), float(R[2,0]), float(R[2,1])
print("  t     legs  z_thorax   up.z    up.x    up.y")
for k in range(int(1.2/dt)):
    be.step()
    if k % max(1,int(0.05/dt))==0:
        obs=be.observe(); L=int((np.asarray(obs.contact_found_raw)>0).sum())
        u=up_dot(); z=float(be.data.xpos[tid][2])
        flag = ""
        if u[0] < 0: flag = "  <== UPSIDE DOWN"
        elif L==0: flag = "  <== no tarsal contact"
        print(f"  {obs.time_s:.2f}  {L:>3}   {z:>7.4f}  {u[0]:>6.3f} {u[1]:>6.3f} {u[2]:>6.3f}{flag}")
