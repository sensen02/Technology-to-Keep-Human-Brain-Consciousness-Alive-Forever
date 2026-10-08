import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model; d=be.data; mass=float(np.sum(np.asarray(m.body_mass))); W=mass*9810
adr=int(m.body_dofadr[1])
dt=cfg.timestep_s
print(f"m*g = {W:.3f} model units.  Terms on the fly's free joint, z component:")
print(f"  {'t':>6}{'constraint':>12}{'actuator':>11}{'passive':>10}{'bias':>10}{'applied':>10}"
      f"{'sum':>9}{'m*a_z':>10}")
for k in range(int(0.30/dt)):
    be.step()
    if k % max(1,int(0.03/dt)): continue
    qc=float(d.qfrc_constraint[adr+2]); qa=float(d.qfrc_actuator[adr+2])
    qp=float(d.qfrc_passive[adr+2]); qb=float(d.qfrc_bias[adr+2]); qx=float(d.qfrc_applied[adr+2])
    q=float(d.qvel[adr+2]); 
    ma = float(np.sum([m.body_mass[b]*float(d.qacc[int(m.body_dofadr[b])+2])
                       for b in range(1,m.nbody) if m.body_jntnum[b]>0 and int(m.jnt_type[int(m.body_jntadr[b])])==int(mujoco.mjtJoint.mjJNT_FREE)]))
    print(f"  {d.time:6.3f}{qc:12.2f}{qa:11.2f}{qp:10.2f}{qb:10.2f}{qx:10.2f}"
          f"{qc+qa+qp-qb+qx:9.2f}{ma:10.3f}")
print("\nMuJoCo's equation for the free joint:  M*qacc = qfrc_applied + qfrc_actuator + qfrc_passive")
print("                                        + qfrc_constraint - qfrc_bias")
print("=> if constraint is ~+150 and the fly does not accelerate, the ACTUATOR term must be ~-150.")
