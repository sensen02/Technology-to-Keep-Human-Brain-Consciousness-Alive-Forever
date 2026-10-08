import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False,
                 add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None)
be = be.attach_cpg_baseline()
# ZERO THE ACTUATORS so this is a pure settling test in THIS scene: with no drive, a fly that
# is standing must report a summed tarsal normal force equal to its own weight, exactly.
be.data.ctrl[:] = 0.0
be.model.actuator_gainprm[:, 0] = 0.0
m = be.model
BW = float(np.sum(np.asarray(m.body_mass, dtype=float))) * 9810.0
print(f"body weight in model force units (m*g, g x mm/s^2) = {BW:.4f}")
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)) for i in range(m.ngeom)]
tars=[i for i,n in enumerate(names) if n and ('tarsus' in n or 'pulvillus' in n)]
print("tarsus-like geoms:", len(tars))
def summed_nf():
    tot = 0.0; ncon=0
    for c in range(be.data.ncon):
        con = be.data.contact[c]
        f = np.zeros(6); mujoco.mj_contactForce(m, be.data, c, f)
        if con.geom1 in tars or con.geom2 in tars:
            tot += abs(float(f[0])); ncon += 1
    return tot, ncon
for k in range(3000):
    be.step()
    if k % 500 == 0:
        tot, ncon = summed_nf()
        print(f"  step {k:5d} t={be.data.time:.3f}s  tarsal contacts {ncon:2d}  "
              f"sum|Fn| {tot:9.3f} units = {tot/BW:6.2f} x body weight")
tot, ncon = summed_nf()
print(f"final: sum|Fn| {tot:.4f} units = {tot/BW:.4f} x body weight ({ncon} tarsal contacts)")
print(f"thorax z = {be.data.xpos[1][2]:.4f}")
