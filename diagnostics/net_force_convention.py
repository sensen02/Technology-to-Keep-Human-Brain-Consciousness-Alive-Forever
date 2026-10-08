import sys; sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np, mujoco
from engine.embodied import BodyBackend, BodyConfig
cfg = BodyConfig(seed=0, scene_preset=None, add_tracking_camera=False, add_world_camera=False, add_vision=False)
be = BodyBackend(cfg, gl_backend=None).attach_cpg_baseline()
m=be.model; mass=float(np.sum(np.asarray(m.body_mass))); W=mass*9810
names=[str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,i)) for i in range(m.ngeom)]
bname=lambda i: str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]))
tars=[i for i,n in enumerate(names) if n and 'tarsus' in n]
dt=cfg.timestep_s
for k in range(int(0.15/dt)):
    be.step()
    if k % max(1,int(0.05/dt)) or k==0: continue
    print(f"\n--- t={be.data.time:.3f}s  m*g={W:.3f} ---")
    # fly COM acceleration straight from the solver: sum m_i * qacc_i over the fly's free body
    com_a = np.zeros(3); 
    for b in range(1, m.nbody):
        if m.body_jntnum[b] > 0:
            adr = int(m.body_dofadr[b])
            n = int(m.body_jntnum[b]) * 3 if False else 0
            # take as many translational dofs as the body actually has
            jt = int(m.jnt_type[int(m.body_jntadr[b])])
            nd = 6 if jt == int(mujoco.mjtJoint.mjJNT_FREE) else (3 if jt == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
            take = np.asarray(be.data.qacc[adr:adr+min(3, nd)], dtype=float)
            if take.size == 3:
                com_a += float(m.body_mass[b]) * take
    com_a /= mass
    print(f"  fly COM acceleration from qacc (body frame): {np.round(com_a,3).tolist()}")
    tot=0.0
    for c in range(be.data.ncon):
        con=be.data.contact[c]; f=np.zeros(6); mujoco.mj_contactForce(m,be.data,c,f)
        if not (con.geom1 in tars or con.geom2 in tars): continue
        fr=np.asarray(con.frame,dtype=float).reshape(3,3)
        nrm=fr[0]                     # MuJoCo: the FIRST ROW of the contact frame is the normal
        g1ok = con.geom1 in tars
        # the force reported acts on geom2 from geom1 (or the reverse); take the one on the fly
        sign = 1.0 if g1ok else -1.0
        tot += sign*float(f[0])*nrm[2]
        print(f"   {bname(con.geom1):<12}{names[con.geom1]:<24} vs {bname(con.geom2):<12}"
              f"{names[con.geom2]:<22} |Fn|={abs(float(f[0])):7.2f}  normal_z={nrm[2]:+.3f}")
    print(f"  signed vertical sum acting on the fly = {tot:8.2f}  (weight {W:.2f})  ratio {tot/W:+.2f}")
