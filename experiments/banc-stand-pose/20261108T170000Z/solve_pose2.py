"""Standing pose with the root height as a free variable, so feet stay on the floor while COM is placed."""
from pathlib import Path
import json, sys, importlib.util
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
WEIGHT = 24.446354755878005
COM_TARGET = 1.35
spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
m = act.load_full_model(); d = mj.MjData(m)
mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
jids = m.actuator_trnid[:, 0]; qa = m.jnt_qposadr[jids]
lo = m.jnt_range[jids, 0] + .01; hi = m.jnt_range[jids, 1] - .01
key = m.key_qpos[0].copy(); q0 = key[qa]
planes = m.geom_type == mj.mjtGeom.mjGEOM_PLANE
gname = [m.geom(i).name or '' for i in range(m.ngeom)]
foot = {}
for leg in LEGS:
    body = m.body(leg + 'Tarsus5').id
    g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH][0]
    mi = int(m.geom_dataid[g])
    v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
    R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
    vb = v @ R.T + m.geom_pos[g]
    vw = vb @ d.xmat[body].reshape(3, 3).T + d.xpos[body]
    foot[leg] = (body, vb[int(np.argmin(vw[:, 2]))])
xy0 = {leg: (d.xmat[foot[leg][0]].reshape(3, 3) @ foot[leg][1] + d.xpos[foot[leg][0]])[:2] for leg in LEGS}


def measure(x):
    q, root_z = x[:42], x[42]
    d.qpos[:] = key; d.qpos[qa] = q; d.qpos[2] = root_z
    mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
    pts = {leg: d.xmat[foot[leg][0]].reshape(3, 3) @ foot[leg][1] + d.xpos[foot[leg][0]] for leg in LEGS}
    low = 1e9
    for g in range(m.ngeom):
        if planes[g]:
            continue
        bn = m.body(int(m.geom_bodyid[g])).name or ''
        if any(bn.startswith(L) and 'Tars' in bn for L in LEGS):
            continue
        b = int(m.geom_bodyid[g])
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mi = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
            R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            vw = (v @ R.T + m.geom_pos[g]) @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
            low = min(low, float(vw[:, 2].min()))
        else:
            low = min(low, float(d.geom_xpos[g][2] - np.linalg.norm(m.geom_size[g])))
    return q, root_z, pts, d.subtree_com[0].copy(), low


def residual(x):
    q, root_z, pts, com, low = measure(x)
    r = [30. * np.array([pts[leg][2] for leg in LEGS])]
    r.append(np.array([30. * (com[2] - COM_TARGET)]))
    for leg in LEGS:
        r.append(2.0 * (pts[leg][:2] - xy0[leg]))
    r.append(np.array([50. * min(0., low - .01)]))
    r.append(0.5 * (q - q0))
    r.append(np.array([0.05 * (root_z - key[2])]))
    return np.concatenate(r)


x0 = np.concatenate([q0, [key[2]]])
lb = np.concatenate([lo, [key[2] - .8]]); ub = np.concatenate([hi, [key[2] + .8]])
res = least_squares(residual, np.clip(x0, lb, ub), bounds=(lb, ub), method='trf', max_nfev=600, xtol=1e-12, ftol=1e-12)
q, root_z, pts, com, low = measure(res.x)
zs = np.array([pts[leg][2] for leg in LEGS])
out = dict(status='SOLVED_2', cost=float(res.cost), nfev=int(res.nfev), com_z=float(com[2]), root_z=float(root_z),
           foot_z=dict(zip(LEGS, np.round(zs, 6).tolist())), coplanarity_max_abs_dev=float(np.abs(zs).max()),
           nonfoot_lowest=float(low), q_rad=np.round(q, 6).tolist(), key_rad=np.round(q0, 6).tolist(),
           joint_margin_min=float(np.min(np.minimum(q - m.jnt_range[jids, 0], m.jnt_range[jids, 1] - q))))
# physical hold from this pose
d.qpos[:] = key; d.qpos[qa] = q; d.qpos[2] = root_z; mj.mj_forward(m, d)
actuator = act.IncrementalActuator(m, d); actuator.reset(d)
shares, nf, cz = [], [], []
for k in range(800):
    for _ in range(40):
        actuator.update(np.zeros(42), .001)
        d.ctrl[:] = actuator.torque(d)
        mj.mj_step(m, d); mj.mj_forward(m, d)
    row = np.zeros(6); n = 0.
    for ci in range(d.ncon):
        c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
        if not (planes[ga] or planes[gb]):
            continue
        geom = gb if planes[ga] else ga
        f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
        normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
        leg = next((j for j, L in enumerate(LEGS) if gname[geom].startswith(L) and 'Tars' in gname[geom]), -1)
        if leg < 0:
            n += normal
        else:
            row[leg] += normal
    shares.append(row / WEIGHT); nf.append(n / WEIGHT); cz.append(float(d.subtree_com[0, 2]))
shares = np.array(shares); nf = np.array(nf); cz = np.array(cz)
tail = shares[len(shares) // 4:]
out['static_hold'] = dict(native_steps=800 * 40, total_share_mean=round(float(tail.sum(1).mean()), 4),
                          per_leg_share=dict(zip(LEGS, np.round(tail.mean(0), 4).tolist())),
                          nonfoot_share_max=round(float(nf.max()), 4),
                          com_z_tail_mean=round(float(cz[len(cz) // 4:].mean()), 4), com_z_min=round(float(cz.min()), 4))
out['acceptance'] = dict(coplanarity=bool(out['coplanarity_max_abs_dev'] <= 1e-3), height=bool(out['com_z'] >= 1.20),
                         clearance=bool(out['nonfoot_lowest'] >= .01), ranges=bool(out['joint_margin_min'] >= 0.),
                         physical=bool(0.95 <= out['static_hold']['total_share_mean'] <= 1.05 and out['static_hold']['nonfoot_share_max'] <= .02))
out['acceptance']['all'] = all(out['acceptance'].values())
(HERE / 'STAND_POSE_2.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps({'com_z': out['com_z'], 'root_z': out['root_z'], 'max_foot_z': out['coplanarity_max_abs_dev'],
                  'nonfoot_lowest': out['nonfoot_lowest'], 'acceptance': out['acceptance'], 'static': out['static_hold']}))
