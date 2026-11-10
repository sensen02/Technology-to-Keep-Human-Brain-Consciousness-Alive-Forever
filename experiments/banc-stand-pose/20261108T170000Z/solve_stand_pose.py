"""Solve a flat, tall standing pose with constrained IK, then verify it can actually bear weight."""
from pathlib import Path
import json, sys, importlib.util
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
WEIGHT = 24.446354755878005
COM_TARGET = 1.30
W = dict(plane=200.0, height=20.0, xy=1.0, clear=20.0, reg=0.05)
spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
m = act.load_full_model(); d = mj.MjData(m)
mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
jids = m.actuator_trnid[:, 0]; qa = m.jnt_qposadr[jids]
lo = m.jnt_range[jids, 0] + .01; hi = m.jnt_range[jids, 1] - .01
q0 = d.qpos[qa].copy()
planes = m.geom_type == mj.mjtGeom.mjGEOM_PLANE
gname = [m.geom(i).name or '' for i in range(m.ngeom)]
# material point per leg: lowest mesh vertex of that leg's Tarsus5 in the key pose
foot_vert = {}
for l, leg in enumerate(LEGS):
    body = m.body(leg + 'Tarsus5').id
    gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
    g = gs[0]; mi = int(m.geom_dataid[g])
    v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
    R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
    vb = v @ R.T + m.geom_pos[g]
    vw = vb @ d.xmat[body].reshape(3, 3).T + d.xpos[body]
    idx = int(np.argmin(vw[:, 2]))
    foot_vert[leg] = (body, g, vb[idx])
xy0 = {}
for leg in LEGS:
    body, g, vb = foot_vert[leg]
    xy0[leg] = (d.xmat[body].reshape(3, 3) @ vb + d.xpos[body])[:2].copy()


def measure(q):
    d.qpos[:] = m.key_qpos[0]
    d.qpos[qa] = q
    mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
    pts = {}
    for leg in LEGS:
        body, g, vb = foot_vert[leg]
        pts[leg] = d.xmat[body].reshape(3, 3) @ vb + d.xpos[body]
    com = d.subtree_com[0].copy()
    lowest_other = 1e9
    for g in range(m.ngeom):
        if planes[g]:
            continue
        bn = m.body(int(m.geom_bodyid[g])).name or ''
        if any(bn.startswith(L) and 'Tars' in bn for L in LEGS):
            continue
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mi = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
            R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            vw = (v @ R.T + m.geom_pos[g]) @ d.xmat[int(m.geom_bodyid[g])].reshape(3, 3).T + d.xpos[int(m.geom_bodyid[g])]
            lowest_other = min(lowest_other, float(vw[:, 2].min()))
        else:
            lowest_other = min(lowest_other, float(d.geom_xpos[g][2] - np.linalg.norm(m.geom_size[g])))
    return pts, com, lowest_other


def residual(q):
    pts, com, low = measure(q)
    zs = np.array([pts[leg][2] for leg in LEGS])
    plane = zs - zs.mean()
    r = [W['plane'] * (plane - plane.mean()) ** 0]
    r = [W['plane'] * plane]
    r.append(W['height'] * np.array([COM_TARGET - com[2]]))
    for leg in LEGS:
        r.append(W['xy'] * (pts[leg][:2] - xy0[leg]))
    r.append(W['clear'] * np.array([min(0., low - zs.mean() - .01)]))
    r.append(W['reg'] * (q - q0))
    return np.concatenate(r)


res = least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi), method='trf', max_nfev=400, xtol=1e-10, ftol=1e-10)
q = res.x
pts, com, low = measure(q)
# the root is fixed during the IK, so raising the body pushes the feet below the floor;
# translate the whole body up until the lowest foot sits exactly on the floor plane (z = 0)
z_shift = -min(pts[leg][2] for leg in LEGS)
root_shift = float(z_shift)
pts, com, low = measure(q)
pts = {leg: pts[leg] + np.array([0., 0., z_shift]) for leg in LEGS}
com = com + np.array([0., 0., z_shift])
low = low + z_shift
zs = np.array([pts[leg][2] for leg in LEGS])
out = dict(status='SOLVED', cost=float(res.cost), nfev=int(res.nfev), com_z=float(com[2]), root_z_shift=root_shift,
           foot_z=dict(zip(LEGS, np.round(zs, 6).tolist())),
           coplanarity_max_abs_dev=float(np.abs(zs - zs.mean()).max()),
           nonfoot_lowest=float(low),
           joint_margin=dict(zip([m.joint(int(j)).name for j in jids],
                                 np.round(np.minimum(q - m.jnt_range[jids, 0], m.jnt_range[jids, 1] - q), 4).tolist())),
           solved_qpos_rad=np.round(q, 6).tolist(),
           root_z_after_shift=float(m.key_qpos[0][2] + root_shift),
           key_qpos_rad=np.round(q0, 6).tolist())
# physical verification: hold with the frozen actuator at rate 0
d.qpos[:] = m.key_qpos[0]; d.qpos[qa] = q; d.qpos[2] = m.key_qpos[0][2] + root_shift
mj.mj_forward(m, d)
actuator = act.IncrementalActuator(m, d); actuator.reset(d)
shares, nonfoot, cz = [], [], []
for k in range(800):
    for _ in range(40):
        actuator.update(np.zeros(42), .001)
        d.ctrl[:] = actuator.torque(d)
        mj.mj_step(m, d); mj.mj_forward(m, d)
    row = np.zeros(6); nf = 0.
    for ci in range(d.ncon):
        c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
        if not (planes[ga] or planes[gb]):
            continue
        geom = gb if planes[ga] else ga
        f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
        normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
        leg = next((j for j, L in enumerate(LEGS) if gname[geom].startswith(L) and 'Tars' in gname[geom]), -1)
        if leg < 0:
            nf += normal
        else:
            row[leg] += normal
    shares.append(row / WEIGHT); nonfoot.append(nf / WEIGHT); cz.append(float(d.subtree_com[0, 2]))
shares = np.array(shares); nonfoot = np.array(nonfoot); cz = np.array(cz)
tail = shares[len(shares) // 4:]
out['static_hold'] = dict(native_steps=800 * 40, total_share_mean=round(float(tail.sum(1).mean()), 4),
                          per_leg_share=dict(zip(LEGS, np.round(tail.mean(0), 4).tolist())),
                          nonfoot_share_max=round(float(nonfoot.max()), 4),
                          com_z_tail_mean=round(float(cz[len(cz) // 4:].mean()), 4), com_z_min=round(float(cz.min()), 4))
out['acceptance'] = dict(
    coplanarity=bool(out['coplanarity_max_abs_dev'] <= .002),
    height=bool(out['com_z'] >= 1.20),
    clearance=bool(out['nonfoot_lowest'] - zs.mean() >= .010),
    ranges=bool(min(out['joint_margin'].values()) >= .0),
    physical=bool(0.95 <= out['static_hold']['total_share_mean'] <= 1.05 and out['static_hold']['nonfoot_share_max'] <= .02))
out['acceptance']['all'] = all(out['acceptance'].values())
(HERE / 'STAND_POSE.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: out[k] for k in ('com_z', 'coplanarity_max_abs_dev', 'nonfoot_lowest')}))
print(json.dumps({'foot_z': out['foot_z'], 'acceptance': out['acceptance'], 'static': out['static_hold']}))
