"""Retarget the 232 long snippets starting from the solved stand pose instead of the built-in key pose."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DATA = ROOT / 'experiments/banc-locomotion-dataset/20261108T011500Z'
POSE = json.loads((ROOT / 'experiments/banc-stand-pose/20261108T170000Z/STAND_POSE_2.json').read_text())
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
SOLVER = dict(max_nfev_first=80, max_nfev_subsequent=16, ftol=1e-7, xtol=1e-7, gtol=1e-7,
              w_land=1.0, w_dir=.05, w_cont=.02, w_key0=.002, step_limit=.06)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rot(q):
    a = np.zeros(9); mj.mju_quat2Mat(a, np.asarray(q)); return a.reshape(3, 3)


def frame(a):
    x = (a[0] + a[3] - a[1] - a[4]) / 2; x = x / np.linalg.norm(x)
    y = a[:3].mean(0) - a[3:].mean(0); y -= x * np.dot(x, y); y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def main():
    t0 = time.time()
    data = np.load(DATA / 'LONG_SNIPPETS.npz')
    lm = data['landmarks']; offsets = data['offsets']; lengths = data['lengths']
    sys.path.insert(0, str(ROOT / 'experiments/s5-ann-incremental/20261107T010000Z'))
    from incremental_actuator import load_full_model
    m = load_full_model(); d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
    th = m.body('Thorax').id
    jids = m.actuator_trnid[:, 0]; qa = m.jnt_qposadr[jids]; da = m.jnt_dofadr[jids]
    bounds = m.jnt_range[jids].copy()
    baseq = d.qpos.copy()
    stand = np.asarray(POSE['q_rad'], float); root_z = float(POSE['root_z'])
    q0 = stand.copy()
    baseq[qa] = stand; baseq[2] = root_z
    d.qpos[:] = baseq; mj.mj_forward(m, d)
    ids = []; local = np.zeros((6, 9, 3))
    for l, leg in enumerate(LEGS):
        names = ['Coxa', 'Femur', 'Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5', 'Tarsus5']
        ids.append([m.body(leg + n).id for n in names])
        b = ids[-1][-1]
        g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH][0]
        mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
        vb = v @ rot(m.geom_quat[g]).T + m.geom_pos[g]
        vw = vb @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
        vi = int(np.argmin(vw[:, 2]))
        local[l, -1] = vb[vi]
    ids = np.array(ids)
    def landmarks():
        rr = d.xmat[th].reshape(3, 3)
        xyz = d.xpos[ids] + np.einsum('...ij,...j->...i', d.xmat[ids].reshape(6, 9, 3, 3), local)
        return (xyz - d.xpos[th]) @ rr
    base = landmarks()
    length = np.linalg.norm(np.diff(base, axis=1), axis=-1); L = length.sum(1)
    R = frame(base[:, 0]) @ np.asarray(data['anatomical_frame'], dtype=float).T
    n_frames = int(lengths.sum())
    Q = np.zeros((n_frames, 42)); err = np.zeros((n_frames, 6)); inflag = np.zeros((n_frames, 6), bool)
    nfev_all = np.zeros((n_frames, 6), int)
    for s in range(len(lengths)):
        a, b = int(offsets[s]), int(offsets[s + 1])
        out = np.zeros((b - a, 42))
        for t in range(b - a):
            raw = lm[a + t]
            u = np.diff(raw, axis=1); u /= np.linalg.norm(u, axis=-1, keepdims=True); u = u @ R.T
            target = np.concatenate([np.broadcast_to(base[:, 0, None], (6, 1, 3)),
                                     np.broadcast_to(base[:, 0, None], (6, 1, 3)) + np.cumsum(u * length[..., None], axis=1)], axis=1)
            for l in range(6):
                ix = np.arange(l * 7, l * 7 + 7)
                prev = q0[ix] if t == 0 else out[t - 1, ix]
                lo = bounds[ix, 0].copy(); hi = bounds[ix, 1].copy()
                if t:
                    lo = np.maximum(lo, prev - SOLVER['step_limit']); hi = np.minimum(hi, prev + SOLVER['step_limit'])
                cache = {}
                def evaluate(q, l=l, ix=ix, prev=prev, target=target):
                    if 'q' in cache and np.array_equal(cache['q'], q):
                        return cache['r'], cache['j']
                    d.qpos[:] = baseq; d.qpos[qa[ix]] = q
                    mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
                    rr = d.xmat[th].reshape(3, 3)
                    world = d.xpos[ids[l]] + np.einsum('nij,nj->ni', d.xmat[ids[l]].reshape(9, 3, 3), local[l])
                    pos = (world - d.xpos[th]) @ rr
                    jac = []
                    for bb, p in zip(ids[l], world):
                        jp = np.zeros((3, m.nv)); mj.mj_jac(m, d, jp, None, p, int(bb)); jac.append(rr.T @ jp[:, da[ix]])
                    jac = np.array(jac)
                    delta = np.diff(pos, axis=0); lens = np.linalg.norm(delta, axis=1)
                    dirs = delta / lens[:, None]
                    du = np.einsum('nij,njk->nik', (np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]) / lens[:, None, None],
                                   np.diff(jac, axis=0))
                    r = np.concatenate([((pos - target[l]) / L[l]).ravel(),
                                        SOLVER['w_dir'] * (dirs - u[l]).ravel(),
                                        SOLVER['w_cont'] * (q - prev), SOLVER['w_key0'] * (q - q0[ix])])
                    J = np.concatenate([(jac / L[l]).reshape(-1, 7), SOLVER['w_dir'] * du.reshape(-1, 7),
                                        SOLVER['w_cont'] * np.eye(7), SOLVER['w_key0'] * np.eye(7)], axis=0)
                    cache.update(q=q.copy(), r=r, j=J, pos=pos)
                    return r, J
                res = least_squares(lambda q: evaluate(q)[0], np.clip(prev, lo + 1e-12, hi - 1e-12),
                                    jac=lambda q: evaluate(q)[1], bounds=(lo, hi), method='trf',
                                    max_nfev=SOLVER['max_nfev_first'] if t == 0 else SOLVER['max_nfev_subsequent'],
                                    ftol=SOLVER['ftol'], xtol=SOLVER['xtol'], gtol=SOLVER['gtol'])
                out[t, ix] = res.x
                evaluate(res.x)
                pos = cache['pos']
                err[a + t, l] = np.linalg.norm(pos - target[l], axis=-1).max() / L[l]
                inflag[a + t, l] = bool(np.all(res.x >= bounds[ix, 0] - 1e-9) and np.all(res.x <= bounds[ix, 1] + 1e-9))
                nfev_all[a + t, l] = res.nfev
            Q[a + t] = out[t]
        if s % 20 == 0:
            print('snippet', s, '/', len(lengths), 'elapsed %.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(HERE / 'RETARGET_STAND.npz', targets_rad=Q, normalized_error=err, in_limit=inflag, nfev=nfev_all,
                        offsets=offsets, lengths=lengths, joint_limits_rad=bounds, leg_lengths_native=L,
                        leg_names=np.array(LEGS), ordered_joint_names=np.array([m.joint(int(j)).name for j in jids]),
                        snippet_indices=data['snippet_indices'], stand_pose_rad=stand, stand_root_z=root_z)
    out = dict(status='RETARGET_STAND_COMPLETE', snippets=int(len(lengths)), frames=int(n_frames), ik_solves=int(n_frames * 6),
               wall_s=time.time() - t0, solver=SOLVER, physics_steps=0,
               start_pose=dict(source=str(ROOT / 'experiments/banc-stand-pose/20261108T170000Z/STAND_POSE_2.json'),
                               root_z=root_z, com_z_at_pose=POSE['com_z']),
               error=dict(max=float(err.max()), p50=float(np.percentile(err, 50)), p95=float(np.percentile(err, 95)),
                          rms=float(np.sqrt(np.mean(err ** 2)))),
               in_limit_fraction=float(inflag.mean()),
               frozen_reference=dict(p50=0.05324859662417217, max=0.38525889686846876, rms=0.0761))
    (HERE / 'RETARGET_STAND.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out['error']))


if __name__ == '__main__':
    main()
