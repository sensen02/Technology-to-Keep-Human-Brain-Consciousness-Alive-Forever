"""Retarget B2: ankle-anchored with ONE reachable folded-tip position target (no physics).

Same solver as the frozen round; only the objective changes: three reachable positions plus one
folded distal-direction term instead of nine positions, six of which this body cannot realise.
The frozen 9-landmark metric is still computed on every solution so the comparison is apples to
apples.
"""
from pathlib import Path
import ast, hashlib, json, sys, time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/flybody-retarget/20261107T182940Z/kinematics'
DATA = ROOT / 'experiments/banc-locomotion-dataset/20261108T011500Z'
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
SOLVER = dict(max_nfev_first=80, max_nfev_subsequent=16, ftol=1e-7, xtol=1e-7, gtol=1e-7,
              w_pos=1.0, w_dir=.2, w_cont=.02, w_key0=.002, step_limit=.06)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rot(q):
    a = np.zeros(9); mj.mju_quat2Mat(a, np.asarray(q)); return a.reshape(3, 3)


def frame(a):
    x = (a[0] + a[3] - a[1] - a[4]) / 2; x = x / np.linalg.norm(x)
    y = a[:3].mean(0) - a[3:].mean(0); y -= x * np.dot(x, y); y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def landmarks(m, d, ids, local, th):
    rr = d.xmat[th].reshape(3, 3)
    xyz = d.xpos[ids] + np.einsum('...ij,...j->...i', d.xmat[ids].reshape(6, 9, 3, 3), local)
    return (xyz - d.xpos[th]) @ rr


def main():
    t0 = time.time()
    data = np.load(DATA / 'LONG_SNIPPETS.npz')
    lm = data['landmarks']; offsets = data['offsets']; lengths = data['lengths']
    sys.path.insert(0, str(ROOT / 'experiments/s5-ann-incremental/20261107T010000Z'))
    from incremental_actuator import load_full_model
    m = load_full_model(); d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
    baseq = d.qpos.copy(); th = m.body('Thorax').id
    jids = m.actuator_trnid[:, 0]; qa = m.jnt_qposadr[jids]; da = m.jnt_dofadr[jids]
    bounds = m.jnt_range[jids].copy(); q0 = baseq[qa].copy()
    ids = []; local = np.zeros((6, 9, 3))
    for l, leg in enumerate(LEGS):
        ids.append([m.body(leg + n).id for n in ['Coxa', 'Trochanter' if leg[1] == 'F' else 'Femur', 'Tibia',
                                                 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5', 'Tarsus5']])
        b = ids[-1][-1]
        gg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
        assert len(gg) == 1
        g = gg[0]; mi = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
        vb = v @ rot(m.geom_quat[g]).T + m.geom_pos[g]
        vw = vb @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
        vi = int(np.argmin(((vw - d.xpos[th]) @ d.xmat[th].reshape(3, 3))[:, 2]))
        local[l, -1] = vb[vi]
    ids = np.array(ids)
    base = landmarks(m, d, ids, local, th)
    length = np.linalg.norm(np.diff(base, axis=1), axis=-1); L = length.sum(1)
    R = frame(base[:, 0]) @ np.asarray(data['anatomical_frame'], dtype=float).T
    # our own reachable ankle -> material-point distance, from the keyframe pose
    base_ach = landmarks(m, d, ids, local, th)
    reach = np.linalg.norm(base_ach[:, 8] - base_ach[:, 2], axis=1)
    n_frames = int(lengths.sum())
    Q = np.zeros((n_frames, 42))
    err_old = np.zeros((n_frames, 6)); err_dir = np.zeros((n_frames, 6))
    inflag = np.zeros((n_frames, 6), bool); nfev_all = np.zeros((n_frames, 6), int)
    for s in range(len(lengths)):
        a, b = int(offsets[s]), int(offsets[s + 1])
        out = np.zeros((b - a, 42))
        for t in range(b - a):
            raw = lm[a + t]
            # frozen 9-landmark target chain (used for the comparison metric)
            u9 = np.diff(raw, axis=1); u9 /= np.linalg.norm(u9, axis=-1, keepdims=True); u9 = u9 @ R.T
            target9 = np.concatenate([np.broadcast_to(base[:, 0, None], (6, 1, 3)),
                                      np.broadcast_to(base[:, 0, None], (6, 1, 3)) + np.cumsum(u9 * length[..., None], axis=1)], axis=1)
            for l in range(6):
                ix = np.arange(l * 7, l * 7 + 7)
                prev = q0[ix] if t == 0 else out[t - 1, ix]
                lo = bounds[ix, 0].copy(); hi = bounds[ix, 1].copy()
                if t:
                    lo = np.maximum(lo, prev - SOLVER['step_limit']); hi = np.minimum(hi, prev + SOLVER['step_limit'])
                pos_target = target9[l, :3]                      # coxa, femur centre, tibia centre
                u_fold = u9[l, 3]                                # direction of the folded distal chain
                cache = {}

                def evaluate(q):
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
                    # our own distal direction: tibia-centre landmark -> material point
                    own_dir = pos[8] - pos[2]; own_dir = own_dir / np.linalg.norm(own_dir)
                    dpos = pos[:3]
                    delta = np.diff(dpos, axis=0); lens = np.linalg.norm(delta, axis=1)
                    dirs = delta / lens[:, None]
                    du = np.einsum('nij,njk->nik', (np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]) / lens[:, None, None],
                                   np.diff(jac, axis=0)[:2])
                    dp = pos[8] - pos[2]; nd = np.linalg.norm(dp)
                    proj = (np.eye(3) - own_dir[:, None] * own_dir[None, :]) / nd
                    dist_jac = proj @ (jac[8] - jac[2])   # unused for a position target, kept for symmetry checks
                    tip_target = target9[l, 2] + u_fold * reach[l]   # our own reachable ankle->tip distance
                    r = np.concatenate([((dpos - pos_target) / L[l]).ravel(),
                                        ((pos[8] - tip_target) / L[l]),
                                        SOLVER['w_cont'] * (q - prev), SOLVER['w_key0'] * (q - q0[ix])])
                    J = np.concatenate([(jac[:3] / L[l]).reshape(-1, 7), (jac[8] / L[l]),
                                        SOLVER['w_cont'] * np.eye(7), SOLVER['w_key0'] * np.eye(7)], axis=0)
                    cache.update(q=q.copy(), r=r, j=J, pos=pos, own_dir=own_dir)
                    return r, J

                res = least_squares(lambda q: evaluate(q)[0], np.clip(prev, lo + 1e-12, hi - 1e-12),
                                    jac=lambda q: evaluate(q)[1], bounds=(lo, hi), method='trf',
                                    max_nfev=SOLVER['max_nfev_first'] if t == 0 else SOLVER['max_nfev_subsequent'],
                                    ftol=SOLVER['ftol'], xtol=SOLVER['xtol'], gtol=SOLVER['gtol'])
                out[t, ix] = res.x; evaluate(res.x)
                ach = cache['pos']
                err_old[a + t, l] = np.linalg.norm(ach - target9[l], axis=-1).max() / L[l]
                tip_err = float(np.linalg.norm(cache['pos'][8] - (target9[l, 2] + u_fold * reach[l])) / L[l])
                err_dir[a + t, l] = tip_err
                inflag[a + t, l] = bool(np.all(res.x >= bounds[ix, 0] - 1e-9) and np.all(res.x <= bounds[ix, 1] + 1e-9))
                nfev_all[a + t, l] = res.nfev
            Q[a + t] = out[t]
        if s % 10 == 0:
            print('snippet', s, '/', len(lengths), 'elapsed %.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(HERE / 'RETARGET_B2.npz', targets_rad=Q, error_old_metric=err_old, error_direction=err_dir,
                        in_limit=inflag, nfev=nfev_all, offsets=offsets, lengths=lengths,
                        joint_limits_rad=bounds, leg_lengths_native=L, leg_names=np.array(LEGS),
                        ordered_joint_names=np.array([m.joint(int(j)).name for j in jids]),
                        snippet_indices=data['snippet_indices'])
    summary = dict(status='RETARGET_B2_COMPLETE', snippets=int(len(lengths)), frames=int(n_frames),
                   ik_solves=int(n_frames * 6), wall_s=time.time() - t0, solver=SOLVER, physics_steps=0,
                   error_old_metric=dict(max=float(err_old.max()), p50=float(np.percentile(err_old, 50)),
                                         p95=float(np.percentile(err_old, 95)), rms=float(np.sqrt(np.mean(err_old ** 2)))),
                   error_folded_tip=dict(max=float(err_dir.max()), p50=float(np.percentile(err_dir, 50)),
                                        p95=float(np.percentile(err_dir, 95))),
                   frozen_solver_reference=str(SRC / 'retarget.py'), frozen_solver_sha256=sha(SRC / 'retarget.py'),
                   note='the old-metric error is computed with the frozen 9-landmark definition on these solutions, so it is directly comparable to the previous .0532 median / .385 max')
    (HERE / 'RETARGET_B2.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
