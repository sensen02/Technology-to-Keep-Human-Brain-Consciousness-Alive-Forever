"""Stage 3: retarget the declared long-snippet set with the frozen solver (no physics).

Same solver and metric as the subsample inventory; the only difference is that whole
snippets are retargeted so that contiguous 0.5 s segments can be extracted afterwards.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SRC = ROOT / 'experiments/flybody-retarget/20261107T182940Z/kinematics'
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
SOLVER = dict(max_nfev_first=80, max_nfev_subsequent=16, ftol=1e-7, xtol=1e-7, gtol=1e-7,
              w_pos=1.0, w_dir=.05, w_cont=.02, w_key0=.002, step_limit=.06)


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
    data = np.load(HERE / 'LONG_SNIPPETS.npz')
    lm_all = data['landmarks']; offsets = data['offsets']; lengths = data['lengths']
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
    n_snip = len(lengths)
    Q_all = np.zeros((int(lengths.sum()), 42)); E_all = np.zeros((int(lengths.sum()), 6))
    nfev_all = np.zeros((int(lengths.sum()), 6), int)
    for s in range(n_snip):
        a, b = int(offsets[s]), int(offsets[s + 1])
        out = np.zeros((b - a, 42))
        for t in range(b - a):
            raw = lm_all[a + t]
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
                    jac = np.array(jac); delta = np.diff(pos, axis=0); lens = np.linalg.norm(delta, axis=1)
                    dirs = delta / lens[:, None]; dj = np.diff(jac, axis=0)
                    du = np.einsum('nij,njk->nik', (np.eye(3)[None] - dirs[:, :, None] * dirs[:, None, :]) / lens[:, None, None], dj)
                    r = np.concatenate([((pos - target[l]) / L[l]).ravel(), (SOLVER['w_dir'] * (dirs - u[l])).ravel(),
                                        SOLVER['w_cont'] * (q - prev), SOLVER['w_key0'] * (q - q0[ix])])
                    J = np.concatenate([(jac / L[l]).reshape(-1, 7), (SOLVER['w_dir'] * du).reshape(-1, 7),
                                        SOLVER['w_cont'] * np.eye(7), SOLVER['w_key0'] * np.eye(7)], axis=0)
                    cache.update(q=q.copy(), r=r, j=J, pos=pos)
                    return r, J

                res = least_squares(lambda q: evaluate(q)[0], np.clip(prev, lo + 1e-12, hi - 1e-12),
                                    jac=lambda q: evaluate(q)[1], bounds=(lo, hi), method='trf',
                                    max_nfev=SOLVER['max_nfev_first'] if t == 0 else SOLVER['max_nfev_subsequent'],
                                    ftol=SOLVER['ftol'], xtol=SOLVER['xtol'], gtol=SOLVER['gtol'])
                out[t, ix] = res.x; evaluate(res.x)
                E_all[a + t, l] = np.linalg.norm(cache['pos'] - target[l], axis=-1).max() / L[l]
                nfev_all[a + t, l] = res.nfev
            Q_all[a + t] = out[t]
        if s % 10 == 0:
            print('snippet', s, '/', n_snip, 'frame', a, 'elapsed %.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(HERE / 'LONG_RETARGET.npz', targets_rad=Q_all, normalized_error=E_all, nfev=nfev_all,
                        offsets=offsets, lengths=lengths, snippet_indices=data['snippet_indices'],
                        joint_limits_rad=bounds, leg_lengths_native=L, leg_names=np.array(LEGS),
                        ordered_joint_names=np.array([m.joint(int(j)).name for j in jids]))
    summary = dict(status='LONG_RETARGET_COMPLETE', snippets=int(n_snip), frames=int(lengths.sum()),
                   ik_solves=int(lengths.sum() * 6), error=dict(max=float(E_all.max()), p50=float(np.percentile(E_all, 50)),
                                                                p95=float(np.percentile(E_all, 95)), rms=float(np.sqrt(np.mean(E_all ** 2)))),
                   wall_s=time.time() - t0, physics_steps=0, neural_steps=0, solver=SOLVER,
                   frozen_solver_source=str(SRC / 'retarget.py'))
    (HERE / 'LONG_RETARGET.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
