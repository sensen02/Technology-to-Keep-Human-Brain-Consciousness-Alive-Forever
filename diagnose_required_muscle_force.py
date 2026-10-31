"""Diagnostic only: how much muscle force would the best searched pose need?

Rebuilds one saved candidate pose (best 6-foot eligible pose of the extended
search), then sweeps a uniform multiplier on the muscle actuator force (the
declared engineering knob: gainprm/biasprm force parameter, which scales both
active and passive muscle force) and reports the static LP residual for each
factor. Nothing is executed as a control; no parameter file is modified.

1 model load, a few forwards, no body integration, no neural steps.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
E = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility-ext/20261028T100000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-pose-force-requirement/20261028T101000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def lp66(A, base, J, L, bias, passive, weight, G, inner):
    n = G.shape[1]
    B = np.column_stack([A @ G, J, L]) / weight
    y = (bias - passive - base) / weight
    nc = J.shape[1] // 3; nl = L.shape[1]
    rows = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
    for j in range(nc):
        mr = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mr[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rows.append(mr); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bounds = [(0., 1.)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=bounds, method='highs')
    return float(sol.fun) if sol.success else float('inf')


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    rows = json.loads((E / 'STAGE_B.json').read_text())
    best = min((r for r in rows if r['eligible']), key=lambda r: r['lp']['trusted66_outer']['residual'] if 'lp' in r else r['lp66_outer']['residual'])
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Diagnostic only: uniform muscle-force multiplier required to make the best searched pose statically feasible. No control executed, no model file changed, no anatomy change.',
        pose=dict(dz=best['dz'], pitch_deg=best['pitch_deg'], roll_deg=best['roll_deg'], fore_aft=best['fore_aft']),
        recorded_residual=best['lp']['trusted66_outer']['residual'] if 'lp' in best else best['lp66_outer']['residual'],
        sweep=[.25, .5, 1., 2., 4., 8., 16., 32.],
        budget=dict(model_load=1, forward=20, body_step=0, neural_step=0),
        source_hashes={str(p): sha(p) for p in [Path(__file__), S / 'MODEL.xml', E / 'STAGE_B.json', S / 'GAINS.json', S / 'MAPPING.json']}))
    weight = float(json.loads((S / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, 66))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    q = np.array(best['qpos'], dtype=float)
    m = _load_mjcf(str(S / 'MODEL.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    lr0 = m.actuator_lengthrange.copy()
    # declared 089 length-range translation at this pose
    d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    Lnow = d.actuator_length.copy(); gp = m.actuator_gainprm
    lmin, lmax = gp[:, 0], gp[:, 1]; width = lr0[:, 1] - lr0[:, 0]; L0 = width / (lmax - lmin)
    new0 = Lnow - (1.0 - lmin) * L0
    m.actuator_lengthrange[:, 0] = new0; m.actuator_lengthrange[:, 1] = new0 + width
    d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0; d.qacc_warmstart[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    base = d.qfrc_actuator.copy()
    M = np.zeros((m.nv, m.nu))
    if d.actuator_moment.size == m.nv * m.nu:
        M = d.actuator_moment.reshape(m.nv, m.nu).copy()
    else:
        for j in range(m.nu):
            adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
            M[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
    d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
    f1 = d.actuator_force.copy(); d.act[:] = .0001
    mujoco.mj_forward(m, d); counts['forward'] += 1
    f0 = d.actuator_force.copy()
    A = M * (f1 - f0)[None, :]
    blocks = []
    for c in range(d.ncon):
        co = d.contact[c]; a, bb = int(co.geom1), int(co.geom2)
        p1 = m.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m.geom_type[bb] == mujoco.mjtGeom.mjGEOM_PLANE
        if not (p1 or p2):
            continue
        g = bb if p1 else a; name = m.geom(g).name
        if not any(name.startswith(l) and 'Tars' in name for l in LEGS):
            continue
        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g])); counts['jac'] += 1
        blocks.append(jp.T)
    J = np.column_stack(blocks)
    E_ = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
    lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    Lrow = E_[lim].T if lim.size else np.zeros((48, 0))
    bias = d.qfrc_bias.copy(); passive = d.qfrc_passive.copy()
    results = []
    for s in [.25, .5, 1., 2., 4., 8., 16., 32.]:
        counts['lps'] += 1
        r_out = lp66(s * A, s * base, J, Lrow, bias, passive, weight, G, False)
        counts['lps'] += 1
        r_in = lp66(s * A, s * base, J, Lrow, bias, passive, weight, G, True)
        results.append(dict(uniform_muscle_force_multiplier=s, residual_outer=r_out, residual_inner=r_in,
                            feasible_outer=bool(r_out <= 1e-6), feasible_inner=bool(r_in <= 1e-6)))
        dump(O / 'SWEEP.json', results)
        print(json.dumps(results[-1]), flush=True)
    np.savez_compressed(O / 'MATRICES.npz', qpos=q, active_force_matrix=A, floor_force=base,
                        contact_force_matrix=J, joint_limit_force_matrix=Lrow, pool_matrix=G)
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0,
                                    note='Uniform multiplier scales the muscle force parameter, which scales active and passive muscle force together. Diagnostic only.'))
    print('COMPLETE', O, flush=True)


if __name__ == '__main__':
    main()
