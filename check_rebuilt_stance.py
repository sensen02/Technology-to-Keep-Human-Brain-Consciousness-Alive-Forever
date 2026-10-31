"""Declared follow-up measurement for the rebuilt muscle geometry.

113000Z restored two-sided authority (42/42 with all 90 actuators, 40/42 with the
trusted 66 pool) but the static LP at that mid-walk handover pose (only 5 contact
points) still failed its fixed criterion of 1e-6.

This measures the SAME rebuilt model at a different, also pre-declared operating
pose: the settled six-foot stance used by the standing-handover protocol, where the
earlier (unrebuilt) model reached 0.0246. The criterion is unchanged (trusted-66
outer residual <= 1e-6); only the pose being tested differs, and both results are
reported. Nothing is tuned to make it pass.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog
from flygym.compose.fly.musculoskeletal import _load_mjcf

REB = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
S74 = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-stance-check/20261028T114000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    mapping = json.loads((REB / 'MAPPING.json').read_text()) if (REB / 'MAPPING.json').exists() else json.loads((S74 / 'MAPPING.json').read_text())
    weight = float(json.loads((S74 / 'GAINS.json').read_text())['weight'])
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    q = np.load(S74 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Same rebuilt model (113000Z geometry, length ranges re-centred at the settled stance), static LP at the settled six-foot stance.',
        criterion='trusted66 outer residual <= 1e-6 (unchanged from 113000Z)',
        pose='s5-walk-settle-neural/20261028T074000Z HANDOVER_BEFORE qpos (12 foot contact points)',
        prior_result=dict(where='113000Z mid-walk handover pose, 5 contacts', residual=0.07055925666933403, passed=False),
        budget=dict(model_load=1, forward=10, body_step=0, neural_step=0),
        source_hashes={str(p): sha(p) for p in [Path(__file__), REB / 'MODEL_REBUILT.xml', S74 / 'HANDOVER_BEFORE.npz', S74 / 'GAINS.json', S74 / 'MAPPING.json']}))
    m = _load_mjcf(str(REB / 'MODEL_REBUILT.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    # re-centre length ranges at THIS pose (declared 089 calibration, translation only)
    lr0 = m.actuator_lengthrange.copy()
    d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    Lnow = d.actuator_length.copy()
    gp = m.actuator_gainprm; lmin, lmax = gp[:, 0], gp[:, 1]
    width = lr0[:, 1] - lr0[:, 0]; L0 = width / (lmax - lmin)
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
    f1 = d.actuator_force.copy(); f0 = None
    d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
    f0 = d.actuator_force.copy()
    A = M * (f1 - f0)[None, :]
    blocks = []
    geoms = []
    for c in range(d.ncon):
        co = d.contact[c]; a1, a2 = int(co.geom1), int(co.geom2)
        p1 = m.geom_type[a1] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m.geom_type[a2] == mujoco.mjtGeom.mjGEOM_PLANE
        if not (p1 or p2):
            continue
        g = a2 if p1 else a1; nm = m.geom(g).name
        if not any(nm.startswith(l) and 'Tars' in nm for l in LEGS):
            continue
        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g])); counts['jac'] += 1
        blocks.append(jp.T); geoms.append(nm)
    J = np.column_stack(blocks)
    E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
    lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    Lrow = E[lim].T if lim.size else np.zeros((48, 0))
    bias = d.qfrc_bias.copy(); passive = d.qfrc_passive.copy()
    lp = {}
    for tag, H in (('trusted66', G), ('all90', np.eye(90))):
        for cone, inner in (('outer', False), ('inner', True)):
            counts['lps'] += 1
            n = H.shape[1]
            B = np.column_stack([A @ H, J, Lrow]) / weight
            y = (bias - passive - base) / weight
            nc = J.shape[1] // 3; nl = Lrow.shape[1]
            rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
            for jj in range(nc):
                mr = np.zeros((4, B.shape[1] + 1))
                dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for k, (sx, sy) in enumerate(dirs):
                    mr[k, n + 3 * jj:n + 3 * jj + 3] = [sx, sy, -1.]
                rws.append(mr); rhs.append(np.zeros(4))
            obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
            bounds = [(0., 1.)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
            sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bounds, method='highs')
            lp[f'{tag}_{cone}'] = dict(residual=float(sol.fun) if sol.success else None,
                                       feasible=bool(sol.success and sol.fun <= 1e-6))
    out = dict(pose='settled six-foot stance', contacts=len(blocks), contact_geoms=sorted(set(geoms)),
               limit_rows=int(Lrow.shape[1]), lp=lp, criterion='trusted66 outer <= 1e-6',
               criterion_met=bool(lp['trusted66_outer']['feasible']))
    dump(O / 'RESULTS.json', out)
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
