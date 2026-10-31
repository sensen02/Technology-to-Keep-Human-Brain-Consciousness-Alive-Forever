"""Independent re-derivation and LP re-solve for the two best searched poses.

Rebuilds the two lowest-residual candidates from their declared parameters
(dz=0.0/spread=1.0 and dz=0.4/spread=1.0), saves the exact matrices, and solves
the four declared LPs again. Any disagreement with the search record is an error.
Static only: 1 model load, no body integration, no neural steps.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
from scipy.optimize import linprog, least_squares
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
SRC = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility/20261028T096000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility-verify/20261028T098000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def solve(A, base, J, L, bias, passive, weight, inner, H):
    n = H.shape[1]
    B = np.column_stack([A @ H, J, L]) / weight
    y = (bias - passive - base) / weight
    nc = J.shape[1] // 3
    nl = L.shape[1]
    rows = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]
    rhs = [y, -y]
    for j in range(nc):
        mrow = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mrow[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rows.append(mrow); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bounds = [(0., 1.)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=bounds, method='highs')
    return float(sol.fun), np.vstack(rows), np.concatenate(rhs)


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0)
    cand = json.loads((SRC / 'CANDIDATES.json').read_text())
    weight = float(json.loads((S / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, 66))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    q0 = np.load(S / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    dump(O / 'PREREGISTRATION.json', dict(scope='Independent matrix re-derivation at the two lowest-residual searched poses (saved qpos used verbatim; no IK re-run) and re-solve of the four declared static LPs.',
                                          candidates=[0, 4], budget=dict(model_load=1, forward=400, body_step=0, neural_step=0),
                                          source_hashes={str(p): sha(p) for p in [Path(__file__), S / 'MODEL.xml', S / 'HANDOVER_BEFORE.npz', S / 'GAINS.json', S / 'MAPPING.json', SRC / 'CANDIDATES.json']}))
    m = _load_mjcf(str(S / 'MODEL.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); va = np.array([m.jnt_dofadr[j] for j in jid])
    lo = m.jnt_range[jid, 0] + 1e-9; hi = m.jnt_range[jid, 1] - 1e-9
    bids = [m.body(l + 'Tarsus5').id for l in LEGS]
    thorax = m.body('Thorax').id
    lr0 = m.actuator_lengthrange.copy()
    out = {}
    for k in (0, 4):
        row = cand[k]
        dz = row['root_height_offset_mm']; spread = row['foot_spread_factor']
        q = np.array(row['qpos'], dtype=float).copy()
        tgt = np.array(row['tarsus_targets'], dtype=float).copy()
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
        mujoco.mj_forward(m, d); counts['forward'] += 1
        # declared 089 length-range translation at this pose
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
        base = d.qfrc_actuator.copy(); A = np.zeros((48, 90))
        for j in range(90):
            d.act[:] = .0001; d.act[j] = 1.
            mujoco.mj_forward(m, d); counts['forward'] += 1
            A[:, j] = d.qfrc_actuator - base
        d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
        blocks = []; contacts = []
        for c in range(d.ncon):
            co = d.contact[c]; a, bb = int(co.geom1), int(co.geom2)
            p1 = m.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m.geom_type[bb] == mujoco.mjtGeom.mjGEOM_PLANE
            if not (p1 or p2):
                continue
            g = bb if p1 else a
            name = m.geom(g).name
            if not any(name.startswith(l) and 'Tars' in name for l in LEGS):
                continue
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g])); counts['jac'] += 1
            blocks.append(jp.T); contacts.append(name)
        E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
        lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        Lrow = E[lim].T if lim.size else np.zeros((48, 0))
        J = np.column_stack(blocks)
        bias = d.qfrc_bias.copy(); passive = d.qfrc_passive.copy()
        res = {}
        for tag, H in (('trusted66', G), ('all90', np.eye(90))):
            for cone, inner in (('outer', False), ('inner', True)):
                counts['lps'] += 1
                val, Aub, bub = solve(A, base, J, Lrow, bias, passive, weight, inner, H)
                recorded = row['lp'][f'{tag}_{cone}']['residual']
                res[f'{tag}_{cone}'] = dict(independently_resolved_residual=val, search_record_residual=recorded,
                                            abs_difference=abs(val - recorded))
                assert abs(val - recorded) < 1e-9, (k, tag, cone, val, recorded)
        np.savez_compressed(O / f'MATRICES_{k}.npz', qpos=q, qpos_hinges=q[qa], qpos_addresses=qa,
                            tarsus_targets=tgt, active_force_matrix=A, floor_force=base,
                            contact_force_matrix=J, joint_limit_force_matrix=Lrow, pool_matrix=G,
                            contact_geoms=np.array(contacts))
        out[k] = dict(parameters=dict(dz=dz, spread=spread), contacts=len(contacts), contact_geoms=sorted(set(contacts)),
                      lp=res, all_match=True)
    dump(O / 'INDEPENDENT_LP_CHECK.json', dict(passed=True, candidates=out,
                                               conclusion='Both lowest-residual poses reproduce exactly; the positive static residual is not an artefact of the search pipeline.'))
    dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
    print('COMPLETE', O, flush=True)


if __name__ == '__main__':
    main()
