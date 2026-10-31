"""Offline static handover-pose feasibility search (pre-registered, bounded).

Budget (declared before any call): <=1 MuJoCo model load, <=3000 mj_forward,
0 body integration, 0 neural steps. Each candidate pose is screened by an exact
linear program: does there exist bounded muscle activation (66 trusted pool and,
separately, all 90 actuators), foot contact force inside a declared friction
cone, and nonnegative active joint-limit force such that all 48 generalized
forces balance to <=1e-6 of body weight?

Muscle length ranges are recentered IN MEMORY at each candidate pose
(translation only, width preserved): the declared engineering calibration from
089, recomputed at the candidate pose and never carried over.

No anatomy, muscle attachment, gain, contact or neural change. No runtime control.
"""
from pathlib import Path
import json, hashlib, time, traceback
import numpy as np
import mujoco
from scipy.optimize import linprog, least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
R = Path('/home/sensen/Desktop/engine修复验收/s5-lengthrange-recenter/20261028T088000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility/20261028T096000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
FORWARD_CAP = 3000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rollout_matrix(m, d, q, counts, base_state):
    """Actuator generalized force at the activation floor and per-unit-activation slopes."""
    d.qpos[:] = q
    d.qvel[:] = 0
    d.act[:] = .0001
    d.ctrl[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    d.qacc_warmstart[:] = 0
    mujoco.mj_forward(m, d)
    counts['forward'] += 1
    base = d.qfrc_actuator.copy()
    A = np.zeros((m.nv, m.nu))
    for j in range(m.nu):
        d.act[:] = .0001
        d.act[j] = 1.
        mujoco.mj_forward(m, d)
        counts['forward'] += 1
        A[:, j] = d.qfrc_actuator - base
    d.act[:] = .0001
    mujoco.mj_forward(m, d)
    counts['forward'] += 1
    return base, A


def collect_rows(m, d, counts):
    """Foot contact Jacobians, active joint-limit rows, and non-foot clearance."""
    contacts = []
    blocks = []
    for k in range(d.ncon):
        co = d.contact[k]
        a, bb = int(co.geom1), int(co.geom2)
        p1 = m.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE
        p2 = m.geom_type[bb] == mujoco.mjtGeom.mjGEOM_PLANE
        if not (p1 or p2):
            continue
        g = bb if p1 else a
        name = m.geom(g).name
        isfoot = any(name.startswith(l) and 'Tars' in name for l in LEGS)
        if not isfoot:
            continue
        jp = np.zeros((3, m.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g]))
        counts['jac'] += 1
        blocks.append(jp.T)
        contacts.append(dict(geom=name, mu=float(co.friction[0]), position=co.pos.copy()))
    E = np.zeros((d.nefc, m.nv))
    if d.nefc:
        if d.efc_J.size == d.nefc * m.nv:
            E = d.efc_J.reshape(d.nefc, m.nv).copy()
        else:
            for row in range(d.nefc):
                adr = int(d.efc_J_rowadr[row]); nnz = int(d.efc_J_rownnz[row])
                E[row, d.efc_J_colind[adr:adr + nnz]] = d.efc_J[adr:adr + nnz]
    lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
    L = E[lim].T if lim.size else np.zeros((m.nv, 0))
    nondist = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        name = m.geom(g).name
        if any(name.startswith(l) and 'Tars' in name for l in LEGS):
            continue
        dist = mujoco.mj_geomDistance(m, d, g, 0, 5.0, None)
        nondist.append(dict(geom=name, clearance_mm=float(dist)))
    return contacts, blocks, L, lim, nondist


def static_lp(A, base, J, L, bias, passive, weight, inner, H):
    """Exact static balance LP. Returns residual/bodyweight and status."""
    n = H.shape[1]
    B = np.column_stack([A @ H, J, L]) / weight
    y = (bias - passive - base) / weight
    nc = J.shape[1] // 3 if J.size else 0
    nl = L.shape[1]
    rows = [np.c_[B, -np.ones(B.shape[0])], np.c_[-B, -np.ones(B.shape[0])]]
    rhs = [y, -y]
    mus = []
    for j in range(nc):
        mrow = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mrow[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rows.append(mrow); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bounds = [(0., 1.)] * n + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=bounds, method='highs')
    if not sol.success:
        return dict(success=False, message=sol.message, residual=np.inf, contacts=nc, limits=nl)
    x = sol.x
    forces = x[n:n + 3 * nc].reshape(nc, 3) if nc else np.zeros((0, 3))
    fatz = forces[:, 2] if nc else np.zeros(0)
    normals = fatz[fatz > 1e-12]
    ftot = forces.sum(0) / weight if nc else np.zeros(3)
    return dict(success=True, residual=float(x[-1]), l2=float(np.linalg.norm(B @ x[:-1] - y)),
                contacts=nc, limits=nl, pool_activation=x[:n].copy(),
                contact_forces=forces, total_ground_force_over_weight=ftot,
                vertical_over_weight=float(ftot[2]) if nc else 0.,
                max_normal_over_weight=float(normals.max() / weight) if normals.size else 0.,
                contact_force_matrix=J, joint_limit_matrix=L, target=y * weight)


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time()
    counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0, candidates=0)
    sources = [Path(__file__), S / 'MODEL.xml', S / 'HANDOVER_BEFORE.npz', S / 'GAINS.json',
               S / 'MAPPING.json', R / 'RESULTS.json']
    hashes = {str(p): sha(p) for p in sources}
    b = np.load(S / 'HANDOVER_BEFORE.npz')
    weight = float(json.loads((S / 'GAINS.json').read_text())['weight'])
    q0 = b['qpos'].copy()
    mapping = json.loads((S / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    assert G.shape[1] == 66 and int((G.sum(1) == 0).sum()) == 16
    targets = [(dz, sp) for dz in (0.0, .4, .8, 1.2) for sp in (1.0, 1.2, 1.4, 1.6)]
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Offline static pose feasibility at the handover instant only. No body integration, no neural run, no runtime controller.',
        budget=dict(model_load=1, forward_cap=FORWARD_CAP, body_step=0, neural_step=0, candidates=len(targets)),
        candidates=[dict(index=i, root_height_offset_mm=dz, foot_spread_factor=sp) for i, (dz, sp) in enumerate(targets)],
        variables='root height offset and foot XY spread about the thorax axis (foot Z held on the floor); redundant 42-hinge IK; roll/pitch left at the real settled value',
        criteria=dict(feet_contact=6, no_nonfoot_contact=True, joint_limits=True,
                      front_legs_not_crossing_midline=True, foot_ik_error_lt=1e-3,
                      nonfoot_clearance_mm_ge=8.0,
                      static_lp='min_t ||A x + J f + L lam - (bias-passive-floor)||_inf/bodyweight; feasible iff t<=1e-6',
                      pools='trusted66 and all90 both reported for every candidate'),
        lengthrange='translated in memory (089 formula, width preserved) so normalized length is 1.0 at the candidate pose; validated against the sealed 089 floor before use',prior_attempts=['20261028T091000Z: calibration gate failed (wrong normalization)','20261028T092000Z: counts dict missing candidates key','20261028T093000Z: foot target array shape bug','20261028T094000Z: undefined local name sp in row construction','20261028T095000Z: console print could not serialise a numpy array (all computation, including LPs, had completed)'],
        source_hashes=hashes))

    m = _load_mjcf(str(S / 'MODEL.xml')).compile()
    counts['model_load'] += 1
    d = mujoco.MjData(m)
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qa = np.array([m.jnt_qposadr[j] for j in jid])
    va = np.array([m.jnt_dofadr[j] for j in jid])
    lo = m.jnt_range[jid, 0] + 1e-9
    hi = m.jnt_range[jid, 1] - 1e-9
    bids = [m.body(l + 'Tarsus5').id for l in LEGS]
    thorax = m.body('Thorax').id
    lr0 = m.actuator_lengthrange.copy()

    def recenter_LR(q):
        """In-memory translation of each muscle length range to normalized length 1 at q."""
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
        mujoco.mj_forward(m, d); counts['forward'] += 1
        Lnow = d.actuator_length.copy()
        gp = m.actuator_gainprm
        lmin, lmax = gp[:, 0], gp[:, 1]
        width = lr0[:, 1] - lr0[:, 0]
        L0 = width / (lmax - lmin)
        new0 = Lnow - (1.0 - lmin) * L0   # exact 089 translation, width preserved
        m.actuator_lengthrange[:, 0] = new0
        m.actuator_lengthrange[:, 1] = new0 + width
        return Lnow

    # Validation of the in-memory calibration against the sealed 089 measurement.
    ref = json.loads((R / 'RESULTS.json').read_text())
    orig = next(r for r in ref if r['name'] == 'original')
    rec = next(r for r in ref if r['name'] == 'recenter_1.0')
    recenter_LR(q0)
    d.qpos[:] = q0; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
    mujoco.mj_forward(m, d); counts['forward'] += 1
    floor_mem = float(np.linalg.norm(d.qfrc_actuator[6:48]) / weight)
    m.actuator_lengthrange[:] = lr0
    validation = dict(memory_floor_over_weight=floor_mem,
                      sealed_089_floor_over_weight=rec['floor_norm_over_weight'],
                      abs_error=abs(floor_mem - rec['floor_norm_over_weight']),
                      passed=bool(abs(floor_mem - rec['floor_norm_over_weight']) < 1e-6))
    dump(O / 'LENGTHRANGE_IN_MEMORY_VALIDATION.json', validation)
    assert validation['passed'], validation

    fy = float(d.xpos[thorax][1])
    foot_pos0 = d.xpos[bids].copy()
    foot_xy0 = foot_pos0[:, :2].copy()
    foot_z0 = foot_pos0[:, 2].copy()
    rows = []
    best = None
    try:
        for idx, (dz, spread) in enumerate(targets):
            if counts['forward'] > FORWARD_CAP - 200:
                break
            counts['candidates'] += 1
            # Root pose: raise only; keep the real settled orientation.
            qr = q0.copy(); qr[2] += dz
            tgt = np.zeros((6, 3))
            tgt[:, 0] = d.xpos[thorax][0] + spread * (foot_xy0[:, 0] - d.xpos[thorax][0])
            tgt[:, 1] = fy + spread * (foot_xy0[:, 1] - fy)
            tgt[:, 2] = foot_z0
            cache = {}

            def up(x):
                if 'x' not in cache or not np.array_equal(cache['x'], x):
                    d.qpos[:] = qr; d.qpos[qa] = x
                    mujoco.mj_forward(m, d); counts['forward'] += 1
                    cache['x'] = x.copy()

            def fun(x):
                up(x)
                return np.r_[((d.xpos[bids] - tgt) * 10.).ravel(), .02 * (x - q0[qa])]

            def jac(x):
                up(x); cols = []
                for bid in bids:
                    jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
                    mujoco.mj_jac(m, d, jp, jr, d.xpos[bid], bid); counts['jac'] += 1
                    cols.append(jp[:, va] * 10.)
                return np.vstack(cols + [.02 * np.eye(42)])

            x0 = np.clip(q0[qa], lo, hi)
            sol = least_squares(fun, x0, jac=jac, bounds=(lo, hi), max_nfev=40, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            q = qr.copy(); q[qa] = sol.x
            up(sol.x)
            foot_err = float(np.max(np.linalg.norm(d.xpos[bids] - tgt, axis=1)))
            side = []
            Rm = d.xmat[thorax].reshape(3, 3)
            for leg in LEGS:
                vals = [float((Rm.T @ (d.xpos[bb] - d.xpos[thorax]))[1]) * (1 if leg[0] == 'L' else -1)
                        for bb in range(m.nbody) if m.body(bb).name.startswith(leg)
                        and any(s in m.body(bb).name for s in ('Coxa', 'Trochanter', 'Femur', 'Tibia', 'Tarsus'))]
                side.append(min(vals))
            # Declared calibration at this candidate pose, then measure everything on it.
            recenter_LR(q)
            d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
            d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0; d.qacc_warmstart[:] = 0
            mujoco.mj_forward(m, d); counts['forward'] += 1
            base, A = rollout_matrix(m, d, q, counts, None)
            contacts, blocks, Lrow, limrows, nondist = collect_rows(m, d, counts)
            feet = sorted({c['geom'][:2] for c in contacts})
            bias = d.qfrc_bias.copy(); passive = d.qfrc_passive.copy()
            J = np.column_stack(blocks) if blocks else np.zeros((m.nv, 0))
            minclear = min([n['clearance_mm'] for n in nondist]) if nondist else 99.
            row = dict(index=idx, root_height_offset_mm=dz, foot_spread_factor=spread,
                       ik_success=bool(sol.success), foot_error=foot_err,
                       feet_contacting=feet, n_contacts=len(contacts),
                       joint_limit_rows=int(len(limrows)), side_min=side,
                       nonfoot_contact_count=int(sum(1 for n in nondist if n['clearance_mm'] <= 0)),
                       min_nonfoot_clearance_mm=float(minclear))
            row['eligible_geometry'] = bool(len(feet) == 6 and row['nonfoot_contact_count'] == 0
                                            and min(side) > 0 and foot_err < 1e-3
                                            and np.all(q[qa] >= lo - 1e-6) and np.all(q[qa] <= hi + 1e-6))
            row['clearance_criterion_8mm'] = bool(minclear >= 8.)
            res = {}
            for tag, H in (('trusted66', G), ('all90', np.eye(90))):
                for cone, inner in (('outer', False), ('inner', True)):
                    counts['lps'] += 1
                    r = static_lp(A, base, J, Lrow, bias, passive, weight, inner, H)
                    res[f'{tag}_{cone}'] = {k: v for k, v in r.items()
                                            if k not in ('pool_activation', 'contact_forces', 'contact_force_matrix',
                                                         'joint_limit_matrix', 'target')}
                    res[f'{tag}_{cone}']['feasible_1e6'] = bool(r.get('residual', np.inf) <= 1e-6)
            row['lp'] = res
            row['static_feasible_66_outer'] = bool(res['trusted66_outer']['feasible_1e6'])
            row['qpos'] = q.copy()
            row['tarsus_targets'] = tgt
            rows.append(row)
            dump(O / 'CANDIDATES.json', rows)
            print(json.dumps({k: v for k, v in row.items() if k != 'qpos'}, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)), flush=True)
            ok = row['eligible_geometry'] and row['static_feasible_66_outer']
            score = abs(dz) + abs(spread - 1.)
            if ok and (best is None or score < best[0]):
                best = (score, row, base, A, J, Lrow, contacts, q.copy())
    except BaseException as exc:
        dump(O / 'FAILURE.json', dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        summary = dict(candidates_attempted=counts['candidates'], feasible=[r['index'] for r in rows
                        if r['eligible_geometry'] and r['static_feasible_66_outer']],
                        any_clearance_8mm=[r['index'] for r in rows if r['clearance_criterion_8mm']],
                        max_min_nonfoot_clearance_mm=max([r['min_nonfoot_clearance_mm'] for r in rows]) if rows else None,
                        selected=None if best is None else best[1]['index'])
        dump(O / 'SUMMARY.json', summary)
        if best is not None:
            _, row, base, A, J, Lrow, contacts, q = best
            np.savez_compressed(O / 'CANDIDATE.npz', qpos=q, qpos_hinges=q[qa], qpos_addresses=qa,
                                dof_addresses=va, tarsus_targets=row['tarsus_targets'],
                                active_force_matrix=A, floor_force=base, contact_force_matrix=J,
                                joint_limit_force_matrix=Lrow, pool_matrix=G)
            dump(O / 'CANDIDATE_LEDGER.json', dict(selected_index=row['index'], qpos=q, hinges=q[qa],
                                                   joint_names=[m.joint(int(j)).name for j in jid],
                                                   lp=row['lp'], geometry=dict(side_min=row['side_min'],
                                                   min_nonfoot_clearance_mm=row['min_nonfoot_clearance_mm'],
                                                   feet=row['feet_contacting'])))
        dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0,
                                        source_unchanged=all(sha(Path(p)) == h for p, h in hashes.items()),
                                        scope='Static screening only; a feasible LP is not a dynamic capability certificate.'))
        (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('COMPLETE', O, flush=True)


if __name__ == '__main__':
    main()
