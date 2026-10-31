"""Extended bounded handover-pose feasibility search (round 14).

Declared budget: 1 model load, <=3000 mj_forward, 0 body integration, 0 neural steps.
Domain (pre-registered, no adaptive expansion): root height offset {-0.4,-0.2,0,+0.2,+0.4}
x pitch {-3,0,+3} deg x roll {-3,0,+3} deg (45 stage-A poses), then the best eight
stage-A poses also get whole-body fore/aft foot displacement {-0.2,0,+0.2} in stage B.

Actuator generalized force is built ANALYTICALLY as moment[:,j]*(F1[j]-F0[j]) from two
forwards (all-activations-at-1 and all-at-floor) plus the sparse actuator moment matrix,
instead of 90 per-actuator probes. This is validated against the saved 90-probe matrices
of the sealed 098 verification before any candidate is judged.

Muscle length ranges are recentered in memory at each candidate pose with the sealed 089
formula (translation only, width preserved).

Static screening only: a feasible LP is not a dynamic capability certificate.
"""
from pathlib import Path
import json, hashlib, time, traceback
import numpy as np
import mujoco
from scipy.optimize import linprog, least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf

S = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
V = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility-verify/20261028T098000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-pose-feasibility-ext/20261028T100000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
CAP = 3000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def rot(pitch_deg, roll_deg):
    p = np.deg2rad(pitch_deg); r = np.deg2rad(roll_deg)
    cp, sp = np.cos(p / 2), np.sin(p / 2); cr, sr = np.cos(r / 2), np.sin(r / 2)
    qp = np.array([cp, sp, 0., 0.]); qr = np.array([cr, 0., sr, 0.])
    w1, x1, y1, z1 = qp; w2, x2, y2, z2 = qr
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def static_lp(A, base, J, L, bias, passive, weight, inner, H):
    n = H.shape[1]
    B = np.column_stack([A @ H, J, L]) / weight
    y = (bias - passive - base) / weight
    nc = J.shape[1] // 3; nl = L.shape[1]
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
    if not sol.success:
        return dict(success=False, residual=float('inf'))
    x = sol.x
    forces = x[n:n + 3 * nc].reshape(nc, 3) if nc else np.zeros((0, 3))
    ftot = forces.sum(0) / weight if nc else np.zeros(3)
    return dict(success=True, residual=float(x[-1]), l2=float(np.linalg.norm(B @ x[:-1] - y)),
                contacts=nc, limits=nl, pool_activation=x[:n].copy(),
                vertical_over_weight=float(ftot[2]) if nc else 0.,
                max_normal_over_weight=float(forces[:, 2].max() / weight) if nc else 0.)


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time()
    counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0, stage_a=0, stage_b=0)
    q0 = np.load(S / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    weight = float(json.loads((S / 'GAINS.json').read_text())['weight'])
    mapping = json.loads((S / 'MAPPING.json').read_text())
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, 66))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    stage_a = [(dz, p, r) for dz in (-.4, -.2, 0., .2, .4) for p in (-3., 0., 3.) for r in (-3., 0., 3.)]
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Extended bounded static pose feasibility. No body integration, no neural run, no runtime controller, no anatomy or muscle-parameter change.',
        budget=dict(model_load=1, forward_cap=CAP, body_step=0, neural_step=0),
        stage_a=dict(domain=dict(root_height_offset=[-.4, -.2, 0., .2, .4], pitch_deg=[-3., 0., 3.], roll_deg=[-3., 0., 3.]),
                     candidates=len(stage_a), lp='trusted66 outer only (screen)'),
        stage_b=dict(rule='best eight stage-A poses by trusted66 outer residual among eligible geometry', fore_aft=[-.2, 0., .2],
                     lp='trusted66 outer, trusted66 inner, all90 outer, all90 inner'),
        criteria=dict(feet_contact=6, no_nonfoot_contact=True, joint_limits=True, front_legs_not_crossing_midline=True,
                      foot_ik_error_lt=1e-3, static_lp_feasible_residual_le=1e-6,
                      clearance='recorded only; the requested 8mm is impossible at this model scale (measured max <1 unit)'),
        analytics='actuator force columns built from two forwards and the sparse moment matrix, validated against sealed 098 90-probe matrices before use',
        lengthrange='in-memory 089 translation at each candidate pose',
        source_hashes={str(p): sha(p) for p in [Path(__file__), S / 'MODEL.xml', S / 'HANDOVER_BEFORE.npz', S / 'GAINS.json', S / 'MAPPING.json', V / 'MATRICES_4.npz']}))

    m = _load_mjcf(str(S / 'MODEL.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); va = np.array([m.jnt_dofadr[j] for j in jid])
    lo = m.jnt_range[jid, 0] + 1e-9; hi = m.jnt_range[jid, 1] - 1e-9
    bids = [m.body(l + 'Tarsus5').id for l in LEGS]
    thorax = m.body('Thorax').id
    lr0 = m.actuator_lengthrange.copy()

    def recenter(q):
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
        mujoco.mj_forward(m, d); counts['forward'] += 1
        Lnow = d.actuator_length.copy()
        gp = m.actuator_gainprm; lmin, lmax = gp[:, 0], gp[:, 1]
        width = lr0[:, 1] - lr0[:, 0]; L0 = width / (lmax - lmin)
        new0 = Lnow - (1.0 - lmin) * L0
        m.actuator_lengthrange[:, 0] = new0; m.actuator_lengthrange[:, 1] = new0 + width

    def moment_dense():
        """Sparse moment lives on MjData; never reshape the sparse storage."""
        M = np.zeros((m.nv, m.nu))
        if d.actuator_moment.size == m.nv * m.nu:
            return d.actuator_moment.reshape(m.nv, m.nu).copy()
        for j in range(m.nu):
            adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
            M[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
        return M

    def build_forces(q):
        """Analytic actuator columns: moment * (force@1 - force@floor)."""
        d.qpos[:] = q; d.qvel[:] = 0; d.ctrl[:] = 0
        d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0; d.qacc_warmstart[:] = 0
        d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
        f0 = d.actuator_force.copy(); base = d.qfrc_actuator.copy()
        mom = moment_dense()
        d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
        f1 = d.actuator_force.copy()
        A = mom * (f1 - f0)[None, :]
        d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
        return base, A

    # ---- validate the analytic columns against the sealed 90-probe matrices ----
    ref = np.load(V / 'MATRICES_4.npz')
    qref = ref['qpos'].copy()
    recenter(qref)
    base_ref, A_ref = build_forces(qref)
    mom_err = float(np.max(np.abs(A_ref - ref['active_force_matrix'])))
    base_err = float(np.max(np.abs(base_ref - ref['floor_force'])))
    validation = dict(max_column_abs_error=mom_err, max_base_abs_error=base_err,
                      tolerance=1e-8, passed=bool(mom_err < 1e-8 and base_err < 1e-8))
    dump(O / 'ANALYTIC_FORCE_VALIDATION.json', validation)
    m.actuator_lengthrange[:] = lr0
    assert validation['passed'], validation

    def evaluate(dz, pitch, roll, fore):
        counts['forward'] += 1
        d.qpos[:] = q0; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        fp = d.xpos[bids].copy(); thr = d.xpos[thorax].copy()
        tgt = fp.copy(); tgt[:, 0] += fore
        qr = q0.copy(); qr[2] += dz
        qr[3:7] = rot(pitch, roll)
        cache = {}

        def up(x):
            if 'x' not in cache or not np.array_equal(cache['x'], x):
                d.qpos[:] = qr; d.qpos[qa] = x
                mujoco.mj_forward(m, d); counts['forward'] += 1
                cache['x'] = x.copy()

        def fun(x):
            up(x); return np.r_[((d.xpos[bids] - tgt) * 10.).ravel(), .02 * (x - q0[qa])]

        def jac(x):
            up(x); cols = []
            for bid in bids:
                jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
                mujoco.mj_jac(m, d, jp, jr, d.xpos[bid], bid); counts['jac'] += 1
                cols.append(jp[:, va] * 10.)
            return np.vstack(cols + [.02 * np.eye(42)])

        sol = least_squares(fun, np.clip(q0[qa], lo, hi), jac=jac, bounds=(lo, hi), max_nfev=25,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q = qr.copy(); q[qa] = sol.x; up(sol.x)
        foot_err = float(np.max(np.linalg.norm(d.xpos[bids] - tgt, axis=1)))
        Rm = d.xmat[thorax].reshape(3, 3)
        side = []
        for leg in LEGS:
            vals = [float((Rm.T @ (d.xpos[bb] - d.xpos[thorax]))[1]) * (1 if leg[0] == 'L' else -1)
                    for bb in range(m.nbody) if m.body(bb).name.startswith(leg)
                    and any(s in m.body(bb).name for s in ('Coxa', 'Trochanter', 'Femur', 'Tibia', 'Tarsus'))]
            side.append(min(vals))
        recenter(q)
        base, A = build_forces(q)
        blocks = []; contacts = []; nondist = []
        for c in range(d.ncon):
            co = d.contact[c]; a, bb = int(co.geom1), int(co.geom2)
            p1 = m.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE; p2 = m.geom_type[bb] == mujoco.mjtGeom.mjGEOM_PLANE
            if not (p1 or p2):
                continue
            g = bb if p1 else a; name = m.geom(g).name
            if any(name.startswith(l) and 'Tars' in name for l in LEGS):
                jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
                mujoco.mj_jac(m, d, jp, jr, co.pos, int(m.geom_bodyid[g])); counts['jac'] += 1
                blocks.append(jp.T); contacts.append(name)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                continue
            name = m.geom(g).name
            if any(name.startswith(l) and 'Tars' in name for l in LEGS):
                continue
            nondist.append(float(mujoco.mj_geomDistance(m, d, g, 0, 5.0, None)))
        feet = sorted({c[:2] for c in contacts})
        J = np.column_stack(blocks) if blocks else np.zeros((48, 0))
        E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
        lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        Lrow = E[lim].T if lim.size else np.zeros((48, 0))
        nf = int(sum(1 for x in nondist if x <= 0))
        row = dict(dz=dz, pitch_deg=pitch, roll_deg=roll, fore_aft=fore,
                   ik_success=bool(sol.success), foot_error=foot_err, feet=feet, n_contacts=len(contacts),
                   n_limit_rows=int(len(lim)), side_min=side, nonfoot_contacts=nf,
                   min_nonfoot_clearance=float(min(nondist)) if nondist else 99.)
        row['eligible'] = bool(len(feet) == 6 and nf == 0 and min(side) > 0 and foot_err < 1e-3
                               and np.all(q[qa] >= lo - 1e-6) and np.all(q[qa] <= hi + 1e-6))
        return row, q, base, A, J, Lrow, d.qfrc_bias.copy(), d.qfrc_passive.copy()

    rows = []; best = None
    try:
        for dz, p, r in stage_a:
            if counts['forward'] > CAP - 600:
                break
            counts['stage_a'] += 1
            row, q, base, A, J, Lrow, bias, passive = evaluate(dz, p, r, 0.)
            counts['lps'] += 1
            lp = static_lp(A, base, J, Lrow, bias, passive, weight, False, G)
            row['lp66_outer'] = lp
            row['feasible'] = bool(row['eligible'] and lp.get('residual', np.inf) <= 1e-6)
            row['qpos'] = q
            rows.append(row); dump(O / 'STAGE_A.json', rows)
            print(json.dumps({k: v for k, v in row.items() if k != 'qpos'}, default=str), flush=True)
            if row['feasible'] and best is None:
                best = dict(row=row, q=q, base=base, A=A, J=J, L=Lrow, bias=bias, passive=passive)
        if best is None:
            pool = [r for r in rows if r['eligible']]
            pool.sort(key=lambda r: r['lp66_outer'].get('residual', np.inf))
            picks = pool[:8]
            dump(O / 'STAGE_B_SELECTION.json', dict(rule='best eight eligible stage-A poses by trusted66 outer residual',
                                                     picked=[[r['dz'], r['pitch_deg'], r['roll_deg'], r['lp66_outer']['residual']] for r in picks]))
            for base_row in picks:
                for fore in (-.2, .2):
                    if counts['forward'] > CAP - 400:
                        break
                    counts['stage_b'] += 1
                    row, q, base, A, J, Lrow, bias, passive = evaluate(base_row['dz'], base_row['pitch_deg'], base_row['roll_deg'], fore)
                    row['stage'] = 'B'
                    lp = {}
                    for tag, H in (('trusted66', G), ('all90', np.eye(90))):
                        for cone, inner in (('outer', False), ('inner', True)):
                            counts['lps'] += 1
                            lp[f'{tag}_{cone}'] = static_lp(A, base, J, Lrow, bias, passive, weight, inner, H)
                    row['lp'] = lp
                    row['feasible'] = bool(row['eligible'] and lp['trusted66_outer'].get('residual', np.inf) <= 1e-6)
                    row['qpos'] = q
                    rows.append(row); dump(O / 'STAGE_B.json', rows)
                    print(json.dumps({k: v for k, v in row.items() if k != 'qpos'}, default=str), flush=True)
                    if row['feasible'] and best is None:
                        best = dict(row=row, q=q, base=base, A=A, J=J, L=Lrow, bias=bias, passive=passive)
    except BaseException as exc:
        dump(O / 'FAILURE.json', dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        allrows = rows
        feas = [r for r in allrows if r.get('feasible')]
        dump(O / 'SUMMARY.json', dict(candidates=int(counts['stage_a']) + int(counts['stage_b']),
                                      stage_a=counts['stage_a'], stage_b=counts['stage_b'],
                                      feasible_count=len(feas),
                                      best_66_outer=None if best is None else best['row'].get('lp66_outer', best['row'].get('lp', {}).get('trusted66_outer')),
                                      selected=None if best is None else dict(dz=best['row']['dz'], pitch_deg=best['row']['pitch_deg'],
                                                                              roll_deg=best['row']['roll_deg'], fore_aft=best['row']['fore_aft']),
                                      min_residual_66_outer=min([r['lp66_outer'].get('residual', np.inf) for r in allrows if 'lp66_outer' in r] or [None]),
                                      note='A negative result here bounds the declared domain only, not every possible pose.'))
        if best is not None:
            np.savez_compressed(O / 'CANDIDATE.npz', qpos=best['q'], qpos_hinges=best['q'][qa], qpos_addresses=qa,
                                active_force_matrix=best['A'], floor_force=best['base'],
                                contact_force_matrix=best['J'], joint_limit_force_matrix=best['L'], pool_matrix=G)
            dump(O / 'CANDIDATE_LEDGER.json', dict(parameters=dict(dz=best['row']['dz'], pitch_deg=best['row']['pitch_deg'],
                                                                   roll_deg=best['row']['roll_deg'], fore_aft=best['row']['fore_aft']),
                                                   geometry=dict(feet=best['row']['feet'], side_min=best['row']['side_min'],
                                                                 min_nonfoot_clearance=best['row']['min_nonfoot_clearance']),
                                                   lp=best['row'].get('lp', best['row'].get('lp66_outer')),
                                                   joint_names=[m.joint(int(j)).name for j in jid]))
        dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
        (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('COMPLETE', O, flush=True)


if __name__ == '__main__':
    main()
