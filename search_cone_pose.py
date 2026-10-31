"""Pose search with the criterion that the residual attribution actually points to.

Attribution result (115000Z): at the settled stance the demand cannot be met by
NONNEGATIVE muscle activation at any magnitude (residual 0.0367 even with activation
allowed up to 13.45x full), while a signed relaxation reaches exactly 0. Muscles are
pull-only, so the achievable torque cone does not contain the demand: that is a
geometry/posture question, not an amplitude question.

This search therefore uses the magnitude-free criterion: with nonnegative UNBOUNDED
activation, contact forces inside the declared cone, and nonnegative active joint-limit
forces, is the demand inside the pull-only cone (residual <= 1e-6)? Bounded-activation
residual is reported as well, but selection uses the magnitude-free test.

Domain and budget are fixed in the pre-registration. Analytic actuator columns keep the
cost near 30 forwards per pose. No anatomy, Fmax, passive bias or neural change.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import mujoco
from scipy.optimize import linprog, least_squares
from flygym.compose.fly.musculoskeletal import _load_mjcf

REB = Path('/home/sensen/Desktop/engine修复验收/s5-muscle-geometry-rebuild/20261028T113000Z')
S74 = Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
O = Path('/home/sensen/Desktop/engine修复验收/s5-cone-pose-search/20261028T116000Z')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
CAP = 3000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                  else v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def lp(B, y, bind, nc, nl, n, inner=False):
    rws = [np.c_[B, -np.ones(48)], np.c_[-B, -np.ones(48)]]; rhs = [y, -y]
    for j in range(nc):
        mr = np.zeros((4, B.shape[1] + 1))
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] if inner else [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (sx, sy) in enumerate(dirs):
            mr[k, n + 3 * j:n + 3 * j + 3] = [sx, sy, -1.]
        rws.append(mr); rhs.append(np.zeros(4))
    obj = np.zeros(B.shape[1] + 1); obj[-1] = 1.
    bnd = list(bind) + [(None, None), (None, None), (0., None)] * nc + [(0., None)] * nl + [(0., None)]
    sol = linprog(obj, A_ub=np.vstack(rws), b_ub=np.concatenate(rhs), bounds=bnd, method='highs')
    return (float(sol.fun) if sol.success else float('inf')), sol.success


def main():
    O.mkdir(parents=True, exist_ok=False)
    started = time.time(); counts = dict(model_load=0, forward=0, jac=0, body_step=0, neural_step=0, lps=0, poses=0)
    mapping = json.loads((S74 / 'MAPPING.json').read_text())
    weight = float(json.loads((S74 / 'GAINS.json').read_text())['weight'])
    groups = {}
    for row in mapping['rows']:
        if row['local_indices']:
            groups.setdefault(tuple(row['pool_root_ids']), []).append(row['actuator_index'])
    G = np.zeros((90, len(groups)))
    for j, ids in enumerate(groups.values()):
        G[ids, j] = 1.
    q0 = np.load(S74 / 'HANDOVER_BEFORE.npz')['qpos'].copy()
    poses = [(dz, p, r, f) for dz in (-.4, -.2, 0., .2, .4) for p in (-3., 0., 3.) for r in (-3., 0., 3.) for f in (-.2, .2)]
    dump(O / 'PREREGISTRATION.json', dict(
        scope='Pose search with the magnitude-free pull-cone criterion at the rebuilt geometry. Static only, no integration, no neural run, no parameter change.',
        budget=dict(model_load=1, forward_cap=CAP, body_step=0, neural_step=0, poses=len(poses)),
        domain=dict(root_height_offset=[-.4, -.2, 0., .2, .4], pitch_deg=[-3., 0., 3.], roll_deg=[-3., 0., 3.], foot_fore_aft=[-.2, .2]),
        criterion=dict(primary='nonnegative UNBOUNDED activation: outer-cone residual <= 1e-6 (magnitude-free; tests whether the demand is inside the pull-only cone)',
                       secondary='bounded 0..1 activation residual, reported not used for selection',
                       geometry='six foot contacts, no non-foot contact, front legs not crossing midline, joint limits respected'),
        note='Selection order: smallest pull-cone residual, then smallest |dz|+|fore|; no post-hoc domain extension.',
        source_hashes={str(p): sha(p) for p in [Path(__file__), REB / 'MODEL_REBUILT.xml', S74 / 'HANDOVER_BEFORE.npz',
                                                S74 / 'GAINS.json', S74 / 'MAPPING.json']}))
    m = _load_mjcf(str(REB / 'MODEL_REBUILT.xml')).compile(); counts['model_load'] += 1
    d = mujoco.MjData(m)
    jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); va = np.array([m.jnt_dofadr[j] for j in jid])
    lo = m.jnt_range[jid, 0] + 1e-9; hi = m.jnt_range[jid, 1] - 1e-9
    bids = [m.body(l + 'Tarsus5').id for l in LEGS]
    thorax = m.body('Thorax').id
    lr0 = m.actuator_lengthrange.copy()

    def rot(pd_, rd):
        p = np.deg2rad(pd_); r = np.deg2rad(rd)
        cp, sp = np.cos(p / 2), np.sin(p / 2); cr, sr = np.cos(r / 2), np.sin(r / 2)
        w1, x1 = cp, sp; w2, y2 = cr, sr
        return np.array([w1 * w2, x1 * w2, w1 * y2, x1 * y2])

    def evaluate(dz, pitch, roll, fore):
        d.qpos[:] = q0; d.qvel[:] = 0; d.act[:] = .0001; d.ctrl[:] = 0
        mujoco.mj_forward(m, d); counts['forward'] += 1
        fp = d.xpos[bids].copy(); tgt = fp.copy(); tgt[:, 0] += fore
        qr = q0.copy(); qr[2] += dz; qr[3:7] = rot(pitch, roll)
        cache = {}

        def up(x):
            if 'x' not in cache or not np.array_equal(cache['x'], x):
                d.qpos[:] = qr; d.qpos[qa] = x; mujoco.mj_forward(m, d); counts['forward'] += 1
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
        side = [min(float((Rm.T @ (d.xpos[bb] - d.xpos[thorax]))[1]) * (1 if leg[0] == 'L' else -1)
                    for bb in range(m.nbody) if m.body(bb).name.startswith(leg)
                    and any(s in m.body(bb).name for s in ('Coxa', 'Trochanter', 'Femur', 'Tibia', 'Tarsus')))
                for leg in LEGS]
        # recentre length ranges at this pose (declared 089 calibration)
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
        Mo = np.zeros((m.nv, m.nu))
        if d.actuator_moment.size == m.nv * m.nu:
            Mo = d.actuator_moment.reshape(m.nv, m.nu).copy()
        else:
            for j in range(m.nu):
                adr = int(d.moment_rowadr[j]); nnz = int(d.moment_rownnz[j])
                Mo[d.moment_colind[adr:adr + nnz], j] = d.actuator_moment[adr:adr + nnz]
        d.act[:] = 1.; mujoco.mj_forward(m, d); counts['forward'] += 1
        f1 = d.actuator_force.copy()
        A = Mo * (f1 - d.actuator_force * 0)[None, :]  # placeholder replaced below
        d.act[:] = .0001; mujoco.mj_forward(m, d); counts['forward'] += 1
        f0 = d.actuator_force.copy()
        A = Mo * (f1 - f0)[None, :]
        blocks = []; geoms = []
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
        near = 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                continue
            nm = m.geom(g).name
            if any(nm.startswith(l) and 'Tars' in nm for l in LEGS):
                continue
            if float(mujoco.mj_geomDistance(m, d, g, 0, 5.0, None)) <= 0:
                near += 1
        J = np.column_stack(blocks) if blocks else np.zeros((48, 0))
        E = d.efc_J.reshape(d.nefc, m.nv).copy() if d.efc_J.size == d.nefc * m.nv else np.zeros((d.nefc, m.nv))
        lim = np.flatnonzero(np.asarray(d.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        Lrow = E[lim].T if lim.size else np.zeros((48, 0))
        nc = J.shape[1] // 3; nl = Lrow.shape[1]
        B = np.column_stack([A @ G, J, Lrow]) / weight
        y = (d.qfrc_bias - d.qfrc_passive - base) / weight
        counts['lps'] += 2
        free, ok1 = lp(B, y, [(0., None)] * 66, nc, nl, 66)
        bounded, ok2 = lp(B, y, [(0., 1.)] * 66, nc, nl, 66)
        feet = sorted({g[:2] for g in geoms})
        row = dict(dz=dz, pitch_deg=pitch, roll_deg=roll, fore_aft=fore, feet=feet, contacts=len(blocks),
                   near_nonfoot=near, side_min=side, foot_error=foot_err,
                   pull_cone_residual=free, bounded_residual=bounded)
        row['eligible'] = bool(len(feet) == 6 and near == 0 and min(side) > 0 and foot_err < 1e-3
                               and np.all(q[qa] >= lo - 1e-6) and np.all(q[qa] <= hi + 1e-6))
        row['pull_cone_feasible'] = bool(row['eligible'] and free <= 1e-6)
        row['qpos'] = q
        return row

    rows = []
    best = None
    try:
        for dz, p, r, f in poses:
            if counts['forward'] > CAP - 120:
                break
            counts['poses'] += 1
            row = evaluate(dz, p, r, f)
            rows.append(row); dump(O / 'POSES.json', rows)
            print(json.dumps({k: v for k, v in row.items() if k != 'qpos'}, default=str), flush=True)
            if row['pull_cone_feasible']:
                score = abs(dz) + abs(f)
                if best is None or score < best[0]:
                    best = (score, row)
    finally:
        fea = [x for x in rows if x['pull_cone_feasible']]
        dump(O / 'SUMMARY.json', dict(poses=int(counts['poses']),
                                      eligible=int(sum(x['eligible'] for x in rows)),
                                      pull_cone_feasible=len(fea),
                                      best_pull_cone=None if best is None else dict(dz=best[1]['dz'], pitch=best[1]['pitch_deg'],
                                                                                    roll=best[1]['roll_deg'], fore=best[1]['fore_aft'],
                                                                                    pull_cone_residual=best[1]['pull_cone_residual'],
                                                                                    bounded_residual=best[1]['bounded_residual'],
                                                                                    feet=best[1]['feet']),
                                      min_pull_cone_residual=min([x['pull_cone_residual'] for x in rows if x['eligible']] or [None]),
                                      min_bounded_residual=min([x['bounded_residual'] for x in rows if x['eligible']] or [None])))
        if best is not None:
            dump(O / 'SELECTED.json', dict(dz=best[1]['dz'], pitch_deg=best[1]['pitch_deg'], roll_deg=best[1]['roll_deg'],
                                           fore_aft=best[1]['fore_aft'], pull_cone_residual=best[1]['pull_cone_residual'],
                                           bounded_residual=best[1]['bounded_residual'], feet=best[1]['feet'],
                                           side_min=best[1]['side_min']))
            (O / 'CANDIDATE.npy').write_bytes(json.dumps({'qpos': best[1]['qpos'].tolist()}).encode())
        dump(O / 'EXECUTION.json', dict(counts=counts, runtime_s=time.time() - started, exit_code=0))
        (O / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('COMPLETE', O, flush=True)


if __name__ == '__main__':
    main()
