"""How much tarsal adhesion would the standing pose need?

WHY THIS QUESTION. The canonical muscle body cannot statically balance its own
standing keyframe even with ideal instantaneous activation of all 90 muscles
(round 15: L1 residual 5.0156 of a 24.4464 body-weight scale; round 16 best
searched posture 3.7285). The XML has NO adhesion: contacts are unilateral
compressive Coulomb friction only (condim 3, mu 1.0), while a real fly stands and
climbs using tarsal adhesion (claws plus pulvilli). This script measures how much
adhesion would have to be available for that pose to become statically feasible.

WHAT THIS IS NOT.
  * The adhesion model here is a bounded TENSILE NORMAL FORCE added as an LP
    variable per contact. It is NOT a contact-mechanics model of pulvilli (no
    capillarity, no peeling, no real contact area, no rate dependence).
  * A positive result would identify a CANDIDATE PHYSICAL GAP. It would NOT
    justify silently editing the model, and it would NOT make the fly stand: a
    static force balance is not stability, not controllability, not behaviour.
  * Frozen-pose, linearized friction cone, actuator force affine in activation.
  * Coordinate signs are model coordinate signs, not flexion/extension labels.

TWO FRICTION COUPLINGS ARE SOLVED, ON PURPOSE.
  STRICT  : the tangential pyramid is tied to the COMPRESSIVE normal component
            only (sum_k f_k <= mu * n_compressive). A pulling contact therefore
            carries no tangential force. This is the conservative model.
  RELAXED : the round-15 generator form (n + mu*d_k) plus a separate tensile
            column, so the tangential bound is tied to (compressive + adhesion).
            A contact that is pulling can still carry tangential load, which is
            OVER-permissive, so its answer is a LOWER bound on what is needed.
The STRICT answer is the one to quote; RELAXED only shows how much the coupling
choice matters.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco as mj                                        # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402
import diag_static_support_feasibility as R15               # noqa: E402

XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
XML_SHA_EXPECTED = '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'
ROUND16 = ROOT / 'outputs/posture_search_round16/posture_search.json'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
OUT = ROOT / 'outputs/adhesion_round17'

K_DIRS = R15.CONE_DIRECTIONS          # 8 pyramid directions, same as round 15
LP_TL = R15.LP_TIME_LIMIT_S           # 60 s
LP_IT = R15.LP_MAXITER                # 20000
TOL = R15.FEASIBLE_SUM_TOL            # 1e-6 force units
SEAT_ITERS = 45
SEAT_TOL_MM = 1e-9

# Grid in BODY WEIGHTS of adhesion per contact. Chosen to bracket both the
# round-15/16 shortfall (<= ~1 body weight) and the literature range (1-2 body
# weights per contact), and to run far past it so "needs a lot" is measurable.
ADHESION_GRID_BW = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0,
                    3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0]
BISECT_ITERS = 26


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def unit_check(model):
    """Establish the force unit from the model itself plus the engine's own docs."""
    mass = float(mj.mj_getTotalmass(model))
    g = abs(float(model.opt.gravity[2]))
    weight = mass * g
    return {
        'length_unit': 'mm',
        'length_evidence': ("the generated XML header says 'the thorax at z=2.0453 mm'; FlyGym documents "
                            "'For numerical reasons, we simulate length in mm, not m. This changes the units of "
                            "other quantities as well, for example acceleration is now in mm/s^2'"),
        'gravity_mm_per_s2': g,
        'total_mass_model_units': mass,
        'total_weight_model_units': weight,
        'mass_unit_inference': 'gram',
        'mass_unit_reasoning': ('length is mm and gravity is 9801 (i.e. 9.801 m/s^2 expressed in mm/s^2), so a '
                                'mass unit of gram makes 1 force unit = 1 g*mm/s^2 = 1e-6 N = 1 uN and the model '
                                'fly weighs 2.494 mg, which is the right order for a fly. A mass unit of mg or kg '
                                'would make the model fly 2.5 ug or 2.5 g, i.e. a thousand-fold wrong.'),
        'force_unit_inference': 'uN (1 model force unit = 1 uN) under the gram/mm/s convention',
        'caveat': ('the XML does not declare a unit system; the gram reading is an inference from the engine '
                   'documentation plus physical plausibility, so every headline result below is ALSO given in '
                   'unit-free body-weight fractions, which do not depend on it.'),
        'body_weight_uN_if_gram': weight,
    }


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def keyframe_qpos(model, data):
    mj.mj_resetData(model, data)
    mj.mj_resetDataKeyframe(model, data, 0)
    return np.asarray(data.qpos, float).copy()


def seat_root(model, data, mujoco, qpos, tarsus_geoms, floor, target_mm):
    """Bisect the root z until the lowest tarsus point sits target_mm above the floor.

    Uses the exact signed geom distance (mj_geomDistance), the same technique the
    round-15 helper uses, because at shallow seating MuJoCo generates almost no
    contacts and a contact-based rule would be unstable.
    """
    def clearance(z):
        data.qpos[:] = qpos
        data.qpos[2] = z
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return R15.foot_clearance_mm(model, data, mujoco, tarsus_geoms, floor)

    base = float(qpos[2])
    lo, hi = base - 2.0, base + 2.0
    for _ in range(200):
        if clearance(lo) is not None and clearance(lo) < target_mm:
            break
        lo -= 0.25
    else:
        raise RuntimeError('no penetrating bracket found')
    for _ in range(200):
        if clearance(hi) is not None and clearance(hi) > target_mm:
            break
        hi += 0.25
    else:
        raise RuntimeError('no clear bracket found')
    for _ in range(SEAT_ITERS):
        mid = 0.5 * (lo + hi)
        d = clearance(mid)
        if d is not None and d <= target_mm:
            lo = mid
        else:
            hi = mid
        if hi - lo < SEAT_TOL_MM:
            break
    z = 0.5 * (lo + hi)
    return z, clearance(z)


def pose_keyframe(model, data):
    return keyframe_qpos(model, data), {'label': 'keyframe_as_shipped'}


def pose_round16_best(model, data, info):
    """Rebuild the round-16 best posture from its recorded, reproducible parameters."""
    q = keyframe_qpos(model, data)
    d_coxa = float(info['d_coxa_rad'])
    root_dz = float(info['root_dz_mm'])
    seat = float(info['seat_penetration_mm'])
    for leg in ('LF', 'RF', 'LM', 'RM', 'LH', 'RH'):
        adr = int(model.jnt_qposadr[model.joint(f'joint_{leg}Coxa_pitch').id])
        q[adr] += d_coxa
    tarsus = [g for g in range(model.ngeom)
              if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    floor = int(np.flatnonzero(model.geom_type == int(mj.mjtGeom.mjGEOM_PLANE))[0])
    z_seated, seat_residual = seat_root(model, data, mj, q, tarsus, floor, -seat)
    q[2] = z_seated + root_dz
    return q, {'label': 'round16_best_pose', 'rebuilt_from': 'round16 posture_search.json best_pose',
               'd_coxa_rad': d_coxa, 'root_dz_mm': root_dz, 'seat_penetration_mm': seat,
               'seated_root_z_mm': float(z_seated), 'seat_residual_mm': float(seat_residual)}


def evaluate_pose(model, data, qpos, sets, A_units, variants=('strict', 'relaxed')):
    """Frozen-pose evaluation: terms, R, target, and adhesion-dependent solves."""
    mj.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.act[:] = 0
    mj.mj_forward(model, data)

    M_dense = R15.mass_matrix(model, data, mj)
    efc_J = (np.asarray(data.efc_J, float).reshape(data.nefc, model.nv)
             if data.nefc else np.zeros((0, model.nv)))
    validation = {
        'mass_matrix_via': 'mj_mulM columns (mjData.M is sparse in this MuJoCo build)',
        'residual_Mqacc_vs_actuator_passive_bias_constraint': float(np.abs(
            M_dense @ np.asarray(data.qacc, float)
            - (np.asarray(data.qfrc_actuator) + np.asarray(data.qfrc_passive)
               - np.asarray(data.qfrc_bias) + np.asarray(data.qfrc_constraint))).max()),
        'efcJ_transpose_force_vs_qfrc_constraint': float(np.abs(
            efc_J.T @ np.asarray(data.efc_force, float)
            - np.asarray(data.qfrc_constraint)).max()),
        'qacc_max': float(np.abs(data.qacc).max()),
    }

    R, act0 = R15.actuator_moment_matrix(model, data, mj)
    mj.mj_forward(model, data)
    target = (np.asarray(data.qfrc_bias, float) - act0
              - np.asarray(data.qfrc_passive, float))
    terms = R15.contact_terms(model, data, mj)
    lims = R15.limit_terms(model, data, mj)
    B = R15.cone_basis(model, data, terms, K_DIRS)
    validation['contact_force_reconstruction_from_our_jacobians'] = \
        contact_force_reconstruction(model, data, mj, terms)

    facts = {
        'contact_count': len(terms),
        'contacts_are_feet_only_tarsus': bool(terms) and all(
            'Tarsus' in t['geom2'] or 'Tarsus' in t['geom1'] for t in terms),
        'contact_pairs': [f"{t['geom1']}|{t['geom2']}" for t in terms],
        'contact_dist_mm': [t['dist_mm'] for t in terms],
        'min_contact_dist_mm': (min(t['dist_mm'] for t in terms) if terms else None),
        'active_joint_limit_rows': len(lims),
        'thorax_height_model': float(data.xpos[model.body('Thorax').id, 2]),
        'root_z_model': float(qpos[2]),
        'qfrc_bias_absmax': float(np.abs(data.qfrc_bias).max()),
        'qfrc_passive_absmax': float(np.abs(data.qfrc_passive).max()),
        'target_absmax': float(np.abs(target).max()),
    }

    solves = {}
    for label, idx in sets.items():
        Rs = R[:, idx] if len(idx) else np.zeros((model.nv, 0))
        for variant in variants:
            key = f'{variant}:A={A_units:.6g}'
            solves[key] = solve_adhesion(Rs, target, terms, lims, B, model.nv, key,
                                         A_units, variant, minimize_usage=False)
    return facts, validation, solves, (R, target, terms, lims, B)


# --------------------------------------------------------------------------- #
# the LP
# --------------------------------------------------------------------------- #
def solve_adhesion(R_set, target, terms, lims, B, nv, label, A_units, variant,
                   minimize_usage):
    """Phase-1 residual (or minimum total adhesion) LP with bounded tensile normal force.

    Per contact the contact force is  (compressive normal) + (tangential) - (adhesion),
    with the adhesion magnitude s_c in [0, A_units] along -n.

    variant 'free3' is the decisive control: per contact, three SIGN-FREE and
    UNBOUNDED force components (normal, t1, t2) with no friction bound and no
    adhesion limit. That is strictly more permissive than any real friction or
    adhesion model, so a non-zero residual under 'free3' cannot be fixed by any
    amount of adhesion.
    """
    m_act = R_set.shape[1]
    n_cont = len(terms)
    cols = [R_set] if m_act else []
    ub_rows, ub_rhs = [], []

    if variant == 'free3':
        for t in terms:
            J = t['J']
            cols.append(J.T @ t['normal'].reshape(-1, 1))
            cols.append(J.T @ t['tangent1'].reshape(-1, 1))
            cols.append(J.T @ t['tangent2'].reshape(-1, 1))
        n_adh = 0
    elif variant == 'relaxed':
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b)
            ub_rows.append(None)
            ub_rhs.append(0.0)
        n_adh = n_cont
    else:
        for t, b in zip(terms, B):
            J, n, mu = t['J'], t['normal'], t['mu']
            col_n = (J.T @ n).reshape(-1, 1)
            idx_n = sum(c.shape[1] for c in cols)
            cols.append(col_n)
            dirs = np.column_stack([b[:, j] - n for j in range(b.shape[1])])
            tan = np.column_stack([J.T @ dirs[:, j] for j in range(b.shape[1])])
            idx_t0 = sum(c.shape[1] for c in cols)
            cols.append(tan)
            idx_t1 = sum(c.shape[1] for c in cols)
            row = np.zeros(idx_t1)
            row[idx_t0:idx_t1] = 1.0
            row[idx_n] = -mu
            ub_rows.append(row); ub_rhs.append(0.0)
        n_adh = n_cont

    if variant != 'free3':
        for t in terms:
            cols.append((-A_units * (t['J'].T @ t['normal'])).reshape(-1, 1))

    for l in lims:
        cols.append(l['direction'].reshape(-1, 1))

    A_struct = np.hstack(cols) if cols else np.zeros((nv, 0))
    n_struct = A_struct.shape[1]
    adh_lo = n_struct - n_adh - len(lims)
    adh_hi = n_struct - len(lims)

    if variant == 'free3':
        contact_bounds = [(None, None)] * (3 * n_cont)
    elif variant == 'relaxed':
        contact_bounds = [b for _ in terms for b in [(0.0, None)] * K_DIRS]
    else:
        contact_bounds = []
        for _ in terms:
            contact_bounds.append((0.0, None))
            contact_bounds.extend([(0.0, None)] * K_DIRS)

    if minimize_usage:
        A_eq = A_struct
        c = np.zeros(n_struct)
        c[adh_lo:adh_hi] = 1.0
        bounds = ([(0.0, 1.0)] * m_act + contact_bounds
                  + [(0.0, 1.0)] * n_adh + [(0.0, None)] * len(lims))
    else:
        A_eq = np.hstack([A_struct, np.eye(nv), -np.eye(nv)])
        c = np.zeros(A_eq.shape[1])
        c[n_struct:] = 1.0
        bounds = ([(0.0, 1.0)] * m_act + contact_bounds
                  + [(0.0, 1.0)] * n_adh + [(0.0, None)] * len(lims)
                  + [(0.0, None)] * 2 * nv)

    A_ub = b_ub = None
    if variant == 'strict':
        A_ub = np.zeros((len(ub_rows), A_eq.shape[1]))
        for i, r in enumerate(ub_rows):
            A_ub[i, :len(r)] = r
        b_ub = np.asarray(ub_rhs)

    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=target, bounds=bounds,
                  method='highs', options={'presolve': True, 'time_limit': LP_TL,
                                           'maxiter': LP_IT})
    dt = time.time() - t0
    if res.x is None:
        return {'label': label, 'variant': variant, 'A_units': float(A_units),
                'usable': False, 'status': int(res.status), 'message': res.message,
                'minimize_usage': bool(minimize_usage), 'solver_seconds': dt}

    x = res.x
    if minimize_usage:
        residual = np.zeros(nv)
    else:
        residual = x[n_struct:n_struct + nv] - x[n_struct + nv:]
    per_contact = [float(v) * A_units for v in x[adh_lo:adh_hi]] if n_adh else []
    return {
        'label': label, 'variant': variant, 'A_units': float(A_units),
        'minimize_usage': bool(minimize_usage), 'usable': True, 'status': int(res.status),
        'solver_seconds': float(dt),
        'objective_L1_residual': float(np.abs(residual).sum()),
        'max_abs_residual': float(np.abs(residual).max()),
        'residual_per_dof': residual.tolist(),
        'total_adhesion_units': float(sum(per_contact)) if n_adh else 0.0,
        'adhesion_used_per_contact_units': per_contact,
        'adhesion_limit_units': float(A_units),
        'feasible_within_tol': bool(np.abs(residual).sum() < TOL),
    }


def contact_force_reconstruction(model, data, mujoco, terms):
    """Two independent checks that our contact parameterization is MuJoCo's own.

    MuJoCo's per-contact efc rows are not literally (J^T n, J^T t1, J^T t2); they are
    a linearly re-scaled basis of the same 3-D per-contact force space. So the right
    check is not row equality but (a) the flat-array identity efc_J @ qvel == efc_vel
    and (b) that MuJoCo's three rows for each contact lie exactly in the span of our
    three contact-frame rows, related by an invertible map.
    """
    if not data.nefc or not terms:
        return {'note': 'no constraints at this pose'}
    Jefc = np.asarray(data.efc_J, float).reshape(data.nefc, model.nv)
    qvel = np.asarray(data.qvel, float).copy()
    data.qvel[:] = np.arange(model.nv) * 1e-3
    mujoco.mj_forward(model, data)
    ident = float(np.abs(Jefc @ np.asarray(data.qvel, float)
                         - np.asarray(data.efc_vel, float)).max())
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    Jefc = np.asarray(data.efc_J, float).reshape(data.nefc, model.nv)
    worst, dets = 0.0, []
    for t in terms:
        a = int(data.contact[t['index']].efc_address)
        if a < 0 or a + 2 >= data.nefc:
            continue
        block = Jefc[a:a + 3].T
        mine = np.column_stack([t['J'].T @ t['normal'], t['J'].T @ t['tangent1'],
                                t['J'].T @ t['tangent2']])
        coef, *_ = np.linalg.lstsq(mine, block, rcond=None)
        worst = max(worst, float(np.abs(mine @ coef - block).max()))
        dets.append(abs(float(np.linalg.det(coef))))
    efc_recon = float(np.abs(
        Jefc.T @ np.asarray(data.efc_force, float)
        - np.asarray(data.qfrc_constraint, float)).max())
    return {'identity_efcJ_qvel_equals_efcvel': ident,
            'worst_residual_projecting_mujoco_rows_into_our_3d_span': worst,
            'min_abs_det_of_the_map': (min(dets) if dets else None),
            'max_abs_det_of_the_map': (max(dets) if dets else None),
            'efcJ_transpose_force_vs_qfrc_constraint': efc_recon,
            'contacts_checked': len(dets),
            'meaning': ('the two parameterizations span the same per-contact 3-D force space '
                        '(projection residual at machine precision, invertible map), so the LP '
                        'contact force set is MuJoCo-s own set, expressed in the contact frame')}


def foot_groups(terms):
    """Group contacts by leg, because adhesion is produced at one pretarsus per leg."""
    groups = {}
    for i, t in enumerate(terms):
        nm = t['geom2'] if 'Tarsus' in t['geom2'] else t['geom1']
        leg = nm[:2]
        groups.setdefault(leg, []).append(i)
    return groups


def solve_footcap(R_set, target, terms, lims, B, nv, label, F_units):
    """STRICT cone plus a per-LEG adhesion budget F_units (tensile normal force).

    This is the version that can be compared directly with a measured per-pretarsus
    adhesion force: every contact may pull by at most F and the pulling forces of all
    contacts on one leg may sum to at most F.
    """
    m_act = R_set.shape[1]
    n_cont = len(terms)
    cols = [R_set] if m_act else []
    ub_rows = []

    for t, b in zip(terms, B):
        J, n, mu = t['J'], t['normal'], t['mu']
        idx_n = sum(c.shape[1] for c in cols)
        cols.append((J.T @ n).reshape(-1, 1))
        dirs = np.column_stack([b[:, j] - n for j in range(b.shape[1])])
        idx_t0 = sum(c.shape[1] for c in cols)
        cols.append(np.column_stack([J.T @ dirs[:, j] for j in range(b.shape[1])]))
        idx_t1 = sum(c.shape[1] for c in cols)
        row = np.zeros(idx_t1)
        row[idx_t0:idx_t1] = 1.0
        row[idx_n] = -mu
        ub_rows.append(row)

    adh_lo = sum(c.shape[1] for c in cols)
    for t in terms:
        cols.append((-F_units * (t['J'].T @ t['normal'])).reshape(-1, 1))
    adh_hi = sum(c.shape[1] for c in cols)

    for l in lims:
        cols.append(l['direction'].reshape(-1, 1))

    A_struct = np.hstack(cols)
    n_struct = A_struct.shape[1]
    A_eq = np.hstack([A_struct, np.eye(nv), -np.eye(nv)])
    c = np.zeros(A_eq.shape[1])
    c[n_struct:] = 1.0

    groups = foot_groups(terms)
    cap_rows = []
    for leg, idxs in groups.items():
        row = np.zeros(A_eq.shape[1])
        for i in idxs:
            row[adh_lo + i] = 1.0
        cap_rows.append(row)
    cone_rows = []
    for row in ub_rows:
        padded = np.zeros(A_eq.shape[1])
        padded[:len(row)] = row
        cone_rows.append(padded)
    A_ub = np.vstack(cone_rows + cap_rows)
    b_ub = np.concatenate([np.zeros(len(cone_rows)), np.ones(len(cap_rows))])

    bounds = ([(0.0, 1.0)] * m_act
              + [b for _ in terms for b in [(0.0, None)] + [(0.0, None)] * K_DIRS]
              + [(0.0, 1.0)] * n_cont + [(0.0, None)] * len(lims)
              + [(0.0, None)] * 2 * nv)

    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=target, bounds=bounds,
                  method='highs', options={'presolve': True, 'time_limit': LP_TL,
                                           'maxiter': LP_IT})
    dt = time.time() - t0
    if res.x is None:
        return {'label': label, 'F_units': float(F_units), 'usable': False,
                'status': int(res.status), 'message': res.message, 'solver_seconds': dt}
    x = res.x
    residual = x[n_struct:n_struct + nv] - x[n_struct + nv:]
    s = x[adh_lo:adh_hi]
    per_leg = {leg: float(sum(s[i] for i in idxs) * F_units) for leg, idxs in groups.items()}
    return {'label': label, 'F_units': float(F_units), 'usable': True,
            'status': int(res.status), 'solver_seconds': float(dt),
            'objective_L1_residual': float(np.abs(residual).sum()),
            'max_abs_residual': float(np.abs(residual).max()),
            'feasible_within_tol': bool(np.abs(residual).sum() < TOL),
            'total_adhesion_units': float(sum(per_leg.values())),
            'adhesion_per_leg_units': per_leg,
            'n_contacts': n_cont, 'n_legs': len(groups)}


def sweep_footcap(sets_R, sets, model, bw_scale, budgets_bw):
    """Minimum per-leg adhesion budget that makes the pose exactly statically feasible."""
    out = []
    for label in ('all90', 'candidate44'):
        idx = sets[label]
        Rs = sets_R[0][:, idx] if len(idx) else np.zeros((model.nv, 0))
        rows, feasible_F = [], None
        for bw in budgets_bw:
            F = bw * bw_scale
            r = solve_footcap(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                              f'{label}|footcap|F={F:.6g}', F)
            rows.append({'per_leg_budget_bw': bw, 'per_leg_budget_units': F,
                         'usable': r['usable'],
                         'objective_L1_residual': r.get('objective_L1_residual'),
                         'feasible_within_tol': r.get('feasible_within_tol', False)})
            if r.get('feasible_within_tol') and feasible_F is None:
                feasible_F = bw
        entry = {'set': label, 'rows': rows}
        if feasible_F is not None:
            lo = 0.0
            for row in rows:
                if row['per_leg_budget_bw'] < feasible_F and not row['feasible_within_tol']:
                    lo = row['per_leg_budget_bw']
            hi = feasible_F
            for _ in range(BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                r = solve_footcap(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                  f'{label}|footcap|bisect', mid * bw_scale)
                if r['usable'] and r['feasible_within_tol']:
                    hi = mid
                else:
                    lo = mid
            r_hi = solve_footcap(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                 f'{label}|footcap|verify', hi * bw_scale)
            r_lo = solve_footcap(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                 f'{label}|footcap|verify', lo * bw_scale)
            entry.update({'feasible': True,
                          'min_per_leg_budget_bw': hi,
                          'min_per_leg_budget_units': hi * bw_scale,
                          'largest_infeasible_budget_bw': lo,
                          'max_feasible_residual_L1': r_hi['objective_L1_residual'],
                          'min_infeasible_residual_L1': r_lo['objective_L1_residual'],
                          'adhesion_per_leg_units_at_min': r_hi['adhesion_per_leg_units'],
                          'adhesion_per_leg_bw_at_min': {k: v / bw_scale for k, v in
                                                         r_hi['adhesion_per_leg_units'].items()},
                          'total_adhesion_at_min_bw': r_hi['total_adhesion_units'] / bw_scale,
                          'n_legs': r_hi['n_legs'], 'n_contacts': r_hi['n_contacts']})
        else:
            entry.update({'feasible': False,
                          'reason': ('no per-leg budget up to %.3g body weights made the pose exactly '
                                     'feasible' % budgets_bw[-1])})
        out.append(entry)
    return out


def sweep_and_bisect(model, data, qpos, sets, bw_scale, variants, sets_R=None):
    """Residual vs adhesion limit, then bisect the smallest feasible limit."""
    out = {'grid': [], 'bisection': {}}
    A_vals = [bw * bw_scale for bw in ADHESION_GRID_BW]
    feasible_A = {}
    for label, idx in sets.items():
        Rs = sets_R[0][:, idx] if len(idx) else np.zeros((model.nv, 0))
        for variant in variants:
            rows = []
            for bw, A in zip(ADHESION_GRID_BW, A_vals):
                r = solve_adhesion(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4],
                                   model.nv, f'{label}|{variant}|A={A:.6g}', A, variant,
                                   minimize_usage=False)
                rows.append({'adhesion_per_contact_bw': bw, 'adhesion_per_contact_units': A,
                             'usable': r['usable'],
                             'objective_L1_residual': r.get('objective_L1_residual'),
                             'max_abs_residual': r.get('max_abs_residual'),
                             'feasible_within_tol': r.get('feasible_within_tol', False)})
            out['grid'].append({'set': label, 'variant': variant, 'rows': rows})
            feas = [row['adhesion_per_contact_bw'] for row in rows if row['feasible_within_tol']]
            if not feas:
                out['bisection'][f'{label}|{variant}'] = {
                    'feasible': False,
                    'reason': ('no grid value up to %.3g body weights per contact made the residual '
                               'fall below the tolerance' % ADHESION_GRID_BW[-1])}
                feasible_A[(label, variant)] = None
                continue
            hi_bw = min(feas)
            i = [r['adhesion_per_contact_bw'] for r in rows].index(hi_bw)
            lo_bw = rows[i - 1]['adhesion_per_contact_bw'] if i > 0 else 0.0
            lo_A, hi_A = lo_bw * bw_scale, hi_bw * bw_scale
            for _ in range(BISECT_ITERS):
                mid = 0.5 * (lo_A + hi_A)
                r = solve_adhesion(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                   f'{label}|{variant}|bisect', mid, variant, minimize_usage=False)
                if r['usable'] and r['feasible_within_tol']:
                    hi_A = mid
                else:
                    lo_A = mid
            # the largest verified-infeasible and smallest verified-feasible limits
            r_hi = solve_adhesion(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                  f'{label}|{variant}|verify', hi_A, variant, minimize_usage=False)
            r_lo = solve_adhesion(Rs, sets_R[1], sets_R[2], sets_R[3], sets_R[4], model.nv,
                                  f'{label}|{variant}|verify', lo_A, variant, minimize_usage=False)
            out['bisection'][f'{label}|{variant}'] = {
                'feasible': True,
                'min_feasible_units': float(hi_A),
                'min_feasible_body_weights_per_contact': float(hi_A / bw_scale),
                'largest_infeasible_units': float(lo_A),
                'largest_infeasible_body_weights_per_contact': float(lo_A / bw_scale),
                'max_feasible_residual_L1': r_hi['objective_L1_residual'],
                'min_infeasible_residual_L1': r_lo['objective_L1_residual'],
                'verified': bool(r_hi['feasible_within_tol'] and not r_lo['feasible_within_tol']),
            }
            feasible_A[(label, variant)] = hi_A
    return out, feasible_A, A_vals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert digest(XML) == XML_SHA_EXPECTED, 'canonical XML changed'
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0   # in memory only: permit a genuine zero command

    units = unit_check(model)
    bw_scale = units['total_weight_model_units']

    manifest = json.loads(MANIFEST.read_text())
    names = [model.actuator(i).name or '' for i in range(model.nu)]
    index_of = {}
    for i, n in enumerate(names):
        index_of.setdefault(n, i)
    cands = [r['actuator'] for r in manifest['actuators']
             if r['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    cand_idx = [index_of[c] for c in cands if c in index_of]
    sets = {'all90': list(range(model.nu)), 'candidate44': cand_idx}

    r16 = json.loads(ROUND16.read_text())
    best = r16['best_pose']
    recorded_L1 = float(best['all90_L1'])

    report = {
        'status': 'ADHESION_REQUIREMENT_FROZEN_POSE_LP',
        'what_this_is': ('frozen-pose static force-balance feasibility with a bounded tensile normal force per '
                         'contact (adhesion) added as an LP variable; NOT a contact-mechanics model of pulvilli, '
                         'NOT a controller, NOT a behaviour, NOT standing'),
        'limitations': [
            'adhesion is modelled as a bounded tensile NORMAL force, not real pulvillus mechanics '
            '(no capillarity, peeling, contact area or rate dependence)',
            'friction cone LINEARIZED (K=%d directions)' % K_DIRS,
            'actuator generalized force treated as affine in activation at a FROZEN pose',
            'force-length, force-velocity and moment arms frozen at a single instant',
            'the RELAXED coupling ties the tangential bound to (compressive + adhesion), which is '
            'over-permissive; its answer is a lower bound, the STRICT coupling is the conservative one',
            'static feasibility is NOT stability, NOT controllability, NOT a controller, NOT standing',
            'a positive result identifies a CANDIDATE PHYSICAL GAP only; it does not justify editing the model, '
            'and the adhesion used here is not attributable to any measured fly structure',
        ],
        'solver_caps': {'linprog_method': 'highs', 'time_limit_s': LP_TL, 'maxiter': LP_IT,
                        'cone_directions': K_DIRS, 'feasible_sum_tolerance': TOL,
                        'bisection_iterations': BISECT_ITERS, 'seat_bisection_iterations': SEAT_ITERS},
        'source_sha256': {str(XML): digest(XML), str(ROUND16): digest(ROUND16),
                          str(MANIFEST): digest(MANIFEST),
                          str(HERE / 'diag_static_support_feasibility.py'):
                              digest(HERE / 'diag_static_support_feasibility.py')},
        'units': units,
        'body_weight_scale_units': bw_scale,
        'adhesion_grid_body_weights_per_contact': ADHESION_GRID_BW,
        'poses': {},
    }

    for name, builder in (('keyframe_as_shipped', lambda: pose_keyframe(model, data)),
                          ('round16_best_pose', lambda: pose_round16_best(model, data, best))):
        qpos, pinfo = builder()
        facts, validation, _, ctx = evaluate_pose(model, data, qpos, {'all90': sets['all90']},
                                                 0.0, variants=('strict',))
        sweep, feasible_A, A_vals = sweep_and_bisect(model, data, qpos, sets, bw_scale,
                                                     ('strict', 'relaxed'), sets_R=ctx)
        entry = dict(pinfo, **facts)
        entry['equilibrium_self_validation'] = validation
        entry['adhesion_sweep'] = sweep
        # DECISIVE CONTROL: unbounded, sign-free 3-component contact force per contact.
        # Strictly more permissive than any friction or adhesion model in this class,
        # so a non-zero residual here cannot be removed by any amount of adhesion.
        free = {}
        for label, idx in sets.items():
            Rs = ctx[0][:, idx] if len(idx) else np.zeros((model.nv, 0))
            rf = solve_adhesion(Rs, ctx[1], ctx[2], ctx[3], ctx[4], model.nv,
                                f'{label}|free3', 0.0, 'free3', minimize_usage=False)
            free[label] = {'usable': rf['usable'],
                           'objective_L1_residual': rf.get('objective_L1_residual'),
                           'max_abs_residual': rf.get('max_abs_residual'),
                           'residual_per_dof': rf.get('residual_per_dof'),
                           'feasible_within_tol': rf.get('feasible_within_tol', False),
                           'solver_seconds': rf.get('solver_seconds')}
        entry['unbounded_contact_force_control'] = free
        entry['actuator_range_per_dof_units'] = {
            'all90_negative': np.minimum(ctx[0], 0).sum(axis=1).tolist(),
            'all90_positive': np.maximum(ctx[0], 0).sum(axis=1).tolist(),
        }
        # minimum total adhesion usage at the smallest feasible limit, per set and variant
        usage = {}
        for (label, variant), A in feasible_A.items():
            if A is None:
                usage[f'{label}|{variant}'] = {'feasible': False}
                continue
            idx = sets[label]
            Rs = ctx[0][:, idx] if len(idx) else np.zeros((model.nv, 0))
            rec = None
            for bump in (1.0, 1.001, 1.01, 1.05, 1.2):
                A_use = A * bump
                r = solve_adhesion(Rs, ctx[1], ctx[2], ctx[3], ctx[4], model.nv,
                                   f'{label}|{variant}|minusage', A_use, variant,
                                   minimize_usage=True)
                if r['usable'] and r['max_abs_residual'] < TOL:
                    rec = {'feasible': True,
                           'A_at_limit_units': A_use,
                           'A_at_limit_body_weights_per_contact': A_use / bw_scale,
                           'A_bump_factor_from_bisection': bump,
                           'min_total_adhesion_units': r['total_adhesion_units'],
                           'min_total_adhesion_body_weights': r['total_adhesion_units'] / bw_scale,
                           'n_contacts': len(ctx[2]),
                           'min_adhesion_per_foot_units': r['total_adhesion_units'] / max(len(ctx[2]), 1),
                           'min_adhesion_per_foot_body_weights': r['total_adhesion_units'] /
                                                                 max(len(ctx[2]), 1) / bw_scale,
                           'per_contact_units': r['adhesion_used_per_contact_units'],
                           'per_contact_body_weights': [v / bw_scale for v in
                                                        r['adhesion_used_per_contact_units']],
                           'max_abs_residual': r['max_abs_residual'],
                           'solver_seconds': r['solver_seconds']}
                    break
            usage[f'{label}|{variant}'] = rec or {
                'feasible': False,
                'reason': ('the tol-feasible limit from the slack LP was not exactly feasible without '
                           'slacks even after a 20 percent bump')}
        entry['minimum_adhesion_usage'] = usage
        # How the MINIMUM TOTAL adhesion depends on the per-contact limit that a real
        # foot could supply. A larger limit is a weaker physical assumption, so the
        # smallest total over the sweep is the least-adhesion solution this model allows.
        trade = []
        for bw_lim in (1.33021666482091, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 40.0):
            A = bw_lim * bw_scale
            row = {'per_contact_limit_bw': bw_lim, 'per_contact_limit_units': A}
            for label in ('all90', 'candidate44'):
                idx = sets[label]
                Rs = ctx[0][:, idx] if len(idx) else np.zeros((model.nv, 0))
                r = solve_adhesion(Rs, ctx[1], ctx[2], ctx[3], ctx[4], model.nv,
                                   f'{label}|trade|A={A:.6g}', A, 'strict', minimize_usage=True)
                if r['usable'] and r['max_abs_residual'] < TOL:
                    per_foot = {}
                    for t, v in zip(ctx[2], r['adhesion_used_per_contact_units']):
                        gname = t['geom2'] if 'Tarsus' in t['geom2'] else t['geom1']
                        per_foot[gname] = per_foot.get(gname, 0.0) + v
                    row[label] = {
                        'feasible': True,
                        'min_total_adhesion_units': r['total_adhesion_units'],
                        'min_total_adhesion_body_weights': r['total_adhesion_units'] / bw_scale,
                        'per_tarsus_units': per_foot,
                        'per_tarsus_body_weights': {k: v / bw_scale for k, v in per_foot.items()},
                        'max_per_tarsus_body_weights': (max(per_foot.values()) / bw_scale
                                                        if per_foot else 0.0),
                        'n_contacts': len(ctx[2]),
                    }
                else:
                    row[label] = {'feasible': False,
                                  'max_abs_residual': r.get('max_abs_residual')}
            trade.append(row)
        entry['minimum_total_adhesion_vs_per_contact_limit'] = trade
        entry['per_leg_adhesion_budget_sweep'] = sweep_footcap(
            ctx, sets, model, bw_scale, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0])
        report['poses'][name] = entry
        print(name, 'contacts', facts['contact_count'],
              'target_absmax %.4f' % facts['target_absmax'], flush=True)
        for k, v in usage.items():
            if v.get('feasible'):
                print('   ', k, 'min feasible per-contact limit %.4f bw -> total usage %.4f bw '
                      '(%.4f unit/foot)' % (v['A_at_limit_body_weights_per_contact'],
                                            v['min_total_adhesion_body_weights'],
                                            v['min_adhesion_per_foot_units']), flush=True)
            else:
                print('   ', k, 'NOT feasible even at the top of the grid', flush=True)

    # reproduction check against the round-16 recorded residual
    pose16 = report['poses']['round16_best_pose']
    repro = {'recorded_round16_all90_L1': recorded_L1,
             'recomputed_all90_L1_at_A0': None, 'delta': None}
    for row in pose16['adhesion_sweep']['grid']:
        if row['set'] == 'all90' and row['variant'] == 'strict':
            a0 = [r for r in row['rows'] if r['adhesion_per_contact_bw'] == 0.0][0]
            repro['recomputed_all90_L1_at_A0'] = a0['objective_L1_residual']
            repro['delta'] = a0['objective_L1_residual'] - recorded_L1
    report['round16_reproduction_check'] = repro
    print('round16 reproduction delta', repro['delta'], flush=True)

    report['literature_reference'] = literature_reference()
    report['xml_sha256_after'] = digest(XML)

    (OUT / 'adhesion_requirement.json').write_text(json.dumps(report, indent=2))
    make_figure(report)
    assert digest(XML) == XML_SHA_EXPECTED, 'canonical XML changed during the run'
    print(OUT)


def literature_reference():
    """Comparable adhesion magnitudes from the engine itself and from the literature."""
    return {
        'framework_native_reference': {
            'source': 'FlyGym demonstration code shipped with the installed engine',
            'path': 'venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/common.py',
            'line': "adhesion_gain: float = 40.0  ->  fly.add_leg_adhesion(gain=adhesion_gain)",
            'meaning': ('the engine adds one adhesion actuator per tarsus5 with gainprm[0] = 40 and ctrl in '
                        '[0,1], so up to 40 model force units = 40 uN of normal attraction per leg at full '
                        'control, applied to every leg'),
            'per_leg_units': 40.0,
            'per_leg_body_weights': 40.0 / 24.446354755878,
            'six_leg_total_body_weights': 6 * 40.0 / 24.446354755878,
            'also_in_this_project': 'engine/embodied/scene.py: SceneConfig.adhesion_gain = 40.0 (same value)',
            'api_doc': ("'Adhesion actuators apply a normal attraction force, enabling the fly to grip surfaces. "
                        "The control input per leg ranges from 0 to 1, where 0 fully releases adhesion and 1 "
                        "applies the configured gain.'"),
            'model_note': ("FlyGym's musculoskeletal (FlyMimic) compose class states 'FlyMimic has no adhesion' "
                           "and ships leg_to_adhesionactuator empty, which is the model used here"),
        },
        'measured_biology_reference': {
            'source': 'Ramdya et al. 2017, Nature Communications 8:14494, doi:10.1038/ncomms14494, Methods',
            'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5321742/',
            'measurement_method': ('the authors did not measure force in newtons; they measured how many '
                                   'pretarsi of real D. melanogaster can hold the fly hanging inverted from a '
                                   'cotton substrate for more than 1 s: two pretarsi 10/10 flies, one pretarsus '
                                   '11/11 flies, no pretarsi 0/10 flies'),
            'derived_unit': ('one pretarsus is therefore sufficient to hold one body weight, so their 100% '
                             'adhesion level = 1 body weight per contact point'),
            'value_used_for_their_optimisation': ('200% adhesion, i.e. twice the insect model weight per '
                                                  'contacting leg, described as the minimal amount required for '
                                                  'gait optimisation to succeed in all travel orientations'),
            'per_leg_body_weights': [1.0, 2.0],
            'their_own_caveat': ('"Although leg adhesion forces have not been formally measured for '
                                 'D. melanogaster, this is likely to be a lower bound based on studies of other '
                                 'species"'),
            'their_implementation': ('constant normal forces acting on the pretarsus while in contact; with 200% '
                                     'adhesion each contacting leg experiences an added normal force equal to '
                                     'twice the weight of the model, and this also raises the friction bound'),
        },
        'comparison_note': ('both references therefore give 1-2 body weights per contacting foot (engine default '
                            '1.64, empirical suspension bound >=1, value used for optimisation 2.0). No direct '
                            'per-foot force measurement in newtons exists for D. melanogaster in either source.'),
    }


def make_figure(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    pose_style = {'keyframe_as_shipped': ('tab:red', 'o', 'XML keyframe as shipped'),
                  'round16_best_pose': ('tab:blue', 's', 'round-16 best posture')}
    set_style = {'all90': ('-', 'o'), 'candidate44': ('--', 's')}

    # panel 1: residual vs available adhesion per contact, with the unbounded-contact floor
    for pname, pose in report['poses'].items():
        col, mark, lab = pose_style[pname]
        floor = pose['unbounded_contact_force_control']['all90']['objective_L1_residual']
        axs[0].axhline(max(floor, 1e-12), color=col, ls=':', lw=1.2,
                       label=f'{lab}: floor with UNBOUNDED contact force')
        for block in pose['adhesion_sweep']['grid']:
            rows = [r for r in block['rows'] if r['usable']]
            ls, mk = set_style[block['set']]
            axs[0].plot([r['adhesion_per_contact_bw'] for r in rows],
                        [max(r['objective_L1_residual'], 1e-12) for r in rows],
                        ls, marker=mk, ms=3.5, color=col,
                        alpha=.5 if block['variant'] == 'relaxed' else 1.0,
                        label=f'{lab} | {block["set"]} | {block["variant"]}')
    axs[0].axvspan(1.0, 2.0, color='green', alpha=.12, label='measured fly 1-2 bw per foot')
    axs[0].axvline(1.644, color='green', ls='-.', label='engine default 1.64 bw per leg')
    axs[0].set_xscale('symlog', linthresh=0.05)
    axs[0].set_yscale('symlog', linthresh=1e-9)
    axs[0].set_xlabel('adhesion limit per contact (body weights)')
    axs[0].set_ylabel('L1 residual (model force units = uN)')
    axs[0].set_title('Residual vs available adhesion\nfree3 floor = best any contact model could do')
    axs[0].legend(fontsize=6.5); axs[0].grid(alpha=.3, which='both')

    # panel 2: per-leg adhesion budget, the value comparable with measured per-foot data
    xs = []
    for entry in report['poses']['round16_best_pose']['per_leg_adhesion_budget_sweep']:
        if entry['set'] != 'all90':
            continue
        rows = [r for r in entry['rows'] if r['usable']]
        xs = [r['per_leg_budget_bw'] for r in rows]
        axs[1].plot(xs, [max(r['objective_L1_residual'], 1e-12) for r in rows], 'o-',
                    color='tab:blue', label='round-16 best posture, all 90 actuators')
        if entry.get('feasible'):
            axs[1].axvline(entry['min_per_leg_budget_bw'], color='tab:blue', ls='--',
                           label='minimum per-leg budget %.2f bw' % entry['min_per_leg_budget_bw'])
    for entry in report['poses']['round16_best_pose']['per_leg_adhesion_budget_sweep']:
        if entry['set'] != 'candidate44':
            continue
        rows = [r for r in entry['rows'] if r['usable']]
        axs[1].plot([r['per_leg_budget_bw'] for r in rows],
                    [max(r['objective_L1_residual'], 1e-12) for r in rows], 's--',
                    color='tab:orange', label='round-16 best posture, 44 candidates')
    axs[1].axhline(TOL, color='k', ls=':', label='feasibility tolerance')
    axs[1].axvspan(1.0, 2.0, color='green', alpha=.12, label='measured fly 1-2 bw per foot')
    axs[1].axvline(1.644, color='green', ls='-.')
    axs[1].set_xscale('symlog', linthresh=0.25)
    axs[1].set_yscale('symlog', linthresh=1e-9)
    axs[1].set_xlabel('adhesion budget per LEG (body weights)')
    axs[1].set_ylabel('L1 residual')
    axs[1].set_title('Per-leg adhesion budget (comparable with measured\nper-pretarsus adhesion)')
    axs[1].legend(fontsize=7); axs[1].grid(alpha=.3, which='both')

    # panel 3: minimum total adhesion vs the per-contact limit
    for pname, pose in report['poses'].items():
        col, mark, lab = pose_style[pname]
        xs, ys = [], []
        for row in pose['minimum_total_adhesion_vs_per_contact_limit']:
            a = row.get('all90', {})
            if a.get('feasible'):
                xs.append(row['per_contact_limit_bw'])
                ys.append(a['min_total_adhesion_body_weights'])
        if xs:
            axs[2].plot(xs, ys, marker=mark, color=col,
                        label=f'{lab}: minimum TOTAL adhesion')
    axs[2].axhspan(6.0, 12.0, color='green', alpha=.12,
                   label='6 feet x measured 1-2 bw per foot = 6-12 bw total')
    axs[2].axhline(6 * 1.644, color='green', ls='-.', label='engine default, 6 legs')
    axs[2].set_xlabel('per-contact adhesion limit (body weights)')
    axs[2].set_ylabel('minimum TOTAL adhesion (body weights)')
    axs[2].set_title('Least adhesion that balances the pose\n(keyframe needs no finite amount)')
    axs[2].legend(fontsize=7); axs[2].grid(alpha=.3)

    fig.suptitle('How much tarsal adhesion would the standing pose need?  Bounded tensile normal force as an LP variable.\n'
                 'A CANDIDATE PHYSICAL GAP at most: not a contact-mechanics model of pulvilli, not a model edit, not standing.')
    fig.tight_layout()
    fig.savefig(OUT / 'adhesion_requirement.png', dpi=110)


if __name__ == '__main__':
    main()
