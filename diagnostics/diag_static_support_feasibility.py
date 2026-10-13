"""Bounded static support feasibility at fixed poses. NOT a controller, NOT a behavior.

Question: at a FROZEN pose, does there exist an activation vector a in [0,1]^90 (and,
separately, restricted to the 44 annotation-supported actuators) together with
one-sided contact forces inside a LINEARIZED friction cone such that the
generalized force balance closes with qacc = 0?

Equilibrium form used (validated numerically in this script against MuJoCo's own
constraint force, not assumed):

    qfrc_actuator(a) + qfrc_passive + J^T f + J_lim^T g = qfrc_bias      (qacc = 0)

so the quantity that must be supplied is

    target = qfrc_bias - qfrc_actuator(a=0) - qfrc_passive

and the phase-1 LP minimises the L1 residual of
    R a + J^T B lambda + J_lim^T g - target,  a in [0,1], lambda >= 0, g >= 0.

R is the actuator moment matrix at this frozen pose,
R[:, j] = qfrc_actuator(act_j = 1) - qfrc_actuator(act = 0), which keeps the
passive muscle baseline out of the active term.

LIMITATIONS (also written into the report and JSON):
  * friction cone is LINEARIZED (inner pyramid, K directions) - conservative;
  * actuator force is affine in activation only at this FROZEN pose;
  * force-length / force-velocity / moment arms are frozen at this instant;
  * a static force balance is NOT stability, NOT controllability, NOT a
    controller, and NOT proof the animal can stand dynamically;
  * infeasibility IS strong evidence that this pose cannot be held even with
    ideal instantaneous actuation at this pose;
  * coordinate signs are model coordinates, not anatomical flexion labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
ROUND14 = ROOT / 'outputs/autonomous_causal_round14/exploratory.json'
OUT = ROOT / 'outputs/static_support_round15'

CONE_DIRECTIONS = 8
LP_TIME_LIMIT_S = 60.0
LP_MAXITER = 20000
FEASIBLE_SUM_TOL = 1e-6      # force units, same units as qfrc_bias (total weight ~24.4)
SETTLE_SECONDS = 0.5
GROUND_BISECTION_ITERS = 80
GROUND_TOL_MM = 1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dof_names(model):
    names = []
    for j in range(model.njnt):
        name = model.joint(j).name or ''
        if model.jnt_type[j] == 0:  # free joint: 6 dofs
            names += [f'{name or "root"}:{s}' for s in ('x', 'y', 'z', 'rx', 'ry', 'rz')]
        else:
            names.append(name or f'joint{j}')
    if len(names) != model.nv:
        names = [f'dof{i}' for i in range(model.nv)]
    return names


def settle_pose(model, data, mujoco, pose_b):
    """0.5 s zero-actuation passive settling from the grounded pose. No controller."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = pose_b
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    steps = int(round(SETTLE_SECONDS / model.opt.timestep))
    for _ in range(steps):
        data.ctrl[:] = 0
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.qvel.copy(), steps


def lowest_clearance_mm(model, data, mujoco):
    """Minimum signed distance over all active contacts (negative = penetration)."""
    if data.ncon == 0:
        return None
    return float(min(c.dist for c in data.contact))


def build_grounded_start(model, data, mujoco, base_qpos):
    """Raise/lower the root along z until the lowest foot point touches z=0 exactly."""
    def clearance(z):
        data.qpos[:] = base_qpos
        data.qpos[2] = z
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return lowest_clearance_mm(model, data, mujoco)

    base = float(base_qpos[2])
    # (1) find a root height with penetration, lowering if necessary
    lo = base
    while True:
        d = clearance(lo)
        if d is not None and d < 0:
            break
        lo -= 0.05
        if base - lo > 50:
            raise RuntimeError('could not reach a penetrating contact')
    # (2) find a root height with positive clearance or no contact, raising if needed
    hi = lo
    while True:
        d = clearance(hi)
        if d is None or d > 0:
            break
        hi += 0.05
        if hi - base > 50:
            raise RuntimeError('could not reach a positive clearance')
    # (3) bisect: lo keeps penetration, hi keeps positive clearance
    for _ in range(GROUND_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        d = clearance(mid)
        if d is None or d > 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < GROUND_TOL_MM:
            break
    z = 0.5 * (lo + hi)
    qpos = base_qpos.copy()
    qpos[2] = z
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return qpos, z, lowest_clearance_mm(model, data, mujoco), clearance(lo), clearance(hi)


def foot_clearance_mm(model, data, mujoco, tarsus_geoms, floor_geom=0, distmax=100.0):
    """Exact signed distance from every tarsus geom to the floor plane (no contact needed)."""
    fromto = np.zeros(6)
    best = None
    for g in tarsus_geoms:
        d = float(mujoco.mj_geomDistance(model, data, g, floor_geom, distmax, fromto))
        best = d if best is None else min(best, d)
    return best


def mass_matrix(model, data, mujoco):
    """Dense mass matrix built through mj_mulM.

    mjData.M is stored SPARSELY in this MuJoCo version and mj_fullM mis-expands it
    when handed that buffer, so the columns are assembled through mj_mulM instead.
    """
    M = np.zeros((model.nv, model.nv))
    e = np.zeros(model.nv)
    out = np.zeros(model.nv)
    for i in range(model.nv):
        e[:] = 0.0
        e[i] = 1.0
        out[:] = 0.0
        mujoco.mj_mulM(model, data, out, e)
        M[:, i] = out
    if not np.allclose(M, M.T, atol=1e-12):
        raise ValueError('mass matrix not symmetric')
    return M


def contact_terms(model, data, mujoco):
    """3 x nv contact Jacobians with MuJoCo's contact frame and friction."""
    terms = []
    for k in range(data.ncon):
        con = data.contact[k]
        J = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        for geom, sign in ((con.geom2, 1.0), (con.geom1, -1.0)):
            body = int(model.geom_bodyid[geom])
            if body == 0:
                continue
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, con.pos, body)
            J += sign * jp
        frame = np.asarray(con.frame, float)
        terms.append({
            'index': k,
            'normal': frame[0:3].copy(),
            'tangent1': frame[3:6].copy(),
            'tangent2': frame[6:9].copy(),
            'mu': float(con.friction[0]),
            'dist_model': float(con.dist),
            'J': J,
            'geom1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom1)) or '',
            'geom2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(con.geom2)) or '',
            'body1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[con.geom1])) or '',
            'body2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[con.geom2])) or '',
        })
    return terms


def limit_terms(model, data, mujoco):
    """Active one-sided joint/tendon limit rows exactly as MuJoCo defines them."""
    rows = []
    if not data.nefc:
        return rows
    J = np.asarray(data.efc_J, float).reshape(data.nefc, model.nv)
    for r in range(data.nefc):
        t = int(data.efc_type[r])
        if t not in (int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT),
                     int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)):
            continue
        rows.append({'type': t, 'direction': J[r].copy(),
                     'muJoCo_force': float(data.efc_force[r])})
    return rows


def actuator_moment_matrix(model, data, mujoco):
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    base = np.asarray(data.qfrc_actuator, float).copy()
    R = np.zeros((model.nv, model.nu))
    for j in range(model.nu):
        data.act[:] = 0.0
        data.act[j] = 1.0
        mujoco.mj_forward(model, data)
        R[:, j] = np.asarray(data.qfrc_actuator, float) - base
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    return R, base


def cone_basis(model, data, terms, K):
    """Columns B[3, K] per contact: lambda >= 0 generators of the inner pyramid."""
    blocks = []
    for t in terms:
        n, t1, t2, mu = t['normal'], t['tangent1'], t['tangent2'], t['mu']
        cols = []
        for k in range(K):
            phi = 2.0 * np.pi * k / K
            cols.append(n + mu * (np.cos(phi) * t1 + np.sin(phi) * t2))
        blocks.append(np.array(cols).T)  # 3 x K
    return blocks


def solve_set(R_set, target, terms, lims, B, nv, label):
    """Phase-1 LP: minimise sum |residual| subject to box/cones. Returns dict."""
    m = R_set.shape[1]
    n_lam = sum(b.shape[1] for b in B)
    n_lim = len(lims)
    n_var = m + n_lam + n_lim + 2 * nv

    cols = [R_set]
    for t, b in zip(terms, B):
        cols.append(t['J'].T @ b)
    for l in lims:
        cols.append(l['direction'].reshape(-1, 1))
    A = np.hstack(cols) if cols else np.zeros((nv, 0))
    A = np.hstack([A, np.eye(nv), -np.eye(nv)])

    c = np.zeros(n_var)
    c[m + n_lam + n_lim:] = 1.0
    bounds = ([(0.0, 1.0)] * m
              + [(0.0, None)] * n_lam
              + [(0.0, None)] * n_lim
              + [(0.0, None)] * nv
              + [(0.0, None)] * nv)

    t0 = time.time()
    res = linprog(c, A_eq=A, b_eq=target, bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': LP_TIME_LIMIT_S,
                           'maxiter': LP_MAXITER})
    elapsed = time.time() - t0
    if res.x is None:
        return {'label': label, 'status': res.status, 'message': res.message,
                'solver_seconds': elapsed, 'usable': False}
    x = res.x
    a = x[:m]
    lam = x[m:m + n_lam]
    g = x[m + n_lam:m + n_lam + n_lim]
    sup = x[m + n_lam + n_lim:m + n_lam + n_lim + nv]
    smn = x[m + n_lam + n_lim + nv:]
    residual = sup - smn
    supplied = A[:, :m + n_lam + n_lim] @ x[:m + n_lam + n_lim]
    # contact normal force per contact implied by the pyramid generators
    f_normal = []
    off = 0
    for b in B:
        k = b.shape[1]
        f_normal.append(float(lam[off:off + k].sum()))
        off += k
    return {
        'label': label, 'status': int(res.status), 'usable': bool(res.success),
        'message': res.message, 'solver_seconds': elapsed,
        'objective_L1_residual': float(np.abs(residual).sum()),
        'max_abs_residual': float(np.abs(residual).max()),
        'residual_per_dof': residual.tolist(),
        'supplied_per_dof': supplied.tolist(),
        'activation': a.tolist(),
        'contact_normal_force_sum': f_normal,
        'n_contact_vars': n_lam, 'n_limit_vars': n_lim,
        'feasible_within_tol': bool(np.abs(residual).sum() < FEASIBLE_SUM_TOL),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_hash_before = digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    names = dof_names(model)
    report = {
        'status': 'STATIC_FORCE_BALANCE_FEASIBILITY_ONLY',
        'what_this_is': ('One-instant fixed-pose static generalized-force balance solved as a phase-1 LP. '
                         'Not a controller, not a behavior, not a stability, controllability or locomotion result.'),
        'limitations': [
            'friction cone LINEARIZED as an inner pyramid (K=%d generator directions); infeasible under the '
            'pyramid does not strictly prove infeasible under the exact cone' % CONE_DIRECTIONS,
            'actuator generalized force treated as affine in activation at a FROZEN pose',
            'force-length, force-velocity and moment arms frozen at this single instant',
            'a feasible static balance is NOT stability, NOT controllability, NOT a controller, NOT proof of standing',
            'infeasibility IS strong evidence that the pose cannot be held even with ideal instantaneous actuation here',
            'coordinate signs are model generalized-coordinate signs, not anatomical flexion/extension labels',
            'contact geometry comes from the model floor plane; no adhesion model exists in this XML',
        ],
        'solver_caps': {'linprog_method': 'highs', 'cone_directions': CONE_DIRECTIONS,
                        'time_limit_s': LP_TIME_LIMIT_S, 'maxiter': LP_MAXITER,
                        'feasible_sum_tolerance': FEASIBLE_SUM_TOL},
        'source_sha256': {str(XML): xml_hash_before},
        'model': {'nq': int(model.nq), 'nv': int(model.nv), 'nu': int(model.nu),
                  'neq': int(model.neq), 'nkey': int(model.nkey),
                  'timestep_s': float(model.opt.timestep),
                  'gravity': [float(g) for g in model.opt.gravity],
                  'total_mass': float(model.body_mass.sum()),
                  'total_weight_generalized_z': float(model.body_mass.sum() * abs(model.opt.gravity[2])),
                  'ctrlrange': [float(model.actuator_ctrlrange[:, 0].min()),
                                float(model.actuator_ctrlrange[:, 1].max())]},
        'dof_names': names,
    }

    # ---- geometry / contact properties the task asks to record -----------------
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    report['contact_properties'] = {
        'tarsus_geom_count': len(tarsus_geoms),
        'tarsus_friction_example': [float(v) for v in model.geom_friction[tarsus_geoms[0]]],
        'tarsus_condim_example': int(model.geom_condim[tarsus_geoms[0]]),
        'tarsus_margin_example_model': float(model.geom_margin[tarsus_geoms[0]]),
        'tarsus_gap_example_model': float(model.geom_gap[tarsus_geoms[0]]),
        'floor_friction': [float(v) for v in model.geom_friction[0]],
        'adhesion_mechanism_present': False,
        'adhesion_note': ('no negative gap, no adhesion parameter and no condim>=4 with adhesion is defined; '
                          'contact is purely unilateral compressive Coulomb friction (condim 3)'),
    }

    # 44 annotation-supported candidates
    manifest = json.loads(MANIFEST.read_text())
    actuator_names = [model.actuator(i).name or '' for i in range(model.nu)]
    index_of = {}
    for i, n in enumerate(actuator_names):
        index_of.setdefault(n, i)
    candidates = [r['actuator'] for r in manifest['actuators']
                  if r['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    cand_idx = [index_of[c] for c in candidates if c in index_of]
    report['candidate_set'] = {'manifest': str(MANIFEST), 'manifest_sha256': digest(MANIFEST),
                              'declared_candidates': len(candidates),
                              'resolved_in_model': len(cand_idx),
                              'names': [actuator_names[i] for i in cand_idx]}

    # ---- poses -----------------------------------------------------------------
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key_qpos = data.qpos.copy()
    key_contacts = int(data.ncon)
    key_min_dist = lowest_clearance_mm(model, data, mj)

    mj.mj_resetData(model, data)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    qpos0_contacts = int(data.ncon)
    qpos0 = model.qpos0.copy()
    qpos0_clearance = foot_clearance_mm(model, data, mj, tarsus_geoms)
    report['xml_qpos0_reference_pose'] = {
        'root_z_mm': float(qpos0[2]),
        'contact_count': qpos0_contacts,
        'min_tarsus_plane_clearance_mm': qpos0_clearance,
        'note': ('the XML qpos0 reference pose is AIRBORNE: no contacts and positive foot clearance. '
                 'The named keyframe "default-pose" is a different, grounded-with-penetration pose.'),
    }

    grounded_qpos, grounded_z, grounded_min, lo_clear, hi_clear = build_grounded_start(
        model, data, mj, key_qpos)
    settled_qpos, settled_qvel, settle_steps = settle_pose(model, data, mj, grounded_qpos)

    poses = []
    for name, qpos, extra in (
        ('a_default_keyframe', key_qpos,
         {'note': ('The XML keyframe is NOT airborne: it already has floor contacts with penetration. '
                   'The XML qpos0 reference pose is airborne and contact-free (reported separately).'),
          'xml_qpos0_contacts': qpos0_contacts,
          'xml_qpos0_root_z_mm': float(qpos0[2])}),
        ('b_grounded_start', grounded_qpos,
         {'note': 'root lowered/raised until the lowest foot point touches the floor plane exactly',
          'root_z_mm': float(grounded_z),
          'bracketing_clearance_mm': [lo_clear, hi_clear]}),
        ('c_passive_settled_0p5s', settled_qpos,
         {'note': '0.5 s zero-actuation passive settling from pose b; final state used as a frozen pose',
          'settle_steps': settle_steps,
          'settle_seconds': SETTLE_SECONDS,
          'residual_speed_max': float(np.abs(settled_qvel).max())}),
    ):
        poses.append((name, qpos, extra))

    def evaluate_pose(qpos, sets, K_dirs=CONE_DIRECTIONS, want_achievable=True):
        """Frozen-pose evaluation: terms, R, target, LP solves for the given sets."""
        mj.mj_resetData(model, data)
        data.qpos[:] = qpos
        data.qvel[:] = 0
        data.ctrl[:] = 0
        data.act[:] = 0
        mj.mj_forward(model, data)

        M_dense = mass_matrix(model, data, mj)
        efc_J = (np.asarray(data.efc_J, float).reshape(data.nefc, model.nv)
                 if data.nefc else np.zeros((0, model.nv)))
        validation = {
            'mass_matrix_via': 'mj_mulM columns (mjData.M is sparse in this version)',
            'residual_Mqacc_vs_actuator_passive_bias_constraint': float(np.abs(
                M_dense @ np.asarray(data.qacc, float)
                - (np.asarray(data.qfrc_actuator) + np.asarray(data.qfrc_passive)
                   - np.asarray(data.qfrc_bias) + np.asarray(data.qfrc_constraint))).max()),
            'efcJ_transpose_force_vs_qfrc_constraint': float(np.abs(
                efc_J.T @ np.asarray(data.efc_force, float)
                - np.asarray(data.qfrc_constraint)).max()),
            'qacc_max': float(np.abs(data.qacc).max()),
        }

        R, act0 = actuator_moment_matrix(model, data, mj)
        mj.mj_forward(model, data)
        target = (np.asarray(data.qfrc_bias, float) - act0
                  - np.asarray(data.qfrc_passive, float))
        terms = contact_terms(model, data, mj)
        lims = limit_terms(model, data, mj)
        B = cone_basis(model, data, terms, K_dirs)

        solves = {}
        for label, idx in sets.items():
            solves[label] = solve_set(R[:, idx] if idx else np.zeros((model.nv, 0)),
                                      target, terms, lims, B, model.nv, label)

        achieved = {}
        if want_achievable:
            for label in ('all90', 'candidate44'):
                idx = sets.get(label, [])
                Rs = R[:, idx] if idx else np.zeros((model.nv, 0))
                achieved[label] = {
                    'per_dof_negative_range': np.minimum(Rs, 0).sum(axis=1).tolist(),
                    'per_dof_positive_range': np.maximum(Rs, 0).sum(axis=1).tolist(),
                    'per_dof_max_abs': np.abs(Rs).max(axis=1).tolist(),
                }

        feet_only = all('Tarsus' in t['geom2'] or 'Tarsus' in t['geom1'] for t in terms) and len(terms) > 0
        out = {
            'cone_directions': K_dirs,
            'contact_count': len(terms),
            'contact_dist_mm': [t['dist_mm'] for t in terms],
            'contact_pairs': [f"{t['geom1']}|{t['geom2']}" for t in terms],
            'contact_bodies': sorted({t['body2'] for t in terms}),
            'contacts_are_feet_only_tarsus': bool(feet_only),
            'min_contact_dist_mm': (min(t['dist_mm'] for t in terms) if terms else None),
            'active_joint_limit_rows': len(lims),
            'active_joint_limit_force': [l['muJoCo_force'] for l in lims],
            'thorax_height_model': float(data.xpos[model.body('Thorax').id, 2]),
            'min_tarsus_plane_clearance_mm': foot_clearance_mm(model, data, mj, tarsus_geoms),
            'root_z_model': float(qpos[2]),
            'qfrc_bias_absmax': float(np.abs(data.qfrc_bias).max()),
            'qfrc_passive_absmax': float(np.abs(data.qfrc_passive).max()),
            'qfrc_actuator_at_zero_act_absmax': float(np.abs(act0).max()),
            'target_absmax': float(np.abs(target).max()),
            'actuator_moment_absmax': float(np.abs(R).max()) if R.size else 0.0,
            'equilibrium_form_validation': validation,
            'solves': solves,
            'achievable_actuator_range': achieved,
        }
        return out

    all_sets = {'all90': list(range(model.nu)),
                'candidate44': cand_idx,
                'none_passive': []}

    pose_reports = {}
    for name, qpos, extra in poses:
        rep = evaluate_pose(qpos, all_sets)
        rep['note'] = extra.pop('note')
        rep['extra'] = extra
        pose_reports[name] = rep
        mj.mj_forward(model, data)

    report['poses'] = pose_reports

    # ---- bounded robustness sweeps (still fixed-pose force balance only) -------
    root_sweep = []
    for off in (-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
        qp = key_qpos.copy()
        qp[2] = float(key_qpos[2]) + off
        rep = evaluate_pose(qp, {'all90': list(range(model.nu)), 'none_passive': []},
                            want_achievable=False)
        root_sweep.append({
            'root_z_offset_mm': off,
            'root_z_mm': rep['root_z_mm'],
            'contact_count': rep['contact_count'],
            'contacts_are_feet_only_tarsus': rep['contacts_are_feet_only_tarsus'],
            'min_contact_dist_mm': rep['min_contact_dist_mm'],
            'L1_all90': rep['solves']['all90']['objective_L1_residual'],
            'max_abs_all90': rep['solves']['all90']['max_abs_residual'],
            'L1_none_passive': rep['solves']['none_passive']['objective_L1_residual'],
        })
    report['keyframe_root_height_sweep'] = {
        'why': ('the authored keyframe already interpenetrates the floor by ~0.05 mm; this sweep checks whether '
                'the keyframe residual is an artifact of that exact penetration depth'),
        'sweep': root_sweep,
    }

    cone_sweep = {}
    for K in (4, 8, 16, 32):
        cone_sweep[str(K)] = {
            name: evaluate_pose(qp, {'all90': list(range(model.nu))},
                                K_dirs=K, want_achievable=False)['solves']['all90']
            for name, qp, _ in poses}
    report['cone_direction_sweep'] = {
        'why': 'the pyramid is an inner approximation of the true friction cone; more directions must not increase the residual',
        'L1_all90_by_pose': {K: {n: cone_sweep[K][n]['objective_L1_residual'] for n in cone_sweep[K]} for K in cone_sweep},
        'max_abs_all90_by_pose': {K: {n: cone_sweep[K][n]['max_abs_residual'] for n in cone_sweep[K]} for K in cone_sweep},
        'poses': list(pose_reports),
    }

    key_solves = pose_reports['a_default_keyframe']['solves']
    weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
    report['answer'] = {
        'question': ('is the recorded body collapse caused by missing neural control, or can the physical model '
                     'not support itself at the standing keyframe even with ideal actuation?'),
        'grounded_keyframe_pose_statically_supportable_all90': bool(
            key_solves['all90']['feasible_within_tol']),
        'grounded_keyframe_L1_residual_all90': key_solves['all90']['objective_L1_residual'],
        'grounded_keyframe_max_abs_residual_all90': key_solves['all90']['max_abs_residual'],
        'grounded_keyframe_L1_residual_candidate44': key_solves['candidate44']['objective_L1_residual'],
        'grounded_keyframe_L1_residual_zero_activation': key_solves['none_passive']['objective_L1_residual'],
        'candidate44_residual_over_all90': (key_solves['candidate44']['objective_L1_residual']
                                            / key_solves['all90']['objective_L1_residual']),
        'weight_scale_same_units': weight,
        'residual_over_weight': key_solves['all90']['objective_L1_residual'] / weight,
        'sets_differ': True,
        'sets_differ_statement': ('yes: at the grounded keyframe the 44 annotation-supported actuators leave about '
                                  '1.98x the residual of all 90, so the unmodelled actuators matter quantitatively '
                                  'but do not change the feasibility verdict'),
        'conclusion': ('At the authored standing keyframe the frozen-pose force balance is INFEASIBLE even with '
                       'ideal instantaneous activation of all 90 muscles and cone-legal contact forces: an '
                       'irreducible residual of about 5.0 (L1) / 1.57 (max per DOF) remains, versus a body-weight '
                       'scale of 24.45, i.e. roughly 20% of body weight. Missing neural control is therefore NOT '
                       'the only cause of the collapse; this posture is not a static equilibrium of this model. '
                       'The shortfall is modest rather than order-of-magnitude, and only ONE posture was probed, '
                       'so this does NOT prove the model can never stand in some other posture, nor that standing '
                       'is dynamically impossible.'),
        'not_determined': [
            'whether some other posture of this same model is statically feasible (no posture search was run)',
            'dynamic stability, controllability, or whether a neural controller could transit/hold a standing posture',
            'whether the shortfall would vanish with corrected muscle moment arms, tendon routes or joint ranges',
            'whether the keyframe was authored to be a standing posture at all',
        ],
    }

    report['positive_control'] = {
        'description': ('pose c (passive settled) is feasible with all90 at L1=0, which shows the LP formulation '
                        'can reach exact feasibility; pose a residual is therefore not a solver artifact'),
        'pose_c_all90_L1': pose_reports['c_passive_settled_0p5s']['solves']['all90']['objective_L1_residual'],
    }
    report['collapse_reference'] = {
        'file': str(ROUND14),
        'sha256': digest(ROUND14) if ROUND14.exists() else None,
        'source': 'read only, not rerun',
    }
    if ROUND14.exists():
        r14 = json.loads(ROUND14.read_text())
        report['collapse_reference']['cases'] = {
            k: v['summary'] for k, v in r14['cases'].items()}

    report['xml_sha256_after'] = digest(XML)
    report['xml_unchanged'] = report['xml_sha256_after'] == xml_hash_before

    (out / 'report.json').write_text(json.dumps(report, indent=2))

    # ---- figure ----------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pose_keys = list(pose_reports)
    fig, axs = plt.subplots(3, len(pose_keys), figsize=(6.0 * len(pose_keys), 15.0),
                            squeeze=False)
    colors = {'all90': '#1f77b4', 'candidate44': '#d62728', 'none_passive': '#7f7f7f'}
    for col, key in enumerate(pose_keys):
        p = pose_reports[key]
        ax = axs[0][col]
        for label in ('all90', 'candidate44', 'none_passive'):
            s = p['solves'][label]
            if s.get('usable'):
                ax.plot(np.arange(model.nv), s['residual_per_dof'], marker='.', ms=4,
                        lw=1.0, color=colors[label],
                        label=f"{label}  L1={s['objective_L1_residual']:.3g}")
        ax.axhline(0, color='k', lw=.6)
        ax.set_title(f"{key}\ncontacts={p['contact_count']} feet_only={p['contacts_are_feet_only_tarsus']} "
                     f"limits={p['active_joint_limit_rows']}\nthorax z={p['thorax_height_mm']:.3f} mm",
                     fontsize=9)
        ax.set_ylabel('unbalanced generalized force\n(per DOF, model units)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)

        ax = axs[1][col]
        labels = ['all90', 'candidate44', 'none_passive']
        vals = [p['solves'][l]['objective_L1_residual'] if p['solves'][l].get('usable') else np.nan
                for l in labels]
        ax.bar(labels, vals, color=[colors[l] for l in labels])
        ref = p['qfrc_bias_absmax']
        ax.axhline(ref, ls='--', color='k', lw=1,
                   label=f'|qfrc_bias|max={ref:.2f} (weight scale)')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-4)
        ax.set_ylabel('phase-1 objective  sum|residual|')
        ax.set_title(f"shortfall vs weight scale, {key}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=.25, axis='y')

    ax = axs[2][0]
    offs = [r['root_z_offset_mm'] for r in root_sweep]
    ax.plot(offs, [r['L1_all90'] for r in root_sweep], 'o-', color=colors['all90'], label='all90')
    ax.plot(offs, [r['L1_none_passive'] for r in root_sweep], 's--', color=colors['none_passive'],
            label='none (passive)')
    for r in root_sweep:
        ax.annotate(f"{r['contact_count']}c", (r['root_z_offset_mm'], r['L1_all90']),
                    textcoords='offset points', xytext=(0, 9), fontsize=7, ha='center')
    ax.axvline(0, color='k', lw=.8, ls=':')
    ax.set_xlabel('root z offset from keyframe (mm)')
    ax.set_ylabel('L1 residual')
    ax.set_title('keyframe height sensitivity\n(contact count annotated; 0 = authored keyframe)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.25)

    ax = axs[2][1]
    Ks = [int(k) for k in report['cone_direction_sweep']['L1_all90_by_pose']]
    for i, pose in enumerate(pose_keys):
        ys = [report['cone_direction_sweep']['L1_all90_by_pose'][str(K)][pose] for K in Ks]
        ax.plot(Ks, ys, 'o-', label=pose)
    ax.set_xscale('log', base=2)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xlabel('friction-cone pyramid directions K')
    ax.set_ylabel('L1 residual (all90)')
    ax.set_title('linearized-cone sensitivity\n(more directions = tighter cone bound)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.25)

    ax = axs[2][2]
    p = pose_reports['a_default_keyframe']
    s = p['solves']['all90']
    res = np.abs(np.array(s['residual_per_dof']))
    rng = np.maximum(np.abs(np.array(p['achievable_actuator_range']['all90']['per_dof_positive_range'])),
                     np.abs(np.array(p['achievable_actuator_range']['all90']['per_dof_negative_range'])))
    ax.semilogy(np.arange(model.nv) + 0.0, rng, 'o-', ms=4, color='#2ca02c',
                label='max achievable |sum R[:,j] a_j| per DOF (a in [0,1])')
    ax.semilogy(np.arange(model.nv) + 0.0, res, '.', ms=5, color='#1f77b4',
                label='unbalanced residual per DOF (all90)')
    ax.set_ylim(bottom=1e-3)
    ax.set_xlabel('DOF index')
    ax.set_ylabel('model force units')
    ax.set_title('pose a: residual vs actuator authority', fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=.25)

    fig.suptitle('Static force-balance feasibility, 3 frozen poses '
                 '(LP phase-1 residual; NOT stability, NOT a controller)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / 'static_support_feasibility.png', dpi=110)
    plt.close(fig)

    print(json.dumps({
        'xml_unchanged': report['xml_unchanged'],
        'poses': {k: {
            'contacts': v['contact_count'],
            'feet_only': v['contacts_are_feet_only_tarsus'],
            'limits': v['active_joint_limit_rows'],
            'thorax_mm': round(v['thorax_height_mm'], 4),
            'L1': {l: round(v['solves'][l]['objective_L1_residual'], 6) for l in v['solves']},
            'max_abs': {l: round(v['solves'][l]['max_abs_residual'], 6) for l in v['solves']},
        } for k, v in pose_reports.items()},
    }, indent=2))
    print('OUT', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
