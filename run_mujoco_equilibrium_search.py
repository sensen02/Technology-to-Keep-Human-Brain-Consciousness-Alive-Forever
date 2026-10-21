"""Declared search for a TRUE MuJoCo equilibrium of the s=1-feasible seated pose A.

Declared question: the static LP says "inside the declared pyramidal (axis-ray diamond) cone the pose is feasible at s <= 1", but a
feasible LP is not a MuJoCo equilibrium. This script closes that gap by asking MuJoCo itself:

    is there an activation vector act (and, in stage 2, also a small bounded pose adjustment) such that with qvel = 0 the
    MuJoCo forward dynamics gives qacc ~ 0, while the contact forces stay inside the DECLARED axis-ray diamond cone?

DECLARED formulation
  stage 1: qpos fixed at the delivered pose A; variables = act in [1e-4, 1] (the actuator ctrlrange; act is the muscle state),
           residual = d.qacc after mj_forward, objective = ||qacc||_2, minimised with scipy least_squares (bounded, several starts).
  stage 2: variables = act (90, in [1e-4, 1]) + root z (mm, +-0.05) + the 42 leg hinge joints (each within its joint range minus a
           5e-3 rad margin and within 0.40 rad of the source pose, the same declared contract as the pose search); same residual.
  The contact forces are NOT free variables: MuJoCo computes them with opt.cone = PYRAMIDAL and per-contact condim = 3, which is
  exactly the DECLARED axis-ray diamond, so any MuJoCo solution automatically satisfies the declared cone; the script verifies this
  by recomputing u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu2*fn) <= 1 for every tarsus contact at the best point.

DECLARED acceptance: equilibrium_found = (max |qacc| <= QACC_TOL_MM_S2) with QACC_TOL_MM_S2 = 1.0 (i.e. about 1e-4 g), reported
together with the raw residual so a reader can apply a different tolerance. Absolute force values and physical units remain
NOT_ESTABLISHED. This is a declared numerical search, not behaviour evidence, and it is NOT a standing/walking conclusion.
"""
from pathlib import Path
import hashlib
import json
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares

DECLARED_GATE = dict(
    max_abs_qacc_mm_s2=1e-3,
    max_abs_qacc_reason=('about 1e-7 g: with mm units and gravity 9801 mm/s^2 this is ~1e-7 of g, i.e. numerically indistinguishable from zero at the precision of a single '
                         'mj_forward on this model (double precision, solver tolerances ~1e-10), while still being 10x tighter than the tightest meaningful dynamic '
                         'criterion used anywhere in this project (1e-2 mm/s^2)'),
    six_tarsi_active_contact=True, zero_nonfoot_contacts=True, all_contacts_u_diamond_le_1=True,
    drive='frozen connectome-derived command archive only (no external CPG, no PD, no preprogrammed action, no per-step pose correction)',
    not_standing='passing this gate may only be called a DECLARED-CONE STATIC EQUILIBRIUM, never standing')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
import sys as _sys
POSE_ARG = None
if '--pose' in _sys.argv and _sys.argv.index('--pose') + 1 < len(_sys.argv):
    POSE_ARG = Path(_sys.argv[_sys.argv.index('--pose') + 1])
LABEL = 'A_min_s_pose'
MULTISTART = 0
if '--multistart' in _sys.argv and _sys.argv.index('--multistart') + 1 < len(_sys.argv):
    MULTISTART = int(_sys.argv[_sys.argv.index('--multistart') + 1])
# declared CLI: --radius <rad> restricts the 42 leg joints to within that distance of the DELIVERED pose (nested probe)
RADIUS = None
if '--radius' in _sys.argv and _sys.argv.index('--radius') + 1 < len(_sys.argv):
    RADIUS = float(_sys.argv[_sys.argv.index('--radius') + 1])
if '--label' in _sys.argv and _sys.argv.index('--label') + 1 < len(_sys.argv):
    LABEL = _sys.argv[_sys.argv.index('--label') + 1]
STATIC_S = {'A_min_s_pose': 0.9097466208040714, 'B_tarsus_only_pose': None, 'C_lowest_s_weight_variant': 0.9391964338719845}
FROZEN_DRIVE = {'A_min_s_pose': dict(T0_step=2398, T0_ms=59.950, root_z_min=0.7266289904906442),
                'B_tarsus_only_pose': dict(T0_step=2316, T0_ms=57.900, root_z_min=-0.14585179021287337),
                'C_lowest_s_weight_variant': dict(T0_step=2399, T0_ms=59.975, root_z_min=0.7224921684347319)}
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
QACC_TOL_MM_S2 = 1.0
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
ROOT_Z_RANGE = 0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = DESK / ('S2-mujoco-equilibrium-%s-' % LABEL.replace('_min_s_pose','').replace('_tarsus_only_pose','')
                  + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    pose_path = POSE_ARG if POSE_ARG is not None else POSE_A
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('declared numerical search for a true MuJoCo equilibrium of the s=1-feasible seated pose A: act in [1e-4,1] and, in '
                         'stage 2, a bounded pose adjustment, minimising ||qacc|| with qvel = 0, while the contact forces automatically satisfy the '
                         'declared axis-ray diamond cone because that is the cone MuJoCo applies'),
                  declared=dict(pose=str(pose_path), label=LABEL,
                                static_s_on_declared_cone=STATIC_S.get(LABEL),
                                frozen_drive_measured=FROZEN_DRIVE.get(LABEL), qacc_tol_mm_s2=QACC_TOL_MM_S2,
                                stage1='qpos fixed; variables act (90) in [1e-4, 1]',
                                stage2=('variables act (90) + root z (+-%.3f mm) + 42 leg hinge joints (joint range minus %.3f rad margin and within '
                                        '%.2f rad of the %s)' % (ROOT_Z_RANGE, JOINT_MARGIN, _dev, ('DELIVERED pose A' if RADIUS is not None else 'SOURCE pose'))),
                                cone=('declared pyramidal axis-ray diamond = u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu2*fn) <= 1, which is what MuJoCo '
                                      'applies with opt.cone = PYRAMIDAL and condim = 3'),
                                not_established='absolute force values and physical units remain NOT_ESTABLISHED'),
                  input_hashes={str(p): sha(p) for p in (Path(__file__), POSE_ARG if POSE_ARG is not None else POSE_A, SOURCE_POSE) if Path(p).exists()})
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        d = mujoco.MjData(m)
        q0 = np.load(pose_path)['projected_seated_qpos'].copy()
        src = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        leg_joint_idx = []
        for L in LEGS:
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
            for suf in JOINT_SUFFIXES:
                leg_joint_idx.append(next(j for j in js if jname(j).endswith(suf)))
        qadr = [int(m.jnt_qposadr[j]) for j in leg_joint_idx]
        _ref_q = q0 if RADIUS is not None else src
        _dev = MAX_DEVIATION if RADIUS is None else RADIUS
        lo_q = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, _ref_q[int(m.jnt_qposadr[j])] - _dev) for j in leg_joint_idx])
        hi_q = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, _ref_q[int(m.jnt_qposadr[j])] + _dev) for j in leg_joint_idx])
        report['nested_radius_rad'] = RADIUS
        act_lo = np.full(m.na, 1e-4)
        act_hi = np.full(m.na, 1.0)
        report['model'] = dict(nq=m.nq, nv=m.nv, na=m.na, nu=m.nu,
                               actuator_ctrlrange=[float(m.actuator_ctrlrange[0, 0]), float(m.actuator_ctrlrange[0, 1])],
                               opt_cone=int(m.opt.cone),
                               tarsus_condim=sorted({int(m.geom_condim[g]) for g in range(m.ngeom) if 'Tarsus' in gname(g)}))

        def forward(qpos, act):
            d.qpos[:] = qpos
            d.qvel[:] = 0.0
            d.act[:] = np.clip(act, act_lo, act_hi)
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            return d

        def gate_evaluate(qpos, act):
            d3 = mujoco.MjData(m)
            d3.qpos[:] = qpos; d3.qvel[:] = 0.0; d3.act[:] = np.clip(act, act_lo, act_hi); d3.ctrl[:] = 0.0
            mujoco.mj_forward(m, d3)
            qa = np.asarray(d3.qacc, dtype=float)
            f6 = np.zeros(6); rows = []; per_leg = {}
            for ci, c in enumerate(d3.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                mujoco.mj_contactForce(m, d3, ci, f6)
                fr = np.asarray(c.friction)[:3]
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                u = (abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)) if fn > 0 else None
                leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                rows.append(dict(geom=sorted((n1, n2)), leg=leg, kind=('tarsus' if 'Tarsus' in (n1 + n2) else 'non-foot'), fn=fn, u_diamond=u))
                if 'Tarsus' in (n1 + n2) and leg in LEGS:
                    per_leg[leg] = u if leg not in per_leg or u is None else max(per_leg[leg] or 0.0, u)
            tarsi_with = [L for L in LEGS if L in per_leg]
            nonfoot = [r for r in rows if r['kind'] == 'non-foot']
            us = [r['u_diamond'] for r in rows if r['u_diamond'] is not None]
            checks = dict(max_abs_qacc=bool(float(np.max(np.abs(qa))) <= DECLARED_GATE['max_abs_qacc_mm_s2']),
                          six_tarsi_active_contact=bool(len(tarsi_with) == 6),
                          zero_nonfoot_contacts=bool(len(nonfoot) == 0),
                          all_contacts_u_diamond_le_1=bool(all(u <= 1.0 + 1e-9 for u in us)))
            return dict(checks=checks, max_abs_qacc_mm_s2=float(np.max(np.abs(qa))), qacc_norm=float(np.linalg.norm(qa)),
                        n_dof_above_1e_minus_3=int(np.sum(np.abs(qa) > 1e-3)), n_dof=int(qa.size),
                        per_leg_max_u_diamond=per_leg, tarsi_with_contact=tarsi_with,
                        n_tarsus_contacts=sum(1 for r in rows if r['kind'] == 'tarsus'), n_nonfoot_contacts=len(nonfoot),
                        max_u_diamond=(None if not us else float(max(us))), gate=DECLARED_GATE,
                        notes='gate declared before the search; passing it may only be called a declared-cone static equilibrium, never standing')

        def describe(qpos, act, tag=''):
            """per-equilibrium detail block, recomputed on a FRESH MjData so it does not depend on solver state.
            The criterion is the declared axis-ray DIAMOND u_diamond = u1 + u2 <= 1; the SQUARE max(u1, u2) <= 1 is
            recorded only as a secondary, strictly larger reading and is NOT the criterion."""
            d4 = mujoco.MjData(m)
            d4.qpos[:] = qpos; d4.qvel[:] = 0.0; d4.act[:] = np.clip(act, act_lo, act_hi); d4.ctrl[:] = 0.0
            mujoco.mj_forward(m, d4)
            qa = np.asarray(d4.qacc, dtype=float)
            order = np.argsort(-np.abs(qa))
            f6 = np.zeros(6); rows = []; sq = 0.0
            for ci, c in enumerate(d4.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                mujoco.mj_contactForce(m, d4, ci, f6)
                fr = np.asarray(c.friction)[:3]
                fnn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                ud = (abs(ft1) / (float(fr[0]) * fnn) + abs(ft2) / (float(fr[1]) * fnn)) if fnn > 0 else None
                if fnn > 0:
                    sq = max(sq, abs(ft1) / (float(fr[0]) * fnn), abs(ft2) / (float(fr[1]) * fnn))
                leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
                rows.append(dict(geom=sorted((n1, n2)), leg=leg,
                                 kind=('tarsus' if 'Tarsus' in (n1 + n2) else 'non-foot'), fn=fnn, u_diamond=ud))
            us = [r['u_diamond'] for r in rows if r['u_diamond'] is not None]
            per_leg = {}
            for r in rows:
                if r['kind'] == 'tarsus' and r['leg'] in LEGS and r['u_diamond'] is not None:
                    per_leg[r['leg']] = max(per_leg.get(r['leg'], 0.0), r['u_diamond'])
            act_c = np.clip(act, act_lo, act_hi)
            return dict(tag=tag, max_abs_qacc_mm_s2=float(np.max(np.abs(qa))), qacc_norm=float(np.linalg.norm(qa)),
                        nv=int(qa.size), n_dof_above_1e_minus_3=int(np.sum(np.abs(qa) > 1e-3)),
                        n_dof_above_1e_minus_6=int(np.sum(np.abs(qa) > 1e-6)),
                        n_dof_above_1e_minus_9=int(np.sum(np.abs(qa) > 1e-9)),
                        top5_abs_qacc=[float(np.abs(qa)[i]) for i in order[:5]],
                        top5_dof_index=[int(i) for i in order[:5]],
                        n_contacts=int(len(d4.contact)),
                        n_tarsus_contacts=sum(1 for r in rows if r['kind'] == 'tarsus'),
                        n_nonfoot_contacts=sum(1 for r in rows if r['kind'] == 'non-foot'),
                        tarsi_with_contact=sorted(per_leg.keys()), six_tarsi_in_contact=bool(len(per_leg) == 6),
                        per_leg_max_u_diamond=per_leg, max_u_diamond=(None if not us else float(max(us))),
                        max_u_square_secondary_NOT_the_criterion=float(sq),
                        root_z_raw=float(qpos[2]), root_z_change_from_delivered_pose_A_mm=float(qpos[2] - q0[2]),
                        max_abs_delta_q_rad_vs_delivered_pose_A=float(np.max(np.abs(qpos[qadr] - q0[qadr]))),
                        max_abs_delta_q_rad_vs_source_pose=float(np.max(np.abs(qpos[qadr] - src[qadr]))),
                        act_min=float(act_c.min()), act_max=float(act_c.max()), act_mean=float(act_c.mean()),
                        act_at_floor_own_atol=int(np.sum(np.abs(act_c - act_lo) <= 1e-12)),
                        act_at_ceiling_own_atol=int(np.sum(np.abs(act_c - act_hi) <= 1e-9)),
                        act_within_1e_minus_9_of_floor=int(np.sum(act_c - act_lo <= 1e-9)),
                        act_within_1e_minus_6_of_floor=int(np.sum(act_c - act_lo <= 1e-6)),
                        act_within_1e_minus_6_of_ceiling=int(np.sum(act_hi - act_c <= 1e-6)))

        def qacc_of(qpos, act):
            forward(qpos, act)
            return np.asarray(d.qacc, dtype=float).copy()

        def diamond_utilisation():
            f6 = np.zeros(6); worst = 0.0; rows = []
            for ci, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                if 'Tarsus' not in (n1 + n2):
                    continue
                mujoco.mj_contactForce(m, d, ci, f6)
                fr = np.asarray(c.friction)[:3]
                fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
                if fn <= 0:
                    continue
                u = abs(ft1) / (float(fr[0]) * fn) + abs(ft2) / (float(fr[1]) * fn)
                worst = max(worst, u)
                rows.append(dict(geom=(n1, n2), fn=fn, ft1=ft1, ft2=ft2, u_diamond=u))
            return worst, rows

        # ---------------- stage 1: act only ----------------
        starts = [np.full(m.na, 1e-4), np.full(m.na, 0.5), np.full(m.na, 1.0),
                  np.linspace(0.2, 1.0, m.na), np.linspace(1.0, 0.2, m.na)]
        hist = []
        best = dict(qacc=np.inf, act=None)
        for x0 in starts:
            res = least_squares(lambda a: qacc_of(q0, a), x0, bounds=(act_lo, act_hi), xtol=1e-14, ftol=1e-14,
                                gtol=1e-14, max_nfev=4000)
            nrm = float(np.linalg.norm(res.fun))
            hist.append(dict(start='uniform(%.2f)' % float(np.mean(x0)), nfev=int(res.nfev),
                             qacc_norm=nrm, qacc_max=float(np.max(np.abs(res.fun)))))
            if nrm < best['qacc']:
                best = dict(qacc=nrm, act=np.clip(res.x, act_lo, act_hi).copy())
            print('stage1 start mean %.2f -> ||qacc|| %.6g max %.6g' % (float(np.mean(x0)), nrm, float(np.max(np.abs(res.fun)))), flush=True)
        s1 = dict(history=hist, best_qacc_norm=float(best['qacc']),
                  best_qacc_max=float(np.max(np.abs(qacc_of(q0, best['act'])))),
                  best_act_stats=dict(min=float(best['act'].min()), max=float(best['act'].max()),
                                      mean=float(best['act'].mean()), n_at_ceiling=int(np.sum(best['act'] >= 1 - 1e-9)),
                                      n_at_floor=int(np.sum(best['act'] <= 1e-4 + 1e-12))))
        u1, rows1 = diamond_utilisation()
        s1['max_u_diamond_at_best'] = float(u1)
        s1['n_tarsus_contacts_at_best'] = len(rows1)
        s1['at_best_front_contacts'] = int(sum(1 for r in rows1 if 'Tarsus' not in r['geom'][0]))
        s1['equilibrium_found'] = bool(s1['best_qacc_max'] <= QACC_TOL_MM_S2)
        report['stage1_act_only'] = s1
        np.savez_compressed(out / 'stage1_best_act.npz', act=best['act'])

        # ---------------- stage 2: act + bounded pose ----------------
        x0 = np.concatenate([best['act'], [0.0], np.clip(q0[qadr], lo_q, hi_q)])
        lb = np.concatenate([act_lo, [-ROOT_Z_RANGE], lo_q])
        ub = np.concatenate([act_hi, [ROOT_Z_RANGE], hi_q])

        def unpack(x):
            qp = q0.copy()
            qp[2] = q0[2] + x[m.na]
            qp[qadr] = x[m.na + 1:]
            return qp, x[:m.na]

        def resid(x):
            qp, act = unpack(x)
            return qacc_of(qp, act)

        rng = np.random.default_rng(20261015)
        ms_hist = []
        best_x = None; best_val = np.inf
        hit_states = []
        HITS_STORED_CAP = 12
        if MULTISTART > 0:
            for i in range(MULTISTART):
                xs = lb + rng.random(lb.size) * (ub - lb)
                try:
                    r_i = least_squares(resid, xs, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
                except Exception:
                    continue
                val = float(np.max(np.abs(r_i.fun)))
                ms_hist.append(val)
                if val <= QACC_TOL_MM_S2 and len(hit_states) < HITS_STORED_CAP:
                    qh, ah = unpack(r_i.x)
                    ev = describe(qh, ah, tag='multistart_start_%d' % i)
                    ev['solver_status'] = int(r_i.status); ev['solver_message'] = str(r_i.message); ev['nfev'] = int(r_i.nfev)
                    ev['found_at_start_index'] = int(i)
                    hit_states.append(ev)
                if val < best_val:
                    best_val = val; best_x = r_i.x.copy()
                if i % 20 == 0:
                    print('multistart %d/%d best max|qacc| %.6g' % (i, MULTISTART, best_val), flush=True)
            report['multistart'] = dict(n_starts=MULTISTART, best_max_abs_qacc_mm_s2=best_val,
                                        n_below_1_mm_s2=int(sum(1 for v in ms_hist if v <= 1.0)),
                                        n_below_1e_minus_3=int(sum(1 for v in ms_hist if v <= 1e-3)),
                                        history_all_starts=[float(v) for v in ms_hist],
                                        history_head=ms_hist[:20],
                                        hits_stored_cap=HITS_STORED_CAP,
                                        n_hits_stored=len(hit_states),
                                        per_equilibrium_hits=hit_states,
                                        note=('history contains every start; hits are the starts whose final max|qacc| is at or below the search '
                                              'tolerance 1.0 mm/s^2, each re-evaluated on a fresh MjData by describe()'))
            if best_x is not None:
                x0 = best_x
        res2 = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000)
        qp2, act2 = unpack(res2.x)
        f2 = resid(res2.x)
        d2 = forward(qp2, act2)
        u2, rows2 = diamond_utilisation()
        s2 = dict(nfev=int(res2.nfev), best_qacc_norm=float(np.linalg.norm(f2)), best_qacc_max=float(np.max(np.abs(f2))),
                  act_stats=dict(min=float(act2.min()), max=float(act2.max()), mean=float(act2.mean()),
                                 n_at_ceiling=int(np.sum(act2 >= 1 - 1e-9)), n_at_floor=int(np.sum(act2 <= 1e-4 + 1e-12))),
                  pose_change=dict(root_z_mm=float(qp2[2] - q0[2]), max_joint_change_rad=float(np.max(np.abs(qp2[qadr] - q0[qadr]))),
                                   any_at_bound=bool(np.any(np.isclose(res2.x[m.na + 1:], lo_q, atol=1e-9)) or
                                                     np.any(np.isclose(res2.x[m.na + 1:], hi_q, atol=1e-9)))),
                  max_u_diamond_at_best=float(u2), n_tarsus_contacts_at_best=len(rows2),
                  equilibrium_found=bool(float(np.max(np.abs(f2))) <= QACC_TOL_MM_S2))
        report['stage2_act_and_bounded_pose'] = s2
        final_point = describe(qp2, act2, tag='stage2_final_point')
        final_point['solver_status'] = int(res2.status); final_point['solver_message'] = str(res2.message)
        final_point['nfev'] = int(res2.nfev)
        report['stage2_final_equilibrium'] = final_point
        np.savez_compressed(out / 'stage2_best_state.npz', qpos=qp2, act=act2)
        gg = gate_evaluate(qp2, act2)
        report['declared_gate_result'] = gg
        report['verdict'] = dict(
            declared_gate_passed=all(v for k, v in gg['checks'].items() if k != 'notes'),
            equilibrium_found=bool(s1['equilibrium_found'] or s2['equilibrium_found']),
            stage1_qacc_max=s1['best_qacc_max'], stage2_qacc_max=s2['best_qacc_max'],
            search_tolerance_mm_s2=QACC_TOL_MM_S2,
            declared_gate_tolerance_mm_s2=DECLARED_GATE['max_abs_qacc_mm_s2'],
            tolerance_note=('search tolerance 1.0 mm/s^2 decides "equilibrium_found"; the DECLARED GATE is 10x tighter at 1e-3 mm/s^2. '
                            'The declared gate is evaluated separately on the stage-2 point in declared_gate_result and the '
                            'recomputed value is in stage2_final_equilibrium'),
            search_range=dict(stage1='act in [1e-4, 1] with qpos fixed at the delivered pose A',
                              stage2=('act in [1e-4, 1] + root z +-%.3f mm + 42 leg hinge joints within [joint range - %.3f rad] and within '
                                      '%.2f rad of the %s' % (ROOT_Z_RANGE, JOINT_MARGIN, _dev,
                                                              ('DELIVERED pose A' if RADIUS is not None else 'SOURCE pose')))),
            contact_cone_check=('every MuJoCo contact automatically satisfies the declared diamond; the maximum u_diamond at the best stage-2 point is '
                                '%.6f over %d tarsus contacts' % (u2, len(rows2))),
            comparison_to_the_gate=('for reference the support gate measured about 4.0e3 mm/s^2 of forward qacc on this pose; the best searched residual is '
                                    '%.6g mm/s^2 (max component)' % s2['best_qacc_max']),
            statement=('this is a declared numerical search only: it does not claim standing, walking or balance, and the absolute force values and units remain '
                       'NOT_ESTABLISHED'))

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        axes[0].bar([h['start'] for h in hist], [h['qacc_norm'] for h in hist])
        axes[0].axhline(QACC_TOL_MM_S2, color='r', ls='--', lw=1, label='declared tolerance')
        axes[0].set_ylabel('||qacc|| at the best point of each start')
        axes[0].set_title('stage 1: act only (qpos fixed)')
        axes[0].tick_params(axis='x', labelrotation=20, labelsize=7); axes[0].legend(fontsize=7)
        axes[1].bar(['stage1', 'stage2'], [s1['best_qacc_max'], s2['best_qacc_max']], color=['tab:blue', 'tab:green'])
        axes[1].set_yscale('log')
        axes[1].axhline(QACC_TOL_MM_S2, color='r', ls='--', lw=1, label='declared tolerance %.0e' % QACC_TOL_MM_S2)
        axes[1].legend(fontsize=7)
        if report.get('multistart', {}).get('history_head'):
            axes[1].scatter([0.5] * len(report['multistart']['history_head']), report['multistart']['history_head'],
                            s=8, color='k', alpha=0.7, label='multi-start residuals')
            axes[1].legend(fontsize=7)
        axes[1].set_ylabel('max |qacc| (mm/s^2)')
        axes[1].set_title('best residual: act only vs act + bounded pose')
        fig.suptitle('Declared MuJoCo equilibrium search on the s=1-feasible pose A - manual research, NOT standing', fontsize=9)
        fig.tight_layout()
        fig.savefig(out / 'equilibrium-search.png', dpi=130)
        plt.close(fig)
        report['status'] = 'PASS_MUJOCO_EQUILIBRIUM_SEARCH_REPORTED'
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                   default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('stage1 max|qacc|', report['stage1_act_only']['best_qacc_max'], 'stage2 max|qacc|',
              report['stage2_act_and_bounded_pose']['best_qacc_max'], 'found:', report['verdict']['equilibrium_found'])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
