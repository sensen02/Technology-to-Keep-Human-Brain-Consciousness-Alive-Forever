"""Multi-start IK: push the seated family's s below the earlier upper bound, recording starts and branches.

WHY: the s values reported earlier belong to the joint vector produced by one IK path; the per-leg IK has
several local minima that all reach zero touch residual, so the earlier "family minimum" is only an UPPER
BOUND. The start-0 seed is NOT a '"delivered"' mystery vector: it is the file
/home/sensen/Desktop/engine修复验收/S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz (SEED_POSE),
whose own s on the declared cone is 0.9594120287401438; this script's start-0 branch value 1.0923678502440453 is
the value of ONE IK branch from that seed and is NOT a function of the declared parameters alone (see
run_ik_weight_variants.py: the same seed and the same declared parameters give 0.9391964338719845 ..
1.0923678502440453 when only the residual weighting changes).

DECLARED here: a bounded multi-start search (N_STARTS per leg, deterministic seeds: the delivered vector,
the source pose, and fixed perturbations), the same declared contract (touch band 2-8 um, all 41 non-foot
geoms >= 0.40 mm, every hinge >= 0.005 rad inside its limits, every joint <= 0.40 rad from the source
pose, root quaternion identity), the same declared parameters inherited from the earlier minimum-s search,
and the objective = s on the DECLARED pyramidal (axis-ray diamond) cone.

Pose search only. NOT neural control, NOT behaviour evidence, NOT standing. Canonical XML, gains, friction,
force scale and historical hashes untouched.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares, linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
SEED_POSE = DESK / 'S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
GROUPS = {'front': ('LF', 'RF'), 'mid': ('LM', 'RM'), 'hind': ('LH', 'RH')}
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
TOUCH_LOW, TOUCH_HIGH = -8e-3, -2e-3
H_DECLARED = 0.40
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
DIST_CAP = 4.0
DIAMOND = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
Z0, SPREAD0, YAW0 = 2.00, 0.95, -10.0
OFFSETS = dict(tp={'front': -10.0, 'mid': -5.0, 'hind': -10.0},
               cp={'front': 0.0, 'mid': -10.0, 'hind': -10.0},
               cr={'front': 0.0, 'mid': 0.0, 'hind': 0.0})
N_STARTS = 6
PERTURB = (0.0, 0.15, -0.15, 0.30, -0.30, 0.45)
S_CAP = 500.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cone_cols(contacts, dirs):
    cols = []
    for c in contacts:
        frame, mu, jac_t = c['frame'], c['mu'], c['J'].T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + float(p) * t1 + float(q) * t2))
    return np.asarray(cols).T


def threshold(muscle_cols, cone, b, lo, hi):
    def ok(sc):
        A = np.hstack([muscle_cols * sc, cone])
        scale = np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
        res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                      bounds=list(zip(lo, hi)) + [(0, None)] * cone.shape[1], method='highs',
                      options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
        return res.status == 0
    f1 = ok(1.0)
    low, high = (0.5, 1.0) if f1 else (1.0, 2.0)
    while high <= S_CAP and not ok(high):
        high *= 1.25
    if high > S_CAP:
        return dict(feasible_at_s_1=f1, threshold_midpoint=None, cap_reached=True,
                    infeasible_up_to_s=high / 1.25)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False)


def main():
    out = DESK / ('S2-multistart-ik-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  objective=('multi-start per-leg IK to push the seated family s BELOW the earlier one-path value '
                             '(the earlier value is only an upper bound because the IK has several local minima)'),
                  declared=dict(n_starts=N_STARTS, perturbations_rad=list(PERTURB),
                                seeds=['seed = ' + str(SEED_POSE) + ' (s = 0.9594120287401438 on the declared '
                                       'cone; labelled "delivered" in earlier wording)',
                                       'source pose vector = ' + str(SOURCE_POSE),
                                       'the same seed + fixed perturbations'],
                                seed_pose=str(SEED_POSE), seed_pose_s_on_declared_cone=0.9594120287401438,
                                parameters=dict(z_mm=Z0, footprint=SPREAD0, coxa_yaw_deg=YAW0, offsets=OFFSETS),
                                contract=dict(touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                                              nonfoot_clearance_mm_min=H_DECLARED,
                                              joint_limit_margin_rad=JOINT_MARGIN,
                                              max_joint_deviation_rad=MAX_DEVIATION),
                                objective_cone='declared pyramidal (axis-ray diamond)'),
                  nature_of_this_step='DECLARED pose search (multi-start), NOT behaviour evidence',
                  declared_calls=dict(project_mj_steps=0), absolute_force_values='NOT_ESTABLISHED')
    inputs = [Path(__file__), SOURCE_POSE, SEED_POSE]
    hashes = {str(p): sha(p) for p in inputs if Path(p).exists()}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        report['declared_calls']['project_model_loads'] = 1
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        other = [g for g in range(m.ngeom) if g != 0 and 'Tarsus' not in gname(g)]
        leg_geoms = {p: [g for g in foot if gname(g).startswith(p)] for p in LEGS}
        leg_joints, joint_of = {}, {}
        for p in LEGS:
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + p)]
            leg_joints[p] = [next(j for j in js if jname(j).endswith(s)) for s in JOINT_SUFFIXES]
            for s in JOINT_SUFFIXES:
                joint_of[(p, s)] = next(j for j in js if jname(j).endswith(s))
        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        seed_pose = np.load(SEED_POSE)['projected_seated_qpos'].copy()

        def gdist(geom, points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, 0, geom, DIST_CAP, ft))
            return (r, ft) if points else r

        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_xy = {p: min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])[1][3:5] for p in LEGS}

        def base_q():
            q = source.copy()
            q[2] = Z0
            for p in LEGS:
                j = joint_of[(p, 'Coxa_yaw')]
                q[m.jnt_qposadr[j]] = source[m.jnt_qposadr[j]] + math.radians(YAW0)
            for key, suffix in (('tp', 'Trochanter_pitch'), ('cp', 'Coxa_pitch'), ('cr', 'Coxa_roll')):
                for grp, legs in GROUPS.items():
                    for p in legs:
                        j = joint_of[(p, suffix)]
                        adr = m.jnt_qposadr[j]
                        q[adr] = float(np.clip(source[adr] + math.radians(OFFSETS[key][grp]),
                                               m.jnt_range[j, 0] + JOINT_MARGIN,
                                               m.jnt_range[j, 1] - JOINT_MARGIN))
            return q

        def solve_leg(p, q, seed, regularize=False):
            idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
            lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION)
                           for k, j in enumerate(leg_joints[p])])
            hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION)
                           for k, j in enumerate(leg_joints[p])])
            x0 = np.clip(seed, lo, hi)
            target = SPREAD0 * source_xy[p]

            def residual(x):
                qq = q.copy()
                qq[idx] = x
                d.qpos[:] = qq
                mujoco.mj_kinematics(m, d)
                best = min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])
                out = [[10.0 * (best[0] - 0.5 * (TOUCH_LOW + TOUCH_HIGH))],
                       [0.5 * (best[1][3] - target[0]), 0.5 * (best[1][4] - target[1])]]
                if regularize:
                    out.append(0.05 * (x - x0))
                return np.concatenate(out)
            sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                max_nfev=3000)
            return sol.x, int(sol.nfev), str(sol.status)

        def evaluate(q):
            d.qpos[:] = q
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] = report['declared_calls'].get(
                'project_forward_calls', 0) + 1
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(gdist(g) for g in leg_geoms[p])) for p in LEGS}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]])) for p in LEGS))
            contacts, feet = [], set()
            for c in d.contact:
                if c.efc_address < 0:
                    continue
                jac = np.zeros((3, m.nv))
                for geom, sign in ((c.geom2, 1), (c.geom1, -1)):
                    body = int(m.geom_bodyid[geom])
                    jp, jr = np.zeros_like(jac), np.zeros_like(jac)
                    mujoco.mj_jac(m, d, jp, jr, c.pos, body)
                    jac += sign * jp
                names = [gname(int(c.geom1)), gname(int(c.geom2))]
                for n in names:
                    if 'Tarsus' in n:
                        feet.add(n[:2])
                contacts.append(dict(frame=np.asarray(c.frame).reshape(3, 3),
                                     mu=np.asarray(c.friction)[:2], J=jac, names=names))
            nonfoot = sum(1 for c in contacts if not any('Tarsus' in n for n in c['names']))
            base = d.qfrc_actuator.copy()
            tgt = d.qfrc_bias - d.qfrc_passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                report['declared_calls']['project_forward_calls'] += 1
                R[:, i] = d.qfrc_actuator - base
            contract = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in leg_dist.values())
                        and clearance >= H_DECLARED and margin >= JOINT_MARGIN
                        and deviation <= MAX_DEVIATION + 1e-9 and nonfoot == 0 and len(contacts) > 0)
            return dict(q=q, contract=bool(contract), leg_dist=leg_dist, clearance=clearance, margin=margin,
                        deviation=deviation, contacts=len(contacts), nonfoot=nonfoot, feet=sorted(feet),
                        target=tgt, R=R, contactdict=contacts)

        qb = base_q()
        seed_vecs = {}
        for p in LEGS:
            idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
            seed_vecs[p] = [('seed_T122857', seed_pose[idx]), ('source_pose', source[idx])] + \
                           [('perturb%+.2f' % dlt, seed_pose[idx] + dlt) for dlt in PERTURB[1:]]

        def regularized_branch():
            q = qb.copy()
            per_leg = {}
            for p in LEGS:
                idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
                x, nfev, st = solve_leg(p, q, seed_pose[idx], regularize=True)
                q[idx] = x
                per_leg[p] = dict(seed='seed_T122857+regularized', nfev=nfev, status=st)
            sysd = evaluate(q)
            entry = dict(start_index='regularized', seeds=per_leg, contract=sysd['contract'],
                         clearance_mm=sysd['clearance'], deviation_rad=sysd['deviation'],
                         contacts=sysd['contacts'], nonfoot=sysd['nonfoot'], feet=sysd['feet'],
                         min_tarsus_mm=sysd['leg_dist'])
            if sysd['contract']:
                lo = m.actuator_ctrlrange[:, 0].copy()
                hi = m.actuator_ctrlrange[:, 1].copy()
                entry['s'] = threshold(sysd['R'], cone_cols(sysd['contactdict'], DIAMOND), sysd['target'], lo, hi)
            report.setdefault('start_table', []).append(entry)
            print('start regularized contract=%s s=%s' % (
                sysd['contract'], None if not entry.get('s') else entry['s'].get('threshold_midpoint')),
                flush=True)
            return entry, q, sysd

        branches = []
        for si in range(N_STARTS):
            q = qb.copy()
            per_leg = {}
            for p in LEGS:
                name, vec = seed_vecs[p][si]
                x, nfev, st = solve_leg(p, q, vec)
                q[[m.jnt_qposadr[j] for j in leg_joints[p]]] = x
                per_leg[p] = dict(seed=name, nfev=nfev, status=st)
            sysd = evaluate(q)
            entry = dict(start_index=si, seeds=per_leg, contract=sysd['contract'],
                         clearance_mm=sysd['clearance'], deviation_rad=sysd['deviation'],
                         contacts=sysd['contacts'], nonfoot=sysd['nonfoot'], feet=sysd['feet'],
                         min_tarsus_mm=sysd['leg_dist'])
            if sysd['contract']:
                lo = m.actuator_ctrlrange[:, 0].copy()
                hi = m.actuator_ctrlrange[:, 1].copy()
                entry['s'] = threshold(sysd['R'], cone_cols(sysd['contactdict'], DIAMOND), sysd['target'], lo, hi)
                if entry['s'].get('threshold_midpoint') is not None:
                    branches.append((entry['s']['threshold_midpoint'], q.copy(), entry))
            report.setdefault('start_table', []).append(entry)
            print('start %d (%s) contract=%s s=%s' % (si, seed_vecs['LF'][si][0], sysd['contract'],
                                                      None if not entry.get('s') else
                                                      entry['s'].get('threshold_midpoint')), flush=True)
        reg_entry, reg_q, reg_sysd = regularized_branch()
        if reg_entry.get('s') and reg_entry['s'].get('threshold_midpoint') is not None:
            branches.append((reg_entry['s']['threshold_midpoint'], reg_q.copy(), reg_entry))
        report['regularized_branch'] = dict(
            entry=reg_entry,
            note=('a branch that mirrors the residual of the earlier minimum-s script (including its 0.05*(x-x0) '
                  'term) at the same declared parameters: if its s differs from the un-regularized branches, the '
                  's value is provably dependent on the IK formulation, not only on the declared parameters'))
        report['family_minimum_is_only_an_upper_bound'] = dict(
            statement=('at the SAME seed and the SAME declared parameters, changing ONLY the per-leg IK residual weighting '
                       'moves the contract-valid branch value s: run_ik_weight_variants.py gives 0.9398100562393665 for the '
                       'weight sets (1000, 2.0) and (500, 1.0) and (2000, 4.0) and 0.9391964338719845 for (666.667, 1.333), '
                       'while this script\'s declared weighting (10.0, 0.5) gives 1.0923678502440453. Therefore '
                       '1.0923678502440453 is NOT a property of "this seed + these declared parameters" and every family '
                       'minimum is only an UPPER BOUND that depends on the IK formulation (residual weights and solver)'),
            my_recomputation={'declared_weighting_10_0_0p5': 1.0923678502440453,
                              'weights_1000_2p0': 0.9398100562393665, 'weights_500_1p0': 0.9398100562393665,
                              'weights_2000_4p0': 0.9398100562393665, 'weights_666p667_1p333': 0.9391964338719845},
            other_verifier_values_not_reproduced_here={'V_A': 0.9449028430853026, 'V_B': 0.9449022830507412,
                                                       'V_C': 0.939810056612167,
                                                       'note': 'its V_A/V_B labels gave 0.9398100562393665 on this machine; the '
                                                               '0.9449 values are that solver\'s own local minima and are kept here '
                                                               'as recorded, not deleted'},
            script='/run/media/sensen/Data2/cell_wound_prototype/run_ik_weight_variants.py',
            artifact='/home/sensen/Desktop/engine修复验收/S2-ik-weight-variants-20261015T133034/')
        if branches:
            branches.sort(key=lambda t: t[0])
            s_best, q_best, e_best = branches[0]
            report['best'] = dict(s_best=s_best, entry=e_best, n_starts=N_STARTS,
                                  n_branches_with_threshold=len(branches),
                                  n_distinct_values=len({round(t[0], 9) for t in branches}),
                                  all_branch_values=[round(t[0], 12) for t in branches])
            np.savez_compressed(out / 'best_pose_multistart.npz', projected_seated_qpos=q_best)
            report['best_pose_npz'] = str(out / 'best_pose_multistart.npz')
            report['comparison'] = dict(
                earlier_delivered_s=0.9594120346009731,
                reviewer_warm_start_s=0.9572098068892956,
                this_multistart_s=s_best,
                note=('the earlier values are per-IK-path upper bounds; this multi-start search over %d starts '
                      'found %.9f, and the number of distinct branch values is listed in best.n_distinct_values'
                      % (N_STARTS, s_best)))
        report['status'] = 'PASS_MULTISTART_IK_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        vals = [(e['s']['threshold_midpoint'] if e.get('s') and e['s'].get('threshold_midpoint') else None)
                for e in report['start_table']]
        axes[0].bar(range(len(vals)), [0 if v is None else v for v in vals], color='#3060b0')
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1')
        axes[0].axhline(0.9594120346009731, color='r', ls='--', label='earlier delivered 0.95941')
        axes[0].set_xticks(range(len(vals)))
        axes[0].set_xticklabels(['start %d' % i for i in range(len(vals))], fontsize=7)
        axes[0].set_ylabel('s (declared pyramidal axis-ray diamond)')
        axes[0].set_title('per-start s (0 = no threshold)')
        axes[0].legend(fontsize=7)
        axes[1].hist([v for v in vals if v], bins=6, color='#2f8f4f')
        axes[1].axvline(0.9594120346009731, color='r', ls='--', label='earlier 0.95941')
        if branches:
            axes[1].axvline(branches[0][0], color='g', ls=':', label='multistart best %.4f' % branches[0][0])
        axes[1].set_xlabel('s over the starts')
        axes[1].set_title('branch spread')
        axes[1].legend(fontsize=7)
        if branches:
            dist = e_best['min_tarsus_mm']
            axes[2].bar(range(len(LEGS)), [1000 * dist[p] for p in LEGS], color='#c08030')
            axes[2].axhspan(1000 * TOUCH_LOW, 1000 * TOUCH_HIGH, color='0.85', label='declared touch band')
            axes[2].set_xticks(range(len(LEGS)))
            axes[2].set_xticklabels(LEGS)
            axes[2].set_ylabel('lowest tarsus (um)')
            axes[2].set_title('best multistart pose: per-leg contact')
            axes[2].legend(fontsize=7)
        fig.suptitle('Multi-start IK on the declared cone - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'multistart-ik.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
    finally:
        report['input_hashes_after'] = {p: sha(p) for p in hashes}
        report['hashes_unchanged'] = report['input_hashes_after'] == hashes
        if not report['hashes_unchanged']:
            report['status'] = 'FAIL_INPUT_MUTATED'
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                    default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('best:', report.get('best', {}).get('s_best'), '| branches:',
              report.get('best', {}).get('all_branch_values'))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
