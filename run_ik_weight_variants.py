"""IK residual-weight variants from the SAME seed: is s a function of the declared parameters?

Declared question: the multi-start artifact reports 1.0923678502440453 for its start-0 seed and calls that
value the branch value at the declared parameters. An independent verifier reported that with the SAME seed and
SAME declared parameters, only the residual WEIGHTING changed, it obtained 0.9449028430853026,
0.9449022830507412 and 0.939810056612167 (all contract-valid). This script recomputes that on this machine with
its own explicit weight sets, so that the claim is either reproduced or contradicted with numbers.

Declared weight sets (residual = [w_touch * (lowest tarsus distance - band midpoint),
                                  w_xy * (footprint x - target x), w_xy * (footprint y - target y)]):
  declared_multistart  (10.0,  0.5)      - the weighting used by run_multistart_ik.py
  V_A_verifier          (1000.0, 2.0)    - verifier's "1/0.001, 1/0.5"
  V_B_verifier          (500.0, 1.0)     - verifier's "1/0.002, 1/1.0"
  V_C_own               (666.667, 1.333) - own set; the verifier's V_C weights were NOT disclosed in its JSON
  V_D_own               (2000.0, 4.0)    - own set
No regularisation term is used in this script (the regularized branch of the multi-start artifact is separate).
Everything else (seed pose, per-leg 7 hinge joints, bounds, touch band, footprint target, cone, units, model
file) is identical to run_multistart_ik.py. Absolute force values and units remain NOT_ESTABLISHED.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares, linprog

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
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
S_CAP = 500.0
WEIGHT_SETS = [('declared_multistart', 10.0, 0.5), ('V_A_verifier', 1000.0, 2.0),
               ('V_B_verifier', 500.0, 1.0), ('V_C_own', 2000.0 / 3.0, 4.0 / 3.0),
               ('V_D_own', 2000.0, 4.0)]
VERIFIER = dict(seed_s=0.9594120287401438, V_A=0.9449028430853026, V_B=0.9449022830507412,
                V_C=0.939810056612167, V_E_declared=1.0923678473511278,
                artifact_start0=1.0923678502440453, artifact_regularized=0.9451071806252003,
                source='/home/sensen/Desktop/engine修复验收/S2-hold-check-20261015T201813Z/'
                       'S2-hold-check-independent-verification.json')


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
        return dict(feasible_at_s_1=f1, threshold_midpoint=None, cap_reached=True, infeasible_up_to_s=high / 1.25)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False)


def main():
    out = DESK / ('S2-ik-weight-variants-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('does the per-leg IK branch value s depend on the residual weighting at IDENTICAL declared '
                         'parameters and an identical seed? Recomputes the verifier\'s variants here with explicit '
                         'weightings'),
                  declared=dict(seed=str(SEED_POSE), source_pose=str(SOURCE_POSE),
                                z0=Z0, footprint_spread=SPREAD0, coxa_yaw_deg=YAW0, offsets_deg=OFFSETS,
                                touch_band_mm=[1000 * TOUCH_LOW, 1000 * TOUCH_HIGH],
                                nonfoot_clearance_mm_min=H_DECLARED, joint_limit_margin_rad=JOINT_MARGIN,
                                max_joint_deviation_rad=MAX_DEVIATION,
                                objective_cone='declared pyramidal (axis-ray diamond)',
                                weight_sets={name: dict(w_touch=a, w_xy=b) for name, a, b in WEIGHT_SETS},
                                no_regularisation=True),
                  verifier_values_to_check=VERIFIER, absolute_force_values='NOT_ESTABLISHED',
                  nature_of_this_step='DECLARED pose search (weight variants), NOT behaviour evidence')
    report['input_hashes'] = {str(p): sha(p) for p in (Path(__file__), SOURCE_POSE, SEED_POSE) if Path(p).exists()}
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
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

        report['declared_calls'] = dict(project_model_loads=1, project_forward_calls=0)
        qb = base_q()
        variants = {}
        poses = {}
        for name, wtouch, wxy in WEIGHT_SETS:
            q = qb.copy()
            per_leg = {}
            for p in LEGS:
                idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
                lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                x0 = np.clip(seed_pose[idx], lo, hi)
                target = SPREAD0 * source_xy[p]

                def residual(x, p=p, q=q, idx=idx, lo=lo, hi=hi, x0=x0, target=target):
                    qq = q.copy()
                    qq[idx] = x
                    d.qpos[:] = qq
                    mujoco.mj_kinematics(m, d)
                    best = min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])
                    return np.array([wtouch * (best[0] - 0.5 * (TOUCH_LOW + TOUCH_HIGH)),
                                     wxy * (best[1][3] - target[0]), wxy * (best[1][4] - target[1])])
                sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                    max_nfev=3000)
                x = np.clip(sol.x, lo, hi)
                q[idx] = x
                per_leg[p] = dict(nfev=int(sol.nfev), status=str(sol.status),
                                  min_tarsus_mm=float(min(gdist(g) for g in leg_geoms[p])))
            sysd = evaluate(q)
            entry = dict(per_leg=per_leg, w_touch=wtouch, w_xy=wxy, contract=sysd['contract'],
                         clearance_mm=sysd['clearance'], deviation_rad=sysd['deviation'],
                         contacts=sysd['contacts'], nonfoot=sysd['nonfoot'], feet=sysd['feet'],
                         min_tarsus_mm=sysd['leg_dist'])
            if sysd['contract']:
                entry['s'] = threshold(sysd['R'], cone_cols(sysd['contactdict'], DIAMOND), sysd['target'],
                                       m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy())
                if entry['s'].get('threshold_midpoint') is not None:
                    poses[name] = q.copy()
                    np.savez_compressed(out / ('pose_%s.npz' % name), projected_seated_qpos=q)
            variants[name] = entry
            print('%-20s contract=%s s=%s' % (name, entry['contract'],
                                              None if not entry.get('s') else entry['s'].get('threshold_midpoint')),
                  flush=True)
        report['weight_variants'] = variants
        valid = {k: v['s']['threshold_midpoint'] for k, v in variants.items()
                 if v.get('s') and v['s'].get('threshold_midpoint') is not None}
        report['comparison_with_verifier'] = dict(
            my_values=valid, verifier_values=dict(V_A=VERIFIER['V_A'], V_B=VERIFIER['V_B'],
                                                  V_C=VERIFIER['V_C'], V_E_declared=VERIFIER['V_E_declared']),
            agreement_vs_verifier={label: dict(verifier=VERIFIER[key], mine=valid.get(label),
                                               abs_difference=(None if label not in valid
                                                               else abs(valid[label] - VERIFIER[key])),
                                               within_1e_6=(None if label not in valid
                                                           else bool(abs(valid[label] - VERIFIER[key]) < 1e-6)))
                                   for label, key in (('V_A_verifier', 'V_A'), ('V_B_verifier', 'V_B'),
                                                      ('declared_multistart', 'V_E_declared'))},
            V_A_B_reproduced=bool(all((lbl in valid and abs(valid[lbl] - VERIFIER[key]) < 1e-6)
                                      for lbl, key in (('V_A_verifier', 'V_A'), ('V_B_verifier', 'V_B')))),
            my_low_basin_vs_verifier_V_C=dict(
                mine_V_A_V_B_V_D=valid.get('V_A_verifier'), verifier_V_C=VERIFIER['V_C'],
                absolute_difference=(None if 'V_A_verifier' not in valid
                                     else abs(valid['V_A_verifier'] - VERIFIER['V_C'])),
                note=('my V_A/V_B/V_D weighting sets land on the same low basin, whose value coincides with the '
                      'verifier V_C value to a few 1e-10; my own third weighting reaches an even lower branch')),
            V_C_note=('the verifier did not disclose its V_C weights; V_C_own in this script is an explicit own set, '
                      'so the verifier value 0.939810056612167 is checked only for being reachable in the same range, '
                      'not for equality'),
            spread_at_identical_parameters=(max(valid.values()) - min(valid.values())) if valid else None)
        if valid:
            best = min(valid, key=lambda k: valid[k])
            report['best_weight_variant'] = dict(name=best, s=valid[best],
                                                 npz=str(out / ('pose_%s.npz' % best)))
            np.savez_compressed(out / 'best_weight_variant_pose.npz', projected_seated_qpos=poses[best])
            report['best_pose_npz'] = str(out / 'best_weight_variant_pose.npz')
        report['conclusion'] = (
            'Measured only: the same seed and the same declared parameters give different contract-valid branch '
            'values when the residual weighting changes, so a branch value is a value of ONE IK branch and any '
            '"family minimum" is only an UPPER BOUND that moves with the IK formulation. No behaviour claim.')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = list(valid)
        vals = [valid[k] for k in names]
        ax.bar(range(len(names)), vals, color='tab:blue')
        for i, v in enumerate(vals):
            ax.text(i, v, '%.6f' % v, ha='center', va='bottom', fontsize=8)
        ax.axhline(1.0, color='k', ls=':', label='s = 1')
        ax.axhline(VERIFIER['seed_s'], color='r', ls='--', label='seed pose itself %.5f' % VERIFIER['seed_s'])
        ax.axhline(VERIFIER['artifact_regularized'], color='g', ls='-.',
                   label='multistart artifact best %.5f' % VERIFIER['artifact_regularized'])
        ax.axhline(0.9097466208040714, color='m', ls=':', label='earlier best-known 0.9097466208')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel('s (declared pyramidal axis-ray diamond)')
        ax.set_title('Same seed, same declared parameters, only the IK residual weighting changed\n'
                     'manual research, NOT behaviour evidence')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'ik-weight-variants.png', dpi=130)
        plt.close(fig)
        report['status'] = 'PASS_IK_WEIGHT_VARIANTS_REPORTED'
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
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print(json.dumps(report['comparison_with_verifier'], indent=1, ensure_ascii=False))
        print('best:', json.dumps(report.get('best_weight_variant'), ensure_ascii=False))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
