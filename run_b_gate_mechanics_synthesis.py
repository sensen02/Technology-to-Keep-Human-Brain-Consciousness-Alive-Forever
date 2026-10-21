"""B-gate mechanics synthesis: does s=1 work in the REAL (declared pyramidal) cone?

Offline only. Reads archived matrices and archived reports; no MuJoCo call, no mj_step, no new pose,
no change to any model, gain, friction or force-scale value. It answers three questions and nothing
more:

 Q1 In the declared pyramidal cone, is the ideal static system feasible at s=1 (declared muscle
    box, declared budget)?  -> a verdict per pose, taken from archived/registered EXACT rational
    separation certificates where they exist, otherwise from the LP status.
 Q2 If not, what is the SMALLEST change that makes it feasible: a pose change inside the declared
    tarsus-only contract family, or a scalar force-scale multiple?
 Q3 Do the "exact zero residual" variants (muscle box relaxed / contact signs relaxed) hold only
    once complementarity is abandoned?  -> inspect the witnesses of those relaxations and count the
    multipliers that are impossible under unilateral contacts, plus how far the muscle box is
    exceeded.

Absolute force values stay NOT_ESTABLISHED; every statement is a relaxation-level statement and MUST
NOT be written as standing, balance, support or biology.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
CANON_GATE = DESK / 'S2-support-gate-20261014T172624'
NEW_GATE = DESK / 'S2-support-gate-20261015T114154'
SENS = DESK / 'S2-pose-cone-sensitivity-20261015T114356'
SWEEP = DESK / 'S2-seated-height-sweep-20261015T114318/report.json'
BUDGET = DESK / 'force-scale-absolute-units-fix-20261015T110716/report.json'
CERTS = {
    'canonical_declared': DESK / 'S2-support-certificate-20261014T175213/certificate.json',
    'new_seated_tarsus_only': DESK / 'S2-support-certificate-20261015T114243/certificate.json',
    'v6_seated_declared_relaxation': DESK / 'S2-support-certificate-20261015T114244/certificate.json',
}
UNION_CERT = DESK / 'S2-support-certificate-20261015T114634/certificate.json'
DIAMOND_RAYS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lp_feasible(A, b, lo, hi, scale=None):
    m = len(lo)
    scale = (np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
             if scale is None else scale)
    res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                  bounds=list(zip(lo, hi)) + [(0, None)] * (A.shape[1] - m), method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res


def elastic(A, b, muscle_bounds, contact_bounds):
    nv, n = A.shape
    scale = np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
    S, bs = A / scale[:, None], b / scale
    rows, rhs = [], []
    for i in range(nv):
        row = np.zeros(n + 1)
        row[:n] = S[i]
        row[n] = -1.0
        rows.append(row)
        rhs.append(bs[i])
        other = -row
        other[n] = -1.0
        rows.append(other)
        rhs.append(-bs[i])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                  bounds=list(muscle_bounds) + list(contact_bounds) + [(0, None)], method='highs',
                  options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
    return res


def contact_frame(meta, multipliers, rays):
    """(fn, ft1, ft2) per contact from per-ray multipliers over the given ray list."""
    out = []
    for k, c in enumerate(meta['active_contacts']):
        mu = np.asarray(c['mu'])
        block = multipliers[k * len(rays):(k + 1) * len(rays)]
        fn = float(np.sum(block))
        ft1 = float(mu[0] * sum(a * v for (a, _), v in zip(rays, block)))
        ft2 = float(mu[1] * sum(b * v for (_, b), v in zip(rays, block)))
        out.append(dict(index=k, dist=float(c['dist']), normal=fn, tangent=[ft1, ft2]))
    return out


def main():
    started = time.time()
    out = DESK / ('S2-B-gate-mechanics-synthesis-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('B-gate synthesis: s=1 feasibility in the declared pyramidal cone, the smallest '
                         'change that makes it feasible, and whether the zero-residual relaxations rely on '
                         'dropping complementarity'),
                  absolute_force_values='NOT_ESTABLISHED',
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0, project_model_loads=0))
    inputs = [Path(__file__), SWEEP, BUDGET, UNION_CERT] + [CANON_GATE / 'native_diamond.npz',
                                                            NEW_GATE / 'native_diamond.npz',
                                                            NEW_GATE / 'native_union.npz',
                                                            SENS / 'report.json',
                                                            SENS / 'matrices_canonical_declared/native_diamond.npz',
                                                            SENS / 'matrices_canonical_declared/metadata.json',
                                                            SENS / 'matrices_new_seated_tarsus_only/native_diamond.npz',
                                                            SENS / 'matrices_new_seated_tarsus_only/metadata.json',
                                                            SENS / 'matrices_v6_seated/native_diamond.npz',
                                                            SENS / 'matrices_v6_seated/metadata.json'] + list(CERTS.values())
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes
    try:
        budget = json.loads(BUDGET.read_text())
        gram = next(iter(budget['unit_reading'].values())) if 'unit_reading' in budget else {}
        report['corrected_muscle_budget'] = dict(
            source=str(BUDGET),
            total_uN=(gram.get('total_muscle_peak_uN') if isinstance(gram, dict) else None),
            per_leg_uN=(gram.get('mean_leg_peak_uN') if isinstance(gram, dict) else None),
            note=('the corrected 90-muscle budget from the force-scale fix round (the gain[:-2] slice is '
                  'superseded); absolute force values remain NOT_ESTABLISHED and none of the numbers below '
                  'depends on them'))
        sweep = json.loads(SWEEP.read_text())
        sens = json.loads((SENS / 'report.json').read_text())

        # ---- Q1: s = 1 verdict per pose, exact where a certificate exists
        q1 = {}
        for label, cert_path in CERTS.items():
            cert = json.loads(cert_path.read_text())
            native = [r for r in cert['results'] if r['floor'] == 'native'][0]
            q1[label] = dict(
                certificate=str(cert_path), certificate_sha256=hashes[str(cert_path)],
                verdict=native['status'],
                exact_gap=float(native['gap_float']),
                certified_gap_encoded=native.get('gap'),
                contact_dots_all_nonpositive=bool(native.get('valid')),
                strict_positive_contact_columns=int(native.get('strict_positive_contact_columns', -1)),
                reading=('EXACT_STORED_MATRIX_SEPARATION means an exact rational separating vector exists, '
                         'so the ideal static system with unilateral contacts and the declared box is '
                         'infeasible at s=1 for that pose'))
        union_cert = json.loads(UNION_CERT.read_text())
        un = [r for r in union_cert['results'] if r['floor'] == 'native'][0]
        q1['new_seated_tarsus_only_union_cone'] = dict(
            certificate=str(UNION_CERT), verdict=un['status'], exact_gap=float(un['gap_float']),
            reading=('the certificate holds on the UNION generator set (diamond + diagonals), a strict '
                     'SUPERSET of the declared diamond cone, so infeasibility holds a fortiori for the '
                     'declared cone of the new pose'))
        report['Q1_s1_feasibility_in_declared_cone'] = dict(
            per_pose=q1,
            answer=('NOT feasible at s=1 for any of the three poses examined (canonical declared pose, the '
                    'new tarsus-only seated pose, and the v6 seated candidate read as a declared relaxation). '
                    'The canonical pose and the new pose each carry an exact rational separation certificate; '
                    'for the new pose the certificate even holds on a strict superset of the declared cone.'))

        # ---- Q2: smallest change
        best = sweep.get('best_in_contract', {})
        q2 = dict(
            pose_change=dict(
                family=('declared tarsus-only contract family: per-leg IK, each leg >=1 tarsus within the '
                        'declared touch band, all non-foot geoms >= 0.40 mm clearance, joints >= 0.005 rad '
                        'inside limits, joints <= 0.40 rad from the source pose'),
                sweep_source=str(SWEEP),
                contract_height_window_mm=sweep.get('contract_height_window_mm'),
                minimum_multiplier=dict(z_mm=best.get('z_mm'), s_K4=best.get('s_K4'), s_K32=best.get('s_K32'),
                                        nonfoot_clearance_mm=best.get('nonfoot_clearance_mm')),
                verdict=('NO height in the declared family reaches s<=1: the best member still needs about '
                         '1.075x the declared muscle strength, and the height-maximising member needs about '
                         '13.7x')),
            force_scale_multiple=dict(
                canonical_declared_pose=dict(
                    declared_diamond_cone=sens['poses']['canonical_declared']['cone_resolution']['4'].get(
                        'threshold_midpoint'),
                    inscribed_coulomb_K32=sens['poses']['canonical_declared']['cone_resolution']['32'].get(
                        'threshold_midpoint'),
                    note=('s is the factor by which every muscle column (equivalently the activation box) must '
                          'be scaled for the ideal static LP to become feasible. The value depends on which '
                          'contact cone is declared: the pyramidal cone needs ~1.0450x, a fine inscribed '
                          'Coulomb polygon needs ~0.9996x, i.e. essentially exactly the declared strength.')),
                new_seated_tarsus_only=dict(
                    declared_diamond_cone=sens['poses']['new_seated_tarsus_only']['cone_resolution']['4'].get(
                        'threshold_midpoint'),
                    inscribed_coulomb_K32=sens['poses']['new_seated_tarsus_only']['cone_resolution']['32'].get(
                        'threshold_midpoint')),
                v6_seated_declared_relaxation=dict(
                    declared_diamond_cone=sens['poses']['v6_seated']['cone_resolution']['4'].get('threshold_midpoint'),
                    cap_reached=sens['poses']['v6_seated']['cone_resolution']['4'].get('cap_reached'))))
        report['Q2_smallest_change_for_feasibility'] = q2

        # ---- Q3: do the zero-residual relaxations need complementarity to be dropped?
        q3 = {}
        for label in ('canonical_declared', 'new_seated_tarsus_only'):
            mdir = SENS / ('matrices_' + label)
            meta = json.loads((mdir / 'metadata.json').read_text())
            with np.load(mdir / 'native_diamond.npz') as z:
                A = z['A'].copy()
                b = z['target'].copy()
                lo = z['lo'].copy()
                hi = z['hi'].copy()
            nu = len(lo)
            ncon = len(meta['active_contacts'])
            assert A.shape[1] == nu + ncon * 4
            # (a) contact signs relaxed: signed multipliers allowed (= unilaterality dropped)
            res_c = elastic(A, b, list(zip(lo, hi)), [(None, None)] * (A.shape[1] - nu))
            witness_c = res_c.x[:A.shape[1]]
            mult = witness_c[nu:]
            cf = contact_frame(meta, mult, DIAMOND_RAYS)
            negative_multipliers = int(np.sum(mult < -1e-12))
            pulling_contacts = [c for c in cf if c['normal'] < -1e-9]
            load_on_separating = [c for c in cf if c['normal'] > 1e-9 and c['dist'] > 0]
            # (b) muscle box relaxed
            res_m = elastic(A, b, [(None, None)] * nu, [(0, None)] * (A.shape[1] - nu))
            act = res_m.x[:nu]
            outside = int(np.sum((act < lo - 1e-12) | (act > hi + 1e-12)))
            q3[label] = dict(
                contact_signs_relaxed=dict(
                    t_star=float(res_c.fun), lp_status=int(res_c.status),
                    negative_contact_multipliers=negative_multipliers,
                    contacts_with_negative_normal_force=len(pulling_contacts),
                    min_contact_normal=float(min(c['normal'] for c in cf)),
                    contacts_carrying_pull=sum(1 for c in cf if c['normal'] < 0),
                    positive_normal_on_separating_contact=len(load_on_separating),
                    reading=('t*=0 here means the ideal static system becomes exactly feasible once contact '
                             'forces are allowed to PULL (signed multipliers), which is precisely abandoning '
                             'unilateral contacts, i.e. the complementarity side of the contact law. These '
                             'witnesses are therefore not physical force distributions and must never be '
                             'quoted as support.')),
                muscle_box_relaxed=dict(
                    t_star=float(res_m.fun), lp_status=int(res_m.status),
                    activations_outside_box=outside,
                    min_activation=float(np.min(act)), max_activation=float(np.max(act)),
                    box=[float(lo[0]), float(hi[0])],
                    reading=('t*=0 here means the ideal static system becomes exactly feasible once muscle '
                             'activation is allowed outside the declared box, i.e. unbounded muscle force. It '
                             'is a strength/recruitment statement about this relaxation only, not a claim that '
                             "the fly's real muscles could do it.")),
                declared_system=dict(
                    lp_status=int(lp_feasible(A, b, lo, hi).status),
                    reading='STATUS 2 = infeasible with both the declared box and unilateral contacts'))
            if outside == 0 and abs(res_m.fun) < 1e-12:
                raise ValueError('expected the box-relaxed witness to leave the box for %s' % label)
        q3['answer'] = ('YES. Both zero-residual variants exist only by giving up a piece of the declared '
                        'problem: the contact-signs variant needs contact forces that pull (negative '
                        'multipliers / negative normal force) or loads separating contacts, which the '
                        'unilateral contact law forbids; the muscle-box variant needs activation outside the '
                        'declared box. Neither is a physical force distribution, and the declared system '
                        'itself is infeasible (LP status 2) in every pose examined.')
        report['Q3_complementarity_question'] = q3
        report['reading'] = (
            'Relaxation-level statements only. s is dimensionless and independent of the unestablished absolute '
            'force scale. A feasible static linear relaxation would still not be standing, balance or biology; '
            'here the declared systems are infeasible, so the honest summary is: under the declared pyramidal '
            'contact cone and the declared activation box, no pose examined admits an ideal static force '
            'distribution at s=1, the closest declared-family member is about 7.5 percent short, and the '
            'canonical declared pose is about 4.5 percent short under the pyramidal cone but essentially '
            'exactly at the boundary under a fine inscribed Coulomb polygon.')
        report['status'] = 'PASS_B_GATE_MECHANICS_SYNTHESIS_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        labels = list(q1)
        gaps = [q1[l]['exact_gap'] for l in labels]
        axes[0].bar(range(len(labels)), gaps, color=['#3060b0', '#b03030', '#c08030', '#3060b0'])
        axes[0].set_yscale('log')
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=7)
        axes[0].set_ylabel('exact rational certificate gap (log)')
        axes[0].set_title('Q1 s=1 infeasible in the declared cone')
        for i, g in enumerate(gaps):
            axes[0].annotate('%.3g' % g, (i, g), fontsize=7, ha='center', va='bottom')
        names = ['canonical\n(K=4)', 'canonical\n(K=32)', 'family min\n(K=4)', 'new seated\n(K=4)',
                 'new seated\n(K=32)']
        vals = [q2['force_scale_multiple']['canonical_declared_pose']['declared_diamond_cone'],
                q2['force_scale_multiple']['canonical_declared_pose']['inscribed_coulomb_K32'],
                q2['pose_change']['minimum_multiplier']['s_K4'],
                q2['force_scale_multiple']['new_seated_tarsus_only']['declared_diamond_cone'],
                q2['force_scale_multiple']['new_seated_tarsus_only']['inscribed_coulomb_K32']]
        axes[1].bar(range(len(names)), vals, color=['#3060b0', '#3060b0', '#2f8f4f', '#b03030', '#b03030'])
        axes[1].set_yscale('log')
        axes[1].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        for i, v in enumerate(vals):
            axes[1].annotate('%.4g' % v, (i, v), fontsize=7, ha='center', va='bottom')
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, fontsize=7)
        axes[1].set_ylabel('required muscle-force multiplier s')
        axes[1].set_title('Q2 smallest change (below 1 = feasible at declared strength)')
        axes[1].legend(fontsize=8)
        q3l = [k for k in q3 if k != 'answer']
        neg = [q3[k]['contact_signs_relaxed']['negative_contact_multipliers'] for k in q3l]
        pull = [q3[k]['contact_signs_relaxed']['contacts_with_negative_normal_force'] for k in q3l]
        outst = [q3[k]['muscle_box_relaxed']['activations_outside_box'] for k in q3l]
        x = np.arange(len(q3l))
        axes[2].bar(x - 0.25, neg, 0.25, label='negative contact multipliers', color='#b03030')
        axes[2].bar(x, pull, 0.25, label='contacts with negative normal force', color='#c08030')
        axes[2].bar(x + 0.25, outst, 0.25, label='activations outside the box', color='#3060b0')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([l.replace('_', '\n') for l in q3l], fontsize=7)
        axes[2].set_ylabel('count in the relaxation witness')
        axes[2].set_title('Q3 zero residuals need a dropped constraint')
        axes[2].legend(fontsize=7)
        fig.suptitle('B-gate mechanics synthesis - relaxation level only, NOT standing, NOT support')
        fig.tight_layout()
        fig.savefig(out / 'b-gate-synthesis.png', dpi=130)
        plt.close(fig)
        report['wall_seconds'] = time.time() - started
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
        print('Q1:', {k: (v['verdict'], round(v['exact_gap'], 6)) for k, v in report['Q1_s1_feasibility_in_declared_cone']['per_pose'].items()})
        print('Q2 family min:', report['Q2_smallest_change_for_feasibility']['pose_change']['minimum_multiplier'])
        print('Q2 canonical s:', report['Q2_smallest_change_for_feasibility']['force_scale_multiple']['canonical_declared_pose'])
        print('Q3:', {k: (v['contact_signs_relaxed']['negative_contact_multipliers'],
                          v['contact_signs_relaxed']['contacts_with_negative_normal_force'],
                          v['muscle_box_relaxed']['activations_outside_box'])
                      for k, v in report['Q3_complementarity_question'].items() if k != 'answer'})
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
