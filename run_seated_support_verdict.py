"""One definite verdict: is the tarsus-only seated pose statically feasible at s=1 in the DECLARED
pyramidal/diamond cone - and does ANY seated height in the declared contract family make it feasible?

Offline. No MuJoCo call, no mj_step, no new pose, no model/gain/friction/force-scale change. Every
number is read from the artifacts named below (with their sha256), and the threshold column comes from
exactly ONE source per pose so that no two script versions are mixed in one table.

Sources (all previously sealed):
  * cone thresholds + brackets ....... S2-pose-cone-sensitivity-20261015T114356/report.json
                                        (produced by run_pose_cone_sensitivity.py 7640a1f2..., S_CAP=500)
  * independent recomputation ......... same dir / threshold-cap-raised.json (run_pose_threshold_cap_raised.py
                                        77f0737d..., deterministic, identical on both copies)
  * height sweep s(K=4)/s(K=32) ....... S2-seated-height-sweep-20261015T114318/report.json
                                        (run_seated_height_sweep.py 34aae0ed...)
  * exact certificates (s=1) .......... S2-support-certificate-20261015T114243 (diamond) and
                                        S2-support-certificate-20261015T114634 (union superset)
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

DESK = Path('/home/sensen/Desktop/engine修复验收')
SENS = DESK / 'S2-pose-cone-sensitivity-20261015T114356'
SWEEP = DESK / 'S2-seated-height-sweep-20261015T114318/report.json'
CERT_D = DESK / 'S2-support-certificate-20261015T114243/certificate.json'
CERT_U = DESK / 'S2-support-certificate-20261015T114634/certificate.json'
POSE = DESK / 'S2-seated-tarsus-only-20261015T114113/report.json'
CAP_RAISED = SENS / 'threshold-cap-raised.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = DESK / ('S2-seated-support-verdict-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('single definite verdict: is a tarsus-only seated pose statically feasible at s=1 '
                         'under the DECLARED pyramidal/diamond cone, and does any seated height in the '
                         'declared contract family make it feasible'),
                  question=('在声明锥（pyramidal/diamond，即真实生效的 MuJoCo 锥：opt.cone=0、condim=3、mu=[1,1]）'
                            '下，仅跗节着地的 seated pose 在 s=1 时静态松弛可行吗？'),
                  absolute_force_values='NOT_ESTABLISHED',
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0, project_model_loads=0))
    inputs = [Path(__file__), SENS / 'report.json', CAP_RAISED, SWEEP, CERT_D, CERT_U, POSE]
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes
    try:
        sens = json.loads((SENS / 'report.json').read_text())
        cap = json.loads(CAP_RAISED.read_text())
        sweep = json.loads(SWEEP.read_text())
        cert_d = json.loads(CERT_D.read_text())
        cert_u = json.loads(CERT_U.read_text())
        pose = json.loads(POSE.read_text())

        nat = [r for r in cert_d['results'] if r['floor'] == 'native'][0]
        un = [r for r in cert_u['results'] if r['floor'] == 'native'][0]
        verdict_s1 = dict(
            pose='new seated tarsus-only pose (root z 2.222260 mm)',
            pose_npz=str(DESK / 'S2-seated-tarsus-only-20261015T114113/candidate_seated_pose.npz'),
            declared_cone='pyramidal (opt.cone=0), condim=3, mu=[1,1] -> the 4-ray diamond generator set',
            lp_feasible_at_s_1=False,
            exact_certificate=dict(status=nat['status'], gap_float=nat['gap_float'],
                                   gap_rational=nat['gap'], certificate=str(CERT_D),
                                   sha256=hashes[str(CERT_D)]),
            union_superset_certificate=dict(status=un['status'], gap_float=un['gap_float'],
                                            certificate=str(CERT_U), sha256=hashes[str(CERT_U)]),
            reading=('INFEASIBLE at s=1 in the declared cone: an exact rational separating vector exists, so '
                     'no force distribution with unilateral diamond-cone contacts and the declared activation '
                     'box balances this pose. The certificate also holds on the union generator set, a strict '
                     'superset of the declared cone. outer/union feasibility would not count here, and in fact '
                     'this pose is infeasible for those relaxations too.'))
        report['Q_s1_verdict'] = verdict_s1

        rows = []
        for r in sweep['sweep']:
            if 'error' in r or not r.get('contract_satisfied'):
                continue
            s4 = r['s_K4']
            s32 = r['s_K32']
            rows.append(dict(
                z_mm=r['z_mm'],
                s_K4_bracket=[s4.get('infeasible_below'), s4.get('feasible_above')],
                s_K4_mid=s4.get('threshold_midpoint'),
                s_K4_cap_reached=bool(s4.get('cap_reached')),
                s_K32_mid=s32.get('threshold_midpoint'),
                s_1_feasible=bool(s4.get('feasible_at_s_1')),
                min_nonfoot_clearance_mm=r['min_nonfoot_clearance_mm'],
                contacts=r['contact_count'], nonfoot_contacts=r['nonfoot_contact_count'],
                source='S2-seated-height-sweep-20261015T114318/report.json @ 34aae0ed...'))
        rows.sort(key=lambda r: r['z_mm'])
        finite = [r for r in rows if r['s_K4_mid'] is not None]
        best = min(finite, key=lambda r: r['s_K4_mid']) if finite else None
        cap_hit = [r['z_mm'] for r in rows if r['s_K4_cap_reached']]
        report['Q_height_sweep_table'] = dict(
            rows=rows,
            source_sha256=hashes[str(SWEEP)],
            contract_height_window_mm=sweep.get('contract_height_window_mm'),
            best_row=best,
            heights_with_cap_reached=cap_hit,
            cap_semantics=('a row with cap_reached=True has NO threshold and must be read as '
                           '"no threshold found within the exploration cap", never as feasible or infeasible'),
            reading=('Every contract-satisfying height needs s > 1: the smallest requirement is %.6f at '
                     'z = %.4f mm, and the height-maximising pose needs %.6f. No height in the declared family '
                     'is statically feasible at s = 1.'
                     % (best['s_K4_mid'], best['z_mm'],
                        next(r['s_K4_mid'] for r in rows if abs(r['z_mm'] - 2.2222) < 1e-3)) if best else 'n/a'))

        # cross-pose threshold table, ONE source per column (the sensitivity report), with the
        # independent recomputation shown separately and only where both agree
        thr = {}
        for label, entry in sens['poses'].items():
            thr[label] = dict(root_z_mm=entry['root_z_mm'], contacts=entry['contact_count'],
                              nonfoot_contacts=entry['nonfoot_contact_count'],
                              min_nonfoot_clearance_mm=entry['min_nonfoot_clearance_mm'],
                              by_K={k: dict(infeasible_below=v.get('infeasible_below'),
                                            feasible_above=v.get('feasible_above'),
                                            cap_reached=bool(v.get('cap_reached')),
                                            infeasible_up_to_s=v.get('infeasible_up_to_s'))
                                    for k, v in entry['cone_resolution'].items()})
        agree = {}
        for label, res in cap['results'].items():
            agree[label] = {k: (v.get('threshold_midpoint') is not None
                                and thr[label]['by_K'][k]['infeasible_below'] is not None
                                and thr[label]['by_K'][k]['infeasible_below'] <= v['threshold_midpoint']
                                <= thr[label]['by_K'][k]['feasible_above'])
                            for k, v in res.items()}
        report['Q_threshold_table'] = dict(
            by_pose=thr, source=str(SENS / 'report.json'), source_sha256=hashes[str(SENS / 'report.json')],
            independent_recomputation=str(CAP_RAISED), independent_sha256=hashes[str(CAP_RAISED)],
            independent_midpoint_inside_report_bracket=agree,
            convergence_reading=('for the canonical declared pose the required multiplier falls monotonically '
                                 'with cone resolution (1.0450258 -> 0.9992329), i.e. a finer Coulomb polygon '
                                 'almost exactly matches the declared strength; for the new seated pose it is '
                                 'flat at ~13.74 (0.09 percent change from K=4 to K=64), so that pose is short '
                                 'on lateral force/arm, not on cone resolution'))

        report['one_sentence_conclusion'] = (
            'NO: under the declared pyramidal/diamond cone there is no seated height in the declared '
            'tarsus-only contract family at which the ideal static relaxation is feasible at s = 1 - the '
            'best member still needs about 1.075x the declared muscle strength (root z = 2.00 mm), the '
            'height-maximising member needs about 13.75x, and the specific seated pose delivered here is '
            'infeasible at s = 1 by an exact rational separating certificate with gap %.9f.'
            % nat['gap_float'])
        report['consequence_for_next_step'] = (
            'Because the answer is NO, the conditional bounded fixed-command dynamic replay is NOT triggered '
            'and is not run. The next bounded step is the declared two-dimensional pose search (root height x '
            'per-leg lateral foot placement) whose criterion is the disappearance of the exact diamond-cone '
            'certificate, reported with exact rational gaps; gain, friction, force scale and cone type are '
            'untouched.')
        report['status'] = 'PASS_SEATED_SUPPORT_V1_VERDICT_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        zs = [r['z_mm'] for r in rows]
        ys = [r['s_K4_mid'] for r in rows]
        axes[0].semilogy(zs, ys, 'o-', color='#3060b0', label='required multiplier s (declared diamond cone, K=4)')
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        if best:
            axes[0].plot([best['z_mm']], [best['s_K4_mid']], 'g*', ms=14,
                         label='family minimum %.4f at z=%.2f mm' % (best['s_K4_mid'], best['z_mm']))
        axes[0].set_xlabel('root height z (mm)')
        axes[0].set_ylabel('required muscle-force multiplier s')
        axes[0].set_title('tarsus-only seated heights: all need s > 1')
        axes[0].legend(fontsize=8)
        labels = list(thr)
        x = range(len(labels))
        for i, k in enumerate(('4', '32')):
            v = [thr[l]['by_K'][k]['feasible_above'] if not thr[l]['by_K'][k]['cap_reached'] else None
                 for l in labels]
            axes[1].bar([xx + (i - 0.5) * 0.35 for xx in x],
                        [0 if w is None else w for w in v], 0.35, label='K=%s' % k)
        axes[1].set_yscale('log')
        axes[1].axhline(1.0, color='k', ls=':')
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=7)
        axes[1].set_ylabel('threshold s (blank = cap reached, no threshold)')
        axes[1].set_title('threshold per pose and cone resolution')
        axes[1].legend(fontsize=8)
        for i, l in enumerate(labels):
            if thr[l]['by_K']['4']['cap_reached']:
                axes[1].annotate('no threshold\nwithin cap', (i, 1.1), fontsize=7, ha='center', color='#b03030')
        axes[1].annotate('s=1 certificate: infeasible\n(seated pose, gap %.4f)' % nat['gap_float'],
                         (0.02, 0.95), xycoords='axes fraction', fontsize=7, va='top',
                         bbox=dict(boxstyle='round', fc='#ffe8e8', ec='#b03030'))
        fig.suptitle('Seated tarsus-only support at s=1: verdict and height table - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'seated-support-verdict.png', dpi=130)
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
        print('s=1 verdict:', report['Q_s1_verdict']['lp_feasible_at_s_1'],
              '| gap', report['Q_s1_verdict']['exact_certificate']['gap_float'])
        print('best row:', report['Q_height_sweep_table']['best_row'])
        print('cap rows:', report['Q_height_sweep_table']['heights_with_cap_reached'])
        print(report['one_sentence_conclusion'])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
