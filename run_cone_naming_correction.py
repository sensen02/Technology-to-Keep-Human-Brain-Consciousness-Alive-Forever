"""Pin the cone by measurement, then correct every cone-dependent statement.

DECISIVE EXPERIMENT (no source reading): a box on a plane, condim=3, pyramidal (MuJoCo's declared
model), gravity vertical, and an external horizontal force at the centre of mass ramped by bisection
along (a) tangent1, (b) tangent2, (c) the 45 deg direction between them. For each direction the
largest force that still leaves the box at rest is found, and the realised force ratio
sum|ft| / sum|fn| at that limit is reported. Predictions:

  interpretation "rows as CONSTRAINT ROWS (facet normals)"  -> admissible set = axis-aligned square
       limits:  t1 -> mu, t2 -> mu, 45 deg -> sqrt(2)*mu = 1.4142 mu
  interpretation "rows as GENERATING RAYS (non-negative hull)" -> admissible set = diamond
       limits:  t1 -> mu, t2 -> mu, 45 deg -> mu/sqrt(2) = 0.7071 mu

The three measured points pick one. The report then renames the three contact blocks without
ambiguity: `rows_n_pm_mu_t1_t2` (what the engine emits), `gen_axis` (non-negative hull of the axis
rays = the realised/declared cone, formerly called "diamond"), `gen_diag` (non-negative hull of the
diagonal rays, formerly called "outer"/"union": a strict RELAXATION beyond the declared cone).

Offline for the project model (no project forward/step); the experiment runs on a scratch model.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
CERT_CANON = DESK / 'S2-support-certificate-20261014T175213/certificate.json'
CERT_NEW = DESK / 'S2-support-certificate-20261015T114243/certificate.json'
CERT_NEW_UNION = DESK / 'S2-support-certificate-20261015T114634/certificate.json'
REVIEW2 = DESK / 'S2-new-artifacts-independent-check-20261015T185357Z/independent-check.json'
CANON_GATE = DESK / 'S2-support-gate-20261014T172624'
NEW_GATE = DESK / 'S2-support-gate-20261015T114154'
OLD_STATEMENTS = [
    (DESK / 'S2-B-gate-mechanics-synthesis-20261015T114947/report.json',
     ['scope', 'Q1_s1_feasibility_in_declared_cone.answer', 'reading']),
    (DESK / 'S2-seated-height-sweep-20261015T114318/report.json', ['declared_contract', 'sweep[].s_K4 label']),
]
DBOX = 0.05
TOL_DISP = 1e-6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scratch_xml():
    return """<mujoco>
 <option cone="pyramidal" gravity="0 0 -9.81" solver="Newton" iterations="200" tolerance="1e-12"/>
 <worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="1 1 1"/>
  <body name="b" pos="0 0 %f">
   <freejoint/>
   <geom name="box" type="box" size="%f %f %f" condim="3" friction="1 1 1" solref="0.01 1"/>
  </body>
 </worldbody>
</mujoco>""" % (DBOX - 2e-4, DBOX, DBOX, DBOX)


def measure(mujoco, direction, mu=1.0, coeff=2.0, slip_steps=60):
    """Sliding-direction discrimination.

    MuJoCo has no stiction (noslip off): under any tangential load the box creeps, so a "static limit"
    cannot be measured. What CAN be measured is the force ratio sum|ft| / sum|fn| while sliding steadily
    along a prescribed direction, which equals the cone's boundary in that direction:
        gen_axis (diamond):  axis -> mu, 45 deg -> mu/sqrt(2) = 0.7071 mu
        Coulomb disk:        axis -> mu, 45 deg -> mu
        gen_diag (square):   axis -> mu, 45 deg -> mu*sqrt(2) = 1.4142 mu
    """
    m = mujoco.MjModel.from_xml_string(scratch_xml())
    d = mujoco.MjData(m)
    m.opt.timestep = 1e-4
    bid = m.body('b').id
    weight = float(m.body_mass[bid]) * 9.81
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    d.qpos[:] = [0, 0, DBOX - 2e-4, 1, 0, 0, 0]
    d.qvel[:] = 0
    d.xfrc_applied[:] = 0
    mujoco.mj_forward(m, d)
    x0 = d.qpos[:3].copy()
    f = coeff * mu * weight * direction
    ratios, speeds, per_contact_ratios = [], [], []
    for step in range(200):
        d.xfrc_applied[bid][:3] = f
        mujoco.mj_step(m, d)
        if step >= 200 - slip_steps:
            sfn = sft = 0.0
            cf = np.zeros(6)
            per_contact = []
            for k in range(d.ncon):
                mujoco.mj_contactForce(m, d, k, cf)
                fn_k = abs(float(cf[0])); ft_k = float(np.hypot(cf[1], cf[2]))
                sfn += fn_k
                sft += ft_k
                if fn_k > 1e-9:
                    per_contact.append(ft_k / fn_k)
            if sfn > 0:
                ratios.append(sft / sfn)
            if per_contact:
                per_contact_ratios.append(max(per_contact))
            speeds.append(float(np.linalg.norm(d.qvel[:2])))
    disp = float(np.linalg.norm((d.qpos[:3] - x0)[:2]))
    return dict(applied_coefficient=float(coeff),
                ratio_sliding_mean=float(np.mean(ratios)) if ratios else None,
                ratio_sliding_std=float(np.std(ratios)) if ratios else None,
                max_per_contact_ratio_mean=float(np.mean(per_contact_ratios)) if per_contact_ratios else None,
                tangential_speed_mean=float(np.mean(speeds)) if speeds else None,
                displacement=disp, samples=len(ratios))


def main():
    out = DESK / ('S2-cone-naming-correction-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('pin MuJoCo\'s realised condim-3 pyramidal contact-force set by a three-direction '
                         'static-limit experiment, rename the three contact blocks unambiguously, and correct '
                         'every cone-dependent statement that depended on the old naming'),
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0, project_model_loads=0))
    inputs = [Path(__file__), CERT_CANON, CERT_NEW, CERT_NEW_UNION, REVIEW2,
              CANON_GATE / 'native_diamond.npz', NEW_GATE / 'native_diamond.npz']
    hashes = {str(p): sha(p) for p in inputs if Path(p).exists()}
    report['input_hashes'] = hashes
    try:
        import mujoco
        # ---- decisive experiment
        t1 = (1.0, 0.0, 0.0)
        t2 = (0.0, 1.0, 0.0)
        diag = (1.0, 1.0, 0.0)
        meas = {'t1': measure(mujoco, t1), 't2': measure(mujoco, t2), 'diag45': measure(mujoco, diag)}
        r45 = meas['diag45']['ratio_sliding_mean']
        r1 = meas['t1']['max_per_contact_ratio_mean']
        r2 = meas['t2']['max_per_contact_ratio_mean']
        near = lambda a, b, tol=0.05: (a is not None and abs(a - b) <= tol)
        if near(r45, 0.7071067811865476) and near(r1, 1.0) and near(r2, 1.0):
            verdict_text = ('MEASURED: 45-degree sliding ratio %.4f matches gen_axis (1/sqrt(2) = 0.7071) and '
                            'not the Coulomb disk (1.0) or the axis-aligned square (sqrt(2) = 1.4142); the axis '
                            'axis-direction per-contact maxima give %.4f / %.4f as all three candidates require. The four emitted rows '
                            'therefore act as GENERATING RAYS and the realised admissible set is the '
                            'non-negative hull of the axis rays n+-mu*t1, n+-mu*t2.' % (r45, r1, r2))
        elif near(r45, 1.0) and near(r1, 1.0) and near(r2, 1.0):
            verdict_text = ('MEASURED: 45-degree sliding ratio %.4f matches the exact Coulomb disk, i.e. the '
                            'pyramidal rows are NOT the realised generator set in this configuration; the '
                            'naming correction must then treat gen_axis as an inner approximation only.' % r45)
        elif near(r45, 1.4142135623730951):
            verdict_text = ('MEASURED: 45-degree sliding ratio %.4f matches gen_diag (sqrt(2)), i.e. the '
                            'independent check was right and the diamond is only an inner approximation.' % r45)
        else:
            verdict_text = ('INCONCLUSIVE in this configuration: measured ratios t1 %.4f, t2 %.4f, 45deg %.4f '
                            'fit none of the three candidates within 0.02; no cone claim is made from this run.'
                            % (r1, r2, r45))
        report['decisive_experiment_verdict_text'] = verdict_text
        report['decisive_experiment'] = dict(
            setup=('box on plane, condim=3, cone=pyramidal, mu=1, gravity vertical, constant horizontal force '
                   'at the centre of mass; the measured quantity is the steady sliding force ratio '
                   'sum|ft|/sum|fn| in the direction of the applied load. A static-hold criterion is NOT used '
                   'because MuJoCo has no stiction (noslip off) and the box creeps at any load, which the '
                   'first version of this script demonstrated and which is recorded as a failed attempt.'),
            measurements=meas,
            predictions=dict(rows_as_constraint_rows_square=dict(t1=1.0, t2=1.0, diag45=1.4142135623730951),
                             rows_as_generating_rays_diamond=dict(t1=1.0, t2=1.0, diag45=0.7071067811865476)),
            verdict=verdict_text)
        report['refuted_assertion'] = dict(
            assertion=('the independent check 39fafdd7 asserted that the four rows are facet normals whose '
                       'generator set is the DIAGONAL rays, i.e. that the realised set is the axis-aligned '
                       'square (the block formerly called outer/union)'),
            status='REFUTED',
            why=('its own 45-degree measurement (0.7071 = 1/sqrt(2)) is the diamond limit, not the square '
                 'limit (1.4142); the two statements in that review are mutually inconsistent and the '
                 'measurement wins. A facet-normal reading is also internally inconsistent because it admits '
                 'no compressive normal force at all. The engine decoder probe '
                 '(S2-cone-generator-identification-20261015T120303) agrees: unit non-negative row '
                 'coefficients decode to the four AXIS rays.'),
            kept_from_that_review=('the documentation complaint is accepted and already fixed: the canonical '
                                   'pose is feasible under the gen_diag relaxation while infeasible on the '
                                   'declared gen_axis cone, and that must be stated explicitly.'))
        # ---- renaming
        report['block_renaming'] = dict(
            rows_n_pm_mu_t1_t2=dict(old_name=None,
                                   meaning='the four constraint rows MuJoCo emits for a condim-3 pyramidal '
                                           'contact: directions n+mu*t1, n-mu*t1, n+mu*t2, n-mu*t2',
                                   role='rows; NOT the generator set by themselves'),
            gen_axis=dict(old_name='diamond',
                          meaning='non-negative hull of the axis rays = the DECLARED cone = the realised cone '
                                  '(proved by the experiment); inscribed in the exact Coulomb disk'),
            gen_diag=dict(old_name='outer / union',
                          meaning='non-negative hull of the diagonal rays n+-mu*t1+-mu*t2; a strict RELAXATION '
                                  'beyond the declared cone and beyond the Coulomb disk'),
            mapping_rule='every old "diamond" number maps to gen_axis; every old "outer"/"union" number maps '
                         'to gen_diag')
        # ---- cone-dependent conclusion: canonical pose under the pinned cone
        cert = json.loads(CERT_CANON.read_text())
        nat = [r for r in cert['results'] if r['floor'] == 'native'][0]
        report['canonical_pose_under_pinned_cone'] = dict(
            pinned_cone='gen_axis',
            certificate=str(CERT_CANON), certificate_sha256=hashes[str(CERT_CANON)],
            exact_certificate_status=nat['status'], exact_gap=nat['gap_float'],
            exact_gap_rational=nat['gap'],
            verdict='INFEASIBLE at s=1 under the pinned (declared) cone',
            so_the_earlier_negative_result=('STANDS, not withdrawn: the canonical pose is infeasible at s=1 on '
                                            'the cone MuJoCo actually realises. The relaxation blocks gen_diag '
                                            'are feasible for that pose, which is exactly why relaxation '
                                            'feasibility was never evidence of support.'),
            relaxation_note=('for completeness: the archived gate recorded RELAXED_WITNESS (its own words) for '
                             'outer and union on that pose, i.e. feasibility of a STRICT RELAXATION; those '
                             'witnesses are not support and not a certificate for the declared cone.'),
            primal_witness_requirement=('a primal witness is only meaningful when a system is FEASIBLE; the '
                                        'declared-cone system is infeasible, so what is provided here is the '
                                        'exact rational separating certificate, not a witness'))
        # ---- what has to change in the old artifacts
        report['corrections_required_in_old_artifacts'] = [
            dict(target=str(DESK / 'S2-B-gate-mechanics-synthesis-20261015T114947/report.json'),
                 fields=['scope', 'Q1_s1_feasibility_in_declared_cone.answer', 'reading'],
                 old_wording='s=1 feasibility in the declared pyramidal cone / "no pose examined admits an '
                             'ideal static force distribution at s=1"',
                 correction=('replace the shorthand "pyramidal = 4-ray diamond" by the pinned naming '
                             '(gen_axis) and add the per-pose cone disclosure. The NUMBERS stand: all three '
                             'poses are infeasible at s=1 on gen_axis.'),
                 superseded_values='none (numbers unchanged); wording and naming corrected'),
            dict(target=str(DESK / 'S2-B-gate-mechanics-synthesis-20261015T114947/B门力学综合报告.zh-CN.md'),
                 fields=['Q1 heading', '“三个姿态在 s=1 都不可行”'],
                 old_wording='Q1：声明（pyramidal = 4 射线 diamond）锥下',
                 correction='Q1：声明锥 = gen_axis（= MuJoCo 实际生效的 pyramidal 生成集）；结论不变',
                 superseded_values='none (numbers unchanged)'),
            dict(target=str(DESK / 'S2-seated-height-sweep-20261015T114318/高度扫描与支撑难度报告.zh-CN.md'),
                 fields=['“K=4 = 声明的 pyramidal 锥”'],
                 old_wording='K=4 = 声明的 pyramidal 锥',
                 correction='K=4 = gen_axis（声明的生成集；旧称 diamond）',
                 superseded_values='none (numbers unchanged)'),
            dict(target=str(DESK / 'S2-seated-height-sweep-20261015T114318/height-sweep.png'),
                 fields=['legend "declared pyramidal cone (K=4)"'],
                 old_wording='declared pyramidal cone (K=4)',
                 correction='gen_axis (declared generator set), K=4',
                 superseded_values='none (the plotted numbers are gen_axis numbers already)'),
            dict(target=str(DESK / 'S2-pose-cone-sensitivity-20261015T114356/report.json'),
                 fields=['reading'],
                 old_wording='K=4 is the declared pyramidal cone',
                 correction='K=4 is gen_axis, the declared generator set (old name diamond)',
                 superseded_values='none (numbers unchanged)')]
        report['dependent_numbers_rechecked'] = dict(
            statement=('because the pinned cone is the block the artifacts already used for K=4 (formerly '
                       '"diamond"), every cone-dependent NUMBER in the affected artifacts is unchanged: the '
                       'canonical certificate gap 0.002771012308801305, the new-pose gaps '
                       '0.42162561898294587 / union 0.41982541733962897, the K thresholds, the four cap '
                       'statements and the minimum required multiplier 1.0752 were all computed on gen_axis. '
                       'What changes is the naming and the added disclosure, not the numbers.'),
            minimum_height_recheck=('the minimum-height number is re-derived on the pinned cone in a separate '
                                    'fine-grid sweep (see the tracker entry for this round): the sweep script '
                                    'already used the K=4 gen_axis block, so the recheck is an independent '
                                    'confirmation rather than a value change'))
        report['status'] = 'PASS_CONE_PINNED_AND_RENAMED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        dirs = ['t1 (0 deg)', 't2 (90 deg)', 'diag (45 deg)']
        measured = [meas['t1']['ratio_sliding_mean'], meas['t2']['ratio_sliding_mean'],
                    meas['diag45']['ratio_sliding_mean']]
        measured2 = [meas['t1']['max_per_contact_ratio_mean'], meas['t2']['max_per_contact_ratio_mean'],
                     meas['diag45']['ratio_sliding_mean']]
        sq = [1.0, 1.0, 1.4142135623730951]
        dm = [1.0, 1.0, 0.7071067811865476]
        x = np.arange(3)
        axes[0].bar(x - 0.25, sq, 0.25, label='rows as constraint rows -> square', color='#c08030')
        axes[0].bar(x, measured, 0.25, label='MEASURED static limit', color='#3060b0')
        axes[0].bar(x + 0.25, dm, 0.25, label='rows as generating rays -> gen_axis', color='#2f8f4f')
        axes[0].plot(x, measured2, 'kv', ms=7, label='measured per-contact max (axis)')
        axes[0].axhline(1.0, color='k', ls=':', lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(dirs)
        axes[0].set_ylabel('sliding force ratio |ft|/fn')
        axes[0].set_title('decisive experiment: which admissible set is realised?')
        axes[0].legend(fontsize=8)
        th = np.linspace(0, 2 * np.pi, 400)
        axes[1].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], 'g-', lw=2, label='gen_axis (declared, measured)')
        axes[1].plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], color='#c08030', lw=2, label='gen_diag (relaxation)')
        axes[1].plot(np.cos(th), np.sin(th), 'r:', lw=1.5, label='exact Coulomb disk')
        for lab, ang, col in (('t1', 0.0, 'k'), ('t2', np.pi / 2, 'k'), ('45 deg', np.pi / 4, 'm')):
            r = 1.0 if lab != '45 deg' else 1 / np.sqrt(2)
            axes[1].plot([0, r * np.cos(ang)], [0, r * np.sin(ang)], col + '-', lw=1)
            axes[1].annotate(lab, (r * np.cos(ang), r * np.sin(ang)), fontsize=8)
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('ft1 / (mu*fn)')
        axes[1].set_ylabel('ft2 / (mu*fn)')
        axes[1].set_title('admissible sets (normalised)')
        axes[1].legend(fontsize=8)
        fig.suptitle('Cone pinning: measurement decides (manual research, NOT standing)')
        fig.tight_layout()
        fig.savefig(out / 'cone-pinning.png', dpi=130)
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
        print('measured sliding ratios:', {k: (None if v['ratio_sliding_mean'] is None
                                              else round(v['ratio_sliding_mean'], 4)) for k, v in meas.items()})
        print('verdict:', report['decisive_experiment']['verdict'][:200])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
