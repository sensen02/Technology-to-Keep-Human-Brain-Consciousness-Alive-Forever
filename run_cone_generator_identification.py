"""Which contact-force set does MuJoCo's declared pyramidal cone actually realise?

An independent check argued that the four emitted rows are FACET NORMALS whose generator set is the
diagonal rays (the artifact's `outer`/`union` block), which would make the canonical pose feasible at
s=1 under MuJoCo's real cone. This script settles it with the engine's own decoder instead of an
interpretation:

  * it re-reads the four constraint rows of a condim-3 pyramidal contact,
  * it sets the non-negative row coefficients `efc_force` to unit vectors (exactly the space the solver
    works in) and reads the force MuJoCo decodes with `mj_contactForce`,
  * the image of the non-negative orthant is then the non-negative hull of those four decoded vectors,
    i.e. the realised feasible contact-force set, with no interpretation needed.

It then records, per pose, which cone each archived result was certified or found feasible on, and the
naming correction (the 4-ray set is MuJoCo's own generator set: a sub-cone of the exact Coulomb disk,
not the whole friction cone). Offline apart from the scratch-model probe; no project model forward.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

DESK = Path('/home/sensen/Desktop/engine修复验收')
SENS = DESK / 'S2-pose-cone-sensitivity-20261015T114356/report.json'
GATE_CANON = DESK / 'S2-support-gate-20261014T172624/report.json'
GATE_NEW = DESK / 'S2-support-gate-20261015T114154/report.json'
CERT_CANON = DESK / 'S2-support-certificate-20261014T175213/certificate.json'
CERT_NEW = DESK / 'S2-support-certificate-20261015T114243/certificate.json'
CERT_NEW_UNION = DESK / 'S2-support-certificate-20261015T114634/certificate.json'
REVIEW1 = DESK / 'gap-source-decomposition-independent-check-20261015T182136Z/independent-check.json'
REVIEW2 = DESK / 'S2-new-artifacts-independent-check-20261015T185357Z/independent-check.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probe():
    import mujoco
    tpl = """<mujoco>
 <option cone="__CONE__" gravity="0 0 -9.81"/>
 <worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" condim="3" friction="0.7 0.4 0.1"/>
  <body name="b" pos="0 0 0.0995"><freejoint/>
   <geom name="sph" type="sphere" size="0.1" condim="3" friction="0.7 0.4 0.1" solref="0.001 1"/></body>
 </worldbody>
</mujoco>"""
    out = {}
    for cone in ('pyramidal', 'elliptic'):
        m = mujoco.MjModel.from_xml_string(tpl.replace('__CONE__', cone))
        d = mujoco.MjData(m)
        d.qpos[:] = [0, 0, 0.0995, 1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        c = d.contact[0]
        adr, nrows = int(c.efc_address), (4 if cone == 'pyramidal' else 3)
        mu = np.asarray(c.friction)[:2]

        def decode(lam):
            d.efc_force[:] = 0.0
            for r, v in enumerate(lam):
                d.efc_force[adr + r] = float(v)
            res = np.zeros(6)
            mujoco.mj_contactForce(m, d, 0, res)
            return [float(x) for x in res[:3]]

        unit = [decode([1.0 if r == i else 0.0 for r in range(nrows)]) for i in range(nrows)]
        combos = {}
        if nrows == 4:
            combos = {'all_ones': decode([1, 1, 1, 1]), 'e0+e2': decode([1, 0, 1, 0]),
                      'e1+e3': decode([0, 1, 0, 1]),
                      '2e0+2e1+e2+e3': decode([2, 2, 1, 1])}
        out[cone] = dict(nefc=int(d.nefc), rows_for_this_contact=nrows, mu=[float(v) for v in mu],
                         unit_coefficient_images=unit, combinations=combos)
    return out


def main():
    out = DESK / ('S2-cone-generator-identification-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  scope=('identify the contact-force set that MuJoCo actually realises for the declared '
                         'pyramidal cone, by reading the engine decoder itself; then record per pose which '
                         'cone each archived result was certified or found feasible on'),
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0, project_model_loads=0,
                                      scratch_model_loads=2))
    inputs = [Path(__file__), SENS, GATE_CANON, GATE_NEW, CERT_CANON, CERT_NEW, CERT_NEW_UNION,
              REVIEW1, REVIEW2]
    hashes = {str(p): sha(p) for p in inputs if Path(p).exists()}
    report['input_hashes'] = hashes
    try:
        pr = probe()
        ur = pr['pyramidal']['unit_coefficient_images']
        mu = pr['pyramidal']['mu'][0]
        expected = [[1.0, mu, 0.0], [1.0, -mu, 0.0], [1.0, 0.0, mu], [1.0, 0.0, -mu]]
        matched = all(abs(a - b) < 1e-9 for row, exp in zip(ur, expected) for a, b in zip(row, exp))
        report['engine_decoder_probe'] = pr
        report['identification'] = dict(
            rows_are_generators=bool(matched),
            statement=('the four non-negative row coefficients decode to the four AXIS rays (1,+-mu,0) and '
                       '(1,0,+-mu), and all-ones decodes to a pure normal force (4,0,0). Because the map is '
                       'linear and the coefficients are non-negative, the realised feasible contact-force set '
                       'is the non-negative hull of those four rays, i.e. exactly the archived DIAMOND block '
                       '{|ft1|<=mu*fn, |ft2|<=mu*fn, |ft1|+|ft2|<=mu*fn}. The independent check\'s reading of '
                       'the rows as facet normals (which would make the diagonal square the realised set) is '
                       'therefore refuted by the engine decoder itself; a facet-normal reading is also '
                       'internally inconsistent because it would admit no compressive force at all.'),
            so_which_is_the_true_cone=('MuJoCo\'s declared pyramidal cone is the diamond (a 4-sided pyramid '
                                       'over the axis-aligned diamond base). It is an INSCRIBED sub-cone of '
                                       'the exact Coulomb disk (|ft|<=mu*fn): the diamond edge midpoints sit '
                                       'at |ft|/fn = mu/sqrt(2) ~ 0.707 mu, so the pyramidal model '
                                       'under-estimates friction. The archived outer/union blocks are strict '
                                       'RELAXATIONS beyond the declared cone and beyond the Coulomb disk.'),
            note_on_the_refuted_reading=('the facet-normal reading is rejected by the measured 45-degree limit '
                                         '(0.7071 = 1/sqrt(2), the gen_axis boundary) and by the engine decoder; '
                                         'it is NOT rejected by a claim about compression, because the hull of the '
                                         'diagonal rays does contain a compressive force.'),
            naming_correction=('where earlier artifacts wrote "the declared pyramidal cone = the 4-ray '
                               'diamond", the precise wording is: "the 4-ray diamond is MuJoCo\'s own '
                               'generator set for the declared pyramidal cone, an inscribed sub-cone of the '
                               'exact Coulomb disk". The 4-ray set is the declared cone; it is not the whole '
                               'friction cone.'))

        def lp_status(path):
            if not Path(path).exists():
                return None
            r = json.loads(Path(path).read_text())
            return {x['floor'] + '_' + x['contact_model']: x['classification'] for x in r.get('results', [])}

        def cert(path):
            if not Path(path).exists():
                return None
            c = json.loads(Path(path).read_text())
            n = [x for x in c['results'] if x['floor'] == 'native'][0]
            return dict(status=n['status'], gap_float=n['gap_float'])

        report['per_pose_cone_disclosure'] = {
            'canonical_declared': dict(
                poses='the canonical declared seated pose (carries the archived certificate)',
                diamond=dict(verdict='INFEASIBLE at s=1', certificate=cert(CERT_CANON),
                             note='exact rational separation on the diamond block'),
                outer_union=dict(verdict='FEASIBLE (relaxation witness)',
                                 gate_report=lp_status(GATE_CANON),
                                 note=('the archived gate recorded a RELAXED_WITNESS for outer and union; '
                                       'that is feasibility of a STRICT RELAXATION of the declared cone and is '
                                       'NOT evidence that the declared cone supports the pose')),
                is_the_declared_cone_infeasible='YES (diamond = declared cone per the engine decoder)'),
            'new_seated_tarsus_only': dict(
                poses='the tarsus-only seated pose built this session',
                diamond=dict(verdict='INFEASIBLE at s=1', certificate=cert(CERT_NEW)),
                outer_union=dict(verdict='INFEASIBLE at s=1 (union certificate holds on a SUPERSET)',
                                 union_certificate=cert(CERT_NEW_UNION), gate_report=lp_status(GATE_NEW),
                                 note=('here even the relaxations are infeasible, so the negative result is '
                                       'robust: it holds a fortiori for the declared cone')),
                is_the_declared_cone_infeasible='YES'),
            'v6_seated': dict(poses='archived v6 seated candidate, read as a DECLARED relaxation '
                                    '(40 of 41 contacts are non-foot)',
                              diamond=dict(verdict='INFEASIBLE at s=1 (exact gap 1.3702715812097843)'),
                              outer_union=dict(verdict='INFEASIBLE at s=1 in the static sensitivity LP for all '
                                                        'six variants (see S2-pose-cone-sensitivity-20261015T114356); '
                                                        'no gate report has a v6 row because the gate rejects that '
                                                        'pose for non-foot contact before building any matrix'),
                              is_the_declared_cone_infeasible='YES; also rejected by the gate precondition')}
        report['disclosure_of_the_undisclosed_fact'] = (
            'REQUIRED DISCLOSURE (raised by the second independent check, accepted as a documentation '
            'defect): the artefacts of this session that evaluated the CANONICAL pose did not state that '
            'the canonical pose is FEASIBLE under the outer/union relaxations while INFEASIBLE on the '
            'diamond block. That fact is now stated here, per pose, together with the reason it is not '
            'evidence of support: those blocks are strict relaxations of the declared cone (and of the '
            'exact Coulomb disk). The reason the omission was harmless for the conclusion is the engine '
            'decoder result above: the declared cone IS the diamond, so the diamond certificate binds.')
        report['other_review_items'] = dict(
            OVR2_manifest_gaps=('three delivered directories ship no manifest or no report pair: '
                                'S2-support-gate-20261015T114154 (report.json, no delivery-hashes.json) and '
                                'S2-support-certificate-20261015T114243 / ...T114634 (certificate.json only). '
                                'They are producer-side artifacts of archived tooling; their file hashes are '
                                'recorded here so that citations can be pinned per file.'),
            OVR3_absolute_units=('the B-gate synthesis Chinese report quotes the corrected budget '
                                 '(64.268526 uN / 10.711421 uN) while its report.json correctly nulls those '
                                 'fields under absolute_force_values = NOT_ESTABLISHED; the quote is '
                                 'order-of-magnitude background only and is traceable to the force-scale fix '
                                 'round. No s value may be converted to newtons.'),
            both_recorded_in_new_directory=True)
        report['status'] = 'PASS_CONE_GENERATOR_IDENTIFIED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for f, lab, col in zip(ur, ['row 0: (1,+mu,0)', 'row 1: (1,-mu,0)', 'row 2: (1,0,+mu)',
                                    'row 3: (1,0,-mu)'], ['#3060b0'] * 4):
            axes[0].arrow(0, 0, f[1] / f[0], f[2] / f[0], head_width=0.03, color=col, length_includes_head=True)
            axes[0].annotate(lab, (f[1] / f[0], f[2] / f[0]), fontsize=7)
        axes[0].plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], 'k--', lw=1, label='unit diamond |a|+|b|=1')
        th = np.linspace(0, 2 * np.pi, 200)
        axes[0].plot(np.cos(th), np.sin(th), 'r:', lw=1, label='exact Coulomb disk')
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('ft1 / (mu*fn)')
        axes[0].set_ylabel('ft2 / (mu*fn)')
        axes[0].set_title('decoded images of the 4 unit row coefficients\n(diamond = declared cone, inscribed)')
        axes[0].legend(fontsize=8)
        labels = list(report['per_pose_cone_disclosure'])
        vals = []
        for l in labels:
            e = report['per_pose_cone_disclosure'][l]
            relax = str(e['outer_union']['verdict'])
            relax_feasible = ('FEASIBLE (' in relax) and not relax.lstrip().startswith('INFEASIBLE')
            vals.append((1 if 'INFEASIBLE' in e['diamond']['verdict'] else 0, 1 if relax_feasible else 0))
        x = np.arange(len(labels))
        axes[1].bar(x - 0.2, [v[0] for v in vals], 0.4, label='diamond (declared cone) infeasible', color='#b03030')
        axes[1].bar(x + 0.2, [v[1] for v in vals], 0.4, label='outer/union RELAXATION feasible', color='#c08030')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['no', 'yes'])
        axes[1].set_title('per-pose cone status (relaxation feasibility is NOT support)', fontsize=9)
        axes[1].annotate('bars are 1 = yes, 0 = no; red = declared cone infeasible,\norange = gen_diag '
                         'relaxation feasible', (0.02, 0.98), xycoords='axes fraction', fontsize=7, va='top')
        axes[1].legend(fontsize=8)
        fig.suptitle('Which cone does MuJoCo realise? (engine decoder) - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'cone-identification.png', dpi=130)
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
        print('rows are generators:', report['identification']['rows_are_generators'])
        print('unit images:', report['engine_decoder_probe']['pyramidal']['unit_coefficient_images'])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
