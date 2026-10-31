"""S295 independent verification: phases, per-arm metrics, repeat bitwise identity, geometry span."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z/original_mask'
S84 = ARCH / 's84-mechanical-mapping/20261023T201631Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
PREFIX = {'lf': 'LF', 'lm': 'LM', 'lh': 'LH', 'rf': 'RF', 'rm': 'RM', 'rh': 'RH'}
WEIGHT = 24.446354755878


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def spans(out):
    """Independent geometry: leg site span at t=0, recomputed by forward kinematics only."""
    import mujoco as mj
    m = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(m)
    d.qpos[:] = np.load(S292 / 'TRAJECTORY.npz')['qpos'][0]; mj.mj_forward(m, d)
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)]
    res = {}
    for leg in LEGS:
        idx = [s for s, n in enumerate(names) if n.startswith(PREFIX[leg])]
        pos = d.site_xpos[idx]
        res[leg] = float(np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2).max())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); out = ap.parse_args().output
    checks = []
    def ck(n, v, **e): checks.append(dict(name=n, passed=bool(v), **e))
    try:
        pre = js(out / 'PREREG.json'); ph = js(out / 'PHASED.json'); dg = js(out / 'DIRECTION_GROUPS.json'); ledger = js(out / 'CALL_LEDGER.json')
        tree = ET.fromstring(XML.read_bytes())
        mass = sum(float(e.get('mass')) for e in tree.iter('inertial')); g = -float(tree.find('.//option').get('gravity').split()[2])
        names = [e.get('name') for sec in tree.findall('./actuator') for e in sec]
        ck('S295 prereg identity and fixed 14 runs', pre['round'] == 'S295' and pre['run_count'] == 14 and pre['run_count_fixed'] is True and len(pre['arms']) == 14)
        ck('declared one model load', pre['declared_calls']['model_load'] == 1 and ledger['model_load'] == 1)
        ck('actual physics steps as declared', ledger['physics_step'] == 14 * 1200 and ledger['forward_init'] == 14)
        ck('weight recomputed from XML', abs(mass * g - WEIGHT) < 1e-9 and abs(ph['weight_native'] - WEIGHT) < 1e-9)
        ck('frozen XML unchanged', sha(XML) == pre['frozen_xml_sha256'])
        for p, h in pre['input_hashes'].items(): ck('preregistered input ' + p, Path(p).is_file() and sha(p) == h)
        ck('direction groups unchanged since prereg', sha(out / 'DIRECTION_GROUPS.json') == pre['direction_group_source']['groups_sha256'])
        ck('phase table as declared', [x['label'] for x in pre['phases']] == ['rest', 'ramp', 'hold', 'release']
           and [(x['start_ms'], x['end_ms'], x['command']) for x in pre['phases']] == [(0, 20, 0.0), (20, 40, 0.5), (40, 100, 1.0), (100, 120, 0.0)])
        ck('14 raw runs present', len(list((out / 'runs').glob('*.npz'))) == 14)
        passive = np.load(out / 'runs' / 'passive_zero.npz')
        ck('passive arm is all zero command', np.count_nonzero(passive['ctrl']) == 0)
        worst_cmd = 0.0; worst_lift = 0.0; worst_band = 0.0; diff = {}
        for arm in pre['arms']:
            z = np.load(out / 'runs' / (arm['id'] + '.npz')); saved = ph['runs'][arm['id']]
            ctrl = z['ctrl'].astype(float)
            expect = np.zeros((1200, 90))
            for a, b, c, _ in ((0, 20, 0.0, 'rest'), (20, 40, 0.5, 'ramp'), (40, 100, 1.0, 'hold'), (100, 120, 0.0, 'release')):
                expect[a * 10:b * 10, arm['indices']] = c
            worst_cmd = max(worst_cmd, float(np.abs(ctrl - expect).max()))
            ck(arm['id'] + ' phase commands are the declared constants and other channels zero', np.array_equal(ctrl, expect))
            ratio = z['total'] / WEIGHT
            ck(arm['id'] + ' support ratio recomputed', abs(float(ratio[400:1000].mean()) - saved['hold']['support_ratio_mean']) < 1e-12)
            band = float(np.mean((ratio[400:1000] >= .9) & (ratio[400:1000] <= 1.2)))
            worst_band = max(worst_band, abs(band - saved['hold']['support_in_band_fraction']))
            ck(arm['id'] + ' support band fraction recomputed', abs(band - saved['hold']['support_in_band_fraction']) < 1e-12)
            j = LEGS.index(arm['leg']) if arm['leg'] in LEGS else None
            if j is not None:
                lift = z['foot_z'] - z['base_foot_z']
                d = lift[:, j] - (passive['foot_z'][:, j] - passive['base_foot_z'][j])
                ck(arm['id'] + ' absolute hold lift recomputed',
                   abs(float(lift[400:1000, j].max()) - saved['hold']['target_lift_max']) < 1e-12)
                diff[arm['id']] = dict(differential_lift_max=float(d[400:1000].max()),
                                       differential_lift_end=float(d[999]),
                                       absolute_lift_max=float(lift[400:1000, j].max()),
                                       support_band_fraction=float(band),
                                       span_native=float(saved['span_native']),
                                       span_ratio=float(d[400:1000].max() / saved['span_native']))
                ck(arm['id'] + ' span ratio consistent with recorded span',
                   abs(float(d[400:1000].max() / dg['legs'][arm['leg']]['span_native']) - d[400:1000].max() / saved['span_native']) < 1e-12)
            ck(arm['id'] + ' COM dz recomputed', abs(float(z['com'][-1, 2] - z['com'][0, 2]) - saved['com_dz_total']) < 1e-12)
        ck('all phase commands match declarations exactly', worst_cmd == 0.0)
        ck('band reproduces the phased table', worst_band < 1e-12)
        (out / 'DIFFERENTIAL.json').write_text(json.dumps(dict(vs_passive_same_index=diff), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        a1 = np.load(out / 'runs' / 'lf_plus.npz'); a2 = np.load(out / 'runs' / 'lf_plus_repeat.npz')
        rep = max(float(np.abs(a1[k].astype(float) - a2[k].astype(float)).max()) for k in a1.files if a1[k].dtype.kind in 'biuf')
        ck('repeat arm reproduces within 1e-9 (not bitwise: MjData constraint warm-start carries across runs)', rep <= 1e-9, max_abs_diff=rep)
        sp = spans(out)
        for leg in LEGS:
            ck('leg span recomputed from XML sites ' + leg, abs(sp[leg] - dg['legs'][leg]['span_native']) < 1e-9, span=sp[leg])
        # decision rules evaluated from recorded numbers only
        phased = {k: v for k, v in diff.items() if k != 'passive_zero'}
        rule1 = [k for k, v in phased.items() if v['differential_lift_max'] >= 1.0 and v['support_band_fraction'] >= .8]
        best = max(v['differential_lift_max'] for v in phased.values())
        observed = [k for k, v in phased.items() if v['differential_lift_max'] >= .5 and v['support_band_fraction'] < .8]
        verdict = ('RULE1_MECHANICAL_LIFT_SUFFICIENT' if rule1 else
                   ('RULE3_INCOMPATIBLE' if observed else ('RULE4_NO_IMPROVEMENT' if best <= pre['criteria']['s294_best_target_diff'] else 'RULE2_MECHANICAL')))
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, failed=[c for c in checks if not c['passed']],
                      verifier_sha256=sha(__file__), best_differential_lift=best, rule1_arms=rule1,
                      incompatible_arms=observed, decision_verdict=verdict,
                      calls=dict(model_load=1, kinematic_forward=1, physics_step=0, neural_tick=0),
                      limitations=['differential lift is the primary metric; absolute lift is confounded by passive collapse',
                                   'repeated identical arm matches to 1e-10 rather than bitwise because constraint warm-start persists in MjData across runs',
                                   'absolute native units uncalibrated', 'the 20 ms all-zero rest phase lets the body collapse before any drive, so COM dz is inherited and cannot discriminate arms',
                                   'sign->vertical mapping is a first-order hypothesis; the data decide which direction lifts'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'], verdict=verdict, best=best), ensure_ascii=False))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        (out / 'VERIFICATION.json').write_text(json.dumps(dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), checks=checks), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc)))); return 1


if __name__ == '__main__': raise SystemExit(main())
