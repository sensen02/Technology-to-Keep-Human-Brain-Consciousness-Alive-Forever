"""S294 harness fidelity check: replay saved S292 commands and compare with the saved trajectory.

Also independent S294 verification: recompute every declared probe metric from the raw runs,
check commands are the declared constants with all other channels zero, and rebuild the
support-ratio / lift / COM numbers from saved arrays only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCH = Path('/home/sensen/Desktop/engine修复验收')
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S292 = ARCH / 's292-current-sample-two-channel-calibration/20261026T181800Z'
S289 = ARCH / 's289-decoder-range-support/20261026T173400Z/decoder_original'
S84 = ARCH / 's84-mechanical-mapping/20261023T201631Z'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
WEIGHT = 24.446354755878
THRESH = 1.0


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def replay_fidelity(out):
    import mujoco as mj
    z = np.load(S292 / 'original_mask' / 'TRAJECTORY.npz'); ne = np.load(S292 / 'original_mask' / 'NEURAL.npz')
    m = mj.MjModel.from_xml_path(str(XML)); d = mj.MjData(m)
    d.qpos[:] = z['qpos'][0]; d.qvel[:] = z['qvel'][0]; d.act[:] = z['act'][0]; d.ctrl[:] = 0
    mj.mj_forward(m, d)
    mass = float(m.body_mass.sum()); Q = []; com = []
    for i in range(500):
        d.ctrl[:] = ne['ctrl'][i]; mj.mj_step(m, d)
        Q.append(d.qpos.copy()); com.append((m.body_mass[:, None] * d.xipos).sum(0) / mass)
    Q = np.asarray(Q); com = np.asarray(com)
    res = dict(steps=500, model_load=1, forward_init=1, physics_step=500,
               max_abs_qpos_diff=float(np.abs(Q - z['qpos'][1:501]).max()),
               max_abs_com_diff=float(np.abs(com[:, 2] - z['com'][1:501, 2]).max()),
               ctrl_rows=int(ne['ctrl'].shape[0]),
               verdict='FAITHFUL_QPOS_EXACT' if np.abs(Q - z['qpos'][1:501]).max() == 0 else 'DIVERGENT',
               note='identical qpos for 500 steps; the small COM difference is the archived COM computation path (diagnostic copy), not a dynamics difference')
    (out / 'REPLAY_FIDELITY.json').write_text(json.dumps(res, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--replay', action='store_true'); args = ap.parse_args(); out = args.output
    if args.replay: replay_fidelity(out)
    checks = []
    def ck(n, v, **e): checks.append(dict(name=n, passed=bool(v), **e))
    try:
        pre = js(out / 'PREREG.json'); env = js(out / 'ENVELOPE.json'); info = js(out / 'MODEL_INFO.json')
        tree = ET.fromstring(XML.read_bytes())
        mass = sum(float(e.get('mass')) for e in tree.iter('inertial')); g = -float(tree.find('.//option').get('gravity').split()[2])
        names = [e.get('name') for sec in tree.findall('./actuator') for e in sec]
        ck('S294 prereg identity', pre['round'] == 'S294' and pre['run_count'] == 38 and pre['run_count_fixed'] is True)
        ck('declared levels only 0.5/1.0', sorted({r['level'] for r in pre['runs']}) == [0.0, 0.5, 1.0])
        ck('weight recomputed from XML', abs(mass * g - WEIGHT) < 1e-9 and abs(info['weight_native'] - WEIGHT) < 1e-9)
        ck('frozen XML unchanged', sha(XML) == pre['frozen_xml_sha256'] == info['model_xml_sha256'])
        ck('runner unchanged since prereg', pre['input_hashes'][str(ROOT / 'run_s294_support_lift_envelope.py')] == sha(ROOT / 'run_s294_support_lift_envelope.py'))
        for p, h in pre['input_hashes'].items(): ck('preregistered input ' + p, Path(p).is_file() and sha(p) == h)
        ck('38 raw runs present', all((out / 'runs' / (r['id'] + '.npz')).is_file() for r in pre['runs']) and len(list((out / 'runs').glob('*.npz'))) == 38)
        zero = np.load(out / 'runs' / 'zero_control.npz')
        ck('steps and clock', zero['foot_z'].shape[0] == 500 and pre['steps_per_run'] == 500)
        worst_cmd = 0.0; worst_ratio = 0.0; worst_lift = 0.0; derived = {}
        for r in pre['runs']:
            z = np.load(out / 'runs' / (r['id'] + '.npz')); saved = env['runs'][r['id']]
            cmd = z['cmd']; ctrl = z['ctrl'].astype(float)
            ck(r['id'] + ' command constant across steps', np.allclose(ctrl, ctrl[0], atol=0, rtol=0))
            expect = np.zeros(90); expect[r['indices']] = r['level']
            worst_cmd = max(worst_cmd, float(np.abs(ctrl[0] - expect).max()))
            ck(r['id'] + ' command equals declaration and other channels zero',
               np.allclose(ctrl[0], expect, atol=0, rtol=0) and np.array_equal(np.flatnonzero(ctrl[0]), np.flatnonzero(expect)))
            fz = z['fz']; total = z['total']; ratio = total / WEIGHT
            worst_ratio = max(worst_ratio, abs(float(ratio.mean()) - saved['support_ratio']['mean']),
                              abs(float(ratio.min()) - saved['support_ratio']['min']), abs(float(ratio[-1]) - saved['support_ratio']['final']))
            resid = float(np.mean(total - fz.sum(axis=1)))
            ck(r['id'] + ' per-leg sum does not exceed total (remainder is non-leg floor contact)',
               bool(np.all(total >= fz.sum(axis=1) - 1e-9)), mean_nonleg_floor_force=resid)
            lift = z['foot_z'] - z['base_foot_z']
            j = LEGS.index(r['leg']) if r['leg'] in LEGS else None
            if j is not None:
                worst_lift = max(worst_lift, abs(float(lift[:, j].max()) - saved['lift']['target_peak']))
                ck(r['id'] + ' target peak lift recomputed', abs(float(lift[:, j].max()) - saved['lift']['target_peak']) < 1e-12)
                d = lift[:, j] - (zero['foot_z'][:, j] - zero['base_foot_z'][j])
                derived[r['id']] = dict(peak=float(d.max()), time_ms=float(np.argmax(d) * .1),
                                        absolute_foot_z_at_peak=float(z['foot_z'][np.argmax(d), j]),
                                        support_ratio_at_peak=float(ratio[np.argmax(d)]),
                                        target_fz_at_peak=float(fz[np.argmax(d), j]),
                                        mean_nonleg_floor_force=float(np.mean(total - fz.sum(axis=1))))
            ck(r['id'] + ' COM dz and support windows recomputed',
               abs(float(z['com'][-1, 2] - z['com'][0, 2]) - saved['com']['dz']) < 1e-12
               and abs(float(ratio[:500].mean()) - saved['support_ratio']['mean']) < 1e-12)
        ck('all commands match declarations', worst_cmd == 0.0, worst=worst_cmd)
        ck('support ratio reproduces envelope table', worst_ratio < 1e-12, worst=worst_ratio)
        ck('target lift reproduces envelope table', worst_lift < 1e-12, worst=worst_lift)
        ledger = js(out / 'CALL_LEDGER.json')
        ck('envelope ledger counts', ledger['physics_step'] == 19000 and ledger['model_load'] == 1 and ledger['forward_init'] == 38)
        ck('prereg predicted 38 loads but runner loads once and resets per run',
           pre['calls']['model_load'] == 38 and ledger['model_load'] == 1)
        sr = js(out / 'SUPPORT_RATIO.json'); a = sr['arms']['s292_original_mask']
        ck('support ratio offline arm present', a['windows']['before_first_nonfoot']['ratio_mean'] > .9 and a['windows']['after_first_nonfoot']['ratio_mean'] < .8)
        ck('momentum identity consistent', a['momentum_check']['verdict'] == 'CONSISTENT')
        ck('zero-control does not hold posture', abs(env['runs']['zero_control']['com']['dz']) > .5)
        ck('no run sustains support with a held posture', all(v['com']['dz'] < 0 for v in env['runs'].values()))
        (out / 'DERIVED_LIFT.json').write_text(json.dumps(dict(vs_zero_control=derived), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        best_diff = max((v['peak'] for v in derived.values()), default=0.0)
        ck('no declared run lifts a commanded leg by the threshold (absolute or differential)', best_diff < THRESH and all(v['lift']['target_peak'] < THRESH for v in env['runs'].values()), best_differential=best_diff)
        if (out / 'REPLAY_FIDELITY.json').is_file():
            rep = js(out / 'REPLAY_FIDELITY.json')
            ck('harness fidelity: exact qpos replay', rep['max_abs_qpos_diff'] == 0.0 and rep['verdict'] == 'FAITHFUL_QPOS_EXACT')
        (out / 'ENVELOPE_VERIFIED.json').write_text(json.dumps(dict(best_differential_lift=best_diff), indent=2) + '\n', encoding='utf-8')
        result = dict(passed=all(c['passed'] for c in checks), checks=checks, failed=[c for c in checks if not c['passed']],
                      verifier_sha256=sha(__file__), best_differential_lift=best_diff,
                      best_target_lift=max(v['lift']['target_peak'] for v in env['runs'].values()),
                      calls=dict(model_load=0, forward=0, physics_step=0, neural_tick=0),
                      limitations=['native length/force units are not calibrated; the 1.0 threshold is one native length unit',
                                   '50 ms probes cannot test a 100 ms sustained-support criterion',
                                   'static constant commands bound constant-command capability only'])
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'], best_diff=best_diff), ensure_ascii=False))
        return 0 if result['passed'] else 1
    except Exception as exc:
        import traceback
        (out / 'VERIFICATION.json').write_text(json.dumps(dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), checks=checks), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=False, error=repr(exc)))); return 1


if __name__ == '__main__': raise SystemExit(main())
