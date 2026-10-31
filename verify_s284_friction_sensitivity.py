"""Offline S284 verifier; reads saved files only and never imports MuJoCo."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args(argv).output
    checks, metrics = [], {}

    def check(name, ok, **extra):
        checks.append({'name': name, 'passed': bool(ok), **extra})

    def load_json(path):
        return json.loads(path.read_text(encoding='utf-8'))

    try:
        prereg = load_json(out / 'PREREG.json')
        arms = list(prereg.get('arms', ['baseline', 'friction_x2']))
        check('round S284', prereg.get('round') == 'S284')
        check('two expected arms', arms == ['baseline', 'friction_x2'])
        check('PREREG source hashes present', bool(prereg.get('source_hashes')))
        check('PREREG fixed clocks/seed/tau', prereg.get('duration_ms') == 400 and prereg.get('seed') == 0 and prereg.get('dt_s') == .0001 and prereg.get('neural_dt_ms') == .5 and prereg.get('exc_tau_ms') == prereg.get('inh_tau_ms') == 3.)
        check('PREREG no auxiliary controller/training/prewarm', all(prereg.get(k) is False for k in ('non_neural_controller', 'auxiliary_force', 'training', 'prewarm')))
        check('no FAILED artifact', not (out / 'FAILED.json').exists())
        complete = load_json(out / 'COMPLETE.json')
        check('complete capture', complete.get('capture_complete') is True and complete.get('source_unchanged') is True)
        expected_budget = {'model_load': 2, 'tier_build': 1, 'network_init': 2,
                           'live_step': 8000, 'neural_tick': 1600,
                           'live_fwd_position': 2, 'live_forward': 0,
                           'diagnostic_forward': 8002}
        ledger = load_json(out / 'CALL_LEDGER.json')
        check('call ledger exact', ledger == expected_budget,
              expected=expected_budget, actual=ledger)

        # Every pre-registered source and both XML identities are checked without loading a model.
        for name, expected in prereg.get('source_hashes', {}).items():
            p = Path(name)
            ok = p.is_file() and sha256(p) == expected
            check('source hash ' + name, ok)
        original = prereg.get('original_xml')
        if original:
            p = Path(original)
            check('original XML hash', p.is_file() and sha256(p) == prereg.get('original_xml_sha256'))
        derived = out / 'MODEL_FOOT_SLIDING_X2.xml'
        check('derived XML exists/hash', derived.is_file() and sha256(derived) == prereg.get('derived_xml_sha256'))
        if original and Path(original).is_file() and derived.is_file():
            source_text = Path(original).read_text(encoding='utf-8')
            derived_text = derived.read_text(encoding='utf-8')
            insertion = ' friction="2 0.005 0.0001"'
            check('derived XML only 30 local friction insertions', derived_text.count(insertion) == 30 and derived_text.replace(insertion, '') == source_text)
        check('COMPLETE call ledger/XML identities', complete.get('calls') == expected_budget and complete.get('derived_xml_sha256') == prereg.get('derived_xml_sha256'))

        # Metadata is saved compiler output; no model import is needed.
        metas = {a: load_json(out / a / 'MODEL_METADATA.json') for a in arms}
        check('metadata model XML hashes', all(
            metas[a].get('model_xml_sha256') == (prereg.get('original_xml_sha256') if a == 'baseline' else prereg.get('derived_xml_sha256'))
            for a in arms))
        check('metadata geometry identical except friction', metas[arms[0]].get('geom_names') == metas[arms[1]].get('geom_names'))
        check('ngeom/nbody/nmesh metadata', all(len(metas[a].get('geom_names', [])) == 72 and len(metas[a].get('body_names', [])) == 73 and metas[a].get('nmesh') == 71 for a in arms))

        params = {a: load_json(out / a / 'PARAMETERS.json') for a in arms}
        check('fixed neural parameters identical', params[arms[0]] == params[arms[1]])
        for a in arms:
            p = params[a]
            check(a + ' tau equal 3 ms', p.get('exc_tau_ms') == 3.0 and p.get('inh_tau_ms') == 3.0)

        target_cells = None
        initial_states = {}
        for arm in arms:
            a = arm
            d = out / arm
            tr = np.load(d / 'TRAJECTORY.npz', allow_pickle=False)
            ne = np.load(d / 'NEURAL.npz', allow_pickle=False)
            ctrlz = np.load(d / 'CTRL.npz', allow_pickle=False)
            qpos, qvel = tr['qpos'], tr['qvel']
            ctrl = np.asarray(ctrlz['ctrl'])
            spikes = np.asarray(ne['spikes'])
            target_cells = np.asarray(ne['target_cells'], dtype=int) if target_cells is None else target_cells
            check(a + ' trajectory/control/neural shapes', qpos.shape[0] == 4001 and ctrl.shape == (4000, 90) and spikes.shape[0] == 800)
            check(a + ' qpos dimension', qpos.shape[1] == 49 and qvel.shape == (4001, 48))
            check(a + ' finite arrays', np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(ctrl).all() and np.isfinite(spikes).all())
            t = np.asarray(tr['time'], float)
            check(a + ' trajectory times', t.shape == (4001,) and np.allclose(t, np.arange(4001) * 0.0001, rtol=0, atol=1e-11))
            nt = np.asarray(ne['time_ms'], float)
            check(a + ' neural times', np.array_equal(nt, np.arange(800) * 0.5 + 0.5))
            check(a + ' command times', np.array_equal(np.asarray(ne['ctrl_time_s']), t[1:]))
            check(a + ' neural commands identical', np.array_equal(ne['ctrl'], ctrl))
            check(a + ' command range and names', np.all((ctrl >= 0) & (ctrl <= 1)) and ctrlz['names'].tolist() == metas[a]['names'])
            with (d / 'STATE.csv').open(newline='', encoding='utf-8') as f:
                state_rows = list(csv.DictReader(f))
            check(a + ' state times', len(state_rows) == 400 and [float(r['t_ms']) for r in state_rows] == list(range(1, 401)))
            initial_states[a] = {k: np.asarray(tr[k])[0].copy() for k in ('qpos', 'qvel', 'act', 'ctrl')}
            check(a + ' command applied', np.array_equal(ctrl, np.asarray(tr['ctrl'])[1:]))
            check(a + ' initial qpos/qvel/act/ctrl invariants', np.array_equal(qpos[0], np.asarray(tr['qpos'])[0]) and np.all(qvel[0] == 0) and np.all(np.asarray(tr['act'])[0] == 0) and np.all(np.asarray(tr['ctrl'])[0] == 0))
            check(a + ' no external/qfrc applied', not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']))
            check(a + ' target cell identity', np.array_equal(ne['target_cells'], metas[a]['target_cells']) and np.array_equal(ne['target_cells'], target_cells) and spikes.shape == (800, len(target_cells)))
            check(a + ' neural sample shapes', all(ne[k].shape == spikes.shape for k in ('ge', 'gi', 'v')) and ne['rates_hz'].shape == (400, 90) and ne['raw_rates_hz'].shape == (400, 2, 90))

            # CONTACTS.json and packed contact_raw/contact_offsets are required to agree exactly.
            contacts = load_json(d / 'CONTACTS.json')
            packed = np.asarray(tr['contact_raw'], float)
            offsets = np.asarray(tr['contact_offsets'], dtype=np.int64)
            counts = np.asarray(tr['contact_count'], dtype=np.int64)
            check(a + ' contact sample count', len(contacts) == 4001 and len(offsets) == 4002 and len(counts) == 4001)
            check(a + ' packed contact schema', packed.ndim == 2 and packed.shape[1] == 21 and np.array_equal(offsets[1:] - offsets[:-1], counts) and offsets[-1] == len(packed))
            contact_consistent = True
            for i, row in enumerate(contacts):
                rr = np.asarray(row, float).reshape((-1, 21)) if row else np.empty((0, 21))
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                if rr.shape != (hi - lo, 21) or not np.array_equal(rr, packed[lo:hi]):
                    contact_consistent = False; break
            check(a + ' CONTACTS/raw packed consistency', contact_consistent)

            meta = metas[a]
            gf = np.asarray(meta.get('geom_friction'), float)
            expected_friction = np.tile([1., .005, .0001], (len(meta['geom_names']), 1))
            if arm == 'friction_x2':
                expected_friction[[int(k) for k in meta['geom_leg']], 0] = 2.
            check(arm + ' compiled friction values', np.array_equal(gf, expected_friction) and np.all(np.asarray(meta['geom_priority']) == 0))
            floor = int(meta.get('floor', -1))
            geom_leg = {int(k): v for k, v in meta.get('geom_leg', {}).items()}
            foot_force = np.zeros((4001, 6, 3), float)
            body_pen = np.zeros(4001, dtype=int)
            cone_ratios = []
            first_pen = None
            for i, row in enumerate(contacts):
                for c in row:
                    c = np.asarray(c, float); g1, g2, dist = int(c[0]), int(c[1]), float(c[2])
                    if floor not in (g1, g2): continue
                    other = g2 if g1 == floor else g1
                    frame = c[6:15].reshape(3, 3)
                    raw = c[15:21]
                    fw = (1.0 if g1 == floor else -1.0) * frame.T.dot(raw[:3])
                    if other in geom_leg:
                        li = ('lf','lm','lh','rf','rm','rh').index(geom_leg[other]); foot_force[i, li] += fw
                    elif dist < 0:
                        body_pen[i] += 1
                        if first_pen is None: first_pen = float(t[i] * 1000)
                    fn = float(raw[0]); ft = float(np.hypot(raw[1], raw[2]))
                    mu = float(max(gf[g1, 0], gf[g2, 0]))
                    if fn > 0 and mu > 0: cone_ratios.append(ft / (mu * fn))
            check(a + ' contact force reconstruction', np.allclose(foot_force, tr['foot_force_world'], rtol=0, atol=2e-12))
            check(a + ' body penetration reconstruction', np.array_equal(body_pen, tr['body_penetrating_contacts']))

            # NEURAL.spikes already contains only the target pool, not global neuron indices.
            target_sp = spikes
            ts = nt[np.any(target_sp > 0, axis=1)]
            q = qpos[:, :]
            quat = q[:, 3:7]
            orient = 2 * np.arccos(np.clip(np.abs(np.sum(quat * quat[0], axis=1)), 0, 1))
            com = np.asarray(tr['com'])
            metrics[a] = {
                'target_spike_count_0_200': int(target_sp[nt <= 200].sum()),
                'target_spike_count_200_400': int(target_sp[nt > 200].sum()),
                'target_last_spike_ms': float(ts[-1]) if len(ts) else None,
                'target_spike_times_ms': ts.tolist(),
                'first_body_penetration_ms': first_pen,
                'COM_dz400_native': float(com[-1, 2] - com[0, 2]),
                'root_dz400_native': float(q[-1, 2] - q[0, 2]),
                'orientation_max_rad': float(orient.max()),
                'friction_cone_ratio_max': float(max(cone_ratios)) if cone_ratios else None,
                'friction_cone_ratio_mean': float(np.mean(cone_ratios)) if cone_ratios else None,
                'friction_ratio_rows': len(cone_ratios),
                'friction_caveat': 'Ft=hypot(raw1,raw2), Fn=raw0>0, mu=max(saved compiled geom sliding friction); equal-priority/no-pair policy. Configured boundary approximation only, not solver saturation or material slip; physical units NOT_ESTABLISHED.',
                'support_claim': 'NOT_ESTABLISHED',
            }

        # Initial state hashes are explicitly arm-independent in the producer.
        inv = load_json(out / 'INITIAL_INVARIANTS.json')
        check('initial invariants report', inv.get('passed') is True and inv.get('neural_parameters_identical') is True)
        if len(arms) == 2:
            for key in ('qpos', 'qvel', 'act', 'ctrl'):
                check('actual initial ' + key + ' identical', np.array_equal(initial_states[arms[0]][key], initial_states[arms[1]][key]))
                for entry in inv.get('arms', []):
                    arm = entry['arm']
                    check(arm + ' recorded initial ' + key + ' hash', hashlib.sha256(initial_states[arm][key].tobytes()).hexdigest() == entry.get(key + '_sha256'))
            for key in ('qpos_sha256', 'qvel_sha256', 'act_sha256', 'ctrl_sha256', 'seed', 'fingerprint'):
                vals = [x.get(key) for x in inv.get('arms', [])]
                check('initial ' + key + ' identical', len(vals) == 2 and vals[0] == vals[1])

        # A saved-data verifier must not import or execute the simulator.
        check('no simulator import', 'mujoco' not in sys.modules)
        result = {'passed': all(c['passed'] for c in checks), 'checks': checks,
                  'metrics': metrics, 'scope': 'offline archive consistency and sensitivity metrics only',
                  'behavior_success': False, 'support_claim': 'NOT_ESTABLISHED',
                  'new_simulation_calls': 0}
        # Plot is intentionally lightweight and generated from saved arrays only.
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(11, 7))
            for a in arms:
                tr = np.load(out / a / 'TRAJECTORY.npz', allow_pickle=False)
                ne = np.load(out / a / 'NEURAL.npz', allow_pickle=False)
                tt = tr['time'] * 1000
                ax[0,0].plot(tt, tr['com'][:,2] - tr['com'][0,2], label=a)
                ax[0,1].plot(tt, tr['qpos'][:,2] - tr['qpos'][0,2], label=a)
                ax[1,0].plot(ne['time_ms'], ne['spikes'].sum(axis=1), label=a)
                ax[1,1].plot(tt, np.asarray(tr['body_penetrating_contacts']), label=a)
            for axis, title in zip(ax.flat, ('COM dz (native)', 'root dz (native)', 'spikes/tick', 'body penetration contacts')):
                axis.set_title(title); axis.set_xlabel('time (ms)'); axis.grid(alpha=.3); axis.legend(fontsize=8)
            fig.suptitle('S284 friction sensitivity — offline saved-data verifier\nNo behavior success claim; support/walking NOT_ESTABLISHED')
            fig.tight_layout(); fig.savefig(out / 'OUTCOMES.png', dpi=140); plt.close(fig)
        except Exception as exc:
            check('OUTCOMES.png generated', False, error=repr(exc))
        result['passed'] = all(c['passed'] for c in checks)
        (out / 'METRICS.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (out / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'passed': result['passed'], 'failed': [c['name'] for c in checks if not c['passed']], 'metrics': metrics}, ensure_ascii=False, indent=2))
        return 0 if result['passed'] else 1
    except Exception as exc:
        print('verifier error:', repr(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
