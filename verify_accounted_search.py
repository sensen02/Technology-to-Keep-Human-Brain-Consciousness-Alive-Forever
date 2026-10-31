"""S173/P1-S2 independent SAVED-ONLY verifier; no producer/engine/optimizer execution.
Reuses immutable S169 numerical verifier via audited textual adaptation, and adds
independent explicit callback/cache/Jacobian accounting and convergence plots.
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

OLD = Path(__file__).with_name('verify_near_initial_support.py')
OLD_SHA256 = '0d78f0d1e4bbf5b829761e921ed8c4267e7af6aeb6ecfdfa664025194a2a8444'


def verify_accounting(out, records, frozen, ledger):
    cfg = json.loads((out/'EXPLICIT_ACCOUNTING_CONFIG.json').read_text())
    a = json.loads((out/'EXPLICIT_EVALUATION_LEDGER.json').read_text())
    assert cfg['jacobian_columns'] == 43 and cfg['forward_per_miss'] == 2
    assert cfg['inner_per_miss'] == 1
    assert cfg['cached_base_full_jacobian_cost'] == 86
    assert cfg['first_base_and_jacobian_cost'] == 88
    np.testing.assert_array_equal(cfg['qpos_indices'], frozen['indices'])
    np.testing.assert_array_equal(cfg['lower'], frozen['lower'])
    width = frozen['upper']-frozen['lower']
    np.testing.assert_array_equal(cfg['physical_width'], width)
    cache = {}; bases = {}; events = []; complete = []; active = None
    eid = 0; used = 1; inner = 0; transitions = 0; previous = None
    def key(x):
        q = frozen['qpos0'].copy(); q[frozen['indices']] = frozen['lower']+x*width
        return q.tobytes()+np.asarray(x, dtype=np.float64).tobytes(), q
    for number, e in enumerate(a['events'], 1):
        assert e['event_id'] == number
        role = e['role']; bid = e.get('base_id')
        if role == 'JACOBIAN_BEGIN':
            assert active is None and bid in bases
            assert e['planned_probes'] == 43
            x = bases[bid]; needed = {key(x)[0]}
            for j in range(43):
                y = x.copy(); h = 1e-5*abs(x[j]) or np.sqrt(np.finfo(float).eps)
                if x[j]+h > 1: h = -h
                y[j] += h; needed.add(key(y)[0])
            required = 2*sum(k not in cache for k in needed)
            assert e['required_forward'] == required and used+required <= 1990
            active = dict(base_id=bid, columns=[], base_seen=False)
        elif role == 'JACOBIAN_COMPLETE':
            assert active is not None and active['base_id'] == bid
            assert active['base_seen'] and active['columns'] == list(range(43))
            assert e['columns'] == 43
            k = key(bases[bid])[0]
            transitions += previous is not None and previous != k; previous = k
            complete.append(dict(base_id=bid, evaluation=cache[k], forward=used))
            active = None
        elif role in ('objective', 'jacobian_base', 'difference_probe'):
            if role == 'objective':
                if 'x' in e: x = np.asarray(e['x'], dtype=np.float64)
                else:
                    assert bid in bases; x = bases[bid].copy()
                if bid in bases: np.testing.assert_array_equal(bases[bid], x)
                else: bases[bid] = x.copy()
            else:
                assert active is not None and active['base_id'] == bid
                x = bases[bid].copy()
                if role == 'jacobian_base':
                    assert not active['base_seen']; active['base_seen'] = True
                else:
                    j = e['variable']; assert j == len(active['columns'])
                    h = 1e-5*abs(x[j]) or np.sqrt(np.finfo(float).eps)
                    if x[j]+h > 1: h = -h
                    y = x.copy(); y[j] += h
                    assert e['perturbation'] == float(y[j]-x[j])
                    x = y; active['columns'].append(j)
                if 'x' in e: np.testing.assert_array_equal(e['x'], x)
            assert x.shape == (43,) and np.isfinite(x).all()
            assert np.all(x >= 0) and np.all(x <= 1)
            k, q = key(x); hit = k in cache
            assert e['cache_hit'] == hit
            if hit:
                assert e['forward_delta'] == e['inner_delta'] == 0
            else:
                assert e['forward_delta'] == 2 and e['inner_delta'] == 1
                eid += 1; used += 2; inner += 1
                assert eid <= len(records)
                with np.load(out/f'EVAL_{eid:03d}.npz', allow_pickle=False) as s:
                    np.testing.assert_array_equal(s['qpos'], q)
                cache[k] = eid
                assert e.get('success', False) or records[eid-1]['accepted']
            events.append(dict(event_id=number, role=role, base_id=bid,
                               evaluation=cache[k], cache_hit=hit, forward=used))
        elif role == 'BUDGET_EXIT_BEFORE_COMPLETE_JACOBIAN':
            assert active is None and e['planned_probes'] == 43
            assert e['remaining'] == 1990-used and e['required_forward'] > e['remaining']
        elif role == 'BUDGET_EXIT_BEFORE_OBJECTIVE':
            assert e['remaining'] == 1990-used and e['remaining'] < 2
        elif role in ('JACOBIAN_LIMIT_EXIT', 'COMPLETE_JACOBIAN_LIMIT'):
            assert len(complete) == 20
        elif role == 'SOLVER_ACCOUNTING_FINISHED': pass
        else:
            # New explicit termination metadata may use distinct limit spelling;
            # never silently accept unknown events.
            assert False, ('Unknown accounting role', role)
    assert active is None or (records and records[-1]['accepted']), 'partial Jacobian without accepted candidate'
    assert eid == len(records) == inner == ledger['inner_solves']
    assert used == ledger['forward'] == a['forward'] <= 1990
    assert a['inner_solves'] == inner
    assert a['complete_jacobians'] == len(complete) <= 20
    assert a['jacobian_base_transitions'] == transitions
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    all_ids = np.arange(1, len(records)+1)
    axes[0].scatter(all_ids, [r['max_abs_native_residual'] for r in records], s=7,
                    alpha=.4, label='All uncached saved evaluations (incl. FD probes)')
    axes[1].scatter(all_ids, [r['objective'] for r in records], s=7, alpha=.4,
                    label='All uncached saved evaluations')
    ids = [v['evaluation'] for v in complete]
    if ids:
        axes[0].plot(ids, [records[i-1]['max_abs_native_residual'] for i in ids],
                     'o-', color='tab:red', label='Completed Jacobian bases ONLY')
        axes[1].plot(ids, [records[i-1]['objective'] for i in ids], 'o-',
                     color='tab:red', label='Completed Jacobian bases ONLY')
    axes[0].set_ylabel('Max abs all48 native residual')
    axes[1].set_ylabel('Scaled objective squared norm')
    axes[1].set_xlabel('Saved EVAL ID (not optimizer accepted-update count)')
    for ax in axes: ax.set_yscale('log'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('S173 saved callback accounting: bases vs all finite-difference probes\n'
                 'Accepted/rejected optimizer updates UNKNOWN; no standing/neural claim')
    fig.tight_layout(); plot = out/'ACCOUNTED_CONVERGENCE.png'
    assert not plot.exists(); fig.savefig(plot, dpi=160); plt.close(fig)
    prereg = json.loads((out/'PREREG.json').read_text())
    return dict(verified=True, producer_prereg_legacy_pose_field=prereg['maximum_pose_evaluations'],
                metadata_note='Inherited producer maximum_pose_evaluations=99 is stale metadata; S173 parent prereg and explicit cap1990/max20Jac govern actual execution, not 99 poses',
                search_forward_cap=1990, total_cap=2000,
                reserve_for_separate_reload=10, actual_search_forward=used,
                complete_jacobians=len(complete), jacobian_bases=complete,
                jacobian_base_transitions=transitions, callback_events=events,
                accepted_optimizer_updates='UNKNOWN', rejected_optimizer_updates='UNKNOWN',
                convergence_plot=str(plot), partial_jacobian_due_to_candidate=active is not None)


def main():
    raw = OLD.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OLD_SHA256, 'Immutable S169 verifier changed'
    source = raw.decode()
    replacements = {
        "parser.add_argument('--output', type=Path, required=True)": "parser.add_argument('--output', '--outputroot', dest='output', type=Path, required=True)",
        "out/'S169_SAVED_INDEPENDENT_CHECK.json'": "out/'S173_SAVED_INDEPENDENT_CHECK.json'",
        "assert prereg['maximum_actual_forwards']==200 and prereg['maximum_pose_evaluations']==99": "assert prereg['maximum_actual_forwards'] in (1990,2000) and prereg['maximum_pose_evaluations'] <= 994",
        "ledger['forward']==len(files)<=200": "ledger['forward']==len(files)<=1990",
        "files=sorted(out.glob('FORWARD_*.npz'))": "files=sorted(out.glob('FORWARD_*.npz'),key=lambda p:int(p.name.split('_')[1]))",
        "ledger['pose_evaluations']==len(records)<=99": "ledger['pose_evaluations']==len(records)<=994",
        "plot=plot_saved(out,baseline,states[bestid],reconstructions[1],reconstructions[bestid],meta,bool(accepted))": "accounted=verify_accounting(out,records,frozen,ledger)\n    plot=plot_saved(out,baseline,states[bestid],reconstructions[1],reconstructions[bestid],meta,bool(accepted))",
        "scope='S169/P1-S2 saved-only all evaluations, all48 residuals and complete ledger'": "scope='S173/P1-S2 saved-only ALL evaluations, all48/nativeB/f0/geometry and explicit accounting', accounting=accounted",
        "S169 bounded near-initial native SOFT search:": "S173 accounted near-initial native SOFT search:",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, ('Adaptation target changed', old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.If)]
    ns = dict(__name__='s173_saved_verification', __file__=str(Path(__file__).resolve()),
              verify_accounting=verify_accounting)
    exec(compile(tree, str(OLD), 'exec'), ns)
    ns['main']()

if __name__ == '__main__':
    main()
