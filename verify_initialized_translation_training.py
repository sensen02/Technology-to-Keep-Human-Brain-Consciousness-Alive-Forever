"""Offline initialized translation-training audit; never imports/runs physics.

Reconstructs FIR, bounded update, original S96 formal proxy and early penalty.
Saved neural dynamics, physical force/contact records and native weight remain
inputs, not independent neural/physics simulations or biological certification.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from verify_neural_initialized_startup import (
    ROOT, INITIAL, MANIFEST, GATE_NAMES, BINS, DT, require, equal, sha,
    load_json, load_npz, fir_recompute, support_metrics, first_nonzero,
)

RUNNER = ROOT / 'run_initialized_translation_training.py'
S96 = Path('/home/sensen/Desktop/engine修复验收/s96-six-gate-training/20261023T214545Z')
BASELINE_WEIGHTS = Path('/home/sensen/Desktop/engine修复验收/s102-tick-training-second/20261023T224501Z/run/round00_minus/TRAJECTORY.npz')
ARMS = ('baseline', 'plus', 'minus', 'updated')
PRE, TICKS, SAMPLES = 80, 800, 4001


def score(a, weight, scales):
    """Independent saved-array arithmetic; no objective-module execution."""
    q, v, time = a['qpos'], a['qvel'], a['time']
    require(weight > 0 and np.isfinite(weight) and q[0, 2] > 0, 'Invalid reference scales')
    norms = np.linalg.norm(q[:, 3:7], axis=1)
    require(np.all(norms > 0), 'Zero quaternion')
    height = q[:, 2] / q[0, 2]
    angle = 2*np.arccos(np.clip(abs(q[:, 3:7] @ q[0, 3:7])/(norms*norms[0]), 0, 1))
    legs = (a['contacts'] > 0).sum(1)
    load = a['foot_force'][:, :, 2].sum(1)/weight
    depth, limit = a['nonfoot_depth'], a['limit_excess']
    violations = np.column_stack((np.maximum(.85-height, 0), np.maximum(angle-.35, 0),
        np.maximum(3-legs, 0), np.maximum(.8-load, 0), np.maximum(depth-1e-12, 0),
        np.maximum(limit-1e-6, 0)))
    gates = np.column_stack((height >= .85, angle <= .35, legs >= 3, load >= .8,
                             depth <= 1e-12, limit <= 1e-6))
    terms = (violations/np.asarray([scales[k] for k in GATE_NAMES]))**2
    formal_parts = np.trapezoid(terms[2000:4001], dx=.0001, axis=0)/.2
    early_terms = np.column_stack(((np.maximum(.8-load, 0)/.8)**2,
                                  (np.maximum(limit-1e-6, 0)/.001)**2))
    early_parts = np.trapezoid(early_terms[:101], dx=.0001, axis=0)/.01
    formal_proxy, early_penalty = float(formal_parts.sum()), float(early_parts.sum())
    phases = {}
    for name, sl in [('early', slice(0, 101)), ('preparation', slice(0, 2001)),
                     ('formal', slice(2000, 4001))]:
        good = gates[sl].all(1)
        occupancy, longest, _ = support_metrics(good, time[sl])
        phases[name] = dict(support_ms=occupancy,
            longest_all_gates_true_endpoint_span_ms=longest, all_samples_pass=bool(good.all()),
            gate_support_ms={k: support_metrics(gates[sl, j], time[sl])[0]
                             for j, k in enumerate(GATE_NAMES)},
            min_height_ratio=float(height[sl].min()),
            max_downward_velocity=float(np.maximum(-v[sl, 2], 0).max()),
            max_angle_rad=float(angle[sl].max()), max_nonfoot_depth=float(depth[sl].max()),
            max_joint_limit_excess=float(limit[sl].max()), min_load_ratio=float(load[sl].min()),
            min_contact_legs=int(legs[sl].min()))
    metrics = dict(optimization_loss=formal_proxy+early_penalty, formal_proxy=formal_proxy,
        proxy_parts=dict(zip(GATE_NAMES, map(float, formal_parts))), early_penalty=early_penalty,
        early_parts=dict(zip(('load', 'limit'), map(float, early_parts))), **phases,
        preparation_terminal20ms_pass=bool(gates[1800:2001].all()),
        first_failures_ms={k: float(time[np.flatnonzero(~gates[:, j])[0]]*1000)
                          if (~gates[:, j]).any() else None for j, k in enumerate(GATE_NAMES)},
        first_nonzero_command_physical_start_ms=first_nonzero(a['applied'], np.arange(TICKS)*DT),
        first_nonzero_activation_ms=first_nonzero(a['act'], time))
    return metrics, dict(load=load, height=height, angle=angle, legs=legs, gates=gates,
                         cumulative=support_metrics(gates.all(1), time)[2])


def recompute_update(weights, plus, minus, lp, lm):
    """Original propose formula, including original zero mask and global guard."""
    active = weights > 0
    secant = np.divide(lp-lm, plus-minus, out=np.zeros_like(weights), where=active)
    raw_step = -.05/(0+1)**.602*secant
    scale = min(1., 1e-7/max(float(np.max(np.abs(raw_step))), np.finfo(float).tiny))
    updated = np.clip(weights+scale*raw_step, np.finfo(float).eps, 4.)
    updated[~active] = 0
    return updated, dict(lp=float(lp), lm=float(lm), guard_scale=float(scale),
        max_actual_step=float(np.max(abs(updated-weights))), active_entries=int(active.sum()),
        formula='secant=divide(lp-lm,plus-minus,out=zeros,where=w>0); raw=-.05/(k+1)**.602*secant; scale=min(1,max_step/max(max(abs(raw)),tiny)); clip(w+scale*raw,eps,4); preserve zero mask',
        gradient_claim=False)


def compare(baseline, candidate):
    safety = {}
    for phase in ('early', 'preparation', 'formal'):
        safety[phase] = {k: bool(candidate[phase][k] >= baseline[phase][k]-1e-12)
                        if k == 'min_height_ratio' else bool(candidate[phase][k] <= baseline[phase][k]+1e-12)
                        for k in ('min_height_ratio', 'max_downward_velocity', 'max_angle_rad',
                                  'max_nonfoot_depth', 'max_joint_limit_excess')}
    support = bool(candidate['formal']['support_ms'] > baseline['formal']['support_ms']+1e-12)
    longest = bool(candidate['formal']['longest_all_gates_true_endpoint_span_ms'] >
                   baseline['formal']['longest_all_gates_true_endpoint_span_ms']+1e-12)
    return dict(proxy_strictly_lower=bool(candidate['optimization_loss'] < baseline['optimization_loss']),
        safety_nonworsening=safety, formal_support_strict_improvement=support,
        formal_longest_strict_improvement=longest,
        physical_joint_acceptance=bool(candidate['optimization_loss'] < baseline['optimization_loss']
            and candidate['formal']['support_ms'] >= baseline['formal']['support_ms']-1e-12
            and candidate['formal']['longest_all_gates_true_endpoint_span_ms'] >=
                baseline['formal']['longest_all_gates_true_endpoint_span_ms']-1e-12
            and (support or longest) and all(all(x.values()) for x in safety.values())))


def verify_csr(csr, label):
    for prefix in ('We', 'Wi'):
        data, indices, indptr, shape = [csr[prefix+'_'+k] for k in ('data', 'indices', 'indptr', 'shape')]
        equal(shape, [17099, 17099], label+' connectivity shape')
        require(data.ndim == indices.ndim == indptr.ndim == 1, label+' CSR ranks')
        require(np.issubdtype(indices.dtype, np.integer) and np.issubdtype(indptr.dtype, np.integer), label+' CSR index dtype')
        require(indptr.shape == (17100,) and indptr[0] == 0 and indptr[-1] == len(data)
                and len(indices) == len(data) and np.all(np.diff(indptr) >= 0), label+' CSR pointers')
        require(np.all((indices >= 0) & (indices < 17099)) and np.all(data >= 0), label+' CSR bounds')


def verify(out):
    out = Path(out).resolve()
    require('mujoco' not in sys.modules, 'Physics already imported')
    require(not (out/'FAILED.json').exists(), 'FAILED.json present')
    require(load_json(out/'COMPLETE.json')['status'] == 'COMPLETE', 'Incomplete runner')
    prereg = load_json(out/'PREREG.json')
    for k, expected in dict(epsilon=1e-6, seed=10901, max_step=1e-7, k=0,
                            early_load_scale=.8, early_limit_scale=.001).items():
        require(prereg[k] == expected, 'Prereg '+k)
    hashes = prereg['source_hashes']
    for path, digest in hashes.items():
        require(sha(path) == digest, 'Source hash mismatch: '+path)
    require(all(str(p) in hashes for p in (RUNNER, INITIAL, MANIFEST)), 'Missing source identities')
    for name in ('objective.py', 'OBJECTIVE_PREREG.json'):
        require(sha(out/name) == sha(S96/name), 'Not byte-identical original S96 '+name)
    scales = load_json(out/'OBJECTIVE_PREREG.json')['scales']
    require(prereg['scales'] == scales and set(scales) == set(GATE_NAMES)
            and all(np.isfinite(x) and x > 0 for x in scales.values()), 'Changed six scales')
    direction = load_npz(out/'DIRECTION.npz'); w = direction['weights']
    require(w.shape == (66, 5) and w.dtype == np.dtype(float), 'Weight shape/dtype')
    require(np.all((w >= 0) & (w <= 4)), 'Baseline weight bounds')
    require(prereg['same_weights'] == str(BASELINE_WEIGHTS), 'Changed starting-weight provenance')
    require(str(BASELINE_WEIGHTS) in hashes, 'Missing baseline saved-weight source hash')
    equal(w, load_npz(BASELINE_WEIGHTS)['weights'], 'Exact saved S102 minus starting weights')
    equal(direction['direction'], np.random.default_rng(10901).choice([-1., 1.], size=w.shape), 'Seeded direction')
    expected_weights = {'baseline': w}
    for label, sign in [('plus', 1), ('minus', -1)]:
        probe = np.clip(w+sign*1e-6*direction['direction'], np.finfo(float).eps, 4.)
        probe[w <= 0] = 0
        equal(direction[label], probe, label+' epsilon arithmetic')
        expected_weights[label] = probe
    expected_weights['updated'] = load_npz(out/'UPDATE.npz')['updated']
    from engine.embodied.motor_translation import FiniteMotorTranslator
    mapping = FiniteMotorTranslator(load_json(MANIFEST), w)
    require(mapping.n == 17099 and mapping.nu == 90 and len(mapping.pools) == 66, 'Pool mapping dimensions')
    require(tuple(mapping.BINS) == BINS and mapping.DT == DT, 'Changed FIR bins/clock')
    results = load_json(out/'RESULTS.json')
    require([r['label'] for r in results] == list(ARMS), 'Results arm order')
    ledger = load_json(out/'CALL_LEDGER.json')
    require(ledger == dict(model_load=4, forward=16004, step=16000, neural_ticks=3520,
                          active=None, completed=list(ARMS)), 'Call ledger mismatch')
    initial_q = load_npz(INITIAL)['new_qpos']
    checks, plotted, archive_hashes = [], [], {}
    baseline_init = baseline_cold = baseline_csr = baseline_observation = baseline_meta = baseline_release_meta = None
    for r in results:
        label = r['label']; dest = out/label; weights = expected_weights[label]
        require(load_json(dest/'RESULT.json') == r, label+' per-arm result mismatch')
        for path, digest in r['hashes'].items():
            require(sha(path) == digest, label+' committed artifact hash '+path)
        require(r['direction_sha256'] == sha(out/'DIRECTION.npz'), label+' direction hash')
        if label == 'updated':
            require(r['update_sha256'] == sha(out/'UPDATE.npz'), label+' update hash')
        for name, key in [('TRAJECTORY.npz', 'trajectory_sha256'), ('INITIALIZATION.npz', 'initialization_sha256')]:
            require(sha(dest/name) == r[key], label+' result archive hash')
        a, init, cold = [load_npz(dest/name) for name in ('TRAJECTORY.npz', 'INITIALIZATION.npz', 'NEURAL_COLD.npz')]
        csr = load_npz(dest/'CONNECTIVITY.npz'); verify_csr(csr, label)
        observation = load_json(dest/'INITIAL_OBSERVATION.json')
        meta = load_json(dest/'NEURAL_COLD_META.json')
        require(isinstance(meta, dict) and bool(meta), label+' missing neural scalar/config metadata')
        # Generic nested metadata accepted; exact cross-arm equality is required.
        serialized_meta = json.dumps(meta, sort_keys=True, allow_nan=False)
        require('backward_euler' in serialized_meta and 'capacitance_nF' in serialized_meta,
                label+' incomplete neural config metadata')
        if baseline_init is None:
            baseline_init, baseline_cold, baseline_csr, baseline_observation = init, cold, csr, observation
            baseline_meta = meta
        require(meta == baseline_meta, label+' cold scalar/config metadata differs')
        release_meta = load_json(dest/'NEURAL_RELEASE_META.json')
        require(isinstance(release_meta, dict) and bool(release_meta), label+' missing release scalar/config metadata')
        json.dumps(release_meta, allow_nan=False)
        if baseline_release_meta is None:
            baseline_release_meta = release_meta
        require(release_meta == baseline_release_meta, label+' release scalar/config metadata differs')
        for snapshot, steps in ((meta, 0), (release_meta, PRE)):
            scalars = snapshot['neural_scalars']
            for key, expected in dict(n=17099, dt=.5, delay_steps=4, steps=steps,
                    t_ms=steps*.5, _ring_i=steps%5, voltage_method='backward_euler').items():
                require(scalars[key] == expected, label+' scalar snapshot '+key)
            require(snapshot['neural_configs'] == meta['neural_configs'], label+' neural params changed')
        require(bool(cold) and set(cold) == set(baseline_cold), label+' cold fields')
        require(set(csr) == set(baseline_csr), label+' connectivity fields')
        for key in csr:
            equal(csr[key], baseline_csr[key], label+' matched CSR '+key)
        for key in cold:
            equal(cold[key], baseline_cold[key], label+' cold '+key)
            equal(init[key], baseline_init[key], label+' release network '+key)
        for key in ('spikes', 'features', 'current', 'history', 'release_features', 'body_qpos', 'body_qvel', 'body_act', 'body_ctrl'):
            equal(init[key], baseline_init[key], label+' matched initialization '+key)
        require(observation == baseline_observation, label+' initial observation differs')
        shapes = dict(time=(4001,), qpos=(4001,49), qvel=(4001,48), act=(4001,90), ctrl=(4001,90),
            contacts=(4001,6), foot_force=(4001,6,3), nonfoot_depth=(4001,), limit_excess=(4001,),
            actuator_force=(4001,90), qfrc_applied=(4001,48), xfrc_applied=(4001,73,6),
            spikes=(800,17099), features=(800,66,5), applied=(800,90), current=(800,17099), weights=(66,5))
        for key, shape in shapes.items():
            require(a[key].shape == shape, label+' shape '+key)
        require(a['warnings'].ndim == 2 and len(a['warnings']) == SAMPLES, label+' warning shape')
        equal(a['time'], np.arange(SAMPLES)*.0001, label+' body clock', 1e-12)
        for key in ('qfrc_applied', 'xfrc_applied', 'warnings'):
            require(not a[key].any(), label+' nonzero '+key)
        for key in ('act', 'ctrl', 'applied'):
            require(np.all((a[key] >= 0) & (a[key] <= 1)), label+' bounds '+key)
        require(np.all(a['contacts'] >= 0) and np.all(a['contacts'] == np.floor(a['contacts'])), label+' contacts')
        require(np.all(a['nonfoot_depth'] >= 0) and np.all(a['limit_excess'] >= 0), label+' nonnegative depth/limit')
        equal(init['body_qpos'], initial_q, label+' initial qpos')
        equal(a['qpos'][0], initial_q, label+' rollout initial qpos')
        for key in ('qvel', 'act', 'ctrl'):
            require(not init['body_'+key].any() and not a[key][0].any(), label+' initial '+key)
        equal(init['weights'], weights, label+' initialization weights')
        equal(a['weights'], weights, label+' rollout weights')
        require(init['spikes'].shape == (PRE,17099), label+' warm spike shape')
        wf, wu, history = fir_recompute(init['spikes'], mapping.pools, mapping.actuator_pool, weights)
        for key, expected in [('features', wf), ('decoded', wu), ('history', history)]:
            equal(init[key], expected, label+' warm '+key, 1e-14)
        release_features = np.stack([history[lo:hi].mean(0) for lo, hi in BINS], axis=1)
        equal(init['release_features'], release_features, label+' release features', 1e-14)
        release = np.zeros(90); mapped = mapping.actuator_pool >= 0
        release[mapped] = np.clip((release_features*weights).sum(1), 0, 1)[mapping.actuator_pool[mapped]]
        equal(init['release_decoded'], release, label+' release command', 1e-14)
        features, commands, _ = fir_recompute(np.concatenate((init['spikes'], a['spikes'])), mapping.pools, mapping.actuator_pool, weights)
        equal(a['features'], features[PRE:], label+' continuous FIR', 1e-14)
        equal(a['applied'], commands[PRE:], label+' continuous commands', 1e-14)
        equal(a['ctrl'][1:], np.repeat(a['applied'], 5, axis=0), label+' five-body-step hold')
        equal(a['current'][:10], np.repeat(init['current'][None], 10, axis=0), label+' initial current')
        equal(a['current'], np.repeat(a['current'][::10], 10, axis=0), label+' 5ms sensory hold')
        schedule = load_json(dest/'SCHEDULE.json'); require(len(schedule) == TICKS, label+' schedule length')
        for k, row in enumerate(schedule):
            require(row['tick'] == k and row['consume_id'] == PRE+k, label+' consume ID')
            for key, expected in dict(source_start_s=(PRE+k)*DT, source_end_s=(PRE+k+1)*DT,
                    physical_start_s=k*DT, physical_end_s=(k+1)*DT, source_offset_s=PRE*DT).items():
                require(abs(row[key]-expected) <= 1e-12, label+' schedule '+key)
        require(r['initialization_ticks'] == PRE and r['source_offset_s'] == .04, label+' result initialization')
        for key, expected in dict(neural_final_s=.44, FIR_final_s=.44, physical_final_s=.4).items():
            require(abs(r[key]-expected) <= 1e-12, label+' final '+key)
        require(r['weight_native'] == results[0]['weight_native'], label+' native weight mismatch')
        metrics, series = score(a, r['weight_native'], scales)
        # Runner score fields must agree rather than override reconstruction.
        saved_scores = r.get('scores', r.get('score', r))
        score_aliases = dict(total=metrics['optimization_loss'], optimization_loss=metrics['optimization_loss'],
            formal_proxy=metrics['formal_proxy'], early_penalty=metrics['early_penalty'],
            early_load=metrics['early_parts']['load'], early_limit=metrics['early_parts']['limit'])
        require(any(k in saved_scores for k in ('total', 'optimization_loss')), label+' missing saved composite score')
        for key, expected in score_aliases.items():
            if key in saved_scores:
                require(np.isclose(saved_scores[key], expected, rtol=1e-12, atol=1e-12), label+' score '+key)
        metrics.update(label=label, passed=True, release_decoded_max=float(release.max()))
        checks.append(metrics); plotted.append((label, a, series))
        for path in sorted(dest.iterdir()):
            if path.is_file() and path.suffix in ('.npz', '.json'):
                archive_hashes[str(path)] = sha(path)
    updated, audit = recompute_update(w, expected_weights['plus'], expected_weights['minus'],
                                      checks[1]['optimization_loss'], checks[2]['optimization_loss'])
    equal(expected_weights['updated'], updated, 'Independent actual updated weights', 1e-15)
    require(np.max(abs(updated-w)) <= 1e-7+1e-15 and np.array_equal(updated[w <= 0], w[w <= 0])
            and np.all((updated[w > 0] >= np.finfo(float).eps) & (updated[w > 0] <= 4)), 'Update mask/bounds')
    historical_scores, _ = score(load_npz(Path(prereg['same_weights'])), results[0]['weight_native'], scales)
    historical_reference = dict(source=prereg['same_weights'], source_sha256=sha(prereg['same_weights']),
        scores=historical_scores, excluded_from_matched_acceptance=True,
        interpretation='Historical cold-start trajectory only; not a matched initialized arm, fresh repeat, or candidate-selection baseline.')
    require('mujoco' not in sys.modules, 'Verifier imported physics')
    return dict(passed=True, verifier=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
        new_physics_calls=0, new_neural_ticks=0, ledger=ledger, source_hashes=hashes,
        observed_archive_hashes=archive_hashes, mapping_digest=mapping.mapping_digest,
        physical_units='NOT_ESTABLISHED', behavior_claim='NONE', checks=checks, update=audit,
        historical_cold_reference=historical_reference,
        comparisons={c['label']: compare(checks[0], c) for c in checks[1:]},
        support_convention='Trapezoid binary occupancy; longest contiguous true-endpoint span; early0:101 preparation0:2001 formal2000:4001; no crossing interpolation.',
        caveats=['40ms held-observation neural/FIR initialization is computational, not physical support.',
            'Network and physical dynamics not rerun/certified; FIR and update arithmetic independently recomputed.',
            'Native weight and aggregated contact/force/depth/limit arrays are trusted saved inputs; physical units NOT_ESTABLISHED.',
            'Hashes and ledger do not independently prove absence of unrecorded runtime callbacks/calls.',
            'Original six-gate formal proxy plus early penalty is optimization only, not standing/walking acceptance.',
            'Source END precedes its physical interval by declared operator split; no future physical observation.']), plotted


def plot(out, plotted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, xmax in [('FULL_COMPARISON.png', 400), ('EARLY_COMPARISON.png', 10)]:
        fig, axes = plt.subplots(6, 1, figsize=(13, 16), sharex=True)
        for label, a, s in plotted:
            t = a['time']*1000
            axes[0].step(np.arange(TICKS)*.5, a['applied'].max(1), where='post', label=label+' command')
            axes[0].plot(t, a['act'].max(1), '--', alpha=.6, label=label+' activation')
            for ax, y in zip(axes[1:], (s['load'], s['height'], s['angle'], a['limit_excess'], s['cumulative'])):
                ax.plot(t, y, label=label)
        for ax, threshold in zip(axes[1:5], (.8, .85, .35, 1e-6)):
            ax.axhline(threshold, color='black', ls=':')
        for ax, ylabel in zip(axes, ('Max command / activation', 'Foot Fz / native weight', 'Height ratio',
                                    'Root angle (rad)', 'Joint limit excess (rad)', 'Six-gate occupancy (ms)')):
            ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); ax.set_xlim(0, xmax)
        axes[-1].set_xlabel('Physical time after release (ms); 40ms computational initialization excluded')
        fig.suptitle('Matched initialized translation training: saved-array independent comparison\nResearch only; NOT standing/walking certification; physical units NOT_ESTABLISHED')
        fig.tight_layout(); fig.savefig(Path(out)/filename, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    report, plotted = verify(out)
    plot(out, plotted)
    report['comparison_pngs'] = [str(out/name) for name in ('FULL_COMPARISON.png', 'EARLY_COMPARISON.png')]
    (out/'INDEPENDENT_CHECK.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
