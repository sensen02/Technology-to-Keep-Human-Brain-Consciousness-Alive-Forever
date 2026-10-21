"""P2 bounded read-only observer. Manual research prototype, NOT a product."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import run_p1_fixed_input_clock_replay as replay
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from engine.embodied.source_snapshot import build_source_snapshot, verify_source_snapshot

HERE = Path(__file__).resolve().parent
PRIOR = HERE/'outputs/model_branches/p1_fixed_input_clock_replay/20261013_234617_bd6f9aa2'
PINS = {'result.json': 'c68c6e7feca42a204c6b104c7749bd1089ca2de077cd6cf781b564e170ae6ff0',
        'traces.npz': '25e4059b9ca3400f984564720970154c92170ea9d161683e2c54e26f3f6227f8',
        'source_snapshot.zip': 'dbc39ea168e91abeca80abd59e37542d619268e17e97e6780ab0e03968fd0832'}
ROOTS = ['720575941354084144', '720575941605019582', '720575941488287797',
         '720575941611937709', '720575941542829085']
TOL = 1e-12  # fixed before simulation, absolute per-state-unit residual


def predict(pre, current, incoming_e, incoming_i, t, dt, p):
    """Independent split BE transition from observed PRE state, no model calls.

    All cells owned; arrivals at START; threshold END; decay AFTER reset.
    The diagnostic matches production arithmetic order to avoid rounding causation.
    """
    ge = pre['ge'] + incoming_e
    gi = pre['gi'] + incoming_i
    active = t + 1e-10 >= pre['refractory_until_ms']
    G = p.leak_uS + ge + gi
    rhs = (p.capacitance_nF / dt * pre['v'] + p.leak_uS * p.rest_mV
           + ge * p.exc_reversal_mV + gi * p.inh_reversal_mV
           + current - pre['adaptation'])
    candidate = np.where(active, rhs/(p.capacitance_nF/dt+G), p.reset_mV)
    spike = active & (candidate >= p.threshold_mV)
    return dict(predicted_pre_reset_v=candidate, active=active, predicted_spike=spike,
                arrived_ge=ge, arrived_gi=gi,
                v=np.where(spike, p.reset_mV, candidate),
                ge=ge*np.exp(-dt/p.exc_tau_ms), gi=gi*np.exp(-dt/p.inh_tau_ms),
                adaptation=pre['adaptation']*np.exp(-dt/p.adaptation_tau_ms)
                           + spike*p.adaptation_increment_nA,
                refractory_until_ms=np.where(spike, t+dt+p.refractory_ms,
                                             pre['refractory_until_ms']))


FIELDS = ('v', 'ge', 'gi', 'adaptation', 'refractory_until_ms')


def observed_step(net, current, ids):
    """Read ring before consumption, never write network or monkeypatch step.

    Save actual ring events plus actual nonzero edge contributions to each target.
    Selected CSR rows preserve each full row's stored summation ordering.
    """
    pre = {key: getattr(net, key)[ids].copy() for key in FIELDS}
    arrived = net._ring[net._ring_i].copy()
    rows_e, rows_i = net.We[ids], net.Wi[ids]
    incoming_e, incoming_i = rows_e @ arrived, rows_i @ arrived
    contributions = []
    for target, (we, wi) in enumerate(zip(rows_e, rows_i)):
        for sign, row in enumerate((we, wi)):
            mask = arrived[row.indices] & (row.data != 0)
            for source, weight in zip(row.indices[mask], row.data[mask]):
                contributions.append((target, int(source), sign, float(weight)))
    start = float(net.t_ms)
    predicted = predict(pre, np.asarray(current)[ids], incoming_e, incoming_i,
                        start, net.dt, net.p)
    emitted = net.step(current_nA=current)
    post = {key: getattr(net, key)[ids].copy() for key in FIELDS}
    trace = {'start_ms': start, 'end_ms': float(net.t_ms), 'current_nA': np.asarray(current)[ids].copy(),
             'incoming_e_uS': incoming_e, 'incoming_i_uS': incoming_i,
             'spike': emitted[ids].copy(), **{f'pre_{k}': v for k, v in pre.items()},
             **{f'post_{k}': v for k, v in post.items()},
             **{k: predicted[k] for k in ('predicted_pre_reset_v', 'active', 'predicted_spike',
                                         'arrived_ge', 'arrived_gi')}}
    residuals = {k: float(np.max(np.abs(post[k]-predicted[k]))) for k in FIELDS}
    trace['state_residual'] = np.asarray([residuals[k] for k in FIELDS])
    trace['threshold_check'] = predicted['predicted_spike'] == emitted[ids]
    return emitted, trace, np.flatnonzero(arrived), contributions


def simulate(built, currents, manifest, config, dt, ids):
    net = ConductanceNetwork(int(built['n']), ConductanceParams(), dt_ms=dt)
    net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
    decoder = MotorOutput(manifest, config)
    pending = np.zeros(manifest['n_actuators'])
    traces = []; contributions = []; arrived_steps = []; arrived_sources = []
    events = []; neurons = []; times = []; counts = []; applied = []; decoded = []; rates = []
    steps = round(5/dt)
    for k, current in enumerate(currents):
        applied.append(pending.copy())
        raster = np.zeros((steps, net.n), dtype=bool)
        for j in range(steps):
            step = k*steps+j
            spike, trace, arrived, incoming = observed_step(net, current, ids)
            raster[j] = spike; traces.append(trace)
            fired = np.flatnonzero(spike)
            events.extend([step]*len(fired)); neurons.extend(fired.tolist()); times.extend([net.t_ms]*len(fired))
            arrived_steps.extend([step]*len(arrived)); arrived_sources.extend(arrived.tolist())
            contributions.extend((step, *entry) for entry in incoming)
        counts.append(raster.sum(axis=0))
        pending, motor_trace = decoder.consume(raster, spike_dt_s=dt/1000,
                                               start_s=k*.005, end_s=k*.005+.005)
        decoded.append(pending.copy()); rates.append(motor_trace['rates_hz'])
    result = {key: np.asarray([row[key] for row in traces]) for key in traces[0]}
    result.update(spike_counts=np.asarray(counts), applied_controls=np.asarray(applied),
                  decoded_next_controls=np.asarray(decoded), motor_rates_hz=np.asarray(rates),
                  event_step=np.asarray(events, dtype=np.int64), event_neuron=np.asarray(neurons, dtype=np.int64),
                  event_time_ms=np.asarray(times), ring_arrived_step=np.asarray(arrived_steps, dtype=np.int64),
                  ring_arrived_source=np.asarray(arrived_sources, dtype=np.int64))
    # typed columns avoid root/index rounding from mixed float event tables
    table = np.asarray(contributions).reshape(-1, 5)
    for index, name in enumerate(('incoming_step', 'incoming_target_column', 'incoming_source', 'incoming_sign', 'incoming_weight_uS')):
        result[name] = table[:, index].astype(float if index == 4 else np.int64)
    return result


def summarize(data, ids):
    result = {}
    for c, root in enumerate(ROOTS):
        spikes = np.flatnonzero(data['spike'][:, c])
        arrivals = np.flatnonzero((data['incoming_e_uS'][:, c] != 0) | (data['incoming_i_uS'][:, c] != 0))
        first = int(spikes[0]) if len(spikes) else None
        result[root] = dict(local_index=int(ids[c]), spike_times_ms=data['end_ms'][spikes].tolist(),
            first_spike_ms=None if first is None else float(data['end_ms'][first]),
            first_incoming_start_ms=None if not len(arrivals) else float(data['start_ms'][arrivals[0]]),
            incoming_events_before_first_threshold=0 if first is None else int(np.sum(
                (data['incoming_target_column'] == c) & (data['incoming_step'] <= first))),
            first_threshold=None if first is None else {key: float(data[key][first, c]) for key in
                ('current_nA', 'pre_v', 'predicted_pre_reset_v', 'incoming_e_uS', 'incoming_i_uS',
                 'arrived_ge', 'arrived_gi', 'pre_adaptation', 'pre_refractory_until_ms')})
    return result


def common_comparison(a, b, dt_a, dt_b):
    """Only equal START/END times, not mismatched row subtraction."""
    ratio = round(dt_a/dt_b)
    result = {}
    for c, root in enumerate(ROOTS):
        first_spike = next((float(t) for t, sa, sb in zip(a['end_ms'], a['spike'][:, c],
                            b['spike'][ratio-1::ratio, c]) if sa != sb), None)
        # Endpoint indicators here omit intermediate fine spikes: explicitly separately report first-spike times.
        rows = []
        for key in ('incoming_e_uS', 'incoming_i_uS', 'pre_v', 'pre_ge', 'pre_gi', 'pre_adaptation'):
            other = b[key][::ratio, c]
            different = np.flatnonzero(np.abs(a[key][:, c]-other) > TOL)
            rows.append((key, None if not len(different) else float(a['start_ms'][different[0]])))
        result[root] = dict(first_common_endpoint_spike_indicator_difference_ms=first_spike,
                            first_common_start_state_difference_ms=dict(rows))
    return result


def run(prior, output_root, desktop_root):
    start = time.monotonic()
    for name, digest in PINS.items():
        replay.validator.require(replay.validator.sha256(prior/name) == digest, 'prior pinned digest '+name)
    with (prior/'result.json').open() as f:
        reference = json.load(f)
    with np.load(prior/'traces.npz', allow_pickle=False) as f:
        old = {key: f[key] for key in f.files}
    sources = dict(reference['source_hashes'])
    replay.validator.require(all(replay.validator.sha256(p) == h for p, h in sources.items()), 'prior live source drift')
    replay.validator.require(all(replay.validator.sha256(p) == h for p, h in reference['input_hashes'].items()), 'prior live input drift')
    import run_neural_muscle_closure as neural
    built = neural.build_neural_tier(neural.TierConfig(background_mean_nA=0, background_sd_nA=0))
    replay.validator.require(replay.network_hashes(built) == reference['network_hashes_before'], 'network identity')
    currents = old['input_current_nA'].copy()
    replay.validator.require(replay.array_hash(currents) == reference['exact_input_array_sha256'], 'input identity')
    roots = list(map(str, built['tier']['root_ids']))
    ids = np.asarray([roots.index(r) for r in ROOTS], dtype=int)
    config = MotorOutputConfig(**reference['motor_translation'])
    with (Path(reference['bundle'])/'result.json').open() as f:
        manifest = json.load(f)['motor_manifest']
    dest = output_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    desktop = desktop_root/('P2-state-trace-'+dest.name)
    desktop.mkdir(parents=True, exist_ok=False)
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(HERE) and not any(
                    x in ('venv', 'venv_body', 'vendor') for x in path.parts):
                if str(path) not in sources:
                    sources[str(path)] = replay.validator.sha256(path)
    for path in (Path(__file__), HERE/'test_p2_state_trace.py', HERE/'P2_STATE_TRACE_PROTOCOL.zh-CN.md'):
        sources[str(path)] = replay.validator.sha256(path)
    build_source_snapshot(sources, dest/'source_snapshot.zip')
    verify_source_snapshot(dest/'source_snapshot.zip', sources)
    input_hashes = {**reference['input_hashes'], **{str(prior/k): v for k, v in PINS.items()}}
    identity = dict(prior=str(prior), prior_pins=PINS, source_hashes=sources, input_hashes=input_hashes,
                    network_hashes=replay.network_hashes(built), input_array_sha256=replay.array_hash(currents),
                    params=asdict(ConductanceParams()), edge_scale=.2, roots=ROOTS, local_indices=ids.tolist(),
                    residual_absolute_tolerance=TOL, ordering='ring arrivals/start current -> active -> BE -> END threshold/reset -> decay -> ring enqueue')
    replay.write_json(dest/'identity.json', identity)
    outputs = {}; summaries = {}; checks = {}; metrics = {}
    for label, dt in (('base', .5), ('half', .25), ('quarter', .125)):
        data = simulate(built, currents, manifest, config, dt, ids)
        outputs[label] = data; summaries[label] = summarize(data, ids)
        for key in ('spike_counts', 'applied_controls', 'decoded_next_controls', 'motor_rates_hz',
                    'event_step', 'event_neuron', 'event_time_ms'):
            checks[label+'_'+key+'_exact_prior'] = bool(np.array_equal(data[key], old[label+'_'+key]))
        checks[label+'_threshold_exact'] = bool(data['threshold_check'].all())
        checks[label+'_post_state_residual'] = bool(data['state_residual'].max() <= TOL)
        # Ring provenance: at start k exactly events emitted at end k-delay-1.
        delay = round(2/dt)
        expect_step = data['event_step']+delay+1
        keep = expect_step < len(data['start_ms'])
        checks[label+'_ring_exact_delayed_events'] = bool(np.array_equal(expect_step[keep], data['ring_arrived_step'])
            and np.array_equal(data['event_neuron'][keep], data['ring_arrived_source']))
        motor = old['motor_indices']
        metrics[label] = dict(total_spikes=int(data['spike_counts'].sum()), motor_spikes=int(data['spike_counts'][:, motor].sum()),
                             max_residual_by_field=dict(zip(FIELDS, data['state_residual'].max(axis=0).tolist())))
        print(label, json.dumps(metrics[label]), flush=True)
    checks['sources_unchanged'] = all(replay.validator.sha256(p) == h for p, h in sources.items())
    checks['inputs_unchanged'] = all(replay.validator.sha256(p) == h for p, h in input_hashes.items())
    checks['network_unchanged'] = identity['network_hashes'] == replay.network_hashes(built)
    checks['current_unchanged'] = identity['input_array_sha256'] == replay.array_hash(currents)
    arrays = dict(selected_root_ids=np.asarray(ROOTS), selected_local_indices=ids, input_current_nA=currents,
                  tier_root_ids=np.asarray(roots))
    for label, data in outputs.items():
        arrays.update({label+'_'+key: value for key, value in data.items()})
    with (dest/'traces.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    comparisons = {'base_half': common_comparison(outputs['base'], outputs['half'], .5, .25),
                   'base_quarter': common_comparison(outputs['base'], outputs['quarter'], .5, .125)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(13, 15))
    for c, root in enumerate(ROOTS):
        for label, data in outputs.items():
            axes[c, 0].plot(data['end_ms'], data['predicted_pre_reset_v'][:, c], label=label)
            axes[c, 1].plot(data['start_ms'], data['arrived_ge'][:, c], label=label+' E')
            axes[c, 1].plot(data['start_ms'], data['arrived_gi'][:, c], linestyle=':', label=label+' I')
        axes[c, 0].axhline(-50, color='black', linestyle='--'); axes[c, 0].set_title(root)
        axes[c, 0].set_ylabel('Pre-reset voltage (mV)'); axes[c, 1].set_ylabel('Arrival-before-decay g (uS)')
    axes[0, 0].legend(); axes[0, 1].legend(ncol=2)
    axes[-1, 0].set_xlabel('END time (ms)'); axes[-1, 1].set_xlabel('START time (ms)')
    fig.suptitle('P2 actual algorithm observer, fixed P1 current; diagnostic only')
    fig.tight_layout(); fig.savefig(dest/'state_trace.png', dpi=140); plt.close(fig)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, identity=identity,
                  summaries=summaries, metrics=metrics, comparisons=comparisons, output_directory=str(dest), desktop_directory=str(desktop),
                  wall_s=time.monotonic()-start, python=sys.executable,
                  artifact_digests={name: replay.validator.sha256(dest/name) for name in
                      ('identity.json', 'source_snapshot.zip', 'traces.npz', 'state_trace.png')},
                  limits=['Manual research implementation, NOT product; no body or behavioral success.',
                          'Posthoc roots; no causal attribution to aggregate upstream firing.',
                          'Independent BE prediction verifies observed post-state and threshold, not biological validity.',
                          'Common endpoint spike indicator ignores fine-only intervening spikes; full spike times also saved.',
                          'No gain/threshold/connectivity/model modification; no counterfactual executed.'])
    replay.write_json(dest/'result.json', result)
    import shutil
    for name in ('result.json', 'state_trace.png', 'identity.json'):
        with (desktop/name).open('xb') as out, (dest/name).open('rb') as src:
            shutil.copyfileobj(src, out)
    print('OUTPUT', dest, 'DESKTOP', desktop, flush=True)
    replay.validator.require(all(checks.values()), 'state trace checks failed '+str(checks))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--output-root', type=Path, default=HERE/'outputs/p2_state_trace')
    parser.add_argument('--desktop-root', type=Path, default=Path('/home/sensen/Desktop/engine修复验收'))
    args = parser.parse_args()
    run(args.prior.resolve(strict=True), args.output_root.resolve(), args.desktop_root.resolve())


if __name__ == '__main__':
    from engine import memory_guard as mg
    with mg.guarded_run('p2_state_trace', **mg.BUDGETS['neural_muscle_closure']):
        main()
