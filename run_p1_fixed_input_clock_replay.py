"""P2 exact P1 current replay: manual research prototype, NOT a finished product.
No body simulation, gain/sign/threshold search, or convergence acceptance claim.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import tools_verify_experiment_platform as validator
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from engine.embodied.source_snapshot import build_source_snapshot, verify_source_snapshot

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d')


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def reconstruct_input(arrays, n):
    """Only exact saved NPZ columns; outside selected neurons is explicitly zero."""
    ids = arrays['sensory_local_indices']
    values = arrays['normal_sensory_current_nA']
    validator.require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(set(ids.tolist())) == len(ids)
                      and np.all((ids >= 0) & (ids < n)), 'invalid saved selected columns')
    validator.require(values.ndim == 2 and values.shape[1] == len(ids) and np.isfinite(values).all(),
                      'invalid exact saved current')
    validator.require(not arrays['normal_outside_sensory_input_max_abs_nA'].any(), 'outside input nonzero')
    currents = np.zeros((len(values), n), dtype=values.dtype)
    currents[:, ids] = values
    return currents


def partition_diagnostic(raster, manifest, config, dt_s=.0005, partitions_ms=(10, 5, 2.5, .5)):
    """Same raster, SAME per-spike sample recurrence, different consume boundaries.

    A spike at t=(step+1)*dt is consumed in its original sample, never rebinned or
    replaced by interval counts. Comparison checkpoints are common 10ms endpoints.
    This tests decoder partition invariance, not neural integration convergence.
    """
    common_steps = round(.010 / dt_s)
    validator.require(len(raster) % common_steps == 0, 'diagnostic duration must align to 10ms')
    results = {}; arrays = {}
    for width in partitions_ms:
        steps = round(width / 1000 / dt_s)
        validator.require(steps > 0 and abs(steps*dt_s-width/1000) < 1e-12
                          and common_steps % steps == 0, 'unaligned decoder partition')
        decoder = MotorOutput(manifest, config)
        rates = []; controls = []
        for start in range(0, len(raster), steps):
            end = start + steps
            ctrl, trace = decoder.consume(raster[start:end], spike_dt_s=dt_s,
                                          start_s=start*dt_s, end_s=end*dt_s)
            if end % common_steps == 0:
                rates.append(trace['rates_hz']); controls.append(ctrl.copy())
        label = str(width) + 'ms'
        arrays[label] = (np.asarray(rates), np.asarray(controls))
    reference = arrays[str(partitions_ms[0])+'ms']
    for label, (rates, controls) in arrays.items():
        results[label] = dict(rates_exact=bool(np.array_equal(rates, reference[0])),
                             commands_exact=bool(np.array_equal(controls, reference[1])),
                             rates_max_abs_difference=float(np.max(abs(rates-reference[0]))),
                             command_max_abs_difference=float(np.max(abs(controls-reference[1]))))
    return results, arrays


def simulate(built, currents, manifest, config, dt_ms, neural_module):
    n = int(built['n']); steps = round(5/dt_ms)
    net = neural_module.ConductanceNetwork(n, neural_module.ConductanceParams(), dt_ms=dt_ms)
    net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
    decoder = MotorOutput(manifest, config)
    pending = np.zeros(manifest['n_actuators'])
    counts = []; applied = []; decoded = []; rates = []; event_steps = []; event_indices = []; event_times = []
    for k, current in enumerate(currents):
        raster = np.zeros((steps, n), dtype=bool)
        applied.append(pending.copy())
        for j in range(steps):
            raster[j] = net.step(current_nA=current)
            ids = np.flatnonzero(raster[j])
            event_steps.extend([k*steps+j]*len(ids)); event_indices.extend(ids.tolist())
            event_times.extend([float(net.t_ms)]*len(ids))
        counts.append(raster.sum(axis=0))
        pending, trace = decoder.consume(raster, spike_dt_s=dt_ms/1000,
                                         start_s=k*.005, end_s=k*.005+.005)
        decoded.append(pending.copy()); rates.append(trace['rates_hz'])
    return dict(spike_counts=np.asarray(counts), applied_controls=np.asarray(applied),
                decoded_next_controls=np.asarray(decoded), motor_rates_hz=np.asarray(rates),
                event_step=np.asarray(event_steps, dtype=np.int64),
                event_neuron=np.asarray(event_indices, dtype=np.int64),
                event_time_ms=np.asarray(event_times),
                interval_start_s=np.arange(len(currents))*.005,
                interval_end_s=(np.arange(len(currents))+1)*.005,
                applied_source_interval=np.arange(len(currents), dtype=np.int64)-1)


def network_hashes(built):
    values = {'root_ids': array_hash(np.asarray(built['tier']['root_ids']))}
    for key in ('We', 'Wi'):
        matrix = built['edges'][key]
        for component in ('data', 'indices', 'indptr'):
            values[key+'_'+component] = array_hash(getattr(matrix, component))
    return values


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def run(bundle, output_root):
    started = time.monotonic()
    validation = validator.verify_bundle(bundle, check_live_sources=True, check_live_inputs=True)
    with (bundle/'result.json').open() as stream:
        p1 = json.load(stream)
    with np.load(bundle/'traces.npz', allow_pickle=False) as loaded:
        old = {key: loaded[key] for key in loaded.files}
    validator.require(len(p1['tier_root_ids']) == 12000 and p1['clocks']['dt_command_s'] == .005
                      and p1['clocks']['dt_neural_s'] == .0005, 'explicit P1 12000/.5/5ms contract')
    currents = reconstruct_input(old, 12000)
    import run_neural_muscle_closure as neural
    built = neural.build_neural_tier(neural.TierConfig(background_mean_nA=0, background_sd_nA=0))
    validator.require(int(built['n']) == 12000 and list(map(str, built['tier']['root_ids'])) ==
                      list(map(str, p1['tier_root_ids'])), 'built tier identity differs from P1')
    config = MotorOutputConfig(**p1['evidence_schema']['motor_translation'])
    validator.require(asdict(config) == asdict(MotorOutputConfig()), 'P1 decoder differs from unchanged defaults')
    manifest = p1['motor_manifest']
    sources = dict(p1['source_hashes'])
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(HERE) and not any(
                    part in ('venv', 'venv_body', 'vendor') for part in path.parts):
                sources[str(path)] = validator.sha256(path)
    sources[str(Path(__file__).resolve())] = validator.sha256(__file__)
    # Never replace a pinned P1 digest with a changed live source.
    validator.require(all(sources[path] == digest for path, digest in p1['source_hashes'].items()), 'P1 source drift')
    input_hashes = dict(p1['input_hashes'])
    for name in ('result.json', 'traces.npz', 'source_snapshot.zip'):
        input_hashes[str(bundle/name)] = validator.sha256(bundle/name)
    source_before = {path: validator.sha256(path) for path in sources}
    input_before = {path: validator.sha256(path) for path in input_hashes}
    network_before = network_hashes(built); current_before = array_hash(currents)
    dest = output_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    build_source_snapshot(source_before, dest/'source_snapshot.zip')
    verify_source_snapshot(dest/'source_snapshot.zip', source_before)
    write_json(dest/'preflight.json', dict(validation=validation, source_hashes=source_before,
                input_hashes=input_before, network_hashes=network_before, current_hash=current_before))
    outputs = {}; experiments = {}
    motor = np.asarray(sorted({i for row in manifest['rows'] for i in row['local_indices']}), dtype=int)
    for label, dt_ms, zero in [('base', .5, False), ('base_repeat', .5, False),
                              ('half', .25, False), ('quarter', .125, False),
                              ('zero_base', .5, True), ('zero_half', .25, True), ('zero_quarter', .125, True)]:
        data = simulate(built, np.zeros_like(currents) if zero else currents, manifest, config, dt_ms, neural)
        outputs[label] = data
        experiments[label] = dict(neural_dt_ms=dt_ms, held_input_window_ms=5, sensory_zero=zero,
            total_spikes=int(data['spike_counts'].sum()), motor_spikes=int(data['spike_counts'][:, motor].sum()),
            motor_counts_per_5ms=data['spike_counts'][:, motor].sum(axis=1).tolist())
        print(label, json.dumps(experiments[label]), flush=True)
    base = outputs['base']
    checks = dict(base_counts_exact_p1=bool(np.array_equal(base['spike_counts'], old['normal_spike_counts'])),
        base_decoded_next_exact_p1=bool(np.array_equal(base['decoded_next_controls'], old['normal_decoded_next_controls'])),
        base_applied_exact_p1=bool(np.array_equal(base['applied_controls'], old['normal_controls'])),
        duplicate_all_arrays_exact=all(np.array_equal(a, outputs['base_repeat'][key]) for key, a in base.items()))
    raster = np.zeros((len(currents)*10, 12000), dtype=bool)
    raster[base['event_step'], base['event_neuron']] = True
    validator.require(np.array_equal(base['event_time_ms'], (base['event_step']+1)*.5), 'base event timestamp identity')
    partitions, partition_arrays = partition_diagnostic(raster, manifest, config)
    checks['decoder_partition_exact'] = all(v['rates_exact'] and v['commands_exact'] for v in partitions.values())
    checks['partition_matches_simulated_rates'] = bool(np.array_equal(partition_arrays['5ms'][0], base['motor_rates_hz'][1::2]))
    checks['sources_unchanged'] = source_before == {p: validator.sha256(p) for p in source_before}
    checks['inputs_unchanged'] = input_before == {p: validator.sha256(p) for p in input_before}
    checks['network_unchanged'] = network_before == network_hashes(built)
    checks['current_unchanged'] = current_before == array_hash(currents)
    arrays = dict(input_current_nA=currents, sensory_local_indices=old['sensory_local_indices'],
                  tier_root_ids=np.asarray(p1['tier_root_ids']), motor_indices=motor)
    for label, data in outputs.items():
        arrays.update({label+'_'+key: value for key, value in data.items()})
    for label, pair in partition_arrays.items():
        arrays['partition_'+label+'_rates_hz'] = pair[0]; arrays['partition_'+label+'_controls'] = pair[1]
    np.savez_compressed(dest/'traces.npz', **arrays)
    comparisons = {}
    for label in ('half', 'quarter', 'zero_base', 'zero_half', 'zero_quarter'):
        other = outputs[label]
        comparisons[label] = dict(all_neuron_count_l1=int(abs(other['spike_counts']-base['spike_counts']).sum()),
            motor_count_l1=int(abs(other['spike_counts'][:, motor]-base['spike_counts'][:, motor]).sum()),
            decoded_max_abs_difference=float(np.max(abs(other['decoded_next_controls']-base['decoded_next_controls']))),
            applied_max_abs_difference=float(np.max(abs(other['applied_controls']-base['applied_controls']))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label in outputs:
        axes[0].plot(np.arange(1, len(currents)+1)*5, experiments[label]['motor_counts_per_5ms'], label=label)
        axes[1].plot(np.arange(1, len(currents)+1)*5, outputs[label]['applied_controls'].max(axis=1), label=label)
    axes[0].set_ylabel('Motor spikes / identical 5ms'); axes[0].legend(ncol=3)
    axes[1].set_ylabel('Max applied control (one-window lag)'); axes[1].set_xlabel('Time (ms)')
    fig.suptitle('P2 fixed P1 currents: diagnostic ONLY; no body simulated')
    fig.tight_layout(); fig.savefig(dest/'clock_replay.png', dpi=150); plt.close(fig)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', label='手动研究原型，不是成品',
        bundle=str(bundle), output_directory=str(dest), checks=checks, experiments=experiments,
        comparisons=comparisons, decoder_partition_diagnostic=partitions,
        decoder_partition_convention='Original .5ms spike recurrence preserved; compare at common 10ms endpoints; no neural resimulation.',
        source_hashes=source_before, input_hashes=input_before, network_hashes_before=network_before,
        network_hashes_after=network_hashes(built), exact_input_array_sha256=current_before,
        source_snapshot_sha256=validator.sha256(dest/'source_snapshot.zip'), traces_sha256=validator.sha256(dest/'traces.npz'),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(),
                         thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        clocks=dict(input_window_ms=5, neural_dt_ms=[.5, .25, .125], decoder_sample_dt_matches_neural=True),
        params=asdict(neural.ConductanceParams()), motor_translation=asdict(config), edge_scale=.2,
        limits=['Fixed saved P1 body-history currents, not new closed-loop feedback or body refinement.',
                'Descriptive count/control differences are not convergence acceptance; timestamp grid equality is not a convergence criterion.',
                'Changing neural dt also changes decoder sample dt; separate fixed-raster test isolates partition boundaries only.',
                'No threshold, gain, sign, edge identity or initial state changes; no body simulated.',
                'No biological calibration, standing, walking, recovery or consciousness claim.'], wall_s=time.monotonic()-started)
    write_json(dest/'result.json', result)
    print('OUTPUT', dest, flush=True)
    validator.require(all(checks.values()), 'replay acceptance failed: '+str(checks))
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output-root', type=Path, default=HERE/'outputs/model_branches/p1_fixed_input_clock_replay')
    args = parser.parse_args(argv)
    run(args.bundle.resolve(strict=True), args.output_root.resolve())
    return 0


if __name__ == '__main__':
    from engine import memory_guard as mg
    with mg.guarded_run('p1_fixed_input_clock_replay', **mg.BUDGETS['neural_muscle_closure']):
        raise SystemExit(main())
