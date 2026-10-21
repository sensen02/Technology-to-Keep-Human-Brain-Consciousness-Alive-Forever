"""P2 isolated numerical diagnostic; manual research prototype, NOT a product."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace
import uuid
import numpy as np
import scipy
import run_p1_fixed_input_clock_replay as replay
from run_p2_state_trace import PRIOR, PINS, ROOTS
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from engine.embodied.source_snapshot import build_source_snapshot, verify_source_snapshot

HERE = Path(__file__).resolve().parent
METHODS = ('backward_euler', 'frozen_exact')
CLOCKS = (('base', .5), ('half', .25), ('quarter', .125), ('base_repeat', .5))
LIMITS = [
    'Manual research implementation, NOT product; fixed saved P1 currents, no body simulation.',
    'frozen_exact freezes step-start coefficients; NOT an exact continuous-network solution.',
    'Conductance/adaptation decay, endpoint threshold/reset, refractory and delayed events remain split.',
    'Changing neural dt also changes decoder sample dt. Comparisons are descriptive, not isolated decoder convergence.',
    'No invented convergence tolerance or claim of assured improvement; posthoc five roots are not a held-out set.',
    'No production replacement, gains/weights/signs/params/initial-state changes, biological or behavioral validation.'
]


def simulate(built, currents, manifest, config, dt_ms, method):
    if method not in METHODS:
        raise ValueError('unknown voltage integration method')
    module = SimpleNamespace(ConductanceParams=ConductanceParams,
                             ConductanceNetwork=partial(ConductanceNetwork, voltage_method=method))
    return replay.simulate(built, currents, manifest, config, dt_ms, module)


def compare(a, b, motor):
    return dict(all_neuron_window_count_l1=int(np.abs(a['spike_counts']-b['spike_counts']).sum()),
                motor_window_count_l1=int(np.abs(a['spike_counts'][:, motor]-b['spike_counts'][:, motor]).sum()),
                motor_rates_max_abs_hz=float(np.max(np.abs(a['motor_rates_hz']-b['motor_rates_hz']))),
                motor_rates_l1_hz=float(np.abs(a['motor_rates_hz']-b['motor_rates_hz']).sum()),
                decoded_l1=float(np.abs(a['decoded_next_controls']-b['decoded_next_controls']).sum()),
                decoded_max_abs=float(np.max(np.abs(a['decoded_next_controls']-b['decoded_next_controls']))),
                applied_max_abs=float(np.max(np.abs(a['applied_controls']-b['applied_controls']))),
                event_neuron_and_time_exact=bool(np.array_equal(a['event_neuron'], b['event_neuron']) and
                                                np.array_equal(a['event_time_ms'], b['event_time_ms'])))


def verify_data(data, dt, manifest, config):
    """Reconstruct every window count and decoder sample from sparse events."""
    count = data['spike_counts']
    n_windows, n = count.shape
    steps = round(5/dt)
    evstep, neuron = data['event_step'], data['event_neuron']
    replay.validator.require(len(evstep) == len(neuron) == len(data['event_time_ms']), 'event lengths')
    replay.validator.require(np.all((evstep >= 0) & (evstep < n_windows*steps)) and
                             np.all((neuron >= 0) & (neuron < n)), 'event bounds')
    rebuilt = np.zeros_like(count)
    np.add.at(rebuilt, (evstep//steps, neuron), 1)
    raster = np.zeros((n_windows*steps, n), dtype=bool)
    raster[evstep, neuron] = True
    replay.validator.require(int(raster.sum()) == len(evstep), 'duplicate sparse event')
    decoder = MotorOutput(manifest, config)
    rates, controls = [], []
    for k in range(n_windows):
        ctrl, trace = decoder.consume(raster[k*steps:(k+1)*steps], spike_dt_s=dt/1000,
                                      start_s=k*.005, end_s=k*.005+.005)
        rates.append(trace['rates_hz']); controls.append(ctrl.copy())
    return dict(counts_exact=bool(np.array_equal(rebuilt, count)),
                timestamp_exact=bool(np.array_equal(data['event_time_ms'], (evstep+1)*dt)),
                rates_exact=bool(np.array_equal(rates, data['motor_rates_hz'])),
                decoded_exact=bool(np.array_equal(controls, data['decoded_next_controls'])),
                applied_lag_exact=bool(not data['applied_controls'][0].any() and np.array_equal(
                    data['applied_controls'][1:], data['decoded_next_controls'][:-1])),
                lag_indices_exact=bool(np.array_equal(data['applied_source_interval'], np.arange(n_windows)-1)))


def verify_directory(dest):
    """Final read-only verification of disk artifacts, sources, pins, counts and lag."""
    with (dest/'result.json').open() as f:
        result = json.load(f)
    identity = result['identity']
    checks = {}
    for name, digest in result['artifact_digests'].items():
        checks['artifact_'+name] = replay.validator.sha256(dest/name) == digest
    checks['sources'] = all(replay.validator.sha256(p) == h for p, h in identity['source_hashes'].items())
    checks['inputs'] = all(replay.validator.sha256(p) == h for p, h in identity['input_hashes'].items())
    verify_source_snapshot(dest/'source_snapshot.zip', identity['source_hashes'])
    checks['zip_members_hashes'] = True
    config = MotorOutputConfig(**identity['motor_translation'])
    with np.load(dest/'traces.npz', allow_pickle=False) as f:
        checks['current_hash'] = replay.array_hash(f['input_current_nA']) == identity['current_array_sha256']
        for label, spec in result['experiments'].items():
            data = {key: f[label+'_'+key] for key in result['data_keys']}
            checks.update({label+'_'+k: v for k, v in verify_data(data, spec['neural_dt_ms'], identity['motor_manifest'], config).items()})
            checks[label+'_total'] = int(data['spike_counts'].sum()) == spec['total_spikes']
            checks[label+'_motor'] = int(data['spike_counts'][:, f['motor_indices']].sum()) == spec['motor_spikes']
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, directory=str(dest))


def run(prior, output_root, desktop_root):
    started = time.monotonic()
    require = replay.validator.require
    for name, digest in PINS.items():
        require(replay.validator.sha256(prior/name) == digest, 'pinned prior '+name)
    with (prior/'result.json').open() as f:
        reference = json.load(f)
    sources = dict(reference['source_hashes'])
    require(all(replay.validator.sha256(p) == h for p, h in sources.items()), 'prior source drift')
    require(all(replay.validator.sha256(p) == h for p, h in reference['input_hashes'].items()), 'prior input drift')
    with np.load(prior/'traces.npz', allow_pickle=False) as f:
        old = {k: f[k] for k in f.files}
    currents = old['input_current_nA'].copy()
    require(currents.shape == (10, 12000), 'bounded 50ms P1 input contract')
    require(replay.array_hash(currents) == reference['exact_input_array_sha256'], 'current identity')
    import run_neural_muscle_closure as neural
    built = neural.build_neural_tier(neural.TierConfig(background_mean_nA=0, background_sd_nA=0))
    require(replay.network_hashes(built) == reference['network_hashes_before'], 'network identity')
    require(asdict(ConductanceParams()) == reference['params'] and reference['edge_scale'] == .2, 'params identity')
    config = MotorOutputConfig(**reference['motor_translation'])
    with (Path(reference['bundle'])/'result.json').open() as f:
        manifest = json.load(f)['motor_manifest']
    roots = list(map(str, built['tier']['root_ids']))
    ids = [roots.index(root) for root in ROOTS]
    motor = old['motor_indices']
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(HERE) and not any(
                    part in ('venv', 'venv_body', 'vendor') for part in path.parts):
                sources.setdefault(str(path), replay.validator.sha256(path))
    for path in (Path(__file__).resolve(), HERE/'test_p2_voltage_method_comparison.py'):
        sources[str(path)] = replay.validator.sha256(path)
    inputs = {**reference['input_hashes'], **{str(prior/k): v for k, v in PINS.items()}}
    dest = output_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    desktop = desktop_root/('P2-voltage-method-'+dest.name)
    desktop.mkdir(parents=True, exist_ok=False)
    identity = dict(prior=str(prior), prior_pins=PINS, source_hashes=sources, input_hashes=inputs,
        network_hashes=replay.network_hashes(built), current_array_sha256=replay.array_hash(currents),
        params=asdict(ConductanceParams()), edge_scale=.2, motor_translation=asdict(config), motor_manifest=manifest,
        methods=list(METHODS), input_window_ms=5, duration_ms=50, selected_roots=ROOTS, selected_local_indices=ids)
    build_source_snapshot(sources, dest/'source_snapshot.zip')
    verify_source_snapshot(dest/'source_snapshot.zip', sources)
    replay.write_json(dest/'identity.json', identity)
    outputs, experiments, checks = {}, {}, {}
    for method in METHODS:
        for clock, dt in CLOCKS:
            label = method+'_'+clock
            data = simulate(built, currents, manifest, config, dt, method)
            outputs[label] = data
            selected = {}
            for root, index in zip(ROOTS, ids):
                times = data['event_time_ms'][data['event_neuron'] == index]
                selected[root] = dict(first_spike_ms=float(times[0]) if len(times) else None, spike_times_ms=times.tolist())
            experiments[label] = dict(method=method, clock=clock, neural_dt_ms=dt,
                total_spikes=int(data['spike_counts'].sum()), motor_spikes=int(data['spike_counts'][:, motor].sum()),
                motor_counts_per_5ms=data['spike_counts'][:, motor].sum(axis=1).tolist(), selected_roots=selected)
            if method == 'backward_euler' and clock != 'base_repeat':
                for key, value in data.items():
                    checks[label+'_'+key+'_exact_prior'] = bool(np.array_equal(value, old[clock+'_'+key]))
            checks.update({label+'_'+k: v for k, v in verify_data(data, dt, manifest, config).items()})
            print(label, json.dumps({k: v for k, v in experiments[label].items() if k != 'selected_roots'}), flush=True)
        checks[method+'_duplicate_all_arrays_exact'] = all(np.array_equal(value, outputs[method+'_base_repeat'][key])
                                                         for key, value in outputs[method+'_base'].items())
    comparisons = {}
    for method in METHODS:
        comparisons[method] = {a+'_'+b: compare(outputs[method+'_'+a], outputs[method+'_'+b], motor)
                              for a, b in (('base', 'half'), ('half', 'quarter'), ('base', 'quarter'))}
    cross = {clock: compare(outputs['backward_euler_'+clock], outputs['frozen_exact_'+clock], motor)
             for clock, _ in CLOCKS if clock != 'base_repeat'}
    checks['sources_unchanged'] = all(replay.validator.sha256(p) == h for p, h in sources.items())
    checks['inputs_unchanged'] = all(replay.validator.sha256(p) == h for p, h in inputs.items())
    checks['network_unchanged'] = replay.network_hashes(built) == identity['network_hashes']
    checks['current_unchanged'] = replay.array_hash(currents) == identity['current_array_sha256']
    arrays = dict(input_current_nA=currents, motor_indices=motor, tier_root_ids=np.asarray(roots),
                  selected_root_ids=np.asarray(ROOTS), selected_local_indices=np.asarray(ids))
    for label, data in outputs.items():
        arrays.update({label+'_'+key: value for key, value in data.items()})
    with (dest/'traces.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for method in METHODS:
        for clock, _ in CLOCKS[:3]:
            label = method+'_'+clock
            axes[0].plot(np.arange(1, 11)*5, experiments[label]['motor_counts_per_5ms'], label=label)
            axes[1].plot(np.arange(1, 11)*5, outputs[label]['applied_controls'].max(axis=1), label=label)
            axes[2].plot(np.arange(5), [experiments[label]['selected_roots'][root]['first_spike_ms'] for root in ROOTS], 'o-', label=label)
    axes[0].set_ylabel('Motor spikes / fixed 5ms'); axes[0].legend(ncol=2)
    axes[1].set_ylabel('Max applied control (5ms lag)'); axes[1].set_xlabel('Time (ms)')
    axes[2].set_ylabel('First spike (ms)'); axes[2].set_xticks(np.arange(5), ROOTS, rotation=12, fontsize=8)
    fig.suptitle('P2 fixed P1 currents: BE vs frozen coefficients; diagnostic ONLY')
    fig.tight_layout(); fig.savefig(dest/'voltage_method.png', dpi=150); plt.close(fig)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks, identity=identity,
        experiments=experiments, comparisons_within_method=comparisons, comparisons_between_methods=cross,
        data_keys=list(next(iter(outputs.values()))), output_directory=str(dest), desktop_directory=str(desktop),
        artifact_digests={name: replay.validator.sha256(dest/name) for name in ('identity.json', 'source_snapshot.zip', 'traces.npz', 'voltage_method.png')},
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform(), threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        wall_s=time.monotonic()-started, limits=LIMITS)
    replay.write_json(dest/'result.json', result)
    verification = verify_directory(dest)
    replay.write_json(dest/'final_readonly_verification.json', verification)
    report = ['# P2电压方法固定输入诊断', '', '手动研究原型，不是成品。工程复现/归档检查：'+result['status']+'；独立只读核验：'+verification['status']+'。',
              '', '固定P1真实电流50ms，5ms输入窗口，三神经时钟及各方法0.5ms重复；无身体积分、无增益/模型替换。', '', '## 实测结果',
              '|方法/时钟|全网络脉冲|运动脉冲|', '|---|---:|---:|']
    for label, spec in experiments.items():
        report.append(f"|{label}|{spec['total_spikes']}|{spec['motor_spikes']}|")
    report += ['', '## 相邻细化（描述性，不设新收敛容差）', '```json', json.dumps(comparisons, indent=2), '```',
               '', '## 具体判定', 'BE三个时钟全部事件/时刻、窗口计数、速率、命令精确复现锁定历史；两个方法基准重复精确。',
               '方法及步长敏感仍存在；frozen_exact不是连续全网络精确解。仅凭这些差异不判收敛，也不承诺改善；未证明站立/行走或生物有效性。',
               '', '## 输出', str(dest), str(desktop), '', '## 边界', *['- '+line for line in LIMITS]]
    with (dest/'report.zh-CN.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(report)+'\n')
    for name in ('result.json', 'identity.json', 'voltage_method.png', 'report.zh-CN.md', 'final_readonly_verification.json'):
        shutil.copyfile(dest/name, desktop/name)
    print('OUTPUT', dest, 'DESKTOP', desktop, flush=True)
    require(all(checks.values()) and verification['status'] == 'PASS', 'diagnostic engineering verification failed')
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--output-root', type=Path, default=HERE/'outputs/p2_voltage_method_comparison')
    parser.add_argument('--desktop-root', type=Path, default=Path('/home/sensen/Desktop/engine修复验收'))
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_directory(args.verify.resolve(strict=True))
        print(json.dumps(result, indent=2)); return 0 if result['status'] == 'PASS' else 1
    run(args.prior.resolve(strict=True), args.output_root.resolve(), args.desktop_root.resolve())
    return 0


if __name__ == '__main__':
    from engine import memory_guard as mg
    with mg.guarded_run('p2_voltage_method_comparison', **mg.BUDGETS['neural_muscle_closure']):
        raise SystemExit(main())
