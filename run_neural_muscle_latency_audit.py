"""Independent latency evidence audit; never edits historical or canonical files.
Run with the project venv Python. PASS means measurement/selftests passed, not closure.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import numpy as np
import scipy
from engine import canonical_physics as cp
import engine.memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
DT = 0.5
LENGTHS = (100., 200., 400., 800.)
WINDOWS = (5., 10., 20.)


def first(mask):
    ids = np.flatnonzero(mask)
    return float((ids[0] + 1) * DT) if len(ids) else None


def causal_rate(counts, pool_n, window_ms):
    """Trailing (t-W,t] rate, zero prestimulus history, fixed denominator."""
    w = int(round(window_ms / DT))
    c = np.r_[0., np.cumsum(counts, dtype=float)]
    i = np.arange(1, len(c))
    return (c[i] - c[np.maximum(i-w, 0)]) * 1000. / (window_ms * pool_n)


def old_quantiles(delta_counts):
    c = np.cumsum(delta_counts, dtype=float)
    final = float(c[-1])
    return {'final_delta_spikes': final, **{
        'cumulative_%dpct_ms' % pct: first(np.sign(final)*c >= abs(final)*pct/100.)
        if abs(final) > 1e-12 else None for pct in (10, 50)}}


def plateau(rate):
    """Operational gate, not proof of asymptotic stationarity: last 4x25ms blocks.
    Require 100ms tail AFTER a complete initial rate window, nonzero mean,
    block range <=10% mean and sample CV <=20%. No passing => no t10/t50.
    """
    n = int(100 / DT)
    if len(rate) < n + int(max(WINDOWS)/DT):
        return {'exists': False, 'reason': 'insufficient independent tail',
                'response_10pct_ms': None, 'response_50pct_ms': None}
    tail = rate[-n:]
    blocks = tail.reshape(4, -1).mean(axis=1)
    level = float(tail.mean())
    spread = float(np.ptp(blocks)/abs(level)) if abs(level) > 1e-9 else None
    cv = float(tail.std()/abs(level)) if abs(level) > 1e-9 else None
    ok = bool(spread is not None and spread <= .1 and cv <= .2)
    return {'exists': ok, 'tail_mean_delta_hz_per_neuron': level,
            'tail_block_means_delta_hz_per_neuron': blocks.tolist(),
            'relative_block_range': spread, 'tail_sample_cv': cv,
            **{'response_%dpct_ms' % pct:
               first(np.sign(level)*rate >= abs(level)*pct/100.) if ok else None
               for pct in (10, 50)}}


def summarize(counts, reference, n):
    delta = counts.astype(float) - reference
    rates = {}
    for w in WINDOWS:
        r = causal_rate(delta, n, w)
        rates[str(w)] = {'first_rate_difference_ms': first(np.abs(r) > 1e-12),
                        'window_ms': w, 'plateau': plateau(r)}
    return {'first_population_count_difference_ms': first(delta != 0),
            'old_total_quantiles_not_latency': old_quantiles(delta),
            'causal_rate': rates}


def selftests():
    rows = []
    for delay in (0., 20.):
        for duration in LENGTHS:
            # Exact deterministic rate mass, not stochastic spikes; boundary
            # interval (delay,delay+DT] is the first responding interval.
            x = np.zeros(int(duration/DT))
            x[int(delay/DT):] = 1.
            s = summarize(x, np.zeros_like(x), 1)
            assert s['first_population_count_difference_ms'] == delay + DT
            old = s['old_total_quantiles_not_latency']['cumulative_50pct_ms']
            assert old == (duration + delay)/2
            p = s['causal_rate']['10.0']['plateau']
            if duration >= 200:
                assert p['exists']
                assert p['response_10pct_ms'] == delay+1
                assert p['response_50pct_ms'] == delay+5
            for w in WINDOWS:
                extended = np.r_[x, np.ones(40)*123]
                assert np.array_equal(causal_rate(x, 1, w), causal_rate(extended, 1, w)[:len(x)])
            rows.append({'input': 'constant_rate_mass', 'known_delay_ms': delay,
                         'duration_ms': duration, 'metrics': s})
    zero = summarize(np.zeros(800), np.zeros(800), 1)
    assert zero['first_population_count_difference_ms'] is None
    assert not zero['causal_rate']['10.0']['plateau']['exists']
    ramp = plateau(np.linspace(0, 100, 800))
    assert not ramp['exists'] and ramp['response_50pct_ms'] is None
    # Opposite signed responses work; cancellation can hide individual differences.
    negative = summarize(-np.ones(800), np.zeros(800), 1)
    assert negative['causal_rate']['10.0']['plateau']['response_50pct_ms'] == 5.
    assert np.any(np.array([True, False]) != np.array([False, True]))
    assert np.array([True, False]).sum() == np.array([False, True]).sum()
    return {'passed': True, 'cases': rows, 'additional_checks': [
        'zero response undefined', 'nonstationary ramp rejects plateau',
        'negative response thresholds', 'future extension does not change causal prefix',
        'individual identity difference may cancel in population count']}


def run_network():
    from engine.embodied.adapters import build_neural_tier, TierConfig
    from engine.embodied.sensory_channels import channel_index_map, CHANNEL_SPEC, LEGS
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    cfg = TierConfig(background_mean_nA=0., background_sd_nA=0.)
    built = build_neural_tier(cfg)
    n = int(built['n'])
    roots = np.asarray(built['tier']['root_ids'])
    with np.load(HERE/'data/flywire/banc_connectome.npz', allow_pickle=True) as z:
        lookup = {int(r): i for i, r in enumerate(z['root_ids'])}
        flow, sup, cls = (np.asarray(z[k]).astype(str) for k in
                          ('ann_flow', 'ann_super_class', 'ann_class'))
        motor = np.array([i for i, r in enumerate(roots) if
            flow[lookup[int(r)]] == 'efferent' or
            'motor' in sup[lookup[int(r)]].lower() or
            'motor' in cls[lookup[int(r)]].lower()], dtype=int)
    pools = {'motor': motor, 'descending_readout': np.asarray(built['readout_local'], int)}
    groups, _ = channel_index_map(n, roots)
    currents = {k: np.zeros(n) for k in ('zero', 'positive_partition', 'negative_partition')}
    for leg in LEGS:
        for ch, (_, _, signed, _) in CHANNEL_SPEC.items():
            if signed:
                ids = np.asarray(groups[leg][ch], int)
                currents['positive_partition'][ids[0::2]] += .3
                currents['negative_partition'][ids[1::2]] += .3
    We, Wi = (built['edges'][k] * .2 for k in ('We', 'Wi'))
    def trial(cur, duration):
        net = ConductanceNetwork(n, params=ConductanceParams(), dt_ms=DT)
        net.set_connectivity(We, Wi)
        a = {name: np.zeros((int(duration/DT), len(idx)), bool) for name, idx in pools.items()}
        for k in range(int(duration/DT)):
            s = net.step(current_nA=cur)
            for name, idx in pools.items():
                a[name][k] = s[idx]
        return a
    trials = {key: trial(cur, max(LENGTHS)) for key, cur in currents.items()}
    prefix = {}
    for key, cur in currents.items():
        short = trial(cur, min(LENGTHS))
        prefix[key] = all(np.array_equal(short[p], trials[key][p][:len(short[p])]) for p in pools)
    assert all(prefix.values())
    rows = []
    arrays = {'times_ms': np.arange(1, int(max(LENGTHS)/DT)+1)*DT, 'tier_root_ids': roots}
    for pool, idx in pools.items():
        arrays[pool+'_local_indices'] = idx
        for condition in currents:
            arrays[pool+'_'+condition+'_spikes'] = trials[condition][pool]
        for condition in ('positive_partition', 'negative_partition'):
            a, ref = trials[condition][pool], trials['zero'][pool]
            for duration in LENGTHS:
                m = int(duration/DT)
                s = summarize(a[:m].sum(axis=1), ref[:m].sum(axis=1), len(idx))
                s['first_individual_difference_spike_ms'] = first(np.any(a[:m] != ref[:m], axis=1))
                rows.append({'pool': pool, 'condition': condition, 'duration_ms': duration, **s})
    np.savez_compressed(OUT/'latency_spike_traces.npz', **arrays)
    return {'tier_n': n, 'pool_n': {p: len(i) for p, i in pools.items()},
            'tier_config': asdict(cfg), 'conductance_params': asdict(ConductanceParams()),
            'connectivity_multiplier': .2, 'current_nA_per_selected_channel': .3,
            'stimulated_cell_counts': {k: int(np.count_nonzero(v)) for k, v in currents.items()},
            'directly_stimulated_motor_count': {k: int(np.count_nonzero(v[motor])) for k,v in currents.items()},
            'independently_restarted_100ms_matches_800ms_prefix': prefix,
            'rows': rows}, arrays


def plot_audit(synthetic, network, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for delay in (0.,20.):
        cases = [r for r in synthetic['cases'] if r['known_delay_ms'] == delay]
        axes[0].plot(LENGTHS, [r['metrics']['old_total_quantiles_not_latency']['cumulative_50pct_ms'] for r in cases], 'o-', label=f'known delay {delay:g} ms')
    axes[0].set(title='Synthetic: old cumulative midpoint', xlabel='Trial duration (ms)', ylabel='Old 50% time (ms)')
    for condition in ('positive_partition','negative_partition'):
        rows = [r for r in network['rows'] if r['pool']=='motor' and r['condition']==condition]
        axes[1].plot(LENGTHS, [r['old_total_quantiles_not_latency']['cumulative_50pct_ms'] for r in rows], 'o-', label=condition)
        counts = arrays['motor_'+condition+'_spikes'].sum(axis=1) - arrays['motor_zero_spikes'].sum(axis=1)
        axes[2].plot(arrays['times_ms'], causal_rate(counts, network['pool_n']['motor'], 10.), label=condition)
    axes[1].set(title='Motor: old cumulative midpoint', xlabel='Trial duration (ms)', ylabel='Old 50% time (ms)')
    axes[2].set(title='Causal trailing 10 ms motor rate', xlabel='Time after current step (ms)', ylabel='Delta Hz / neuron')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'latency_audit.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    protected = [cp.CANONICAL_XML, HERE/'engine/canonical_physics.py',
                 HERE/'run_neural_rebuild_23_step_latency.py', HERE/'run_neural_rebuild_24_motor_path.py',
                 HERE/'outputs/neural_rebuild/neural_rebuild_24_motor_path.json']
    dependencies = [Path(__file__), HERE/'engine/neural_cond.py', HERE/'engine/embodied/adapters.py',
                    HERE/'engine/embodied/sensory_channels.py', HERE/'engine/neck_cut_data.py',
                    HERE/'data/flywire/banc_connectome.npz',
                    HERE/'outputs/neural_rebuild/sensory_channel_lookup.json']
    hashes = {str(p): cp.sha256_file(p) for p in protected+dependencies}
    assert hashes[str(cp.CANONICAL_XML)] == cp.CANONICAL_XML_SHA256
    synthetic = selftests()  # Always complete BEFORE building/running the network.
    measured = {'synthetic_selftests': synthetic}
    if not args.self_test_only:
        network, arrays = run_network()
        measured['network'] = network
        plot_audit(synthetic, network, arrays)
    assert all(cp.sha256_file(p) == hashes[str(p)] for p in protected)
    artifact_hashes = {str(p): cp.sha256_file(p) for p in
                      (OUT/'latency_spike_traces.npz', OUT/'latency_audit.png') if p.exists()} if not args.self_test_only else {}
    output = OUT/('latency_selftest.json' if args.self_test_only else 'latency_audit.json')
    cp.write_result(output, question='Do old cumulative quantiles measure neural feedback latency?',
        status='PASS', measured=measured,
        provenance={'command': shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
                    'cwd': str(Path.cwd()), 'interpreter': sys.executable,
                    'python_version': platform.python_version(), 'numpy_version': np.__version__,
                    'scipy_version': scipy.__version__, 'sha256': hashes,
                    'artifact_sha256': artifact_hashes, 'protected_files_unchanged': True,
                    'elapsed_s': time.monotonic()-t0,
                    'sign_partition': 'step24 idx[0::2] versus idx[1::2], artificial, NOT biological direction labels'},
        what_this_does_not_support=[
            'PASS denotes audit execution/selftests, not neural-muscle closure or stability.',
            'Old cumulative 10/50% depend on final trial total and are not response latency.',
            'Earliest deterministic difference is not a robust decoder, muscle activation, torque or feedback latency.',
            'No MuJoCo, muscle mapping, neuromuscular junction or body feedback integrated.',
            'Artificial alternating-index partitions cannot establish biological direction coding.',
            'No noise trials, parameter sweep, physiological calibration or asymptotic steady-state proof.'],
        units={'time': 'ms', 'execution_elapsed': 's', 'rate': 'Hz per neuron',
               'spike_count': 'spikes', 'current': 'nA', 'conductance': 'uS',
               'voltage': 'mV', 'capacitance': 'nF'},
        metric_units={'timestamps': 'ms', 'causal_rates': 'Hz per neuron', 'dt_ms': DT},
        protocol={'durations_ms': LENGTHS, 'causal_windows_ms': WINDOWS,
                  'onset_ms': 0, 'prehistory': 'zero; all neurons at rest, no warmup, as step24',
                  'timestamps': 'input at interval start, spike recorded at interval end',
                  'plateau_gate': 'last 4x25ms; >=120ms observation; nonzero mean; block range/abs(mean)<=0.1; sample std/abs(mean)<=0.2',
                  't10_t50': 'first signed crossing relative to accepted rate plateau, includes window filtering delay; not cumulative quantiles',
                  'length_test': '100/200/400/800ms prefixes plus independently restarted 100ms for all three conditions'},
        illustrative_vs_measured='Analytic synthetic rate-mass selftests and simulator measurements with illustrative uncalibrated conductance parameters; not animal measurements.')
    print(output)
    print('PASS: selftests; audit only, NOT closure')


if __name__ == '__main__':
    with mg.guarded_run('latency_audit', **mg.BUDGETS['latency_audit']):
        main()
