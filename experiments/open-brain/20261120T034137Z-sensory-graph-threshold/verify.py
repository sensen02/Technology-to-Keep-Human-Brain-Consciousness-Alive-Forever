"""Compare archived fixed dynamics and replay the four published-edge interventions."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_neural import (ConductanceNetwork, DT_MS, HERE, THRESHOLD_CANDIDATE,
                        load_graph, simulate)

OLD = HERE.with_name('20261120T031017Z-minimal-stance')


def equal_key_arrays(expected, observed):
    assert set(expected) == set(observed)
    for key in expected:
        np.testing.assert_array_equal(expected[key], observed[key], err_msg=key)


def check_frozen_old(graph):
    roots, exc, inh = graph
    settings = json.loads((OLD / 'NEURAL_INPUT.json').read_text())
    with np.load(OLD / 'banc_tier.npz', allow_pickle=False) as recorded:
        np.testing.assert_array_equal(roots, recorded['root_ids'])
        for label, matrix in (('exc', exc), ('inh', inh)):
            for key in ('data', 'indices', 'indptr'):
                np.testing.assert_array_equal(getattr(matrix, key), recorded[f'{label}_{key}'])
    assert settings['dt_ms'] == DT_MS
    ticks = round(settings['duration_ms'] / DT_MS)
    for label, indices in settings['probes'].items():
        engine = ConductanceNetwork(len(roots), dt_ms=DT_MS).set_connectivity(exc, inh)
        current = np.zeros(len(roots))
        current[indices] = settings['probe_current_nA']
        events = np.stack([engine.step(current if settings['onset_ms'] <= step * DT_MS < settings['offset_ms']
                                       else 0.) for step in range(ticks)])
        with np.load(OLD / 'results' / f'neural_{label}.npz', allow_pickle=False) as actual:
            equal_key_arrays(dict(root_ids=roots, spikes=events, spike_count=engine.spike_count,
                                  voltage_end_mV=engine.v), actual)
    return len(settings['probes'])


def check_dynamic():
    dt = DT_MS
    fixed = ConductanceNetwork(1, dt_ms=dt)
    zero = ConductanceNetwork(1, dt_ms=dt, threshold_adaptation=((10., 200.), (0., 0.)))
    adaptive = ConductanceNetwork(1, dt_ms=dt, threshold_adaptation=THRESHOLD_CANDIDATE)
    response = {'fixed': [], 'adaptive': []}
    tau, amplitude = np.array(THRESHOLD_CANDIDATE)
    max_error = 0.
    for tick in range(3000):
        current = .045 if 40 <= tick < 600 else 0.
        basic_event = bool(fixed.step(current)[0])
        assert bool(zero.step(current)[0]) == basic_event
        np.testing.assert_array_equal(zero.v, fixed.v)
        if basic_event:
            response['fixed'].append(tick)
        if adaptive.step(current)[0]:
            response['adaptive'].append(tick)
        history = np.array(response['adaptive'])
        decay = (tick - history[None, :]) * dt / tau[:, None]
        independent = amplitude * np.exp(-decay).sum(axis=1)
        error = np.max(np.abs(adaptive._threshold_history[:, 0] - independent))
        max_error = max(max_error, float(error))
    assert max_error < 1e-12
    assert response['fixed'][0] == response['adaptive'][0]
    assert len(response['adaptive']) < len(response['fixed'])
    assert not np.any(np.asarray(response['adaptive']) < 40)
    per_cell = ConductanceNetwork(2, dt_ms=dt, threshold_adaptation=((10.,), [[1., 2.]]))
    assert per_cell.step([1., 1.]).all()
    np.testing.assert_array_equal(per_cell._threshold_history, [[1., 2.]])
    per_cell.step(enabled=np.array([False, True]))
    np.testing.assert_array_equal(per_cell._threshold_history, [[1., 2. * np.exp(-dt / 10)]])
    return dict(max_analytic_threshold_error_mV=max_error,
                example_first_spike_ms=(response['fixed'][0] + 1) * dt,
                spike_times_ms={key: ((np.array(ticks) + 1) * dt).tolist() for key, ticks in response.items()},
                zero_increment_is_exact=True)


def check_edges(out):
    graph = json.loads((HERE / 'GRAPH.json').read_text())
    full = load_graph('augmented')
    old = load_graph('legacy')
    quiet = ConductanceNetwork(len(full[0]), dt_ms=DT_MS).set_connectivity(*full[1:])
    assert all(not quiet.step().any() for _ in range(round(120 / DT_MS)))
    reports = []
    for case in graph['graph_probes']:
        pre, post = (case[f'{end}_local'] for end in ('pre', 'post'))
        disconnected = load_graph('augmented', disconnected_edge=(post, pre))
        expected_weight = case['count'] * graph['weight_uS_per_source_synapse']
        assert old[1][post, pre] == disconnected[1][post, pre] == 0.
        assert full[1][post, pre] == expected_weight
        for key, definition in (('legacy', (old, None)), ('augmented', (full, None)),
                                ('edge_cut', (disconnected, None)), ('adaptive', (full, THRESHOLD_CANDIDATE))):
            network, threshold = definition
            result = simulate(network, pre, post, threshold)
            np.savez_compressed(out / f"{case['group']}_{case['side']}_{key}.npz", **result)
            if key == 'augmented':
                connected = result
            elif key == 'edge_cut':
                cut = result
            elif key == 'adaptive':
                adaptive = result
        first_spike = int(np.flatnonzero(connected['spikes'][:, pre])[0])
        assert np.flatnonzero(adaptive['spikes'][:, pre])[0] == first_spike
        steps_delay = ConductanceNetwork(1, dt_ms=DT_MS).delay_steps
        first_arrival = first_spike + steps_delay + 1
        voltage_delta = connected['target_mV'] - cut['target_mV']
        measured = np.flatnonzero(voltage_delta != 0.)
        assert measured.size and measured[0] == first_arrival
        assert voltage_delta[first_arrival] > 0
        measured_ge = connected['target_ge_uS'] - cut['target_ge_uS']
        assert np.all(measured_ge[:first_arrival] == 0.)
        np.testing.assert_allclose(measured_ge[first_arrival], expected_weight * np.exp(-DT_MS / 3), atol=1e-15, rtol=0.)
        reports.append(dict(group=case['group'], side=case['side'], source_root=case['pre'],
            target_root=case['post'], first_source_end_ms=(first_spike + 1) * DT_MS,
            first_target_response_end_ms=(first_arrival + 1) * DT_MS,
            isolated_edge_first_delta_mV=float(voltage_delta[first_arrival]),
            isolated_edge_first_delta_ge_uS=float(measured_ge[first_arrival]),
            fixed_spikes=int(connected['spikes'].sum()), adaptive_spikes=int(adaptive['spikes'].sum())))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    old_cases = check_frozen_old(load_graph('legacy'))
    report = dict(legacy_exact_cases=old_cases, adaptive_single_cell=check_dynamic(),
                  edge_interventions=check_edges(out), physiology_training_updates=0, native_body_steps=0)
    (out / 'VERIFICATION.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"{old_cases} frozen cases exact, {len(report['edge_interventions'])} sourced interventions passed")


if __name__ == '__main__':
    main()
