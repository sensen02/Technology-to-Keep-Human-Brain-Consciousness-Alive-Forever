"""Standalone historical BANC conductance model, with manual diagnostic stimuli."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from neural_cond import ConductanceNetwork, ConductanceParams

HERE = Path(__file__).resolve().parent


def main(out):
    config = json.loads((HERE / 'NEURAL_INPUT.json').read_text())
    dt = config['dt_ms']
    params = ConductanceParams()
    with np.load(HERE / 'banc_tier.npz', allow_pickle=False) as graph:
        roots = graph['root_ids']
        n = roots.size
        weights = [csr_matrix(tuple(graph[f'{label}_{key}'] for key in ('data', 'indices', 'indptr')),
                              shape=(n, n)) for label in ('exc', 'inh')]
    assert n == len(set(roots)) == 12000
    ticks = round(config['duration_ms'] / dt)
    first_tick = round(config['onset_ms'] / dt)
    last_tick = round(config['offset_ms'] / dt)
    report = dict(params=asdict(params), dt_ms=dt, ticks_per_trial=ticks, trials={})
    out.mkdir(parents=True, exist_ok=True)
    for label, indices in config['probes'].items():
        network = ConductanceNetwork(n, params, dt).set_connectivity(*weights)
        current = np.zeros(n)
        current[indices] = config['probe_current_nA']
        spikes = np.empty((ticks, n), dtype=bool)
        for tick in range(ticks):
            spikes[tick] = network.step(current if first_tick <= tick < last_tick else 0.)
        assert np.isfinite(network.v).all()
        assert np.array_equal(network.spike_count, spikes.sum(axis=0))
        assert not np.any(spikes[:first_tick])
        rest = np.ones(n, dtype=bool)
        rest[indices] = False
        report['trials'][label] = dict(input_cells=len(indices), all_spikes=int(spikes.sum()),
            spiking_cells=int(np.count_nonzero(network.spike_count)),
            directly_driven_spikes=int(spikes[:, indices].sum()),
            other_spikes=int(spikes[:, rest].sum()))
        np.savez_compressed(out / f'neural_{label}.npz', root_ids=roots, spikes=spikes,
                            spike_count=network.spike_count, voltage_end_mV=network.v)
    assert report['trials']['quiet']['all_spikes'] == 0
    for key, trial in report['trials'].items():
        if key != 'quiet':
            assert trial['directly_driven_spikes'] > 0
    (out / 'NEURAL.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
