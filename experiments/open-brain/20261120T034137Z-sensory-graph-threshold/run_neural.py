"""Standalone neural probe for the frozen or augmented BANC circuit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] if len(HERE.parents) >= 3 else HERE
sys.path.insert(0, str(HERE if (HERE / 'engine/neural_cond.py').is_file() else PROJECT))
from engine.neural_cond import ConductanceNetwork

THRESHOLD_CANDIDATE = ((10., 200.), (1.5, 2.0))
DT_MS = .5


def load_graph(which, disconnected_edge=None):
    with np.load(HERE / 'NETWORK.npz', allow_pickle=False) as graph:
        roots = graph['root_ids']
        tags = ('exc', 'inh', 'added_exc')
        matrices = {label: csr_matrix(tuple(graph[f'{label}_{key}'] for key in ('data', 'indices', 'indptr')),
                                       shape=(roots.size, roots.size)) for label in tags}
        legacy_count = int(graph['legacy_count'])
    if which == 'legacy':
        return roots[:legacy_count], matrices['exc'][:legacy_count, :legacy_count], matrices['inh'][:legacy_count, :legacy_count]
    delta = matrices['added_exc']
    if disconnected_edge is not None:
        post, pre = disconnected_edge
        assert delta[post, pre] > 0.
        delta = delta.copy()
        delta[post, pre] = 0.
        delta.eliminate_zeros()
    return roots, matrices['exc'] + delta, matrices['inh']


def simulate(graph, source, observed, threshold_adaptation=None):
    roots, ge, gi = graph
    assert 0 <= source < len(roots) and 0 <= observed < len(roots)
    engine = ConductanceNetwork(len(roots), dt_ms=DT_MS, threshold_adaptation=threshold_adaptation)
    engine.set_connectivity(ge, gi)
    current = np.zeros(len(roots))
    current[source] = .08
    ticks = round(120 / DT_MS)
    spikes = np.empty((ticks, len(roots)), dtype=bool)
    voltage = np.empty(ticks)
    conductance = np.empty(ticks)
    for tick in range(ticks):
        spikes[tick] = engine.step(current if 10 <= tick * DT_MS < 40 else 0.)
        voltage[tick] = engine.v[observed]
        conductance[tick] = engine.ge[observed]
    assert not np.any(spikes[:round(10 / DT_MS)])
    assert np.isfinite(voltage).all() and spikes[:, source].any()
    return dict(root_ids=roots, spikes=spikes, target_mV=voltage, target_ge_uS=conductance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--graph', choices=('legacy', 'augmented'), default='augmented')
    parser.add_argument('--threshold', choices=('fixed', 'adaptive'), default='fixed')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graph = load_graph(args.graph)
    metadata = json.loads((HERE / 'GRAPH.json').read_text())
    adaptation = THRESHOLD_CANDIDATE if args.threshold == 'adaptive' else None
    for probe in metadata['graph_probes']:
        name = f"{probe['group']}_{probe['side']}"
        traces = simulate(graph, probe['pre_local'], probe['post_local'], adaptation)
        np.savez_compressed(args.output / f'{name}.npz', **traces)
    (args.output / 'CONFIG.json').write_text(json.dumps({'graph': args.graph,
        'dt_ms': DT_MS, 'threshold': args.threshold, 'threshold_adaptation': adaptation}, indent=2) + '\n')


if __name__ == '__main__':
    main()
