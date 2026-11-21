"""Independent neural test of restored excitation and inhibition in the local VNC."""
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

DT_MS = .5


def load():
    with np.load(HERE / 'NETWORK.npz', allow_pickle=False) as archive:
        roots = archive['root_ids']
        weights = {tag: csr_matrix(tuple(archive[f'{tag}_{key}'] for key in ('data', 'indices', 'indptr')),
                                   shape=(len(roots), len(roots)))
                   for tag in ('exc', 'inh', 'repaired_exc', 'repaired_inh')}
    return roots, weights


def run(weights, roots, source, target):
    net = ConductanceNetwork(len(roots), dt_ms=DT_MS).set_connectivity(*weights)
    current = np.zeros(len(roots))
    current[source] = .08
    ticks = round(60 / DT_MS)
    events = np.empty((ticks, len(roots)), dtype=bool)
    states = np.empty((ticks, 3))
    for k in range(ticks):
        events[k] = net.step(current if 10 <= k * DT_MS < 25 else 0.)
        states[k] = net.v[target], net.ge[target], net.gi[target]
    assert np.isfinite(states).all()
    return events, states


def main(out):
    roots, matrices = load()
    mapping = {str(root): index for index, root in enumerate(roots)}
    graph = json.loads((HERE / 'LOCAL_GRAPH.json').read_text())
    exc, inh = matrices['exc'], matrices['inh']
    delta_e, delta_i = matrices['repaired_exc'], matrices['repaired_inh']
    for old, diff in ((exc + inh, delta_e + delta_i), (delta_e, delta_i)):
        assert old.multiply(diff).nnz == 0
    original = exc, inh
    repaired = exc + delta_e, inh + delta_i
    assert sum(m.nnz for m in original) == graph['original_edges']
    assert sum(m.nnz for m in repaired) == graph['all_edges']
    quiet = ConductanceNetwork(len(roots), dt_ms=DT_MS).set_connectivity(*repaired)
    assert all(not quiet.step().any() for _ in range(120))
    reports = []
    for candidate in graph['probes']:
        src, dst = (mapping[candidate[name]] for name in ('pre', 'post'))
        sign = candidate['sign']
        select = 0 if sign == 'exc' else 1
        cut = list(repaired)
        cut[select] = cut[select].copy()
        weight = cut[select][dst, src]
        assert original[select][dst, src] == 0. and weight == .0005 * candidate['count']
        cut[select][dst, src] = 0.
        cut[select].eliminate_zeros()
        responses = {}
        for label, weights in (('original', original), ('restored', repaired), ('single_edge_cut', cut)):
            events, states = run(weights, roots, src, dst)
            responses[label] = (events, states)
            np.savez_compressed(out / f'{sign}_{label}.npz', root_ids=roots, spikes=events, target_states=states)
        events, states = responses['restored']
        cut_events, old_states = responses['single_edge_cut']
        first_spike = int(np.flatnonzero(events[:, src])[0])
        difference = states - old_states
        arrival = int(np.flatnonzero(difference[:, select + 1])[0])
        assert arrival == first_spike + round(2 / DT_MS) + 1
        assert not np.any(difference[:arrival])
        if sign == 'exc':
            assert difference[arrival, 0] > 0 or (events[arrival, dst] and not cut_events[arrival, dst])
        else:
            assert difference[arrival, 0] < 0
        tau = 3. if select == 0 else 8.
        np.testing.assert_allclose(difference[arrival, select + 1], weight * np.exp(-DT_MS / tau), rtol=0., atol=1e-14)
        reports.append(dict(**candidate, first_source_spike_ms=(first_spike + 1)*DT_MS,
                            first_target_conductance_difference_ms=(arrival + 1)*DT_MS,
                            delta_first_target_voltage_mV=float(difference[arrival, 0]),
                            target_spiked_on_arrival=bool(events[arrival, dst]),
                            cut_target_spiked_on_arrival=bool(cut_events[arrival, dst]),
                            total_spikes={k: int(v[0].sum()) for k, v in responses.items()}))
    (out / 'NEURAL_VERIFICATION.json').write_text(json.dumps(dict(cuts=reports, quiet=True, neural_ticks=840,
        body_command_generated=False, train_updates=0), indent=2) + '\n')
    print(reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    path = parser.parse_args().output
    path.mkdir(parents=True, exist_ok=True)
    main(path)
