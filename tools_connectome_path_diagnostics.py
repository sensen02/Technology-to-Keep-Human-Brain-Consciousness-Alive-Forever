"""Directed paths for W[target, source]. No edge creation or sign inference."""
import numpy as np
from scipy import sparse


def distances_from_sources(weights, sources):
    graph = sparse.csc_matrix(weights)
    distance = np.full(graph.shape[0], -1, dtype=int)
    frontier = sorted(set(map(int, sources)))
    distance[frontier] = 0
    depth = 0
    while frontier:
        following = set()
        for source in frontier:
            targets = graph.indices[graph.indptr[source]:graph.indptr[source+1]]
            following.update(int(t) for t in targets if distance[t] < 0)
        depth += 1
        frontier = sorted(following)
        distance[frontier] = depth
    return distance


def diagnose(net, sources, motor, fired_counts, roots, sensory, peak_v, peak_ge, peak_gi):
    # We/Wi are nonnegative conductance matrices; union includes inhibitory paths.
    distances = distances_from_sources(net.We+net.Wi, sources)
    fired = np.flatnonzero(fired_counts)
    nonsensory = set(range(net.n))-set(map(int, sensory))
    source_set = set(map(int, sources))
    motor_set = set(map(int, motor))
    connections = []
    for source in fired:
        for sign, weights in [('excitatory', net.We), ('inhibitory', net.Wi)]:
            col = weights[:, int(source)].tocoo()
            for target, value in zip(col.row, col.data):
                if int(target) in motor_set:
                    connections.append({'source_root_id': roots[int(source)], 'target_root_id': roots[int(target)],
                                        'kind': sign, 'conductance_increment_uS': float(value),
                                        'source_spikes': int(fired_counts[source])})
    return {'edge_orientation': 'W[target, source], as used by ConductanceNetwork.step',
            'path_limit': 'Unsigned reachability includes inhibitory paths; NOT proof of functional excitation.',
            'fired_nonsensory_cells': [{'root_id': roots[int(i)], 'tier_index': int(i), 'spikes': int(fired_counts[i])}
                                     for i in fired if int(i) in nonsensory],
            'fired_source_to_motor_edges': connections,
            'motor_cells': [{'root_id': roots[int(i)], 'tier_index': int(i),
                             'is_directly_stimulated': int(i) in source_set,
                             'shortest_unsigned_hops': int(distances[i]) if distances[i] >= 0 else None,
                             'spikes': int(fired_counts[i]), 'peak_post_step_voltage_mV': float(peak_v[j]),
                             'threshold_gap_mV': float(net.p.threshold_mV-peak_v[j]),
                             'peak_post_decay_ge_uS': float(peak_ge[j]),
                             'peak_post_decay_gi_uS': float(peak_gi[j])} for j, i in enumerate(motor)]}
