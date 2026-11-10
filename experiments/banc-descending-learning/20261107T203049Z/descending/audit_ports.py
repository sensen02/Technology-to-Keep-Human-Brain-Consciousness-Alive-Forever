"""Round11 source-only DNp09 graph audit. Run with venv_body/bin/python -B.

Writes only sibling PORTS.json. No neural/physical runtime imports or steps.
CSR BFS is checked against scipy shortest_path(method='D', unweighted=True),
with only two source rows at a time, never a dense N x N distance matrix.
"""
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
HERE = Path(__file__).resolve().parent
FROZEN = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'
PATHS = {
    'source_connectome': PROJECT / 'data/flywire/banc_connectome.npz',
    'local_candidate': FROZEN / 'LOCAL_MODULE_CANDIDATE.npz',
    'groups': FROZEN / 'GROUP_CACHE_INDICES.npz',
    'motor_ports': FROZEN / 'LOCAL_MOTOR_READOUT_PORTS.csv',
    'manifest': FROZEN / 'NEURAL_INTERFACE_MANIFEST.json',
    'prior_candidates': Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/named-dn/CANDIDATES.csv'),
    'source_runtime_definition': PROJECT / 'experiments/banc-local-learning/20261107T185910Z/dynamics/module.py',
}
PURE = ('acetylcholine', 'gaba', 'glutamate')
EXPECTED = [(1414, 10567, '720575941433155799', 181, 114),
            (13170, 99854, '720575941566493282', 161, 112)]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a, dtype='<i8'):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def bfs(graph, sources):
    """Independent FIFO BFS; CSR only. -1 marks unreachable; roots are zero."""
    dist = np.full(graph.shape[0], -1, dtype=np.int32)
    parent = np.full(graph.shape[0], -1, dtype=np.int32)
    roots = np.unique(np.asarray(sources, dtype=np.int64))
    dist[roots] = 0
    queue = deque(roots.tolist())
    while queue:
        u = queue.popleft()
        neighbors = graph.indices[graph.indptr[u]:graph.indptr[u + 1]]
        fresh = neighbors[dist[neighbors] < 0]
        dist[fresh] = dist[u] + 1
        parent[fresh] = u
        queue.extend(fresh.tolist())
    return dist, parent


def histogram(dist):
    dist = np.asarray(dist)
    result = {'total': int(dist.size), 'hop0': int((dist == 0).sum()),
              'hop1': int((dist == 1).sum()), 'hop2': int((dist == 2).sum()),
              'hop3': int((dist == 3).sum()), 'hop_ge4': int((dist >= 4).sum()),
              'unreachable': int((dist < 0).sum()), 'reachable_positive_hops': int((dist > 0).sum())}
    assert sum(result[k] for k in ('hop0', 'hop1', 'hop2', 'hop3', 'hop_ge4', 'unreachable')) == dist.size
    return result


def union_dist(rows):
    stack = np.stack(rows)
    return np.where(np.all(stack < 0, axis=0), -1, np.where(stack < 0, 2**30, stack).min(axis=0))


def main():
    sources = {k: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for k, p in PATHS.items()}
    a, c, groups = (load_npz(PATHS[k]) for k in ('source_connectome', 'local_candidate', 'groups'))
    manifest = json.loads(PATHS['manifest'].read_text())
    assert sources['source_connectome']['sha256'] == manifest['sources']['cache']['sha256']
    for key in ('local_candidate', 'groups', 'motor_ports'):
        assert sources[key]['sha256'] == manifest['artifacts'][PATHS[key].name]['sha256']
    prior = {r['root_id_string']: r for r in read_csv(PATHS['prior_candidates'])}
    motor_rows = read_csv(PATHS['motor_ports'])
    rid, pre, post, syn = (a[k] for k in ('root_ids', 'pre', 'post', 'syn'))
    n = len(rid)
    assert rid.dtype.kind in 'iu' and np.unique(rid).size == n
    assert np.all(np.isfinite(syn)) and np.all(syn > 0) and np.all(syn == np.floor(syn))
    selected = c['cache_indices']
    local = np.full(n, -1, dtype=np.int32)
    local[selected] = np.arange(len(selected))
    assert len(selected) == 20000 and np.array_equal(selected, groups['local_module_selected'])
    assert np.array_equal(rid[selected], c['root_ids'])
    nt = a['ann_nt_verified']
    strict = np.isin(nt[pre], PURE)
    induced = (local[pre] >= 0) & (local[post] >= 0)
    erows = np.flatnonzero(induced)
    srows = np.flatnonzero(induced & strict)
    assert np.array_equal(erows, c['source_edge_rows']) and len(erows) == 705068
    assert np.array_equal(local[pre[erows]], c['pre_local'])
    assert np.array_equal(local[post[erows]], c['post_local'])
    assert np.array_equal(syn[erows], c['synapse_counts'])
    assert np.array_equal(nt[pre[erows]], c['source_verified_nt'])
    assert np.array_equal(strict[erows], c['pure_verified_label_mask'])
    assert np.array_equal(srows, c['strict_source_edge_rows']) and len(srows) == 435126
    all_mn = np.array([int(r['cache_index_0based']) for r in motor_rows], dtype=np.int64)
    exported = np.array([int(r['cache_index_0based']) for r in motor_rows if r['readout_eligible_single_leg'] == 'True'], dtype=np.int64)
    assert len(exported) == 323 and len(all_mn) == 391
    assert np.array_equal(all_mn, groups['leg_motor_class'])
    assert np.array_equal(all_mn, np.flatnonzero(a['ann_class'] == 'leg_motor_neuron'))
    for row in motor_rows:
        g, l = int(row['cache_index_0based']), int(row['local_index_0based'])
        assert str(rid[g]) == row['root_id_string'] and local[g] == l
    targets = {'exported323': exported, 'all391_leg_motor': all_mn}
    dns = np.array([e[1] for e in EXPECTED], dtype=np.int64)
    full = csr_matrix((np.ones(strict.sum(), dtype=np.uint8), (pre[strict], post[strict])), shape=(n, n))
    loc = csr_matrix((np.ones(len(srows), dtype=np.uint8), (local[pre[srows]], local[post[srows]])), shape=(20000, 20000))
    assert full.nnz == strict.sum() and loc.nnz == len(srows), 'source contains duplicate pairs'
    distances, predecessors, verification = {}, {}, []
    for name, graph, indices, nodes in [('full_strict', full, dns, np.arange(n)), ('local_strict_induced', loc, local[dns], selected)]:
        distances[name], predecessors[name] = {}, {}
        for direction, graph_d in [('downstream', graph), ('upstream', graph.T.tocsr())]:
            rows, pars = zip(*(bfs(graph_d, [source]) for source in indices))
            independent = shortest_path(graph_d, method='D', directed=True, unweighted=True, indices=indices)
            for j in range(2):
                comparison = np.where(np.isfinite(independent[j]), independent[j], -1)
                assert np.array_equal(rows[j], comparison), (name, direction, j)
            verification.append({'graph': name, 'direction': direction, 'sources': 2,
                                 'compared_distances': int(independent.size), 'exact': True,
                                 'maximum_dense_distance_shape': list(independent.shape)})
            expanded = []
            for d, p in zip(rows, pars):
                out = np.full(n, -1, dtype=np.int32)
                out[nodes] = d
                expanded.append(out)
            distances[name][direction] = expanded + [union_dist(expanded)]
            predecessors[name][direction] = pars
    def identity(g):
        g = int(g)
        return {'global_index': g, 'local_index': int(local[g]) if local[g] >= 0 else None,
                'root_id_string': str(rid[g]), 'primary_type': str(a['ann_primary_type'][g]),
                'class': str(a['ann_class'][g]), 'superclass': str(a['ann_super_class'][g]),
                'source_verified_nt': str(nt[g]), 'in_descending_boundary': bool(g in groups['descending_boundary'])}
    def label_counts(nodes):
        return {k: dict(sorted(Counter(a[field][nodes].tolist()).items())) for k, field in
                [('primary_type', 'ann_primary_type'), ('class', 'ann_class'), ('superclass', 'ann_super_class'), ('source_verified_nt', 'ann_nt_verified')]}
    def edge_summary(rows, endpoint):
        nodes = np.unique(endpoint[rows])
        return {'edge_pairs': len(rows), 'unique_neighbor_cells': len(nodes),
                'synapse_sum': int(syn[rows].sum(dtype=np.float64)), 'neighbor_labels': label_counts(nodes),
                'source_nt_edge_counts': dict(sorted(Counter(nt[pre[rows]].tolist()).items()))}
    candidates = []
    for j, (l, g, root, expected_full, expected_local) in enumerate(EXPECTED):
        assert int(selected[l]) == g and str(rid[g]) == root
        ident = identity(g)
        assert (ident['primary_type'], ident['class'], ident['superclass'], ident['source_verified_nt'], ident['in_descending_boundary']) == ('DNp09', 'descending_neuron', 'descending', 'acetylcholine', True)
        assert int(prior[root]['local_index']) == l and int(prior[root]['cache_index']) == g
        assert int(prior[root]['full_outgoing_edges']) == expected_full
        assert int(prior[root]['strict_local_outgoing_edges']) == expected_local
        item = {'identity': ident, 'left_right_homology_behavioral_function': 'NOT_ESTABLISHED_FROM_DNp09_ANNOTATION'}
        for direction, mask, endpoint in [('outgoing', pre == g, post), ('incoming', post == g, pre)]:
            masks = {'full_original': mask, 'full_strict': mask & strict,
                     'selected_original': mask & induced, 'selected_strict': mask & induced & strict,
                     'strict_lost_by_selection': mask & strict & ~induced,
                     'full_excluded_nonpure_source_label': mask & ~strict}
            summaries = {key: edge_summary(np.flatnonzero(value), endpoint) for key, value in masks.items()}
            summaries['source_edge_records'] = [
                {'source_edge_row': int(e), 'pre_global_index': int(pre[e]), 'post_global_index': int(post[e]),
                 'synapse_count': int(syn[e]), 'strict_source_eligible': bool(strict[e]),
                 'retained_in_selected_induced_graph': bool(induced[e]), 'neighbor': identity(endpoint[e])}
                for e in np.flatnonzero(mask)]
            item[direction] = summaries
        assert item['outgoing']['full_original']['edge_pairs'] == expected_full
        assert item['outgoing']['selected_strict']['edge_pairs'] == expected_local
        candidates.append(item)
    reach = {}
    for label, j in [('local1414', 0), ('local13170', 1), ('union_min_hops', 2)]:
        result = {}
        for direction in ('downstream', 'upstream'):
            fd = distances['full_strict'][direction][j]
            ld = distances['local_strict_induced'][direction][j]
            data = {'full_graph_all_nodes': histogram(fd), 'full_graph_selected_nodes': histogram(fd[selected]),
                    'local_graph_selected_nodes': histogram(ld[selected]),
                    'full_reachable_outside_selected_positive_hops': int(((fd > 0) & (local < 0)).sum()),
                    'selected_reachable_full_but_unreachable_local': int(((fd[selected] > 0) & (ld[selected] < 0)).sum()),
                    'selected_shorter_full_than_local': int(((fd[selected] >= 0) & (ld[selected] > fd[selected])).sum()),
                    'full_reachable_positive_labels': label_counts(np.flatnonzero(fd > 0)),
                    'local_reachable_positive_labels': label_counts(np.flatnonzero(ld > 0)), 'targets': {}}
            for key, target in targets.items():
                f, d = fd[target], ld[target]
                data['targets'][key] = {'full_strict': histogram(f), 'local_strict_induced': histogram(d),
                    'lost_reachability_in_local': int(((f >= 0) & (d < 0)).sum()),
                    'longer_shortest_path_in_local': int(((f >= 0) & (d > f)).sum())}
            result[direction] = data
        reach[label] = result
    # Multi-source reverse BFS identifies full DN->motor route corridors without N x N matrices.
    corridors = {}
    for target_name, target in targets.items():
        to_motor, _ = bfs(full.T.tocsr(), target)
        corridor = (distances['full_strict']['downstream'][2] >= 0) & (to_motor >= 0)
        corridor[dns] = False
        corridor[target] = False
        extra = np.flatnonzero(corridor & (local < 0))
        corridors[target_name] = {'definition': 'Non-endpoint cells reachable from either DN and able to reach any target; existence on a directed walk, not necessarily a shortest/simple path.',
            'full_route_intermediate_cells': int(corridor.sum()), 'outside_selected_intermediate_cells': len(extra),
            'outside_selected_global_indices': extra.tolist(), 'outside_selected_index_sha256_i64le': array_sha(extra),
            'outside_selected_labels': label_counts(extra), 'outside_selected_identity_sample_first20': [identity(g) for g in extra[:20]]}
    # Exact target identities and per-DN and union distances permit independent endpoint reconstruction.
    target_records = []
    exported_set = set(exported.tolist())
    for g in all_mn:
        row = {'identity': identity(g), 'in_exact_exported323': int(g) in exported_set}
        for graph in distances:
            row[graph] = {direction: {label: int(distances[graph][direction][j][g]) for j, label in enumerate(['local1414', 'local13170', 'union_min_hops'])}
                          for direction in ('downstream', 'upstream')}
        target_records.append(row)
    incoming = np.bincount(local[post[srows]], weights=syn[srows].astype(np.float64), minlength=20000)
    candidate_incoming = np.bincount(c['post_local'][c['pure_verified_label_mask']], weights=c['synapse_counts'][c['pure_verified_label_mask']].astype(np.float64), minlength=20000)
    assert np.array_equal(incoming, candidate_incoming)
    weight = .02 * syn[srows].astype(np.float64) / incoming[local[post[srows]]]
    totals = np.bincount(local[post[srows]], weights=weight, minlength=20000)
    assert np.allclose(totals[incoming > 0], .02, rtol=0, atol=1e-15)
    assert all(sha(PATHS[k]) == v['sha256'] for k, v in sources.items())
    result = {
        'schema': 'round11-dnp09-source-graph-audit/1', 'status': 'VERIFIED_SOURCE_GRAPH_ONLY',
        'task_relation': 'S3/S4 named descending source interface; no functional or behavioral validation',
        'scope': {'neural_steps': 0, 'physical_steps': 0, 'training_updates': 0, 'external_research': 0, 'new_synapses': 0},
        'source_hashes': sources, 'audit_script': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
        'definitions': {'direction': 'pre -> post; upstream means reverse-graph distance, i.e. target/ancestor -> DN',
            'strict_rule': 'Keep an edge iff original source ann_nt_verified is exactly acetylcholine, gaba, or glutamate; destination label need not qualify. No predicted, mixed or unknown source labels admitted.',
            'local_graph': 'Exact source graph induced by frozen 20000 cache indices, then same strict rule.',
            'union': 'Minimum finite directed hop distance over the two DN seeds; not sum of counts.',
            'unreachable': '-1 in target records; missing strict path is not proof of biological absence.',
            'synapse_counts': 'Original cache aggregated pair syn counts, distinct from edge-pair and neighbor counts.',
            'root_ids': 'Loaded as int64; exported as exact decimal strings, never floats.',
            'limits': 'Graph reachability alone establishes neither receptor signs, physiological recruitment, laterality, homology nor DNp09 behavioral function. Full graph uses cells absent from runtime; audit does not expand runtime.'},
        'graphs': {'original': {'nodes': n, 'edge_pairs': len(pre), 'synapse_sum': int(syn.sum(dtype=np.float64))},
            'full_strict': {'nodes': n, 'edge_pairs': int(strict.sum()), 'synapse_sum': int(syn[strict].sum(dtype=np.float64)), 'source_nt_edge_counts': dict(Counter(nt[pre[strict]].tolist()))},
            'local_original_induced': {'nodes': 20000, 'edge_pairs': len(erows), 'synapse_sum': int(syn[erows].sum(dtype=np.float64))},
            'local_strict_induced': {'nodes': 20000, 'edge_pairs': len(srows), 'synapse_sum': int(syn[srows].sum(dtype=np.float64)), 'source_nt_edge_counts': dict(Counter(nt[pre[srows]].tolist()))}},
        'runtime_normalization_preserved': {'changed': False, 'total_incoming_conductance_uS': .02,
            'formula': '0.02 * original_syn[e] / sum(original_syn over ALL frozen strict local edges with same post)',
            'denominator_does_not_limit_to_DN_edges': True, 'denominator_does_not_use_full_graph': True,
            'denominator_exact_candidate_match': True, 'incoming_denominator_sha256_f64le': array_sha(incoming, '<f8'),
            'strict_source_edge_rows_sha256_i64le': array_sha(srows), 'normalized_weight_sha256_f64le': array_sha(weight, '<f8'),
            'positive_denominator_nodes': int((incoming > 0).sum()), 'max_total_conductance_error_uS': float(np.max(np.abs(totals[incoming > 0] - .02)))},
        'candidates': candidates, 'directed_reachability': reach, 'full_graph_extra_intermediates': corridors,
        'exact_motor_target_records': target_records,
        'verification': {'all_source_hashes_unchanged': True, 'candidate_arrays_exact_source_match': True,
            'expected_DN_identities_and_edge_counts': True, 'independent_method': 'FIFO CSR BFS versus scipy directed unweighted shortest_path(method=D); all distances checked for both DN roots, both graphs, both directions',
            'comparisons': verification, 'dense_N_by_N_allocated': False, 'numpy_version': np.__version__, 'scipy_version': scipy.__version__}}
    (HERE / 'PORTS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'graphs': result['graphs'],
        'candidates': [{'identity': x['identity'], 'outgoing': {k: x['outgoing'][k]['edge_pairs'] for k in ['full_original', 'full_strict', 'selected_strict']},
                        'incoming': {k: x['incoming'][k]['edge_pairs'] for k in ['full_original', 'full_strict', 'selected_strict']}} for x in candidates],
        'motor_reachability': {k: v['downstream']['targets'] for k, v in reach.items()},
        'output_sha256': sha(HERE / 'PORTS.json')}, indent=2))


if __name__ == '__main__':
    main()
