"""Restore source-grounded local outputs for the 600 previously terminal VNC targets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pyarrow import array, compute as pc, feather
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'experiments/open-brain/20261120T034137Z-sensory-graph-threshold'
META = ROOT / 'data/banc888/banc_888_meta.feather'
PUBLISHED = ROOT / 'experiments/visual-orientation/20261116T023521Z/banc_888_edgelist_simple_v2.feather'
CONDUCTANCE = .0005
TRANSMITTER = {'acetylcholine': 'exc', 'gaba': 'inh'}
LEG_AREAS = {'front_leg', 'middle_leg', 'hind_leg'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matrices(data, keys):
    n = len(data['root_ids'])
    return {tag: csr_matrix(tuple(data[f'{tag}_{field}'] for field in ('data', 'indices', 'indptr')),
                            shape=(n, n)) for tag in keys}


def permitted_receiver(meta, root, sensory_roots):
    return meta['region'] == 'ventral_nerve_cord' and (
        meta['super_class'] == 'ventral_nerve_cord_intrinsic' or root in sensory_roots or
        (meta['super_class'] == 'motor' and meta['body_part_effector'] in LEG_AREAS))


def cycles(matrix, terminal_indices):
    reduced = matrix[terminal_indices, :][:, terminal_indices].tocsr()
    size, labels = connected_components(reduced, directed=True, connection='strong')
    memberships = np.bincount(labels, minlength=size)
    return dict(cyclic_nodes=int(memberships[memberships > 1].sum()), largest_cycle_component=int(memberships.max()),
                internal_edges=reduced.nnz)


def build(out):
    parent_file = PARENT / 'NETWORK.npz'
    source_report = json.loads((PARENT / 'GRAPH.json').read_text())
    survey = ROOT / 'experiments/open-brain/20261120T032302Z-peripheral-sources/SOURCE_AUDIT.json'
    pinned = {str(path): sha(path) for path in (parent_file, META, PUBLISHED)}
    assert pinned[str(parent_file)] == source_report['artifact_sha256']
    assert pinned[str(META)] == source_report['input_sha256'][str(META)]
    assert pinned[str(PUBLISHED)] == json.loads(survey.read_text())['local_inputs'][str(PUBLISHED)]
    with np.load(parent_file, allow_pickle=False) as data:
        roots = data['root_ids']
        prior = matrices(data, ('exc', 'inh', 'added_exc'))
        legacy_count = int(data['legacy_count'])
    mapped = {str(root): i for i, root in enumerate(roots)}
    sensory_roots = set(source_report['source_memberships'])
    terminal = sorted(set(mapped) - sensory_roots - {str(root) for root in roots[:legacy_count]}, key=int)
    indices = np.array([mapped[root] for root in terminal])
    original = {'exc': prior['exc'] + prior['added_exc'], 'inh': prior['inh']}
    old_edges = (original['exc'] + original['inh']).tocsc()
    assert len(terminal) == 600 and np.all(np.diff(old_edges.indptr)[indices] == 0)
    assert np.all(np.diff(old_edges.tocsr().indptr)[indices] > 0)

    columns = ['banc_888_id', 'super_class', 'region', 'proofread', 'neurotransmitter_verified',
               'body_part_effector', 'side', 'cell_type']
    metadata = feather.read_table(META, columns=columns).to_pylist()
    meta = {row['banc_888_id']: row for row in metadata if row['banc_888_id'] in mapped}
    assert len(meta) == len(mapped)
    table = feather.read_table(PUBLISHED, columns=['pre', 'post', 'count'])
    source_edges = table.filter(pc.is_in(table['pre'], value_set=array(terminal))).to_pylist()
    known = {root: TRANSMITTER[meta[root]['neurotransmitter_verified']]
             for root in terminal if meta[root]['proofread'] == 'TRUE'
             and meta[root]['region'] == 'ventral_nerve_cord'
             and meta[root]['super_class'] == 'ventral_nerve_cord_intrinsic'
             and meta[root]['neurotransmitter_verified'] in TRANSMITTER}
    evaluated = Counter()
    records = []
    raw_sources = set()
    for row in source_edges:
        pre, post = row['pre'], row['post']
        raw_sources.add(pre)
        if pre not in known:
            evaluated['source_identity_or_sign_unresolved'] += 1
        elif pre == post:
            evaluated['self_edge'] += 1
        elif post not in mapped:
            evaluated['outside_current_graph'] += 1
        elif not permitted_receiver(meta[post], post, sensory_roots):
            evaluated['outside_local_receiver_scope'] += 1
        else:
            evaluated['restored'] += 1
            records.append(dict(**row, sign=known[pre]))
    assert len(source_edges) == sum(evaluated.values())
    assert len({(r['pre'], r['post']) for r in records}) == len(records)
    assert all(r['count'] > 0 for r in records)
    additions = {}
    saved = dict(root_ids=roots)
    for sign in ('exc', 'inh'):
        edges = [r for r in records if r['sign'] == sign]
        additions[sign] = csr_matrix((np.array([r['count'] for r in edges]) * CONDUCTANCE,
            ([mapped[r['post']] for r in edges], [mapped[r['pre']] for r in edges])), shape=old_edges.shape)
        assert old_edges.multiply(additions[sign]).nnz == 0
        for label, matrix in ((sign, original[sign]), (f'repaired_{sign}', additions[sign])):
            saved.update({f'{label}_{field}': getattr(matrix, field) for field in ('data', 'indices', 'indptr')})
    now = old_edges + additions['exc'] + additions['inh']
    assert len(records) == now.nnz - old_edges.nnz
    recovered = {r['pre'] for r in records}
    assert recovered <= set(terminal) and len(recovered) == np.count_nonzero(np.diff(now.tocsc().indptr)[indices])
    report = dict(nodes=len(roots), original_edges=old_edges.nnz, repaired_edges=len(records), all_edges=now.nnz,
        original_terminal_count=len(terminal), terminal_with_published_outgoing=len(raw_sources),
        sources_with_eligible_transmitter=len(known), restored_source_count=len(recovered),
        still_terminal_count=len(terminal)-len(recovered), excluded_by_reason=dict(evaluated),
        added_by_sign={key: mat.nnz for key, mat in additions.items()},
        motor_target_edges=sum(meta[r['post']]['super_class'] == 'motor' for r in records),
        candidate_sensory_target_edges=sum(r['post'] in sensory_roots for r in records),
        components_before=cycles(old_edges, indices), components_after=cycles(now, indices),
        input_sha256=pinned, terminal_roots=terminal, remaining_terminal_roots=sorted(set(terminal)-recovered, key=int),
        probes=[min((r for r in records if r['sign'] == tag), key=lambda r: (-r['count'], r['pre'], r['post']))
                for tag in TRANSMITTER.values()])
    assert pinned == {str(path): sha(path) for path in (parent_file, META, PUBLISHED)}
    np.savez_compressed(out / 'NETWORK.npz', **saved)
    (out / 'LOCAL_GRAPH.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (out / 'LOCAL_EDGES.json').write_text(json.dumps(records, separators=(',', ':')) + '\n')
    print({k: report[k] for k in ('restored_source_count','still_terminal_count','added_by_sign','components_after')})


if __name__ == '__main__':
    build(Path(sys.argv[1]).resolve())
