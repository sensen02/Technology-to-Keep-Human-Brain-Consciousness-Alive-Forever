"""Add a source-checked set of BANC v888 VNC edges to the frozen historical graph."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from pyarrow import feather

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.with_name('20261120T031017Z-minimal-stance')
SURVEY = HERE.with_name('20261120T032302Z-peripheral-sources')
META = ROOT / 'data/banc888/banc_888_meta.feather'
WEIGHT_US = 5e-4
LEG_PARTS = {'front_leg', 'middle_leg', 'hind_leg'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_inputs(paths):
    return {str(path): sha(path) for path in paths}


def allowed_target(row):
    return row['region'] == 'ventral_nerve_cord' and (
        row['super_class'] == 'ventral_nerve_cord_intrinsic' or
        (row['super_class'] == 'motor' and row['body_part_effector'] in LEG_PARTS))


def audit_source(row):
    if row['neurotransmitter_verified'] != 'acetylcholine':
        return 'transmitter_not_verified_ACh'
    if (row['region'] != 'ventral_nerve_cord' or row['proofread'] != 'TRUE'
            or not row['nerve'] or row['flow'] != 'afferent'):
        return 'source_identity_or_coverage'
    return 'verified_ACh'


def build():
    frozen = OLD / 'banc_tier.npz'
    edge_path = SURVEY / 'STANCE_CANDIDATE_EDGES.feather'
    source_audit = SURVEY / 'SOURCE_AUDIT.json'
    paths = [frozen, edge_path, source_audit, META]
    pins = digest_inputs(paths)
    for row in (SURVEY / 'RESULTS.sha256').read_text().splitlines():
        expected, path = row.split(maxsplit=1)
        if path in pins:
            assert pins[path] == expected
    published = json.loads(source_audit.read_text())
    assert pins[str(META)] == published['local_inputs'][str(META)]
    assert pins[str(frozen)] == published['local_inputs'][str(frozen)]

    columns = ['banc_888_id', 'region', 'root_region', 'flow', 'nerve', 'side', 'super_class',
               'cell_type', 'body_part_sensory', 'body_part_effector', 'proofread',
               'neurotransmitter_verified', 'neurotransmitter_predicted']
    records = feather.read_table(META, columns=columns).to_pylist()
    annotations = {r['banc_888_id']: r for r in records}
    assert len(annotations) == len(records)
    source_edges = feather.read_table(edge_path).to_pylist()
    source_groups = {r['pre']: r['presynaptic_group'] for r in source_edges}
    assert len(source_edges) == published['stance_source_edges']['rows']
    assert len(source_groups) == published['stance_source_edges']['source_roots']
    with np.load(frozen, allow_pickle=False) as data:
        legacy_roots = data['root_ids']
        size = len(legacy_roots)
        legacy = {tag: sparse.csr_matrix(tuple(data[f'{tag}_{k}'] for k in ('data', 'indices', 'indptr')),
                                        shape=(size, size)) for tag in ('exc', 'inh')}
    existing = {str(root): index for index, root in enumerate(legacy_roots)}
    assert len(existing) == size and set(existing) <= set(annotations)
    qualifications = {root: audit_source(annotations[root]) for root in source_groups}
    new_records = []
    edge_gate = Counter()
    for row in source_edges:
        pre, post = row['pre'], row['post']
        why = qualifications[pre]
        if why == 'verified_ACh':
            if pre == post:
                why = 'v888_self_edge'
            elif not allowed_target(annotations[post]):
                why = 'outside_VNC_local_or_leg_motor_targets'
            elif pre in existing and post in existing:
                address = existing[post], existing[pre]
                if legacy['inh'][address] != 0:
                    why = 'legacy_sign_conflict'
                elif legacy['exc'][address] != 0:
                    why = 'legacy_existing_exc_edge'
        if why == 'verified_ACh':
            new_records.append(row)
            why = 'admitted_new_edge'
        edge_gate[why] += 1

    append = sorted((set(source_groups) | {r['post'] for r in new_records}) - set(existing), key=int)
    roots = np.concatenate([legacy_roots, np.array(append, dtype=np.int64)])
    local = {str(root): index for index, root in enumerate(roots)}
    assert len(local) == len(roots)
    sources = np.array([local[r['pre']] for r in new_records], dtype=np.int64)
    targets = np.array([local[r['post']] for r in new_records], dtype=np.int64)
    counts = np.array([r['count'] for r in new_records], dtype=np.int64)
    assert np.all(counts > 0)
    n = len(roots)
    delta = sparse.csr_matrix((counts * WEIGHT_US, (targets, sources)), shape=(n, n))
    assert delta.nnz == len(counts), 'published source-target pairs must be unique'
    for matrix in legacy.values():
        matrix.resize((n, n))
        assert matrix.multiply(delta).nnz == 0

    graph = dict(root_ids=roots, legacy_count=size, added_pre=sources, added_post=targets, added_counts=counts)
    for tag, matrix in dict(**legacy, added_exc=delta).items():
        graph.update({f'{tag}_{k}': getattr(matrix, k) for k in ('data', 'indices', 'indptr')})
    np.savez_compressed(HERE / 'NETWORK.npz', **graph)

    probes = []
    for group in sorted(set(source_groups.values())):
        for side in ('left', 'right'):
            options = [r for r in new_records if source_groups[r['pre']] == group
                       and annotations[r['pre']]['side'] == side
                       and annotations[r['post']]['side'] == side and r['post'] in existing
                       and r['pre'] in existing]
            assert options, (group, side)
            row = min(options, key=lambda r: (annotations[r['post']]['super_class'] != 'motor',
                                              -r['count'], r['pre'], r['post']))
            probes.append(dict(group=group, side=side, **row, pre_local=local[row['pre']],
                               post_local=local[row['post']], pre_annotation=annotations[row['pre']],
                               post_annotation=annotations[row['post']]))
    memberships = {root: dict(group=source_groups[root], local=local[root], gate=qualifications[root],
                              source_annotation=annotations[root]) for root in sorted(source_groups, key=int)}
    assert len(source_edges) == sum(edge_gate.values()) and pins == digest_inputs(paths)
    report = dict(input_sha256=pins, legacy_nodes=size, derived_nodes=n,
        appended_sources=len(set(append) & set(source_groups)),
        appended_other_targets=len(set(append) - set(source_groups)),
        new_source_roots=len(set(sources)), new_target_roots=len(set(targets)),
        new_motor_edges=sum(annotations[r['post']]['super_class'] == 'motor' for r in new_records),
        new_inhibitory_edges=0, legacy_exc_edges=legacy['exc'].nnz, legacy_inh_edges=legacy['inh'].nnz,
        source_gates=dict(Counter(qualifications.values())), edge_gates=dict(edge_gate),
        admitted_source_group=dict(Counter(source_groups[str(roots[i])] for i in set(sources))),
        weight_uS_per_source_synapse=WEIGHT_US, graph_probes=probes,
        source_memberships=memberships, artifact_sha256=sha(HERE / 'NETWORK.npz'))
    (HERE / 'GRAPH.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    result = build()
    print({key: result[key] for key in ('legacy_nodes', 'derived_nodes', 'new_source_roots',
                                      'new_target_roots', 'new_motor_edges', 'edge_gates')})
