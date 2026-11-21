"""Freeze the historical BANC circuit and audit source identities without a body bridge."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from pyarrow import feather

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from engine.embodied.adapters import TierConfig, build_neural_tier
from engine.embodied.motor_output import build_motor_manifest

BANC = ROOT / 'data/flywire/banc_connectome.npz'
META = ROOT / 'data/banc888/banc_888_meta.feather'
MOTOR = ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
FINGERPRINT = '5ab78f550d682e2849749587cb6f7551'
LEG_NAMES = {'F': 'front_leg', 'M': 'middle_leg', 'H': 'hind_leg'}


def source_rows(manifest):
    columns = ['root_id', 'banc_888_id', 'side', 'body_part_effector',
               'peripheral_target_type', 'super_class', 'flow', 'nerve']
    table = feather.read_table(META, columns=columns)
    root_set = {str(root) for row in manifest['rows'] for root in row['pool_root_ids']}
    by_root = {}
    by_target = defaultdict(list)
    for row in table.to_pylist():
        if row['root_id'] in root_set:
            assert row['root_id'] not in by_root
            by_root[row['root_id']] = row
        target = row['peripheral_target_type']
        if target and row['flow'] == 'efferent':
            by_target[target, row['body_part_effector'], row['side']].append(int(row['root_id']))
    assert set(by_root) == root_set
    for row in manifest['rows']:
        key = row['muscle_target'], LEG_NAMES[row['leg'][1]], row['side']
        assert set(row['pool_root_ids']) == set(by_target[key]), row['actuator']
        row['source_root_annotations'] = [by_root[str(root)] for root in row['pool_root_ids']]
    return by_root


def distances(graph, groups):
    return {label: dijkstra(graph, directed=True, indices=indices, unweighted=True, min_only=True)
            for label, indices in groups.items()}


def minimum_lengths(from_groups, sources):
    hits = {}
    for label, dist in from_groups.items():
        known = [int(dist[src]) for src in sources if np.isfinite(dist[src])]
        hits[label] = min(known, default=None)
    return hits


def peripheral_coverage(a, gi, modeled_degree):
    region = a['ann_body_part']
    nerve = a['ann_nerve']
    cls = a['ann_super_class']
    afferent = a['ann_flow'] == 'afferent'
    groups = {
        'brain_intrinsic': cls == 'central_brain_intrinsic',
        'optic_lobe_intrinsic': cls == 'optic_lobe_intrinsic',
        'ventral_nerve_cord_intrinsic': cls == 'ventral_nerve_cord_intrinsic',
        'six_leg_peripheral_afferent': afferent & np.isin(region, list(LEG_NAMES.values())),
        'abdomen_annotations': np.isin(region, ['abdomen', 'abdominal_wall', 'thoracic_abdominal']),
        'abdominal_nerve_afferent': afferent & (np.char.find(nerve, 'abdominal_nerve') >= 0),
        'body_wall_afferent_labels': afferent & np.isin(region, ['abdominal_wall', 'thorax', 'neck']),
        'digestive_labels': np.isin(region, ['anterior_digestive_tract', 'crop']) | (np.char.find(region, 'pharynx') >= 0),
        'anterior_digestive_tract': region == 'anterior_digestive_tract',
        'crop': region == 'crop',
        'pharynx_labels': np.char.find(region, 'pharynx') >= 0,
        'midgut_or_hindgut_specific_label': (np.char.find(region, 'midgut') >= 0) | (np.char.find(region, 'hindgut') >= 0),
    }
    return {key: dict(cache=int(mask.sum()), tier=int(mask[gi].sum()),
                      tier_with_modeled_outgoing=int(np.count_nonzero(mask[gi] & (modeled_degree > 0))),
                      cache_flow=dict(Counter(a['ann_flow'][mask].tolist())),
                      tier_root_ids=[str(root) for root in a['root_ids'][gi[mask[gi]]]])
            for key, mask in groups.items()}


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    built = build_neural_tier(TierConfig(), data_path=str(BANC))
    assert built['fingerprint'] == FINGERPRINT, 'snapshot-historical graph differs'
    a, gi, n = built['data'], built['global_index'], built['n']
    root_ids = a['root_ids'][gi]
    local_of_global = built['local_of_global']
    exc = built['edges']['We']
    inh = built['edges']['Wi']
    modeled = (exc + inh).T.tocsr()
    out_degree = np.diff(modeled.indptr)
    source = json.loads(MOTOR.read_text())
    body = json.loads((out / 'BODY.json').read_text())
    manifest = build_motor_manifest(source, root_ids, body['actuator_names'])
    annotations = source_rows(manifest)

    pg, qg = a['pre'], a['post']
    valid = (pg >= 0) & (qg >= 0)
    raw = sparse.csr_matrix((np.ones(valid.sum(), dtype=bool), (pg[valid], qg[valid])),
                            shape=(a['root_ids'].size,) * 2)
    induced = raw[gi, :][:, gi].tocsr()
    assert modeled.multiply(induced).nnz == modeled.nnz

    groups = {leg.upper(): indices for leg, indices in built['sensory'].items()}
    groups['crossing_descending'] = built['descending_local']
    groups['crossing_ascending'] = built['boundary']['crossing_ascending']['local_indices']
    raw_groups = {leg: np.flatnonzero(a['ann_nerve'] == nerve) for leg, nerve in
                  ((leg, a['ann_nerve'][gi[indices[0]]]) for leg, indices in groups.items() if len(leg) == 2)}
    mech = built['groups_global']['route_overlapping']['body_touch_sensor']
    raw_groups = {leg: indices[mech[indices]] for leg, indices in raw_groups.items()}
    for key in ('crossing_descending', 'crossing_ascending'):
        raw_groups[key] = np.flatnonzero(built['groups_global']['route_overlapping'][key])
    for leg, global_indices in raw_groups.items():
        assert set(gi[groups[leg]]) <= set(global_indices)
    distance_raw = distances(raw, raw_groups)
    distance_tier = distances(induced, groups)
    distance_modeled = distances(modeled, groups)
    global_index = {int(root): i for i, root in enumerate(a['root_ids'])}
    for row in manifest['rows']:
        global_sources = [global_index[root] for root in row['pool_root_ids'] if root in global_index]
        row['source_roots_missing_from_cache'] = [root for root in row['pool_root_ids'] if root not in global_index]
        row['minimum_edges_raw_cache'] = minimum_lengths(distance_raw, global_sources)
        row['minimum_edges_unsigned_tier'] = minimum_lengths(distance_tier, row['local_indices'])
        row['minimum_edges_modeled_tier'] = minimum_lengths(distance_modeled, row['local_indices'])
        row['modeled_in_degree_by_root'] = np.diff((exc + inh).tocsr().indptr)[row['local_indices']].tolist()
        for key in ('pool_root_ids', 'tier_root_ids', 'missing_tier_roots', 'source_roots_missing_from_cache'):
            if key in row:
                row[key] = [str(root) for root in row[key]]
    for row in manifest['shared_pools']:
        row['pool_root_ids'] = [str(root) for root in row['pool_root_ids']]

    candidates = groups['crossing_descending']
    group_dn = int(candidates[np.argmax(out_degree[candidates])])
    probes = dict(quiet=[], **{leg: group.tolist() for leg, group in groups.items() if len(leg) == 2},
                  single_descending=[group_dn])
    graph = dict(root_ids=root_ids)
    for label, mat in (('exc', exc), ('inh', inh)):
        graph.update({f'{label}_{key}': getattr(mat, key) for key in ('data', 'indices', 'indptr')})
    np.savez_compressed(HERE / 'banc_tier.npz', **graph)
    (HERE / 'NEURAL_INPUT.json').write_text(json.dumps(dict(dt_ms=.5, fingerprint=FINGERPRINT,
        duration_ms=120, onset_ms=10, offset_ms=40, probe_current_nA=.08, probes=probes), indent=2) + '\n')
    classes = peripheral_coverage(a, gi, out_degree)
    report = dict(cache_neurons=int(a['root_ids'].size), cache_pair_count=raw.nnz,
                  edge_rules=built['coverage']['edges'], tier_neurons=n, tier_raw_pairs=induced.nnz,
                  excitatory_pairs=exc.nnz, inhibitory_pairs=inh.nnz, modeled_pairs=modeled.nnz,
                  group_counts={label: dict(cache=len(raw_groups[label]), tier=len(group),
                      tier_with_modeled_outgoing=int(np.count_nonzero(out_degree[group])))
                      for label, group in groups.items()},
                  source_pool_root_count=len(annotations), source_pool_roots_missing_current_888=[
                      root for root, row in annotations.items() if row['banc_888_id'] != root],
                  single_descending_probe=dict(root_id=str(root_ids[group_dn]),
                      annotated_type=str(a['ann_primary_type'][gi[group_dn]]),
                      modeled_outgoing=int(out_degree[group_dn])),
                  peripheral_annotations=classes, motor_manifest=manifest)
    (out / 'WIRING.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
