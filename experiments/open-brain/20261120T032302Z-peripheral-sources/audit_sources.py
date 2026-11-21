"""Audit published peripheral IDs and same-specimen BANC candidates without editing the runtime graph."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UPSTREAM = HERE / 'upstream'
OLD = ROOT / 'experiments/open-brain/20261120T031017Z-minimal-stance'
META = ROOT / 'data/banc888/banc_888_meta.feather'
EDGES = ROOT / 'experiments/visual-orientation/20261116T023521Z/banc_888_edgelist_simple_v2.feather'
EON = ROOT / 'experiments/open-brain/20261117T045901Z-foundation/upstream/data'
LEG_LABELS = ('front_leg', 'middle_leg', 'hind_leg')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(name):
    with (UPSTREAM / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def candidate_group(row):
    body = row['body_part_sensory'] or ''
    organ = row['cell_class'] or ''
    subtype = row['cell_sub_class'] or ''
    leg = next((name for name in LEG_LABELS if name in body or name in subtype), '')
    if organ == 'crop_motor_neuron' and row['flow'] == 'efferent':
        return 'crop_motor', leg
    if row['flow'] != 'afferent':
        return None
    if leg and organ == 'hair_plate_neuron':
        return 'leg_hair_plate', leg
    if leg and organ == 'campaniform_sensillum_neuron':
        return 'leg_campaniform', leg
    if leg and organ == 'bristle_neuron':
        return 'leg_tactile_bristle', leg
    if leg and body in LEG_LABELS:
        for kind in ('claw', 'hook', 'club'):
            if subtype == f'{leg}_{kind}_chordotonal_organ_neuron':
                return f'leg_FeCO_{kind}', leg
    if body in ('abdomen', 'abdominal_wall', 'thoracic_abdominal', 'thorax', 'neck'):
        if organ == 'bristle_neuron':
            return 'trunk_tactile_bristle', leg
        if 'multidendritic' in organ or 'multidendritic' in subtype:
            return 'trunk_multidendritic', leg
    if body == 'anterior_digestive_tract' or 'pharynx' in body:
        return 'digestive_afferent', leg
    if organ == 'campaniform_sensillum_neuron' and body.startswith(('wing', 'haltere')):
        return 'wing_or_haltere_campaniform', leg
    return None


def inspect_banc():
    columns = ['banc_888_id', 'root_626', 'flow', 'side', 'nerve', 'body_part_sensory',
               'body_part_effector', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'fafb_match',
               'peripheral_target_type', 'neurotransmitter_verified', 'neurotransmitter_predicted']
    meta = feather.read_table(META, columns=columns).to_pylist()
    current_roots = {row['banc_888_id'] for row in meta}
    assert None not in current_roots and len(current_roots) == len(meta)
    with np.load(OLD / 'banc_tier.npz', allow_pickle=False) as data:
        original_ids = [str(root) for root in data['root_ids']]
        old_roots = set(original_ids)
        modeled_pre = np.union1d(data['exc_indices'], data['inh_indices'])
        old_with_modeled_outgoing = {original_ids[i] for i in modeled_pre}
    rows = []
    for row in meta:
        identity = candidate_group(row)
        if identity:
            group, leg = identity
            rows.append(dict(group=group, leg=leg, **row))
    edge_table = feather.read_table(EDGES, columns=['pre', 'post', 'count'])
    edge_rows = edge_table.num_rows
    candidate_roots = pa.array([r['banc_888_id'] for r in rows], type=pa.string())
    results = {}
    for field in ('pre', 'post'):
        selected = edge_table.filter(pc.is_in(edge_table[field], value_set=candidate_roots))
        by_id = selected.group_by(field, use_threads=False).aggregate([('count', 'count'), ('count', 'sum')])
        results[field] = {r[field]: (r['count_count'], r['count_sum']) for r in by_id.to_pylist()}
    grouped = defaultdict(list)
    for row in rows:
        root = row['banc_888_id']
        row.update(exact_root_in_old_tier=root in old_roots,
                   old_tier_match_via_root_626=row['root_626'] in old_roots,
                   exact_legacy_root_has_modeled_outgoing=root in old_with_modeled_outgoing,
                   outgoing_pairs=results['pre'].get(root, (0, 0))[0],
                   outgoing_synapses=results['pre'].get(root, (0, 0))[1],
                   incoming_pairs=results['post'].get(root, (0, 0))[0],
                   incoming_synapses=results['post'].get(root, (0, 0))[1])
        grouped[row['group']].append(row)
    summary = {}
    for group, records in sorted(grouped.items()):
        summary[group] = dict(roots=len(records), with_outgoing=sum(r['outgoing_pairs'] > 0 for r in records),
                              with_incoming=sum(r['incoming_pairs'] > 0 for r in records),
                              exact_root_in_legacy_tier=sum(r['exact_root_in_old_tier'] for r in records),
                              matched_via_626_in_legacy_tier=sum(r['old_tier_match_via_root_626'] for r in records),
                              exact_legacy_root_with_modeled_outgoing=sum(
                                  r['exact_legacy_root_has_modeled_outgoing'] for r in records),
                              published_no_exact_legacy_root_with_edges=sum(
                                  not r['exact_root_in_old_tier'] and r['outgoing_pairs'] > 0 for r in records),
                              by_leg_and_side={'/'.join(k): v for k, v in sorted(Counter(
                                  (r['leg'], r['side'] or 'unknown') for r in records).items())})
    stance_groups = {'leg_campaniform', 'leg_hair_plate'}
    stance_by_id = {r['banc_888_id']: r['group'] for r in rows if r['group'] in stance_groups}
    selected = edge_table.filter(pc.is_in(edge_table['pre'], value_set=pa.array(list(stance_by_id))))
    expected_pairs = sum(r['outgoing_pairs'] for r in rows if r['group'] in stance_groups)
    assert selected.num_rows == expected_pairs
    pre, post = selected['pre'].to_pylist(), selected['post'].to_pylist()
    target_annotations = {r['banc_888_id']: r for r in meta}
    stance = dict(source_roots=len(stance_by_id), rows=selected.num_rows,
        edges_to_identified_motor=sum(target_annotations[target]['super_class'] == 'motor'
                                      for target in post),
        unique_target_roots=len(set(post)), targets_outside_metadata=len(set(post) - current_roots),
        scope='Source outgoing directed weights only, no runtime receptor dynamics or sign.')
    candidate_edges = selected.append_column('presynaptic_group', pa.array([stance_by_id[root] for root in pre]))
    feather.write_feather(candidate_edges, HERE / 'STANCE_CANDIDATE_EDGES.feather')
    return rows, meta, edge_rows, summary, stance


def inspect_external(meta, candidate_rows):
    feco = [r for r in read_csv('lee_feco_cells.csv') if r['valid'] == 't']
    feco_ids = {r['pt_root_id'] for r in feco}
    assert len(feco) == len(feco_ids)
    leg_counts = dict(sorted(Counter(r['classification_system'] for r in feco).items()))
    assert set(leg_counts) == {'T1L', 'T1R', 'T2L', 'T2R', 'T3L', 'T3R'}
    feco_edges = [r for r in read_csv('lee_feco_edges.csv') if r['valid'] == 't']
    feco_next = {r['post_pt_root_id'] for r in feco_edges if r['pre_pt_root_id'] in feco_ids}
    gut = read_csv('giakoumas_aPhN_PhN.csv')
    gut_ids = {r['root_id'] for r in gut}
    assert len(gut) == len(gut_ids) and {r['flow'] for r in gut} == {'afferent'}
    with (EON / '2025_Completeness_783.csv').open(newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        eon_ids = {row[0] for row in reader}
    gut_banc = defaultdict(list)
    for r in meta:
        if r['fafb_match'] in gut_ids:
            gut_banc[r['fafb_match']].append(r['banc_888_id'])
    selected = {r['banc_888_id'] for r in candidate_rows if r['group'] == 'digestive_afferent'}
    published = read_csv('pugliese_banc829_cells.csv')
    roots = {r['banc_888_id'] for r in meta}
    motor = [r for r in published if r['class'] == 'motor neuron' and r['body_part_effector'] == 'front_leg']
    feco_829_matches = [r for r in published if r['fanc_1116_match_id'] in feco_ids]
    previous = json.loads((OLD / 'results/MUSCLE_CHAIN.json').read_text())
    front_gaps = [r for r in previous if r['status'] == 'unresolved' and r['leg'].endswith('F')]
    front_gap_matches = {r['actuator']: sorted(x['pt_root_id'] for x in motor
        if x['side'] == r['side'] and x['peripheral_target_type'] == r['muscle_target']) for r in front_gaps}
    return dict(
        lee_FANC_FeCO=dict(roots=len(feco_ids), labels=dict(Counter(r['cell_type'] for r in feco)),
                           roots_by_leg=leg_counts,
                           valid_downloaded_synapse_rows=len(feco_edges),
                           matched_pre_roots=len({r['pre_pt_root_id'] for r in feco_edges} & feco_ids),
                           matched_post_roots=len(feco_next),
                           note='FANC identities belong to a different specimen than BANC.'),
        giakoumas_FlyWire_v783=dict(roots=len(gut_ids), indexed_in_frozen_Eon=len(gut_ids & eon_ids),
            with_BANC_FAFB_match_field=len(gut_banc), unique_BANC_match_field=sum(len(ids) == 1 for ids in gut_banc.values()),
            with_BANC_digestive_candidate=sum(bool(set(ids) & selected) for ids in gut_banc.values()),
            no_version_or_specimen_inference=True,
            missing_from_frozen_Eon=sorted(gut_ids - eon_ids),
            BANC_matching_field={root: sorted(ids) for root, ids in sorted(gut_banc.items())}),
        pugliese_BANC_v829=dict(rows=len(published), exact_ids_in_local_v888=sum(r['pt_root_id'] in roots for r in published),
            front_leg_motor_rows=len(motor), front_leg_motor_exact_ids_in_888=sum(r['pt_root_id'] in roots for r in motor),
            exact_FANC_root_mentions_in_829_table=len(feco_829_matches),
            unresolved_front_actuator_exact_muscle_matches=front_gap_matches,
            note='Exact overlaps only; no inference for changed IDs or approved cross-specimen edges.'))


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def main():
    source_files = json.loads((UPSTREAM / 'SOURCES.json').read_text())
    for name, record in source_files.items():
        assert sha(UPSTREAM / name) == record['sha256']
    local = [META, EDGES, OLD / 'banc_tier.npz', OLD / 'results/MUSCLE_CHAIN.json',
             EON / '2025_Completeness_783.csv']
    hashes = {str(path): sha(path) for path in local}
    candidates, meta, edges, groups, stance = inspect_banc()
    external = inspect_external(meta, candidates)
    assert hashes == {str(path): sha(path) for path in local}
    save('BANC888_CANDIDATES.json', candidates)
    save('SOURCE_AUDIT.json', dict(metadata_rows=len(meta), edge_rows=edges,
        selection='Published metadata annotation filters only; central projection/synapse counts, no peripheral skeleton inferred.',
        local_inputs=hashes, inspected_external_tables=external, groups=groups, stance_source_edges=stance,
        runtime_updates=0, model_training_updates=0))
    print(f'{len(candidates)} candidate annotations; scanned {edges:,} BANC edge rows.')


if __name__ == '__main__':
    main()
