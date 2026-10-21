"""P3 read-only version-column projection, NOT an authoritative revised model.
Exact Arrow strings/integers only; row association is not split/merge validation.
No simulation imports, root replacement, pool publishing, or network access.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
VERSIONS = ['banc_888_id', 'root_626', 'root_850', 'root_888', 'root_890']
ANNOTATIONS = ['side', 'flow', 'super_class', 'cell_type', 'body_part_effector', 'peripheral_target_type']
LEG = {'F': 'front_leg', 'M': 'middle_leg', 'H': 'hind_leg'}
LABEL = 'READONLY VERSION-COLUMN PROJECTION — NOT AUTHORITATIVE REVISED MODEL'


def identity(value):
    """Never round a large ID through float; invalid/missing IDs stay missing."""
    if value is None or value == '':
        return None
    if isinstance(value, (float, np.floating, bool)):
        raise ValueError('floating/boolean root ID forbidden')
    s = str(value)
    if not s.isascii() or not s.isdecimal():
        raise ValueError('invalid root ID: ' + s)
    if int(s) == 0:  # Source table uses string zero as a missing-root sentinel.
        return None
    return str(int(s))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def crosswalk(records, declared):
    """Retain all duplicate rows; never select an arbitrary first ambiguous row."""
    indexes = {c: defaultdict(list) for c in ['root_id'] + VERSIONS}
    for i, row in enumerate(records):
        for c in indexes:
            v = identity(row.get(c))
            if v is not None:
                indexes[c][v].append(i)
    result = []
    for r in sorted({identity(x) for x in declared}, key=int):
        positions = indexes['root_id'].get(r, [])
        candidates = []
        for i in positions:
            row = records[i]
            values = {c: identity(row.get(c)) for c in VERSIONS}
            candidates.append({'feather_row_0based': i, 'root_id': r, 'version_ids': values,
                               'annotation': {c: row.get(c) for c in ANNOTATIONS},
                               'version_id_global_row_counts': {c: len(indexes[c].get(v, [])) if v else 0 for c, v in values.items()},
                               'version_id_global_other_root_ids': {c: sorted({identity(records[j].get('root_id')) for j in indexes[c].get(v, []) if identity(records[j].get('root_id')) not in (None, r)}, key=int) if v else [] for c, v in values.items()}})
        status = 'missing_row' if not positions else 'multiple_rows' if len(positions) > 1 else 'missing_version_id' if any(v is None for v in candidates[0]['version_ids'].values()) else 'same_value' if all(v == r for v in candidates[0]['version_ids'].values()) else 'different_value_association'
        result.append({'declared_root_id': r, 'association_status': status, 'row_count': len(positions), 'candidates': candidates,
                       'resolution': 'unresolved' if status != 'same_value' else 'same_exact_string_not_biological_validation',
                       'root_equivalence_established': False, 'split_merge_validated': False})
    collisions = {}
    for c in VERSIONS:
        reverse = defaultdict(set)
        for item in result:
            for row in item['candidates']:
                v = row['version_ids'][c]
                if v is not None:
                    reverse[v].add(item['declared_root_id'])
        collisions[c] = {v: sorted(rr, key=int) for v, rr in reverse.items() if len(rr) > 1}
    return result, collisions


def project(pool, lookup, column):
    original = {identity(r) for r in pool}
    projected = set()
    unresolved = []
    for r in sorted(original, key=int):
        item = lookup[r]
        if item['row_count'] != 1 or item['candidates'][0]['version_ids'][column] is None:
            unresolved.append(r)
        else:
            projected.add(item['candidates'][0]['version_ids'][column])
    return {'original_ids': sorted(original, key=int), 'projected_ids': sorted(projected, key=int),
            'unprojectable_declared_ids': unresolved, 'removed_ids': sorted(original - projected, key=int),
            'added_ids': sorted(projected - original, key=int), 'original_unique_count': len(original),
            'projected_unique_count': len(projected), 'cardinality_delta': len(projected) - len(original),
            'complete_projection': not unresolved}


def scan_csv(path, key, ids, max_rows, deadline):
    matches = {r: [] for r in ids}
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', newline='') as f:
        for n, row in enumerate(csv.DictReader(f), 1):
            if n > max_rows or time.monotonic() > deadline:
                raise RuntimeError('bounded annotation scan exceeded limits; no complete output published')
            if row[key] in matches:
                matches[row[key]].append({'line_1based': n + 1, 'row': row})
    return {'rows_scanned': n, 'matches': matches}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p1', type=Path, default=ROOT / 'outputs/experiment_platform/20261013_233237_7be93a1d/result.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-annotation-rows', type=int, default=250000)
    parser.add_argument('--scan-seconds', type=float, default=90)
    args = parser.parse_args(argv)
    import pyarrow.feather as feather
    paths = {'p1': args.p1, 'declared_pool': ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',
             'feather': ROOT / 'data/banc888/banc_888_meta.feather', 'csv': ROOT / 'data/flywire/banc_neurons.csv.gz',
             'supplement': ROOT / 'data/banc_dataverse/supplemental_data_2.txt', 'cache': ROOT / 'data/flywire/banc_connectome.npz',
             'raw_connections': ROOT / 'data/flywire/banc_connections_princeton.csv.gz',
             'npy_association': ROOT / 'data/banc888/b888_to_root.npy', 'generator': ROOT / 'run_neural_rebuild_11_mn_pools.py',
             'tool': Path(__file__), 'test': ROOT / 'test_p3_motor_version_crosswalk.py'}
    before = {k: {'path': str(p.resolve()), 'sha256': sha(p), 'bytes': p.stat().st_size} for k, p in paths.items()}
    p1 = json.loads(args.p1.read_text())
    declared_file = json.loads(paths['declared_pool'].read_text())
    rows = [r for r in p1['motor_manifest']['rows'] if 'leg' in r]
    sources = {r['actuator']: r for r in declared_file['measured']['resolved']}
    assert len(rows) == len(sources) == 74
    for r in rows:
        assert r['source_annotation'] == sources[r['actuator']]
        assert set(r['pool_root_ids']) == set(sources[r['actuator']]['pool_root_ids'])
    for k in ['declared_pool', 'cache']:
        assert p1['input_hashes'][str(paths[k])] == before[k]['sha256']
    declared = {identity(x) for r in rows for x in r['pool_root_ids']}
    tier = {identity(x) for x in p1['tier_root_ids']}
    current = declared & tier
    assert len(declared) == 208 and len(current) == 202 and len(tier) == 12000
    table = feather.read_table(paths['feather'], columns=['root_id'] + VERSIONS + ANNOTATIONS)
    records = table.to_pylist()  # Arrow string identity: no pandas JSON float conversion.
    items, collisions = crosswalk(records, declared)
    lookup = {r['declared_root_id']: r for r in items}
    all_ids = declared | {v for item in items for row in item['candidates'] for v in row['version_ids'].values() if v}
    deadline = time.monotonic() + args.scan_seconds
    csvs = {'original_csv': scan_csv(paths['csv'], 'Root ID', all_ids, args.max_annotation_rows, deadline),
            'official_local_supplement': scan_csv(paths['supplement'], 'root_id', all_ids, args.max_annotation_rows, deadline)}
    with np.load(paths['cache'], allow_pickle=False) as z:
        cache_array = z['root_ids']; cache = {identity(x) for x in cache_array}
        pre, post, syn = z['pre'], z['post'], z['syn']
        verified = z['ann_nt_verified']
    cache_index = {identity(r): i for i, r in enumerate(cache_array)}
    tier_mask = np.isin(cache_array, np.asarray(p1['tier_root_ids'], dtype=np.int64))
    incoming = (syn > 0) & tier_mask[pre]
    strict_incoming = incoming & np.isin(verified[pre], ['acetylcholine', 'gaba', 'glutamate'])
    incoming_counts = np.bincount(post[incoming], minlength=len(cache_array))
    strict_counts = np.bincount(post[strict_incoming], minlength=len(cache_array))
    def presence(r):
        g = cache_index.get(r)
        return {'in_original_declared_pool': r in declared, 'in_current_202_motor_roots': r in current,
                'in_p1_tier': r in tier, 'in_cached_node_universe': r in cache,
                'original_csv_row_count': len(csvs['original_csv']['matches'][r]),
                'supplement_row_count': len(csvs['official_local_supplement']['matches'][r]),
                'cached_positive_incoming_rows_from_p1_tier': int(incoming_counts[g]) if g is not None else 0,
                'cached_verified_source_incoming_rows_from_p1_tier': int(strict_counts[g]) if g is not None else 0}
    association = np.load(paths['npy_association'], allow_pickle=False)
    npy_pairs = Counter((identity(a), identity(b)) for a, b in association)
    annotations_issues = []
    actuator_rows = []
    for row in rows:
        rids = {identity(r) for r in row['pool_root_ids']}
        expectations = {'side': row['side'], 'peripheral_target_type': row['muscle_target'],
                        'body_part_effector': LEG[row['leg'][1]], 'flow': 'efferent'}
        checks = []
        for r in sorted(rids, key=int):
            item = lookup[r]
            for candidate in item['candidates']:
                mismatches = {c: {'expected': v, 'actual': candidate['annotation'][c]} for c, v in expectations.items() if candidate['annotation'][c] != v}
                if candidate['annotation']['super_class'] != 'motor':
                    mismatches['super_class'] = {'expected': 'motor', 'actual': candidate['annotation']['super_class']}
                check = {'declared_root_id': r, 'feather_row_0based': candidate['feather_row_0based'], 'consistent': not mismatches, 'mismatches': mismatches}
                checks.append(check)
                if mismatches:
                    annotations_issues.append({'actuator': row['actuator'], **check})
        projections = {}
        for column in VERSIONS:
            projected = project(rids, lookup, column)
            ids = set(projected['projected_ids'])
            projected.update(in_p1_tier_count=len(ids & tier), in_cache_count=len(ids & cache),
                             new_ids_overlapping_current_202=sorted(set(projected['added_ids']) & current, key=int),
                             status_if_only_version_column_projected='complete' if projected['complete_projection'] and ids <= tier else 'partial',
                             original_p1_status=row['status'])
            projections[column] = projected
        actuator_rows.append({'actuator': row['actuator'], 'leg': row['leg'], 'side': row['side'], 'muscle_target': row['muscle_target'],
                              'original_p1_status': row['status'], 'annotation_checks': checks, 'readonly_projections': projections})
    for item in items:
        item['declared_presence'] = presence(item['declared_root_id'])
        item['actuators'] = [r['actuator'] for r in rows if int(item['declared_root_id']) in r['pool_root_ids']]
        for candidate in item['candidates']:
            candidate['version_presence'] = {c: presence(v) if v else None for c, v in candidate['version_ids'].items()}
            candidate['npy_banc888_to_declared_pair_rows'] = npy_pairs[(candidate['version_ids']['banc_888_id'], item['declared_root_id'])]
    summary_columns = {}
    for c in VERSIONS:
        projection = project(declared, lookup, c)
        ids = set(projection['projected_ids'])
        summary_columns[c] = {**projection, 'in_cache_count': len(ids & cache), 'in_p1_tier_count': len(ids & tier),
                              'added_ids_overlapping_current_202': sorted(set(projection['added_ids']) & current, key=int),
                              'actuator_status_counts': dict(Counter(r['readonly_projections'][c]['status_if_only_version_column_projected'] for r in actuator_rows)),
                              'actuators_with_changed_ids': [r['actuator'] for r in actuator_rows if r['readonly_projections'][c]['added_ids'] or r['readonly_projections'][c]['removed_ids']]}
    # Hash-only inheritance of prior original connection scan; this CLI does NOT rescan 4M edges.
    for k, path in paths.items():
        assert sha(path) == before[k]['sha256'], 'input mutated during audit: ' + str(path)
    report = {'label': LABEL, 'manual_research_not_product': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'P3/S3 exact same-row association only; current P1 unchanged; no replacement/branch pools/dynamics/services/network',
              'limitations': ['root_id generation revision and original CSV export materialization/date unknown', 'association NOT split/merge or biological identity validation',
                             'cached incoming row counts are structural adjacency, NOT path reachability/dynamic efficacy', 'current P1 tier fixed; no counterfactual tier expansion',
                             'original connectivity CSV hash pinned, not rescanned'],
              'input_files': before, 'feather_rows': table.num_rows, 'identity_arrow_types': {c: str(table.schema.field(c).type) for c in ['root_id'] + VERSIONS},
              'summary': {'declared_unique_roots': len(declared), 'actual_mapped_actuator_pools': len(rows), 'current_motor_roots_in_p1_tier': len(current),
                          'association_status_counts': dict(Counter(i['association_status'] for i in items)), 'original_p1_actuator_status_counts': dict(Counter(r['status'] for r in rows)),
                          'annotation_issue_count': len(annotations_issues), 'version_projections': summary_columns},
              'crosswalk': items, 'different_declared_root_version_collisions': collisions, 'annotation_issues': annotations_issues,
              'actuator_pools': actuator_rows, 'exact_csv_source_evidence': csvs,
              'verification': {'p1_source_annotations_equal_original_declared_pool': True, 'p1_pool_and_cache_hashes_match': True, 'inputs_unchanged_after_readonly_audit': True},
              'next_action': 'Request immutable feather URL/revision/checksum and exact root_id/version-column generation definitions; original neuron/connection CSV export manifest (datastack/materialization/UTC); versioned supervoxel/nucleus lineage for two unresolved pairs including split/merge operations and cardinality. Do not replace roots.'}
    out = args.output or ROOT / 'outputs/p3_motor_version_crosswalk' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    text = '# P3 全池版本身份只读审计\n\n**手动研究/测试原型，不是成品。' + LABEL + '**\n\n'
    compact_summary = {k: v for k, v in report['summary'].items() if k != 'version_projections'}
    compact_summary['version_projections'] = {c: {k: v for k, v in s.items() if k not in ['original_ids', 'projected_ids']} for c, s in summary_columns.items()}
    text += '同一feather行关联≠split/merge验证；不换根、不发布派生池，当前P1不变。\n\n```json\n' + json.dumps(compact_summary, indent=2, ensure_ascii=False) + '\n```\n\n'
    text += '纠正前轮tier存在性：P1 tier_root_ids存储为JSON字符串；前轮以int去查字符串列表导致假阴性。两同行版本ID均在冻结12000根tier中，仍未声明为当前202运动探针。版本列投影是假设结构覆盖204，不追写当前P1。18个其他根只在root_626列不同，banc888/850/888/890均同声明值。\n\n'
    text += '## 两个未解决关联及全池差异\n\n'
    for i in items:
        if i['association_status'] != 'same_value':
            text += '- `' + i['declared_root_id'] + '` → `' + str(i['candidates'][0]['version_ids'] if i['candidates'] else None) + '`，resolution=unresolved；同行版本ID的存在性见result.json。\n'
    text += '\n结构覆盖是固定当前tier上的只读版本列投影，不是权威修订模型。缓存传入边行数非可达路径，也非活动。已哈希原连接CSV但未重扫。\n\n下一最小动作：' + report['next_action'] + '\n\n## 固定来源\n\n'
    text += '\n'.join('- `' + v['path'] + '` SHA256 `' + v['sha256'] + '`' for v in before.values()) + '\n'
    (out / 'report.zh-CN.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    names = ['original'] + VERSIONS
    ax.bar(np.arange(6) - .2, [len(declared)] + [summary_columns[c]['projected_unique_count'] for c in VERSIONS], .4, label='unique declared / projected IDs')
    ax.bar(np.arange(6) + .2, [len(current)] + [summary_columns[c]['in_p1_tier_count'] for c in VERSIONS], .4, label='in unchanged P1 tier')
    ax.set_xticks(np.arange(6), names); ax.set_ylim(0, 240); ax.set_ylabel('Unique roots (shared pools deduplicated)')
    ax.set_title('READONLY VERSION-COLUMN PROJECTION\nNot an authoritative revised model; two associations unresolved')
    ax.legend(loc='lower right')
    for j, (unique, retained) in enumerate(zip([len(declared)] + [summary_columns[c]['projected_unique_count'] for c in VERSIONS], [len(current)] + [summary_columns[c]['in_p1_tier_count'] for c in VERSIONS])):
        ax.text(j - .2, unique + 3, str(unique), ha='center'); ax.text(j + .2, retained - 12, str(retained), ha='center')
    fig.tight_layout(); fig.savefig(out / 'projection.png', dpi=160); plt.close(fig)
    desktop = Path('/home/sensen/Desktop/engine修复验收') / ('P3-motor-version-crosswalk-' + out.name)
    shutil.copytree(out, desktop)
    for p in [Path(__file__), paths['test']]:
        shutil.copy2(p, desktop / p.name)
    print(json.dumps({'status': 'PASS', 'output': str(out), 'desktop': str(desktop), 'summary': report['summary']}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
