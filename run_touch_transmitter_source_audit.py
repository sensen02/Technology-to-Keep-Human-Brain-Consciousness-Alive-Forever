"""Read-only exact-root transmitter provenance audit; no sign assignment or model build.
Run with /run/media/sensen/Data2/cell_wound_prototype/venv/bin/python.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import signal
import numpy as np
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
NT_MAP = {'ACH': 'acetylcholine', 'GABA': 'gaba', 'GLUT': 'glutamate',
          'DA': 'dopamine', 'OCT': 'octopamine', 'SER': 'serotonin'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def root_id(value):
    """Never route 18-digit IDs through float; reject malformed/non-integer inputs."""
    if value is None or (isinstance(value, str) and not value.strip()) or (
            isinstance(value, (float, np.floating)) and math.isnan(value)):
        return None
    if isinstance(value, (float, np.floating, bool)):
        raise ValueError('float/bool root IDs are unsafe')
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError('nondecimal root ID: ' + text)
    return str(int(text))


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (float, np.floating)) and math.isnan(value))


def clean(value):
    return None if missing(value) else value


def normalized_nt(value):
    if missing(value):
        return None
    return NT_MAP.get(value, str(value).lower())


def hist(values):
    return dict(sorted(Counter('<missing>' if missing(x) else str(x) for x in values).items()))


def stream_neurons(path, targets):
    rows = {r: [] for r in targets}
    all_ids = set()
    with gzip.open(path, 'rt', encoding='utf-8', errors='strict', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        assert 'Root ID' in headers and 'Verified NT type' in headers
        cols = [c for c in headers if 'nt' in c.lower().split() or 'neurotransmitter' in c.lower() or 'neuropeptide' in c.lower()]
        count = 0
        for lineno, row in enumerate(reader, 2):
            count += 1
            rid = root_id(row['Root ID'])
            all_ids.add(rid)
            if rid in rows:
                rows[rid].append({'csv_record_number_including_header': lineno,
                                  'values': {c: row[c] for c in cols}})
    return rows, all_ids, {'headers_verbatim': headers, 'transmitter_headers': cols,
                          'rows': count, 'unique_root_ids': len(all_ids)}


def feather_audit(path, targets, csv_ids):
    # Inspect schema without materializing whole table; pandas loads selected columns only.
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pandas as pd
    with pa.memory_map(str(path), 'r') as source:
        schema = ipc.open_file(source).schema
    ids = [c for c in schema.names if c == 'banc_888_id' or c == 'root_id' or c.startswith('root_') and c[5:].isdigit()]
    nts = [c for c in schema.names if 'neurotransmitter' in c or 'neuropeptide' in c]
    frame = pd.read_feather(path, columns=ids + nts, use_threads=False)
    summary = {'rows': len(frame), 'schema': {f.name: str(f.type) for f in schema},
               'transmitter_headers': nts, 'id_columns': ids, 'overlap_by_id_column': {}}
    selected = {r: {} for r in targets}
    for col in ids:
        canon = [root_id(x) for x in frame[col]]
        pool = set(canon) - {None}
        hits = Counter(x for x in canon if x in targets)
        summary['overlap_by_id_column'][col] = {
            'unique_ids': len(pool), 'csv_all_unique_overlap': len(pool & csv_ids),
            'exact450_unique_overlap': len(pool & targets),
            'exact450_missing_ids': sorted(targets - pool),
            'exact450_duplicate_matches': {r: n for r, n in hits.items() if n != 1}}
        for pos, rid in enumerate(canon):
            if rid in targets:
                record = {c: clean(frame.iloc[pos][c]) for c in ids + nts}
                record['feather_zero_based_row'] = pos
                selected[rid].setdefault(col, []).append(record)
    # Explicit join key, not silent fallback between versions.
    summary['comparison_join_key'] = 'banc_888_id'
    summary['exact450_histograms_on_comparison_join'] = {
        c: hist(row[c] for r in targets for row in selected[r].get('banc_888_id', [])) for c in nts}
    return selected, summary


def main(report):
    paths = [OUT/'touch_sign_exclusion.json', OUT/'touch_path_audit_spikes.npz',
             HERE/'data/flywire/banc_neurons.csv.gz', HERE/'data/flywire/banc_connectome.npz',
             HERE/'data/banc888/banc_888_meta.feather', HERE/'build_banc_cns.py',
             HERE/'flywire_download.py', HERE/'launch_banc.sh', Path(__file__).resolve()]
    before = {str(p): sha(p) for p in paths}
    previous = json.loads(paths[0].read_text())
    targets = {root_id(r) for r in previous['lf_bristle_root_ids']}
    assert len(targets) == 450
    with np.load(paths[1], allow_pickle=False) as data:
        replay = data['root_ids'][data['lf_bristle_indices']]
        assert set(map(root_id, replay)) == targets
    cached = {}
    with np.load(paths[3], allow_pickle=False) as data:
        roots = data['root_ids']
        labels = data['ann_nt_verified']
        for rid, value in zip(roots, labels):
            rid = root_id(rid)
            if rid in targets:
                assert rid not in cached
                cached[rid] = str(value)
    raw, csv_ids, csv_summary = stream_neurons(paths[2], targets)
    feather, feather_summary = feather_audit(paths[4], targets, csv_ids)
    per_root = []
    for rid in sorted(targets):
        values = raw[rid]
        record = {'root_id': rid, 'cached_ann_nt_verified': cached.get(rid),
                  'cache_present': rid in cached, 'raw_csv_matches': values,
                  'feather_matches_by_id_column': feather[rid]}
        record['raw_csv_verified_equals_cached'] = (len(values) == 1 and rid in cached and
            values[0]['values']['Verified NT type'] == cached[rid])
        matched = feather[rid].get('banc_888_id', [])
        if len(values) == len(matched) == 1:
            record['csv_feather_predicted_label_agree'] = normalized_nt(values[0]['values']['Predicted NT type']) == normalized_nt(matched[0]['neurotransmitter_predicted'])
            record['csv_feather_verified_agree_null_normalized'] = normalized_nt(values[0]['values']['Verified NT type']) == normalized_nt(matched[0]['neurotransmitter_verified'])
        per_root.append(record)
    csv_summary['exact450_histograms'] = {c: hist(row['values'][c] for rows in raw.values() for row in rows) for c in csv_summary['transmitter_headers']}
    csv_summary['confidence_by_predicted_label'] = {}
    for label in sorted({row['values']['Predicted NT type'] for rows in raw.values() for row in rows}):
        vals = [row['values']['Predicted NT confidence'] for rows in raw.values() for row in rows if row['values']['Predicted NT type'] == label]
        nums = [float(x) for x in vals if not missing(x)]
        csv_summary['confidence_by_predicted_label'][label or '<missing>'] = {
            'roots': len(vals), 'missing_confidence': len(vals)-len(nums),
            'quantiles_min_q25_median_q75_max': np.quantile(nums, [0, .25, .5, .75, 1]).tolist() if nums else None}
    consistency = {
        'exact450_cache_present': len(cached),
        'csv_missing_roots': [r for r in targets if not raw[r]],
        'csv_duplicate_roots': [r for r in targets if len(raw[r]) > 1],
        'csv_cached_verified_mismatches': [r['root_id'] for r in per_root if not r['raw_csv_verified_equals_cached']],
        'csv_feather_predicted_disagreements': [r['root_id'] for r in per_root if r.get('csv_feather_predicted_label_agree') is False],
        'csv_feather_verified_disagreements': [r['root_id'] for r in per_root if r.get('csv_feather_verified_agree_null_normalized') is False]}
    after = {str(p): sha(p) for p in paths}
    assert before == after, 'Source changed during audit'
    assert not consistency['csv_missing_roots'] and not consistency['csv_duplicate_roots']
    assert not consistency['csv_cached_verified_mismatches']
    memory = dict(report())
    if not math.isfinite(memory['previous_soft_limit_mb']):
        memory['previous_soft_limit_mb'] = 'unlimited'
    payload = {'status': 'MEASURED_READ_ONLY_SOURCE_AUDIT_NOT_SIGN_REPAIR',
        'source_sha256_before': before, 'source_sha256_after': after, 'all_sources_unchanged': True,
        'roots_count': len(targets), 'csv': csv_summary, 'feather': feather_summary,
        'consistency': consistency, 'per_root': per_root,
        'ingestion_evidence': {'file': str(paths[5]), 'lines': '91-101',
            'mapping': {'Verified NT type': 'ann_nt_verified'},
            'verdict': 'No verified-column ingestion mismatch for these exact450 roots; alternative predicted columns are separate evidence, not verified replacements.'},
        'provider_provenance': {'csv_neurons_url_without_auth': 'https://codex.flywire.ai/api/download_resource?data_product=neurons&dataset=banc',
            'local_evidence': [str(paths[6]) + ':43-49', str(paths[7]) + ':5'],
            'csv_materialization': 'Not explicitly pinned in downloader/build; measured cross-version ID overlaps only, not proof of snapshot date.',
            'feather_version': 'Filename claims banc888; each root/version column audited independently; local file hash identifies exact bytes.'},
        'limits': ['Confidence values preserved verbatim; summary quantiles are numeric descriptions, not calibrated probability or universal eligibility threshold.',
                   'Empty verified annotation is missing evidence, not absence of biological transmitter.',
                   'Prediction/verification disagreement must not be resolved from sensory labels.',
                   'No download, network/model construction, source mutation, sign fill, or build/sign code change.'],
        'next_step': 'Retain strict verified-only exclusion. Parent may separately approve a prediction-provenance/eligibility investigation; resolve version-specific disagreements and score semantics before any exploratory signed mode.',
        'memory_guard': memory}
    dest = OUT/'touch_transmitter_source_audit.json'
    dest.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(dest), 'csv_hist': csv_summary['exact450_histograms'],
                      'feather_hist': feather_summary['exact450_histograms_on_comparison_join'],
                      'overlap': {c: v['exact450_unique_overlap'] for c, v in feather_summary['overlap_by_id_column'].items()},
                      'consistency': consistency, 'memory': payload['memory_guard']}, indent=2))


if __name__ == '__main__':
    def deadline(*_):
        raise TimeoutError('180-second audit bound exceeded')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    with mg.guarded_run('touch_transmitter_source_audit', need_mb=512, cap_mb=2048) as report:
        main(report)
    signal.alarm(0)
