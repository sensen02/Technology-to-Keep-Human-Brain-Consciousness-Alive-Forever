"""S4 read-only, receipt-locked calcium adaptation. No fitting or production changes.

CLI writes a NEW directory beneath outputs/s4_real_calcium_adaptation. Each strict
contract contains one primary group plus the first opposite-split anchor group.
Anchors are packaging duplicates, never additional independent observations.
Arrow and NumPy are imported only when needed, not at module import.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote
import uuid
from datetime import datetime, timezone

from tools_s4_calcium_paired_contract import SCHEMA, PROCESS_KEYS, SYNC_KEYS, validate

PROJECT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT / 'outputs/s4_real_calcium_adaptation'
DEFAULT_INPUT = Path('/home/sensen/Desktop/engine修复验收/S4-calcium-input')
PROTOCOL = PROJECT / 'S4_REAL_CALCIUM_ADAPTATION_PROTOCOL.zh-CN.md'
SOURCES = {
    'claw_magnet_Mamiya2018.parquet': {
        'sha256': '7fc8a108258529d52815dc6e28b84958afffa073362ae41d75f551f2d83d01da',
        'rows': 26990, 'groups': 60, 'driver': 'R73D10'},
    'hook_flexion_01_magnet_Mamiya2018.parquet': {
        'sha256': 'ea61eee2d26a5e6fddb7aebaee06e13e11c9b87427d0f70b350760ab0ae545be',
        'rows': 25102, 'groups': 56, 'driver': 'R21D12'},
}
COLUMNS = {'driver', 'roi', 'animal_id', 'trial', 'frame', 'time',
           'stimulus_type', 'analyze', 'calcium', 'L1C_flex', 'L1C_flex_vel'}
NUMERIC = {'frame', 'time', 'analyze', 'calcium', 'L1C_flex', 'L1C_flex_vel'}
POLICY = 'per-file sorted highest two animal IDs holdout; others train; no validation'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def animal_integer(value):
    """Accept ordinary Arrow doubles holding exact small positive integer IDs, not bool."""
    if type(value) not in (int, float):
        raise ValueError('animal_id must be a finite ordinary number, not bool')
    if not 0 < value <= 2**31 - 1 or not math.isfinite(value) or int(value) != value:
        raise ValueError('animal_id must be an exact small positive integral value')
    return int(value)


def exact_text(value, field):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512 or value.lower() == 'unknown':
        raise ValueError(f'{field}: exact bounded known identity required')
    return value


def namespaced(name, kind, value):
    return name + '/' + kind + '/' + quote(str(value), safe='')


def group_rows(name, rows):
    """Validate ALL rows, preserve within-group source order, sort group identities."""
    if name not in SOURCES:
        raise ValueError('unknown source file')
    groups = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != COLUMNS:
            raise ValueError(f'{name} row {index}: unknown/missing columns')
        animal = animal_integer(row['animal_id'])
        for field in ('trial', 'roi', 'driver', 'stimulus_type'):
            exact_text(row[field], field)
        if row['driver'] != SOURCES[name]['driver']:
            raise ValueError('unexpected source driver')
        for field in NUMERIC:
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or (type(value) is int and abs(value) > 2**53):
                raise ValueError(f'{field}: nonfinite/non-numeric or unsafe value')
        if row['analyze'] != 1:
            raise ValueError('analyze != 1: no silent exclusion permitted')
        key = (animal, row['trial'], row['roi'])
        groups.setdefault(key, []).append((index, copy.deepcopy(row)))
    animals = sorted({key[0] for key in groups})
    if len(animals) < 3:
        raise ValueError('at least three animals needed for train and highest-two holdout')
    holdout = set(animals[-2:])
    result = []
    for (animal, trial, roi), indexed in sorted(groups.items()):
        sample = [row for _, row in indexed]
        times = [row['time'] for row in sample]
        if len(times) < 2 or not all(b > a for a, b in zip(times, times[1:])):
            raise ValueError('each group needs >=2 strictly increasing source-order times')
        if len({row['stimulus_type'] for row in sample}) != 1:
            raise ValueError('stimulus_type changes within source group')
        subject = namespaced(name, 'animal', animal)
        recording = subject + '/recording/' + quote(trial, safe='')
        identity = recording + '/roi/' + quote(roi, safe='')
        result.append(dict(group_id=identity, source_file=name, animal_id=animal,
                           trial=trial, roi=roi, subject_id=subject,
                           raw_recording_id=recording, split='holdout' if animal in holdout else 'train',
                           source_row_indices=[i for i, _ in indexed], rows=sample))
    return result


def to_trial(group):
    import numpy as np
    rows = group['rows']
    times = [row['time'] for row in rows]
    return dict(subject_id=group['subject_id'],
                trial_id=namespaced(group['source_file'], 'trial', group['trial']),
                roi_id=namespaced(group['source_file'], 'roi', group['roi']),
                raw_recording_id=group['raw_recording_id'], split=group['split'],
                identity_level='roi_population', roi_identity='unknown',
                response_origin='measured', leg='unknown', joint='unknown',
                angle_convention='source L1C_flex absolute degrees converted with np.deg2rad; sign/side unknown; no zero shift',
                sync={key: 'unknown' for key in sorted(SYNC_KEYS)},
                processing={key: 'unknown' for key in sorted(PROCESS_KEYS)},
                time_s=times, fluorescence_time_s=list(times),
                angle_rad=np.deg2rad([row['L1C_flex'] for row in rows]).tolist(),
                fluorescence=[row['calcium'] for row in rows])


def make_package(primary, anchor, source):
    if primary['split'] == anchor['split'] or primary['group_id'] == anchor['group_id']:
        raise ValueError('opposite split anchor required')
    return dict(schema=SCHEMA, data_kind='biological', dataset_id=primary['group_id'],
                species='unknown', license='unknown', fluorescence_metric='unknown', reporter='unknown',
                split_policy=POLICY, units=dict(time='s', angle='rad', fluorescence='unknown'),
                source=source, trials=[to_trial(primary), to_trial(anchor)])


def plan_packages(groups):
    """One primary per unique group; deterministic first opposite split anchor."""
    if len({g['group_id'] for g in groups}) != len(groups):
        raise ValueError('duplicate global group identity')
    ordered = sorted(groups, key=lambda g: (g['source_file'], g['animal_id'], g['trial'], g['roi']))
    subject_splits = {}
    recording_splits = {}
    for g in ordered:
        for registry, identity in ((subject_splits, g['subject_id']), (recording_splits, g['raw_recording_id'])):
            if identity in registry and registry[identity] != g['split']:
                raise ValueError('global subject/recording split leakage')
            registry[identity] = g['split']
    if set(subject_splits.values()) != {'train', 'holdout'}:
        raise ValueError('both global splits required')
    pairs = []
    for primary in ordered:
        anchor = next(g for g in ordered if g['source_file'] == primary['source_file'] and g['split'] != primary['split'])
        pairs.append((primary, anchor))
    return pairs


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def run(input_dir, receipt_path, output_dir):
    input_dir, receipt_path, output_dir = map(lambda p: Path(p).resolve(), (input_dir, receipt_path, output_dir))
    if OUTPUT_ROOT.resolve() not in output_dir.parents:
        raise ValueError(f'output must be a NEW child directory under {OUTPUT_ROOT}')
    if output_dir.exists():
        raise ValueError('output directory already exists; refusing overwrite')
    provenance_paths = (Path(__file__), PROTOCOL, receipt_path,
                        PROJECT / 'tools_s4_calcium_paired_contract.py',
                        PROJECT / 'tools_s4_paired_data_contract.py')
    provenance_before = {str(path): sha256(path) for path in provenance_paths}
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt_files = {}
    for item in receipt['files']:
        name = Path(item['path']).name
        if name in receipt_files:
            raise ValueError('duplicate receipt file identity')
        receipt_files[name] = item
    # Verify fixed constants AND receipt before importing Arrow or reading tables.
    source_info = {}
    for name, expected in SOURCES.items():
        path = input_dir / name
        item = receipt_files.get(name, {})
        digest = sha256(path)
        if digest != expected['sha256'] or digest != item.get('sha256') or path.stat().st_size != item.get('bytes'):
            raise ValueError(f'{name}: fixed/receipt hash or byte count mismatch')
        source_info[name] = dict(uri=path.as_uri(), revision='received Parquet bytes; upstream revision unknown',
                                 sha256=digest, asset_bytes_verified=True)
    import pyarrow.parquet as pq
    groups, file_summary = [], {}
    for name, expected in SOURCES.items():
        rows = pq.read_table(input_dir / name).to_pylist()
        file_groups = group_rows(name, rows)
        if len(rows) != expected['rows'] or len(file_groups) != expected['groups']:
            raise ValueError('fixed source row/group counts changed')
        groups.extend(file_groups)
        animals = sorted({g['animal_id'] for g in file_groups})
        file_summary[name] = dict(rows=len(rows), groups=len(file_groups), train_animals=animals[:-2],
                                  holdout_animals=animals[-2:], source=source_info[name])
    if len(groups) != 116 or sum(len(g['rows']) for g in groups) != 52092:
        raise ValueError('global unique source count mismatch')
    pairs = plan_packages(groups)
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger, packages = [], []
    for index, (primary, anchor) in enumerate(pairs):
        stem = f'package-{index:04d}'
        package = make_package(primary, anchor, source_info[primary['source_file']])
        # Multi-file anchors carry their own provenance in the global manifest.
        report = validate(package)
        write_json(output_dir / (stem + '.json'), package)
        write_json(output_dir / (stem + '.validate.json'), report)
        if not report['contract_valid']:
            raise ValueError(f'{stem} rejected: {report["errors"]}')
        packages.append(dict(package=stem + '.json', validation=stem + '.validate.json',
                             sha256=sha256(output_dir / (stem + '.json')),
                             primary_group_id=primary['group_id'], anchor_group_id=anchor['group_id'],
                             anchor_is_packaging_duplicate=True, anchor_is_new_sample=False,
                             total_points=report['total_points']))
    for group in sorted(groups, key=lambda g: g['group_id']):
        trial = to_trial(group)
        entry = {key: value for key, value in group.items() if key != 'rows'}
        entry.update(samples=len(group['rows']), primary_occurrences=1,
                     anchor_occurrences=sum(a['group_id'] == group['group_id'] for _, a in pairs),
                     source_rows_sha256=canonical_hash(group['rows']), adapted_trial_sha256=canonical_hash(trial),
                     driver=group['rows'][0]['driver'], stimulus_type=group['rows'][0]['stimulus_type'])
        ledger.append(entry)
    provenance = {str(path): sha256(path) for path in provenance_paths}
    if provenance != provenance_before:
        raise ValueError('adapter/protocol/receipt/contract changed during adaptation')
    for name, expected in SOURCES.items():
        if sha256(input_dir / name) != expected['sha256']:
            raise ValueError('source bytes changed during adaptation')
    summary = dict(schema='s4_real_calcium_adaptation_v1', package_count=len(packages),
                   unique_group_count=len(groups), unique_row_count=sum(len(g['rows']) for g in groups),
                   packaged_rows_including_duplicate_anchors=sum(len(p['rows']) + len(a['rows']) for p, a in pairs),
                   split_policy=POLICY, subject_disjoint=True, recording_disjoint=True,
                   namespace_does_not_prove_cross_file_animals_distinct=True,
                   no_normalization=True, no_resampling=True, no_zero_shift=True,
                   unknown_processing_preserved=True, calibration_ready=False,
                   calibration_executed=False, biological_validation=False, files=file_summary,
                   provenance_sha256=provenance, fixed_source_constants=SOURCES,
                   anchor_policy='first sorted opposite-split group within same file; duplicate packaging NOT new sample',
                   before_after_input_and_provenance_hashes_equal=True,
                   scope='manual research adaptation; structural admission only; no author model or production changes')
    write_json(output_dir / 'unique_identity_manifest.json', dict(groups=ledger, packages=packages))
    write_json(output_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--receipt', type=Path, help='defaults to INPUT_DIR/receipt.json')
    parser.add_argument('--output-dir', type=Path, help='new child of project outputs/s4_real_calcium_adaptation')
    args = parser.parse_args(argv)
    output = args.output_dir or OUTPUT_ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:10])
    try:
        summary = run(args.input_dir, args.receipt or args.input_dir / 'receipt.json', output)
    except (ValueError, OSError, TypeError, KeyError, ImportError, StopIteration) as exc:
        parser.exit(1, f'REJECTED: {exc}\n')
    print(json.dumps(dict(output_dir=str(output.resolve()), unique_groups=summary['unique_group_count'],
                          unique_rows=summary['unique_row_count'], packages=summary['package_count'],
                          calibration_ready=False, biological_validation=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
