"""Validate a bounded inline claw/hook paired-data manifest, not biological validity."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

MAX_BYTES = 8 * 1024 * 1024
MAX_TRIALS = 1000
MAX_POINTS = 100000


def validate(data):
    errors = []
    def require(condition, path, message):
        if not condition:
            errors.append({'path': path, 'message': message})
        return bool(condition)
    def text(obj, key, path):
        return require(isinstance(obj.get(key), str) and bool(obj[key].strip()),
                       path + '/' + key, 'nonempty string required')
    def number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if not isinstance(data, dict):
        return {'contract_valid': False, 'errors': [{'path': '/', 'message': 'object required'}],
                'biological_validation': False}
    require(type(data.get('schema_version')) is int and data['schema_version'] == 1,
            '/schema_version', 'integer version 1 required')
    require(data.get('data_kind') in ('synthetic', 'biological'), '/data_kind', 'explicit data kind required')
    for key in ('dataset_id', 'species', 'license', 'preprocessing', 'split_policy'):
        text(data, key, '')
    require(data.get('scope') == 'sensory_claw_hook', '/scope', 'only sensory_claw_hook supported')
    require(data.get('units') == {'time': 's', 'angle': 'rad', 'velocity': 'rad/s', 'events': 'spike'},
            '/units', 'canonical units required; conversions must be performed upstream and documented')
    source = data.get('source')
    if require(isinstance(source, dict), '/source', 'source object required'):
        for key in ('uri', 'revision', 'identity_evidence', 'coordinate_convention'):
            text(source, key, '/source')
        require(isinstance(source.get('sha256'), str) and bool(re.fullmatch('[0-9a-f]{64}', source['sha256'])),
                '/source/sha256', 'lowercase SHA256 required (declaration, not independently fetched)')
    trials = data.get('trials')
    if not require(isinstance(trials, list) and 2 <= len(trials) <= MAX_TRIALS,
                   '/trials', '2..1000 trial records required'):
        trials = []
    ids, raw_groups, splits = set(), {}, set()
    total_points = 0
    for i, trial in enumerate(trials):
        path = f'/trials/{i}'
        if not require(isinstance(trial, dict), path, 'trial object required'):
            continue
        for key in ('trial_id', 'subject_id', 'raw_recording_id', 'joint', 'channel', 'leg'):
            text(trial, key, path)
        ident = trial.get('trial_id')
        if isinstance(ident, str):
            require(ident not in ids, path+'/trial_id', 'duplicate trial identity')
            ids.add(ident)
        require(trial.get('channel') in ('claw_position', 'hook_direction'), path+'/channel', 'unsupported channel')
        require(trial.get('leg') in ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'), path+'/leg', 'unsupported leg')
        root = trial.get('cell_root_id')
        require(isinstance(root, str) and bool(re.fullmatch('[1-9][0-9]*', root)),
                path+'/cell_root_id', 'exact positive decimal string required; floats/integers rejected')
        split = trial.get('split')
        require(split in ('train', 'validation', 'holdout'), path+'/split', 'explicit split required')
        if isinstance(split, str):
            splits.add(split)
        group = trial.get('raw_recording_id')
        if isinstance(group, str):
            if group in raw_groups:
                require(raw_groups[group] == split, path+'/split', 'same recording cannot cross splits')
            raw_groups[group] = split
        sync = trial.get('sync')
        if require(isinstance(sync, dict), path+'/sync', 'synchronization object required'):
            for key in ('common_clock_id', 'method', 'error_evidence', 'missing_data_policy'):
                text(sync, key, path+'/sync')
            for key in ('neural_offset_s', 'neural_scale', 'max_error_s'):
                require(number(sync.get(key)), path+'/sync/'+key, 'finite numeric value required, not bool')
            if number(sync.get('neural_scale')):
                require(sync['neural_scale'] > 0, path+'/sync/neural_scale', 'positive clock scale required')
            if number(sync.get('max_error_s')):
                require(sync['max_error_s'] >= 0, path+'/sync/max_error_s', 'nonnegative error required; no biological tolerance implied')
        arrays = {}
        for key in ('time_s', 'angle_rad', 'velocity_rad_s', 'spike_time_neural_s'):
            value = trial.get(key)
            ok = isinstance(value, list) and len(value) <= MAX_POINTS and all(number(v) for v in value)
            if require(ok, path+'/'+key, 'bounded finite numeric array required'):
                arrays[key] = value
                total_points += len(value)
        time = arrays.get('time_s', [])
        require(len(time) >= 2, path+'/time_s', 'at least two body samples required')
        require(all(b > a for a, b in zip(time, time[1:])), path+'/time_s', 'strictly increasing samples required')
        for key in ('angle_rad', 'velocity_rad_s'):
            if key in arrays:
                require(len(arrays[key]) == len(time), path+'/'+key, 'paired body sample count mismatch')
        events = arrays.get('spike_time_neural_s', [])
        require(all(b > a for a, b in zip(events, events[1:])), path+'/spike_time_neural_s',
                'single-cell events must strictly increase; duplicates rejected')
        if time and isinstance(sync, dict) and number(sync.get('neural_offset_s')) and number(sync.get('neural_scale')):
            aligned = [v*sync['neural_scale']+sync['neural_offset_s'] for v in events]
            require(all(math.isfinite(v) and time[0] <= v <= time[-1] for v in aligned),
                    path+'/spike_time_neural_s', 'mapped events outside sampled body window')
    require(total_points <= MAX_POINTS, '/trials', 'aggregate point budget exceeded')
    require({'train', 'holdout'}.issubset(splits), '/trials', 'train and independent holdout required')
    return {'schema_version': 1, 'contract_valid': not errors, 'errors': errors,
            'trial_count': len(trials), 'data_kind': data.get('data_kind'),
            'biological_validation': False, 'calibration_executed': False,
            'scope': 'structural admission only; identity, source digest and synchronization evidence are declarations',
            'limitations': ['No external asset fetched or digest verified',
                            'No synchronization accuracy or cell lineage scientifically verified',
                            'Recording-disjoint split checked; subject-disjointness not required by v1',
                            'No numerical convergence, biological prediction or behavior validated']}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def load(path):
    with Path(path).open('rb') as f:
        raw = f.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise ValueError('manifest byte budget exceeded')
    data = json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON constant: '+v)))
    return data, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        data, digest = load(args.manifest)
        report = validate(data)
        report.update(manifest=str(args.manifest.resolve()), manifest_sha256=digest)
    except (ValueError, OSError, TypeError) as error:
        report = {'contract_valid': False, 'biological_validation': False,
                  'errors': [{'path': '/', 'message': str(error)}]}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('STRUCTURAL_PASS' if report['contract_valid'] else 'REJECTED')
    raise SystemExit(0 if report['contract_valid'] else 1)


if __name__ == '__main__':
    main()
