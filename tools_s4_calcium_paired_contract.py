"""Bounded inline calcium/angle admission; never infer spikes or EM identities."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from tools_s4_paired_data_contract import load, MAX_POINTS, MAX_TRIALS

SCHEMA = 's4_calcium_paired_v1'
MAX_TEXT = 4096
TOP_KEYS = {'schema', 'data_kind', 'dataset_id', 'species', 'license', 'source',
            'units', 'fluorescence_metric', 'reporter', 'split_policy', 'trials'}
TRIAL_KEYS = {'subject_id', 'trial_id', 'roi_id', 'raw_recording_id', 'split',
              'identity_level', 'roi_identity', 'response_origin', 'leg', 'joint',
              'angle_convention', 'sync', 'processing', 'time_s', 'angle_rad',
              'fluorescence_time_s', 'fluorescence'}
SYNC_KEYS = {'common_clock_id', 'method', 'fluorescence_offset_s',
             'fluorescence_scale', 'max_error_s', 'error_evidence', 'missing_data_policy'}
PROCESS_KEYS = {'baseline_method', 'normalization', 'filtering', 'resampling',
                'motion_correction', 'observation_model'}


def validate(data):
    errors, gaps, identities = [], [], []
    def require(ok, path, message):
        if not ok:
            errors.append({'path': path, 'message': message})
        return bool(ok)
    def obj(value, keys, path):
        if not require(isinstance(value, dict), path, 'object required'):
            return {}
        require(set(value) <= keys, path, 'unsupported fields (including spike/BANC claims) rejected')
        return value
    def text(value, path, identity=False):
        ok = isinstance(value, str) and 0 < len(value) <= MAX_TEXT and bool(value.strip())
        require(ok, path, 'bounded nonempty string declaration required')
        if ok and value.strip().lower() == 'unknown':
            if identity:
                require(False, path, 'split identity cannot be unknown')
                return False
            gaps.append({'path': path, 'message': 'explicit unknown'})
        if ok and identity:
            return require(value == value.strip(), path, 'exact identity cannot have surrounding whitespace')
        return ok
    def number(value):
        # Reject bool and integer magnitudes not exactly portable to binary64.
        return (type(value) is int and abs(value) <= 2**53 or
                type(value) is float and math.isfinite(value))
    def numeric_or_unknown(value, path):
        if value == 'unknown':
            gaps.append({'path': path, 'message': 'explicit unknown; clock mapping not established'})
            return False
        return require(number(value), path, 'finite numeric value or explicit unknown required, not bool')
    data = obj(data, TOP_KEYS, '/')
    require(data.get('schema') == SCHEMA, '/schema', 's4_calcium_paired_v1 required')
    require(data.get('data_kind') in ('synthetic_fixture', 'biological'), '/data_kind', 'explicit kind required')
    for key in ('dataset_id', 'species', 'license', 'fluorescence_metric', 'reporter', 'split_policy'):
        text(data.get(key), '/'+key, identity=key == 'dataset_id')
    units = obj(data.get('units'), {'time', 'angle', 'fluorescence'}, '/units')
    require(units.get('time') == 's' and units.get('angle') == 'rad', '/units',
            'canonical time s and angle rad required; convert upstream, never guessed')
    text(units.get('fluorescence'), '/units/fluorescence')
    source = obj(data.get('source'), {'uri', 'revision', 'sha256', 'asset_bytes_verified'}, '/source')
    for key in ('uri', 'revision', 'sha256'):
        text(source.get(key), '/source/'+key)
    digest = source.get('sha256')
    if isinstance(digest, str) and digest != 'unknown':
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                '/source/sha256', 'lowercase SHA256 or unknown required; not independently verified')
    require(type(source.get('asset_bytes_verified')) is bool, '/source/asset_bytes_verified', 'bool declaration required')
    if source.get('asset_bytes_verified') is not True:
        gaps.append({'path': '/source', 'message': 'asset bytes not declared inspected'})
    trials = data.get('trials')
    if not require(isinstance(trials, list) and 2 <= len(trials) <= MAX_TRIALS,
                   '/trials', '2..1000 paired ROI records required'):
        trials = []
    records, recordings, trial_recordings, rois, splits = set(), {}, {}, {}, set()
    total = 0
    for i, value in enumerate(trials):
        p = f'/trials/{i}'
        t = obj(value, TRIAL_KEYS, p)
        valid_ids = all([text(t.get(k), p+'/'+k, True) for k in
                         ('subject_id', 'trial_id', 'roi_id', 'raw_recording_id')])
        split = t.get('split')
        require(split in ('train', 'validation', 'holdout'), p+'/split', 'explicit split required')
        if isinstance(split, str):
            splits.add(split)
        for k in ('roi_identity', 'leg', 'joint', 'angle_convention'):
            text(t.get(k), p+'/'+k)
        level = t.get('identity_level')
        require(level in ('roi_population', 'single_cell', 'unknown'), p+'/identity_level', 'explicit ROI identity level required')
        if level != 'single_cell':
            gaps.append({'path': p+'/identity_level', 'message': 'single-cell identity not established; ROI preserved'})
        origin = t.get('response_origin')
        require(origin in ('measured', 'predicted', 'unknown'), p+'/response_origin', 'measured/predicted/unknown required')
        if origin != 'measured':
            gaps.append({'path': p+'/response_origin', 'message': 'not declared measured'})
        if valid_ids:
            ident = tuple(t[k] for k in ('subject_id', 'trial_id', 'roi_id', 'raw_recording_id'))
            require(ident not in records, p, 'duplicate subject/trial/ROI/recording identity')
            records.add(ident)
            raw = t['raw_recording_id']
            grouping = (t['subject_id'], split)
            require(raw not in recordings or recordings[raw] == grouping, p+'/raw_recording_id',
                    'raw recording cannot change subject or cross splits')
            recordings[raw] = grouping
            trial_key = (t['subject_id'], t['trial_id'])
            require(trial_key not in trial_recordings or trial_recordings[trial_key] == raw,
                    p+'/trial_id', 'same subject/trial cannot change raw recording')
            trial_recordings[trial_key] = raw
            roi_key = (t['subject_id'], raw, t['roi_id'])
            declaration = (level, t.get('roi_identity'))
            require(roi_key not in rois or rois[roi_key] == declaration, p+'/roi_identity',
                    'same recording ROI identity declaration must stay consistent')
            rois[roi_key] = declaration
        identities.append({k: copy.deepcopy(t.get(k)) for k in
                           ('subject_id', 'trial_id', 'roi_id', 'raw_recording_id', 'identity_level', 'roi_identity', 'response_origin', 'split')})
        sync = obj(t.get('sync'), SYNC_KEYS, p+'/sync')
        for k in ('common_clock_id', 'method', 'error_evidence', 'missing_data_policy'):
            text(sync.get(k), p+'/sync/'+k)
        known = {k: numeric_or_unknown(sync.get(k), p+'/sync/'+k) for k in
                 ('fluorescence_offset_s', 'fluorescence_scale', 'max_error_s')}
        if known['fluorescence_scale']:
            require(sync['fluorescence_scale'] > 0, p+'/sync/fluorescence_scale', 'positive clock scale required')
        if known['max_error_s']:
            require(sync['max_error_s'] >= 0, p+'/sync/max_error_s', 'nonnegative declaration; no biological tolerance implied')
        processing = obj(t.get('processing'), PROCESS_KEYS, p+'/processing')
        for k in sorted(PROCESS_KEYS):
            text(processing.get(k), p+'/processing/'+k)
        arrays = {}
        for k in ('time_s', 'angle_rad', 'fluorescence_time_s', 'fluorescence'):
            a = t.get(k)
            if require(isinstance(a, list) and 2 <= len(a) <= MAX_POINTS, p+'/'+k, 'bounded list of at least two samples required'):
                total += len(a)
                if require(all(number(v) for v in a), p+'/'+k, 'finite numeric samples required; bool and oversized integers rejected'):
                    arrays[k] = a
        counts = [len(a) for a in arrays.values()]
        require(len(set(counts)) <= 1, p, 'paired sample count mismatch')
        for k in ('time_s', 'fluorescence_time_s'):
            a = arrays.get(k, [])
            require(all(b > a for a, b in zip(a, a[1:])), p+'/'+k, 'strictly increasing times required (no rounded/collapsed clock)')
        if all(known[k] for k in ('fluorescence_offset_s', 'fluorescence_scale')) and 'fluorescence_time_s' in arrays and 'time_s' in arrays:
            try:
                mapped = [v * sync['fluorescence_scale'] + sync['fluorescence_offset_s'] for v in arrays['fluorescence_time_s']]
                require(all(number(v) for v in mapped) and all(b > a for a, b in zip(mapped, mapped[1:])),
                        p+'/sync', 'mapped times must be finite and strictly increasing')
                require(mapped == arrays['time_s'], p+'/sync', 'mapped fluorescence times must exactly equal paired time_s; resample upstream explicitly')
            except (OverflowError, ValueError):
                require(False, p+'/sync', 'clock transform overflow')
    require(total <= MAX_POINTS, '/trials', 'aggregate point budget exceeded')
    require({'train', 'holdout'} <= splits, '/trials', 'recording-disjoint train and holdout required')
    return dict(schema=SCHEMA, contract_valid=not errors, calibration_ready=False,
                biological_validation=False, calibration_executed=False, errors=errors, gaps=gaps,
                data_kind=data.get('data_kind'), trial_count=len(trials), total_points=total,
                roi_records=identities, fluorescence_metric=copy.deepcopy(data.get('fluorescence_metric')),
                reporter=copy.deepcopy(data.get('reporter')), units=copy.deepcopy(units),
                scope='structural admission only; declarations not independently scientifically verified',
                limitations=['No spikes or BANC roots inferred; ROI is not an EM neuron',
                             'No assets fetched, executed, fitted or production configuration mutated',
                             'Recording IDs are declared: hidden aliases cannot be detected',
                             'Subject-disjointness not required; holdout is not biological validation',
                             'No biological error threshold or fluorescence-to-activation inference'])


def synthetic_fixture():
    """Pure engineering fixture: NOT animal data, NOT a BANC/single-cell record."""
    data = dict(schema=SCHEMA, data_kind='synthetic_fixture', dataset_id='NOT_ANIMAL_fixture',
                species='NOT_ANIMAL', license='engineering fixture only', reporter='unknown',
                fluorescence_metric='delta_f_over_f', split_policy='raw-recording-disjoint',
                units={'time': 's', 'angle': 'rad', 'fluorescence': 'dimensionless'},
                source={'uri': 'synthetic:NOT_ANIMAL', 'revision': 'fixture-v1',
                        'sha256': 'unknown', 'asset_bytes_verified': False}, trials=[])
    for split in ('train', 'holdout'):
        data['trials'].append(dict(subject_id='NOT_ANIMAL_subject', trial_id=split+'_trial',
            roi_id='population_roi', raw_recording_id=split+'_recording', split=split,
            identity_level='roi_population', roi_identity='synthetic population; NOT animal',
            response_origin='predicted', leg='unknown', joint='unknown', angle_convention='synthetic radians',
            sync=dict(common_clock_id='fixture_clock', method='synthetic common grid',
                      fluorescence_offset_s=0.0, fluorescence_scale=1.0, max_error_s=0.0,
                      error_evidence='engineering construction, NOT animal synchronization', missing_data_policy='no missing fixture points'),
            processing={k: 'synthetic fixture; no animal processing' for k in PROCESS_KEYS},
            time_s=[0.0, 0.5, 1.0], angle_rad=[0.0, 0.1, -0.1],
            fluorescence_time_s=[0.0, 0.5, 1.0], fluorescence=[0.0, 0.2, -0.1]))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        data, digest = load(args.manifest)
        report = validate(data)
        report.update(manifest=str(args.manifest.resolve()), manifest_sha256=digest)
    except (ValueError, OSError, TypeError, RecursionError) as e:
        report = dict(contract_valid=False, calibration_ready=False, biological_validation=False,
                      calibration_executed=False, errors=[{'path': '/', 'message': str(e)}], gaps=[])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('STRUCTURAL_PASS_WITH_GAPS' if report['contract_valid'] else 'REJECTED')
    raise SystemExit(0 if report['contract_valid'] else 1)


if __name__ == '__main__':
    main()
