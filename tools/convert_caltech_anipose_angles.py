#!/usr/bin/env python3
"""Convert audited Caltech model-bundle features, without executing author code.
This is a derived angle supplement, never a raw 3-D or pristine holdout dataset.
"""
import argparse
import csv
import hashlib
import io
import json
import pickle
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
from numpy._core.multiarray import _reconstruct

LEGS = ('L1', 'L2', 'L3', 'R1', 'R2', 'R3')
MEMBER = 'code/data_and_weights.pickle'
MD5 = '06be8182b5e73b635930216b5ed5e3fb'
RAW_SHA = '95b312b4c6efe397350f923257032ad58f18a1f0dd980b8cdba2c8168330f0e3'
DEFAULT_ROOT = Path('/home/sensen/Desktop/engine修复验收/s1-continuous-fly-kinematics/20261102T004326Z')

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): _reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent references forbidden')


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_source(path):
    assert digest(path, 'md5') == MD5, 'ZIP MD5 mismatch'
    with zipfile.ZipFile(path) as z:
        raw = z.read(MEMBER)  # Never extract or persist a new pickle copy.
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA, 'Pickle SHA256 mismatch'
    data = RestrictedUnpickler(io.BytesIO(raw)).load()
    assert set(data) == set(LEGS)
    return data


def identities(bnums):
    animal, trial = [], []
    for b in bnums:
        m = re.fullmatch(r'(.+ Fly \d+)_([^ ]+) b(\d+)', str(b))
        if not m:
            raise ValueError(f'Unrecognized bout identity: {b!r}')
        animal.append(m[1])
        trial.append(m[1] + '_' + m[2])
    return np.asarray(animal), np.asarray(trial)


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def convert(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = load_source(args.source)
    checks = {'alignment': {}, 'source_feature_shapes': {}, 'pair_checks': [], 'warnings': []}
    names = [n for leg in LEGS for n in data[leg]['angle_names']]
    all_b = np.concatenate([np.asarray(data['L1'][s][1], dtype=str) for s in ('train', 'test')])
    all_animals, _ = identities(all_b)
    animals = sorted(set(all_animals), key=lambda a: hashlib.sha256(('caltech-v1:' + a).encode()).hexdigest())
    assert len(animals) >= 3
    ntest = max(1, round(len(animals) * .15))
    nval = max(1, round(len(animals) * .15))
    split_map = {a: ('test' if i < ntest else 'val' if i < ntest + nval else 'train') for i, a in enumerate(animals)}
    bout_rows, summaries = [], {}
    for source in ('train', 'test'):
        bnums = np.asarray(data['L1'][source][1], dtype=str)
        context = np.asarray(data['L1'][source][0][0])[:, -5:-2]
        # Check before constructing any cross-leg record.
        for leg in LEGS:
            xy, b = data[leg][source]
            assert np.array_equal(np.asarray(b, dtype=str), bnums), (source, leg, 'bnums mismatch')
            assert np.array_equal(xy[0][:, -5:-2], context, equal_nan=True), (source, leg, 'context mismatch')
        checks['alignment'][source] = {'all_six_bnums_strictly_equal': True, 'all_six_context_exactly_equal': True}
        animal, trial = identities(bnums)
        split = np.asarray([split_map[a] for a in animal])
        nrow = len(bnums)
        angles = np.full((nrow, len(names)), np.nan, dtype=np.float32)
        mask = np.zeros_like(angles, dtype=bool)
        admission = np.ones(nrow, dtype=bool) if source == 'train' else np.zeros(nrow, dtype=bool)
        arrays = {'angle_names': np.asarray(names), 'angles_rad': angles, 'angle_mask': mask,
                  'admission_mask': admission, 'bnum': bnums, 'animal_id': animal, 'trial_id': trial,
                  'source_partition': np.full(nrow, source), 'split': split, 'source_row_index': np.arange(nrow),
                  'context': context, 'context_finite_mask': np.isfinite(context), 'nominal_hz': np.asarray(300.),
                  'frame_index_is_original': np.asarray(False)}
        offset = 0
        for leg in LEGS:
            xy, _ = data[leg][source]
            x, y, sx, sy = xy
            n = len(data[leg]['angle_names'])
            width_valid = x.shape[1] == 3 * n + 5
            arrays[f'{leg}_X'] = x
            arrays[f'{leg}_y'] = y
            arrays[f'{leg}_X_finite_mask'] = np.isfinite(x)
            arrays[f'{leg}_y_finite_mask'] = np.isfinite(y)
            arrays[f'{leg}_angle_names'] = np.asarray(data[leg]['angle_names'])
            for label, scale in [('X', sx), ('y', sy)]:
                for i, v in enumerate(scale):
                    arrays[f'{leg}_{label}_scale_{i}'] = np.asarray(v)
            checks['source_feature_shapes'][source + ':' + leg] = {'X': list(x.shape), 'y': list(y.shape), 'angle_width_valid': width_valid}
            if width_valid:
                c, s = x[:, :n], x[:, n:2*n]
                valid = np.isfinite(c) & np.isfinite(s) & ((c != 0) | (s != 0))
                angles[:, offset:offset+n] = np.arctan2(s, c)
                mask[:, offset:offset+n] = valid
                arrays[f'{leg}_derivative_source_units'] = x[:, 2*n:3*n]
            else:
                checks['warnings'].append(f'{source}:{leg}: X width {x.shape[1]} disagrees with {n} named angles; angle values NaN/mask false; original X retained.')
            arrays[f'{leg}_phase_rad'] = np.arctan2(x[:, -1], x[:, -2])
            arrays[f'{leg}_phase_mask'] = np.isfinite(x[:, -2:]).all(axis=1) & np.any(x[:, -2:] != 0, axis=1)
            offset += n
        for i, left in enumerate(LEGS):
            for right in LEGS[i+1:]:
                a, b = arrays[left + '_X'], arrays[right + '_X']
                same_shape = a.shape == b.shape
                exact = same_shape and np.array_equal(a, b, equal_nan=True)
                # Mirrored numerical equality means sign-inverted angles at aligned rows.
                na, nb = len(data[left]['angle_names']), len(data[right]['angle_names'])
                mirrored = False
                comparable = same_shape and na == nb and a.shape[1] == 3*na+5
                if comparable:
                    aa, ab = np.arctan2(a[:, na:2*na], a[:, :na]), np.arctan2(b[:, nb:2*nb], b[:, :nb])
                    mirrored = bool(np.allclose(aa, -ab, atol=1e-7, rtol=0, equal_nan=True))
                checks['pair_checks'].append({'source': source, 'left': left, 'right': right,
                    'X_exact_duplicate': bool(exact),
                    'y_exact_duplicate': bool(arrays[left+'_y'].shape == arrays[right+'_y'].shape and np.array_equal(arrays[left+'_y'], arrays[right+'_y'], equal_nan=True)),
                    'sign_mirror_comparable': comparable, 'angle_sign_mirror': mirrored})
        # Cross-leg same-suffix angle equality and shifts: never compare across bout boundaries.
        checks.setdefault('same_angle_cross_leg_checks', [])
        for li, left in enumerate(LEGS):
            for right in LEGS[li+1:]:
                x1, x2 = arrays[left+'_X'], arrays[right+'_X']
                names1, names2 = data[left]['angle_names'], data[right]['angle_names']
                n1, n2 = len(names1), len(names2)
                if x1.shape[1] != 3*n1+5 or x2.shape[1] != 3*n2+5:
                    continue
                for j1, name1 in enumerate(names1):
                    suffix = name1[2:]
                    if right + suffix not in names2:
                        continue
                    j2 = names2.index(right + suffix)
                    a = np.arctan2(x1[:,n1+j1],x1[:,j1])
                    b = np.arctan2(x2[:,n2+j2],x2[:,j2])
                    row = {'source':source,'left':name1,'right':right+suffix,
                           'exact_equal':bool(np.array_equal(a,b,equal_nan=True)),
                           'sign_mirror':bool(np.allclose(a,-b,atol=1e-7,rtol=0,equal_nan=True)),
                           'shift_checks':[]}
                    for lag in (-300,-30,-10,-2,-1,1,2,10,30,300):
                        lo, hi = max(0,-lag), min(nrow,nrow-lag)
                        ix = np.arange(lo,hi)
                        ix = ix[bnums[ix] == bnums[ix+lag]]
                        row['shift_checks'].append({'lag_rows':lag,'compared_rows':len(ix),
                            'exact_equal':bool(len(ix) and np.array_equal(a[ix],b[ix+lag],equal_nan=True))})
                    checks['same_angle_cross_leg_checks'].append(row)
        # Test features are identical across all legs: labels cannot be trusted even if width fits.
        if source == 'test':
            arrays['angle_decode_mask'] = mask.copy()
            mask[:] = False
            checks['warnings'].append('All upstream test rows quarantined: six leg X/y feature arrays are exact duplicates, and front-leg widths invalid. Numerically decoded values are retained only as untrusted derived values; angle_mask/admission_mask false.')
        else:
            arrays['angle_decode_mask'] = mask.copy()
        row_in_bout = np.zeros(nrow, dtype=np.int64)
        bout_hashes = {}
        for b in dict.fromkeys(bnums):
            ix = np.flatnonzero(bnums == b)
            row_in_bout[ix] = np.arange(len(ix))
            packed = np.hstack([arrays[l+'_X'][ix] for l in LEGS])
            h = hashlib.sha256(packed.tobytes()).hexdigest()
            repeated = bout_hashes.get(h, '')
            bout_hashes[h] = str(b)
            bout_rows.append({'source': source, 'bnum': b, 'animal_id': animal[ix[0]], 'trial_id': trial[ix[0]],
                'split': split[ix[0]], 'rows': len(ix), 'nominal_duration_s': len(ix)/300,
                'nominal_first_last_span_s': (len(ix)-1)/300, 'nominal_span_at_least_10s': bool(len(ix)>=3001),
                'source_rows_contiguous': bool(np.all(np.diff(ix) == 1)),
                'X_nonfinite_values': int((~np.isfinite(packed)).sum()),
                'decoded_angle_nonfinite_values': int((~np.isfinite(angles[ix])).sum()),
                'usable_angle_values': int(mask[ix].sum()), 'X_sha256': h, 'exact_duplicate_bout': repeated})
        arrays['row_in_bout'] = row_in_bout
        assert not any(np.asarray(v).dtype.hasobject for v in arrays.values())
        np.savez_compressed(out / (source + '_neutral.npz'), **arrays)
        summaries[source] = {'rows': nrow, 'bouts': len(set(bnums)), 'animals': len(set(animal)),
            'angle_values_decoded_finite': int(np.isfinite(angles).sum()), 'angle_values_usable': int(mask.sum()),
            'admitted_rows': int(admission.sum()), 'X_nonfinite_values': sum(int((~np.isfinite(arrays[l+'_X'])).sum()) for l in LEGS),
            'y_nonfinite_values': sum(int((~np.isfinite(arrays[l+'_y'])).sum()) for l in LEGS),
            'split_rows': {s: int(np.sum(split == s)) for s in ('train', 'val', 'test')},
            'bout_length_min': min(r['rows'] for r in bout_rows if r['source'] == source),
            'bout_length_max': max(r['rows'] for r in bout_rows if r['source'] == source),
            'nominal_span_max_s': max(r['nominal_first_last_span_s'] for r in bout_rows if r['source'] == source),
            'bouts_nominal_span_at_least_10s': sum(r['nominal_span_at_least_10s'] for r in bout_rows if r['source'] == source)}
        if source == 'train':
            first = bnums[0]
            ix = np.flatnonzero(bnums == first)
            write_csv(out / 'example_first_bout_six_legs.csv', ({'bnum': bnums[i], 'animal_id': animal[i],
                'trial_id': trial[i], 'row_in_bout': row_in_bout[i], 'nominal_time_s': row_in_bout[i]/300,
                **{name + '_rad': float(angles[i,j]) for j,name in enumerate(names)},
                **{name + '_mask': bool(mask[i,j]) for j,name in enumerate(names)}} for i in ix))
    write_csv(out / 'bout_checks.csv', bout_rows)
    checks['summary'] = summaries
    checks['duplicate_bout_count'] = sum(bool(r['exact_duplicate_bout']) for r in bout_rows)
    checks['source_test_train_bnum_overlap'] = len(set(data['L1']['train'][1]) & set(data['L1']['test'][1]))
    checks['source_test_train_animal_overlap'] = len(set(identities(data['L1']['train'][1])[0]) & set(identities(data['L1']['test'][1])[0]))
    dump(out / 'NUMERICAL_CHECKS.json', checks)
    metadata = {'dataset_kind': 'Derived angles from already-trained model feature bundle, not raw xyz',
        'source_zip': str(args.source.resolve()), 'zip_md5': MD5, 'zip_sha256': digest(args.source),
        'pickle_member': MEMBER, 'pickle_sha256': RAW_SHA,
        'author_feature_definition': str((args.source.parent / 'AUTHOR_code_tools_trajgen_tools.py').resolve()),
        'angle_formula': 'atan2(X[:,n:2*n], X[:,:n]) in radians; n from per-leg angle_names',
        'angle_names': names, 'nominal_hz': 300,
        'nominal_rate_evidence': str((args.source.parent/'AUTHOR_code_main.py').resolve()) + ':23 Ts=1/300 (trajectory generator rate, not proof of original observed frame dt)',
        'stage': 'S1 real-motion derived-angle reference supplement; research implementation, not a finished physical walking result',
        'clip_stitching_risk': 'No synthetic bout concatenation. Each bnum retained separately; original dropped-frame gaps or prior clip stitching inside author features cannot be ruled out. source_row_index and row_in_bout are feature row indices only.',
        'feature_layout': 'n cos, n sin, n derivatives in source units, 3 context, phase cos/sin',
        'no_xyz': True, 'no_root_trajectory': True, 'no_contact': True, 'no_original_frame_numbers': True,
        'timing_limit': 'Only row order within bnum is known. Deleted frames inside bouts and original dt cannot be guaranteed. nominal_time_s is not observed timestamp.',
        'identity': 'Full bnum retained. trial_id removes final bNUMBER. animal_id also removes _trial; date+Fly number is an inferred animal identity, not externally validated across dates.',
        'split_policy': 'Pool upstream train/test identities; deterministic SHA256 animal allocation approximately 70/15/15. All trials/bouts of inferred animal share one split. Upstream test is not a new blind holdout; full bundle already associated with trained models.',
        'split_animal_map': split_map,
        'split_limitation': 'New test only evaluates downstream separation of inferred animal groups; no claim of upstream unseen data. Upstream test rows remain quarantined regardless of new split.',
        'upstream_test_defect': 'All six X/y exact duplicates with 14 X columns; front-leg 4-angle labels require 17. All test angle_mask/admission_mask false. angle_decode_mask records width/finite numerical decodability, not biological validity.',
        'mask_semantics': 'angle_mask=usable named angle; angle_decode_mask=width-compatible finite nonzero cos/sin decoding; raw X/y finite masks retained. NaN invalid values never imputed.',
        'relationship_to_parent': 'Supplemental long angle target alongside parent NMF original xyz; cannot be called a true original 3D record.',
        'summary': summaries, 'cleanup': {'actually_deleted_files': 0, 'reason': 'No disposable files created; failed executions retained as evidence.'},
        'retained_file_importance': {
            str(Path(__file__).resolve()): {'importance':'核心必需','purpose':'Restricted conversion and independent-process verification entrypoint'},
            **{str((out / f).resolve()): {'importance': '核心必需' if f.endswith('.npz') or f == 'METADATA.json' else '重要辅助', 'purpose': purpose} for f,purpose in {
                'train_neutral.npz':'All upstream train rows with neutral arrays, identities, split, masks',
                'test_neutral.npz':'All upstream test raw rows retained with quarantined angle masks',
                'METADATA.json':'Source caveats, schema, importance and provenance',
                'NUMERICAL_CHECKS.json':'Alignment, finite, length, duplicate and mirror checks',
                'bout_checks.csv':'One row per source bout with length, finite counts and SHA256',
                'example_first_bout_six_legs.csv':'First complete six-leg bout example, radians and masks',
                'INDEPENDENT_VERIFY.json':'Separate-process raw-to-NPZ verification result',
                'EXECUTION.json':'Execution exit status and retained failure evidence',
                'HASHES.json':'Local converter/output hashes, not parent overall manifest',
            }.items()}}}
    dump(out / 'METADATA.json', metadata)
    return summaries


def verify(args):
    # Fresh process: reload audited raw bytes and independently recompute direct slices.
    raw = load_source(args.source)
    report = {'passed': True, 'process_pid': __import__('os').getpid(), 'verification': [], 'max_abs_angle_error_rad': 0.0}
    for source in ('train', 'test'):
        with np.load(args.output / (source + '_neutral.npz'), allow_pickle=False) as z:
            for k in z.files:
                assert not z[k].dtype.hasobject, k
            off = 0
            for leg in LEGS:
                xy, bn = raw[leg][source]
                x, y, sx, sy = xy
                assert np.array_equal(z['bnum'], np.asarray(bn,dtype=str))
                assert np.array_equal(z[leg+'_angle_names'],np.asarray(raw[leg]['angle_names']))
                assert np.array_equal(z['context'],x[:,-5:-2],equal_nan=True)
                expected_animal, expected_trial = identities(bn)
                assert np.array_equal(z['animal_id'],expected_animal)
                assert np.array_equal(z['trial_id'],expected_trial)
                assert np.array_equal(z[leg+'_X'], x, equal_nan=True)
                assert np.array_equal(z[leg+'_y'], y, equal_nan=True)
                assert np.array_equal(z[leg+'_X_finite_mask'], np.isfinite(x))
                assert np.array_equal(z[leg+'_y_finite_mask'], np.isfinite(y))
                for label, values in [('X',sx),('y',sy)]:
                    for i,v in enumerate(values):
                        assert np.array_equal(z[f'{leg}_{label}_scale_{i}'],v,equal_nan=True)
                n = len(raw[leg]['angle_names'])
                actual = z['angles_rad'][:, off:off+n]
                if x.shape[1] == 3*n+5:
                    # atan2 is recomputed independently; do not call converter logic.
                    expected = np.arctan2(x[:, n:2*n], x[:, 0:n])
                    np.testing.assert_array_equal(actual, expected)
                    error = float(np.max(np.abs(actual - expected)))
                    report['max_abs_angle_error_rad'] = max(error, report['max_abs_angle_error_rad'])
                    decode_mask = np.isfinite(x[:, :n]) & np.isfinite(x[:, n:2*n]) & ((x[:, :n] != 0) | (x[:, n:2*n] != 0))
                    assert np.array_equal(z['angle_decode_mask'][:,off:off+n],decode_mask)
                    if source == 'train':
                        assert np.array_equal(z['angle_mask'][:,off:off+n],decode_mask)
                else:
                    assert np.isnan(actual).all()
                    assert not z['angle_decode_mask'][:,off:off+n].any()
                off += n
                report['verification'].append({'source':source, 'leg':leg, 'rows':len(bn), 'raw_X_y_scales_identical':True, 'named_angles':n, 'width_compatible':x.shape[1] == 3*n+5})
            if source == 'test':
                assert not z['angle_mask'].any() and not z['admission_mask'].any()
            a = z['animal_id']
            for identity in set(a):
                assert len(set(z['split'][a == identity])) == 1
    train = np.load(args.output/'train_neutral.npz',allow_pickle=False)
    test = np.load(args.output/'test_neutral.npz',allow_pickle=False)
    for s1 in ('train','val','test'):
        a1 = set(train['animal_id'][train['split']==s1]) | set(test['animal_id'][test['split']==s1])
        for s2 in ('train','val','test'):
            if s1 != s2:
                a2 = set(train['animal_id'][train['split']==s2]) | set(test['animal_id'][test['split']==s2])
                assert not (a1 & a2)
    train.close(); test.close()
    report['animal_split_overlap'] = 0
    dump(args.output/'INDEPENDENT_VERIFY.json',report)
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_ROOT/'code.zip')
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT/'caltech_angles')
    p.add_argument('--verify-only',action='store_true')
    args = p.parse_args()
    if args.verify_only:
        verify(args)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    execution = {'converter':str(Path(__file__).resolve()),'failures_retained':[
        {'stage':'initial environment probe','command':'python - (read-only structure probe)', 'exit_code':127, 'stderr':'python: command not found; used python3 successfully thereafter'}]}
    try:
        result = convert(args)
        child = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(args.source),'--output',str(args.output),'--verify-only'],capture_output=True,text=True)
        execution['independent_verifier'] = {'command':[sys.executable,str(Path(__file__).resolve()),'--verify-only'], 'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
        if child.returncode:
            raise RuntimeError('Independent verifier failed')
        execution['passed'] = True
        dump(args.output/'EXECUTION.json',execution)
        outputs = [p for p in args.output.iterdir() if p.is_file() and p.name != 'HASHES.json']
        hashes = {str(p.resolve()):{'sha256':digest(p),'bytes':p.stat().st_size} for p in outputs + [Path(__file__).resolve()]}
        dump(args.output/'HASHES.json',{'scope':'Only this conversion task, not parent final manifest','files':hashes})
        print(json.dumps(result,indent=2))
    except Exception:
        execution['passed'] = False
        execution['failures_retained'].append({'stage':'conversion or verification','traceback':traceback.format_exc()})
        dump(args.output/'EXECUTION.json',execution)
        raise

if __name__ == '__main__':
    main()
