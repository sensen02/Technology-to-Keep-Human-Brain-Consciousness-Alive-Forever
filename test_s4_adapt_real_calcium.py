"""Synthetic-only tests; never run the actual received-data CLI here."""
import copy
import math
from pathlib import Path
import subprocess
import sys
import unittest

import tools_s4_adapt_real_calcium as adapter


NAMES = list(adapter.SOURCES)


def rows(name=NAMES[0]):
    result = []
    for animal in (3.0, 1, 2.0, 4):
        for index, time in enumerate((0.125, 0.251, 0.377)):
            result.append(dict(driver=adapter.SOURCES[name]['driver'], roi='L1_x',
                               animal_id=animal, trial=f'trial {int(animal)}', frame=float(index),
                               time=time, stimulus_type='ramp_hold_flex_first', analyze=1.0,
                               calcium=-0.2 + index * 1.7, L1C_flex=90.0 + index * 45,
                               L1C_flex_vel=-0.25))
    return result


class AdaptationTests(unittest.TestCase):
    def test_integer_identity_not_bool_or_fraction(self):
        for value in (1, 1.0, 123, 123.0):
            self.assertEqual(adapter.animal_integer(value), int(value))
        for value in (True, False, '1', 0, -1, 1.25, float('nan'), float('inf'), 2**31):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.animal_integer(value)

    def test_deterministic_groups_and_subject_disjoint_split(self):
        source = rows()
        saved = copy.deepcopy(source)
        groups = adapter.group_rows(NAMES[0], source)
        self.assertEqual(source, saved)
        self.assertEqual([g['animal_id'] for g in groups], [1, 2, 3, 4])
        self.assertEqual([g['split'] for g in groups], ['train', 'train', 'holdout', 'holdout'])
        self.assertEqual(groups, adapter.group_rows(NAMES[0], source))
        train = {g['subject_id'] for g in groups if g['split'] == 'train'}
        holdout = {g['subject_id'] for g in groups if g['split'] == 'holdout'}
        self.assertFalse(train & holdout)
        for group in groups:
            self.assertEqual(group['rows'], [source[i] for i in group['source_row_indices']])

    def test_multiple_rois_share_recording_and_keep_subject_split(self):
        source = rows()
        extra = copy.deepcopy(source[:3])
        for row in extra:
            row['roi'] = 'L1_y'
        groups = adapter.group_rows(NAMES[0], source + extra)
        matching = [g for g in groups if g['animal_id'] == 3]
        self.assertEqual(len(matching), 2)
        self.assertEqual(len({g['raw_recording_id'] for g in matching}), 1)
        self.assertEqual({g['split'] for g in matching}, {'holdout'})
        self.assertEqual(len({g['group_id'] for g in matching}), 2)

    def test_namespace_every_identity(self):
        a = adapter.group_rows(NAMES[0], rows(NAMES[0]))[0]
        b = adapter.group_rows(NAMES[1], rows(NAMES[1]))[0]
        self.assertEqual(a['animal_id'], b['animal_id'])
        for key in ('group_id', 'subject_id', 'raw_recording_id'):
            self.assertNotEqual(a[key], b[key])
        ta, tb = adapter.to_trial(a), adapter.to_trial(b)
        for key in ('trial_id', 'roi_id'):
            self.assertNotEqual(ta[key], tb[key])

    def test_preservation_conversion_and_unknowns(self):
        group = adapter.group_rows(NAMES[0], rows())[0]
        trial = adapter.to_trial(group)
        self.assertEqual(trial['time_s'], [r['time'] for r in group['rows']])
        self.assertEqual(trial['fluorescence_time_s'], trial['time_s'])
        self.assertEqual(trial['fluorescence'], [r['calcium'] for r in group['rows']])
        self.assertEqual(trial['angle_rad'], [math.pi / 2, math.pi * 3 / 4, math.pi])
        self.assertEqual(trial['identity_level'], 'roi_population')
        self.assertEqual(trial['response_origin'], 'measured')
        for key in ('leg', 'joint', 'roi_identity'):
            self.assertEqual(trial[key], 'unknown')
        self.assertEqual(set(trial['sync'].values()), {'unknown'})
        self.assertEqual(set(trial['processing'].values()), {'unknown'})

    def test_each_primary_once_anchor_same_split_and_source(self):
        groups = sum((adapter.group_rows(name, rows(name)) for name in NAMES), [])
        pairs = adapter.plan_packages(groups)
        self.assertEqual(len(pairs), len(groups))
        self.assertEqual({p['group_id'] for p, _ in pairs}, {g['group_id'] for g in groups})
        splits = {g['group_id']: g['split'] for g in groups}
        source = dict(uri='synthetic:NOT_ANIMAL', revision='test', sha256='unknown', asset_bytes_verified=False)
        for primary, anchor in pairs:
            self.assertNotEqual(primary['split'], anchor['split'])
            self.assertEqual(primary['source_file'], anchor['source_file'])
            first = next(g for g in groups if g['source_file'] == primary['source_file'] and g['split'] != primary['split'])
            self.assertEqual(anchor['group_id'], first['group_id'])
            self.assertEqual(anchor['split'], splits[anchor['group_id']])
            package = adapter.make_package(primary, anchor, source)
            report = adapter.validate(package)
            self.assertTrue(report['contract_valid'], report['errors'])
            self.assertEqual(report['total_points'], 24)
            self.assertFalse(report['calibration_ready'])
            self.assertFalse(report['biological_validation'])
            self.assertEqual(package['fluorescence_metric'], 'unknown')
            self.assertEqual(package['reporter'], 'unknown')

    def test_fail_closed_bad_source_rows(self):
        mutations = [('time', float('nan')), ('calcium', float('inf')), ('analyze', 0),
                     ('analyze', True), ('animal_id', True), ('animal_id', 1.25),
                     ('trial', 'unknown'), ('roi', None), ('driver', 'wrong'),
                     ('stimulus_type', 'unknown'), ('frame', False)]
        for key, value in mutations:
            source = rows()
            source[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                adapter.group_rows(NAMES[0], source)
        for transform in ('extra', 'missing', 'reverse', 'few_animals'):
            source = rows()
            if transform == 'extra':
                source[0]['prediction'] = 1
            elif transform == 'missing':
                del source[0]['calcium']
            elif transform == 'reverse':
                source.reverse()
            else:
                source = [r for r in source if r['animal_id'] <= 2]
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                adapter.group_rows(NAMES[0], source)
        with self.assertRaises(ValueError):
            adapter.group_rows('unrecognized.parquet', rows())

    def test_global_leakage_or_duplicate_rejected(self):
        groups = adapter.group_rows(NAMES[0], rows())
        with self.assertRaises(ValueError):
            adapter.plan_packages(groups + [groups[0]])
        bad = copy.deepcopy(groups)
        bad[2]['subject_id'] = bad[0]['subject_id']
        with self.assertRaises(ValueError):
            adapter.plan_packages(bad)
        bad = copy.deepcopy(groups)
        bad[2]['raw_recording_id'] = bad[0]['raw_recording_id']
        with self.assertRaises(ValueError):
            adapter.plan_packages(bad)

    def test_contract_budget_not_relaxed(self):
        groups = adapter.group_rows(NAMES[0], rows())
        package = adapter.make_package(groups[0], groups[2], dict(uri='test:x', revision='test', sha256='unknown', asset_bytes_verified=False))
        for trial in package['trials']:
            for key in ('time_s', 'fluorescence_time_s', 'angle_rad', 'fluorescence'):
                trial[key] = list(range(12501))
        report = adapter.validate(package)
        self.assertFalse(report['contract_valid'])
        self.assertIn('aggregate point budget exceeded', [e['message'] for e in report['errors']])

    def test_import_and_help_do_not_require_arrow(self):
        code = "import sys; sys.modules['pyarrow'] = None; import tools_s4_adapt_real_calcium; assert 'numpy' not in sys.modules"
        completed = subprocess.run([sys.executable, '-c', code], cwd=adapter.PROJECT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run([sys.executable, str(Path(adapter.__file__)), '--help'], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--output-dir', completed.stdout)


if __name__ == '__main__':
    unittest.main()
