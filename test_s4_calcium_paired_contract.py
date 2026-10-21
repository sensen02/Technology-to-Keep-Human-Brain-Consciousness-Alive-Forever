"""Engineering-only adversarial tests; fixture is NOT animal data."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import tools_s4_calcium_paired_contract as c


class CalciumContractTests(unittest.TestCase):
    def reject(self, data):
        r = c.validate(data)
        self.assertFalse(r['contract_valid'], r)
        self.assertFalse(r['calibration_ready'])
        self.assertFalse(r['biological_validation'])
        return r

    def test_fixture_preserves_population_and_origin(self):
        d = c.synthetic_fixture()
        before = copy.deepcopy(d)
        r = c.validate(d)
        self.assertTrue(r['contract_valid'], r['errors'])
        self.assertEqual(d, before)
        self.assertEqual(r['roi_records'][0]['identity_level'], 'roi_population')
        self.assertEqual(r['roi_records'][0]['response_origin'], 'predicted')
        self.assertEqual(r['data_kind'], 'synthetic_fixture')
        self.assertFalse(r['calibration_ready'])
        self.assertFalse(r['biological_validation'])
        self.assertTrue(r['gaps'])
        self.assertNotIn('cell_root_id', json.dumps(d))

    def test_unknown_metadata_admitted_with_explicit_gaps(self):
        d = c.synthetic_fixture()
        d['fluorescence_metric'] = 'unknown'
        for t in d['trials']:
            t['identity_level'] = 'unknown'
            t['roi_identity'] = 'unknown'
            t['response_origin'] = 'unknown'
            t['processing'] = {k: 'unknown' for k in c.PROCESS_KEYS}
            t['sync'] = {k: 'unknown' for k in c.SYNC_KEYS}
        r = c.validate(d)
        self.assertTrue(r['contract_valid'], r['errors'])
        self.assertGreaterEqual(len(r['gaps']), 30)
        self.assertEqual(r['roi_records'][0]['roi_identity'], 'unknown')

    def test_measured_is_still_not_validation(self):
        d = c.synthetic_fixture()
        d['data_kind'] = 'biological'
        for t in d['trials']:
            t['response_origin'] = 'measured'
        r = c.validate(d)
        self.assertTrue(r['contract_valid'])
        self.assertFalse(r['calibration_ready'])
        self.assertFalse(r['biological_validation'])

    def test_malformed_fields_29_cases(self):
        mutations = [
            ('schema', True), ('data_kind', 'synthetic'), ('dataset_id', 'unknown'),
            ('reporter', None), ('fluorescence_metric', []), ('license', ''),
            ('species', ' '), ('split_policy', {}), ('units', {'time': 'ms', 'angle': 'rad', 'fluorescence': 'a.u.'}),
            ('source', []), ('trials', []), ('trials', [None, None]),
        ]
        trial_mutations = [('identity_level', []), ('response_origin', {}), ('roi_identity', 123),
            ('subject_id', 'unknown'), ('trial_id', ' trial'), ('roi_id', 3),
            ('raw_recording_id', False), ('split', []), ('leg', None), ('joint', ''),
            ('angle_convention', None), ('sync', []), ('processing', None),
            ('time_s', True), ('angle_rad', '0,1'), ('fluorescence', []), ('fluorescence_time_s', {})]
        self.assertEqual(len(mutations)+len(trial_mutations), 29)
        for k, v in mutations:
            with self.subTest(top=k, value=v):
                d = c.synthetic_fixture(); d[k] = v; self.reject(d)
        for k, v in trial_mutations:
            with self.subTest(trial=k, value=v):
                d = c.synthetic_fixture(); d['trials'][0][k] = v; self.reject(d)

    def test_non_objects(self):
        for value in (None, False, 3, 'x', []):
            with self.subTest(value=value):
                self.reject(value)

    def test_spike_root_and_unknown_fields_rejected(self):
        for k, v in (('cell_root_id', '1'), ('banc_mapping', {'root_id': '1'}),
                     ('spike_time_s', [0.1]), ('modality', 'spike'), ('extra', 'x')):
            with self.subTest(k=k):
                d = c.synthetic_fixture(); d['trials'][0][k] = v; self.reject(d)
        d = c.synthetic_fixture(); d['calibration_ready'] = True; self.reject(d)

    def test_nonfinite_bool_and_precision_8_cases(self):
        for v in (True, False, float('nan'), float('inf'), -float('inf'), 2**53+1, 10**400, '0.5'):
            with self.subTest(v=v):
                d = c.synthetic_fixture(); d['trials'][0]['fluorescence'][1] = v; self.reject(d)

    def test_timing_and_counts_9_cases(self):
        cases = [('time_s', [0.0, 0.0, 1.0]), ('time_s', [1, 0.5, 0]),
                 ('time_s', [1e16, 1e16+1, 1e16+2]), ('fluorescence_time_s', [0, 1, 0.5]),
                 ('fluorescence_time_s', [0, 0.4, 1]), ('angle_rad', [0, 1]),
                 ('fluorescence', [0, 1]), ('time_s', [0]), ('fluorescence_time_s', [0])]
        for k, v in cases:
            with self.subTest(k=k, v=v):
                d = c.synthetic_fixture(); d['trials'][0][k] = v; self.reject(d)

    def test_sync_numeric_8_cases(self):
        for k, v in [('fluorescence_scale', 0), ('fluorescence_scale', -1),
                     ('max_error_s', -1), ('max_error_s', True), ('fluorescence_offset_s', []),
                     ('fluorescence_offset_s', 0.1), ('fluorescence_scale', 1e308),
                     ('fluorescence_offset_s', 1e16)]:
            with self.subTest(k=k, v=v):
                d = c.synthetic_fixture(); d['trials'][0]['sync'][k] = v; self.reject(d)

    def test_exact_known_clock_conversion(self):
        d = c.synthetic_fixture()
        t = d['trials'][0]
        t['fluorescence_time_s'] = [1, 2, 3]
        t['sync'].update(fluorescence_scale=0.5, fluorescence_offset_s=-0.5)
        self.assertTrue(c.validate(d)['contract_valid'])

    def test_recording_split_and_subject_identity(self):
        d = c.synthetic_fixture()
        d['trials'][1]['raw_recording_id'] = d['trials'][0]['raw_recording_id']
        self.reject(d)
        d = c.synthetic_fixture()
        t = copy.deepcopy(d['trials'][0]); t['subject_id'] = 'other'; t['trial_id'] = 'other'
        d['trials'].append(t); self.reject(d)

    def test_trial_recording_and_duplicate_identity(self):
        d = c.synthetic_fixture()
        d['trials'].append(copy.deepcopy(d['trials'][0])); self.reject(d)
        d = c.synthetic_fixture()
        t = copy.deepcopy(d['trials'][0]); t['raw_recording_id'] = 'other'; t['roi_id'] = 'other'
        d['trials'].append(t); self.reject(d)

    def test_population_multi_roi_and_consistency(self):
        d = c.synthetic_fixture()
        t = copy.deepcopy(d['trials'][0]); t['roi_id'] = 'second_roi'
        d['trials'].append(t)
        self.assertTrue(c.validate(d)['contract_valid'])
        t = copy.deepcopy(d['trials'][0]); t['trial_id'] = 'another'; t['roi_identity'] = 'different'
        d['trials'].append(t); self.reject(d)

    def test_bounds(self):
        with patch.object(c, 'MAX_POINTS', 23):
            self.reject(c.synthetic_fixture())  # 24 scalar samples aggregate
        with patch.object(c, 'MAX_TRIALS', 1):
            self.reject(c.synthetic_fixture())
        d = c.synthetic_fixture(); d['reporter'] = 'x'*(c.MAX_TEXT+1); self.reject(d)

    def test_loader_duplicate_keys_constants_overflow(self):
        cases = [b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}']
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'input.json'
            for raw in cases:
                with self.subTest(raw=raw):
                    p.write_bytes(raw)
                    with self.assertRaises(ValueError): c.load(p)
            p.write_bytes(b'{"x":1e999}')
            data, digest = c.load(p)
            self.assertEqual(digest, hashlib.sha256(p.read_bytes()).hexdigest())
            d = c.synthetic_fixture(); d['trials'][0]['fluorescence'][1] = data['x']; self.reject(d)

    def test_loader_byte_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'input.json'; p.write_bytes(b' '*33)
            with patch('tools_s4_paired_data_contract.MAX_BYTES', 32):
                with self.assertRaisesRegex(ValueError, 'byte budget'): c.load(p)

    def test_cli_pass_reject_duplicate_and_no_overwrite(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/'input.json', Path(tmp)/'report.json'
            inp.write_text(json.dumps(c.synthetic_fixture()))
            cmd = [sys.executable, str(Path(c.__file__).resolve()), str(inp), '--output', str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            r = json.loads(out.read_text()); self.assertFalse(r['calibration_ready'])
            saved = out.read_bytes()
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0); self.assertEqual(out.read_bytes(), saved)
            out.unlink(); inp.write_text('{"schema":1,"schema":2}')
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            r = json.loads(out.read_text()); self.assertFalse(r['contract_valid'])
            self.assertIn('duplicate JSON key', r['errors'][0]['message'])


if __name__ == '__main__':
    unittest.main()
