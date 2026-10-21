import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools_s4_paired_data_contract import validate, load


def synthetic_fixture():
    # Arbitrary illustrative numbers, NOT animal data or a verified connectome mapping.
    trials = []
    for i, split in enumerate(('train', 'holdout')):
        trials.append(dict(trial_id=f'synthetic_trial_{i}', subject_id='synthetic_subject',
                           raw_recording_id=f'synthetic_recording_{i}', joint='synthetic_tibia',
                           channel='claw_position', leg='lf', cell_root_id='1', split=split,
                           sync=dict(common_clock_id='synthetic_clock', method='constructed identity transform',
                                     error_evidence='synthetic zero error by construction, not measured',
                                     missing_data_policy='no missing samples by construction',
                                     neural_offset_s=0.0, neural_scale=1.0, max_error_s=0.0),
                           time_s=[0.0, .01, .02], angle_rad=[-.1, 0., .1],
                           velocity_rad_s=[10., 10., 10.], spike_time_neural_s=[.005, .015]))
    return dict(schema_version=1, data_kind='synthetic', dataset_id='S4_SYNTHETIC_CONTRACT_ONLY',
                scope='sensory_claw_hook', species='synthetic (no animal)', license='synthetic test fixture',
                preprocessing='constructed arrays; no measurement or parameter inference',
                split_policy='independent synthetic recording IDs; not biological holdout validation',
                units=dict(time='s', angle='rad', velocity='rad/s', events='spike'),
                source=dict(uri='synthetic://constructed-fixture', revision='fixture-v1', sha256='0'*64,
                            identity_evidence='placeholder root 1, not a BANC identity claim',
                            coordinate_convention='synthetic signed tibia angle'), trials=trials)


class ContractTests(unittest.TestCase):
    def test_synthetic_admitted_not_biology(self):
        r = validate(synthetic_fixture())
        self.assertTrue(r['contract_valid'], r['errors'])
        self.assertFalse(r['biological_validation'])
        self.assertFalse(r['calibration_executed'])

    def reject_mutation(self, mutate):
        data = synthetic_fixture()
        mutate(data)
        self.assertFalse(validate(data)['contract_valid'])

    def test_invalid_variants(self):
        variants = {
            'wrong_units': lambda d: d['units'].update(angle='deg'),
            'float_root': lambda d: d['trials'][0].update(cell_root_id=720575941458992031.0),
            'integer_root': lambda d: d['trials'][0].update(cell_root_id=1),
            'nonfinite': lambda d: d['trials'][0].update(angle_rad=[0, float('nan'), 1]),
            'boolean': lambda d: d['trials'][0]['sync'].update(neural_scale=True),
            'time_order': lambda d: d['trials'][0].update(time_s=[0, .02, .01]),
            'mismatched_samples': lambda d: d['trials'][0].update(angle_rad=[0]),
            'duplicate_events': lambda d: d['trials'][0].update(spike_time_neural_s=[.005, .005]),
            'outside_window': lambda d: d['trials'][0]['sync'].update(neural_offset_s=1),
            'negative_error': lambda d: d['trials'][0]['sync'].update(max_error_s=-1),
            'missing_evidence': lambda d: d['source'].pop('identity_evidence'),
            'missing_holdout': lambda d: d['trials'][1].update(split='train'),
            'recording_leak': lambda d: d['trials'][1].update(raw_recording_id='synthetic_recording_0'),
            'recording_leak_changed_subject': lambda d: d['trials'][1].update(raw_recording_id='synthetic_recording_0', subject_id='another_subject'),
            'duplicate_trial': lambda d: d['trials'][1].update(trial_id='synthetic_trial_0'),
            'unknown_channel': lambda d: d['trials'][0].update(channel='bristle_touch'),
            'malformed_trial': lambda d: d['trials'].__setitem__(0, []),
        }
        for name, mutation in variants.items():
            with self.subTest(name=name):
                self.reject_mutation(mutation)

    def test_declared_biological_still_not_validated(self):
        d = synthetic_fixture()
        d['data_kind'] = 'biological'
        result = validate(d)
        self.assertTrue(result['contract_valid'])
        self.assertFalse(result['biological_validation'])

    def test_large_decimal_root_preserved(self):
        d = synthetic_fixture()
        d['trials'][0]['cell_root_id'] = '720575941458992031'
        self.assertTrue(validate(d)['contract_valid'])

    def test_json_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'input.json'
            for content in ('{"x":1,"x":2}', '{"x":NaN}'):
                p.write_text(content)
                with self.assertRaises(ValueError):
                    load(p)

    def test_nonobject_and_absent_fields(self):
        for d in (None, [], {}, {'trials': []}):
            self.assertFalse(validate(d)['contract_valid'])


if __name__ == '__main__':
    unittest.main()
