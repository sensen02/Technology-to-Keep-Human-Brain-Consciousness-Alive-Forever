import unittest
from test_s4_paired_data_contract import synthetic_fixture as sensory_fixture
from tools_s4_motor_data_contract import validate


def synthetic_fixture():
    data = sensory_fixture()
    data.update(scope='motor_muscle', dataset_id='S4_MOTOR_SYNTHETIC_ONLY')
    for trial in data['trials']:
        trial.pop('channel')
        trial.update(muscle_bundle_id='synthetic_bundle', recruitment_evidence='constructed pairing, not anatomy',
                     pool_coverage='unknown', observation=dict(
                         kind='calcium', unit='deltaF/F', time_s=[0, .01, .02], values=[0, .1, .2],
                         measurement_model='synthetic calcium observation, not direct activation',
                         model_revision='synthetic-v1', calibration_evidence='constructed test values only',
                         clock_mapping_evidence='identity synthetic clock', max_sync_error_s=0., sensor_delay_s=0.))
    return data


class MotorContractTests(unittest.TestCase):
    def test_synthetic_pass_not_biology(self):
        result = validate(synthetic_fixture())
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertFalse(result['biological_validation'])
        self.assertFalse(result['calibration_executed'])

    def test_all_supported_modalities(self):
        for kind, unit in [('emg','V'),('calcium','deltaF/F'),('activation_estimate','1'),('force','N'),('torque','N*m')]:
            with self.subTest(kind=kind):
                data = synthetic_fixture()
                data['trials'][0]['observation'].update(kind=kind, unit=unit)
                self.assertTrue(validate(data)['contract_valid'])

    def test_rejections(self):
        mutations = {
            'wrong_scope': lambda d: d.update(scope='sensory_claw_hook'),
            'missing_model': lambda d: d['trials'][0]['observation'].pop('measurement_model'),
            'wrong_unit': lambda d: d['trials'][0]['observation'].update(unit='1'),
            'unknown_modality': lambda d: d['trials'][0]['observation'].update(kind='command'),
            'negative_delay': lambda d: d['trials'][0]['observation'].update(sensor_delay_s=-1),
            'boolean_sync': lambda d: d['trials'][0]['observation'].update(max_sync_error_s=True),
            'outside_window': lambda d: d['trials'][0]['observation'].update(time_s=[0, .01, 1]),
            'wrong_count': lambda d: d['trials'][0]['observation'].update(values=[0]),
            'nonfinite': lambda d: d['trials'][0]['observation'].update(values=[0,float('inf'),0]),
            'activation_range': lambda d: d['trials'][0]['observation'].update(kind='activation_estimate',unit='1',values=[0,2,0]),
            'missing_recruitment': lambda d: d['trials'][0].pop('recruitment_evidence'),
            'wrong_coverage': lambda d: d['trials'][0].update(pool_coverage='verified'),
            'recording_leak': lambda d: d['trials'][1].update(raw_recording_id=d['trials'][0]['raw_recording_id']),
            'float_identity': lambda d: d['trials'][0].update(cell_root_id=1.0),
            'missing_clock_evidence': lambda d: d['trials'][0]['observation'].pop('clock_mapping_evidence'),
            'malformed_obs': lambda d: d['trials'][0].update(observation=[]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = synthetic_fixture()
                mutate(data)
                self.assertFalse(validate(data)['contract_valid'])

    def test_malformed_top_level(self):
        for value in ([], None, {}, {'trials': []}, {'trials': [None,None]}):
            self.assertFalse(validate(value)['contract_valid'])

    def test_aggregate_budget(self):
        data = synthetic_fixture()
        for trial in data['trials']:
            trial['observation'].update(time_s=[i*1e-7 for i in range(26000)],values=[0]*26000)
        self.assertFalse(validate(data)['contract_valid'])


if __name__ == '__main__':
    unittest.main()
