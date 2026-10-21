"""Data-free entry validation plus JSON-safe memory-report regression."""
import json
from types import SimpleNamespace
import numpy as np
import unittest
from unittest.mock import patch
import run_experiment_platform as platform
from engine import memory_guard as mg


class PlatformTests(unittest.TestCase):
    def test_duration_validation(self):
        self.assertEqual(platform.validate_duration(.05), .05)
        for value in (0, .015, .021, 1.005, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                platform.validate_duration(value)

    def fixtures(self):
        arrays = {}
        for arm in ('normal', 'normal_repeat', 'replay_baseline', 'feedback_push',
                    'replay_push', 'sensory_cut', 'motor_cut', 'output_silence'):
            arrays[arm+'_trajectory'] = np.ones((5, 2))
            arrays[arm+'_controls'] = np.ones((5, 1))
            arrays[arm+'_spike_counts'] = np.ones((5, 1))
        for arm in ('sensory_cut', 'output_silence'):
            arrays[arm+'_controls'][:] = 0
        arrays['motor_cut_controls'][3:] = 0
        return arrays, SimpleNamespace(actuator_ctrlrange=np.array([[0., 1.]])), {
            'rows': [{'local_indices': [0]}]}

    def test_checks_require_real_neural_output_and_repeatability(self):
        arrays, model, manifest = self.fixtures()
        with patch.object(platform.loop.cp, 'sha256_file', return_value=platform.loop.cp.CANONICAL_XML_SHA256):
            self.assertTrue(all(platform.engineering_checks(arrays, model, manifest).values()))
            arrays['normal_controls'][:] = 0
            self.assertFalse(platform.engineering_checks(arrays, model, manifest)['neural_commands_present'])
            arrays['normal_spike_counts'][:] = 0
            self.assertFalse(platform.engineering_checks(arrays, model, manifest)['motor_spikes_present'])
            arrays['normal_repeat_trajectory'][0, 0] += 1
            self.assertFalse(platform.engineering_checks(arrays, model, manifest)['repeat_exact'])

    def test_nonfinite_traces_rejected(self):
        arrays, model, manifest = self.fixtures()
        arrays['normal_trajectory'][0, 0] = float('nan')
        self.assertFalse(platform.engineering_checks(arrays, model, manifest)['finite_traces'])

    def test_feedback_measurements_do_not_claim_recovery(self):
        arrays, _, _ = self.fixtures()
        measured = platform.feedback_measurements(arrays)
        self.assertEqual(measured['feedback_vs_replay_command_max_difference'], 0)
        arrays['feedback_push_controls'][2, 0] += .25
        self.assertEqual(platform.feedback_measurements(arrays)['feedback_vs_replay_command_max_difference'], .25)

    def test_guard_report_unlimited_is_json_safe(self):
        with patch.object(mg, 'cap_address_space_mb', return_value=float('inf')), \
             patch.object(mg, 'require_available_mb', return_value={}):
            with mg.guarded_run('json_test', need_mb=1, cap_mb=4096, lock=False) as report:
                info = report()
                json.dumps(info, allow_nan=False)
                self.assertIsNone(info['previous_soft_limit_mb'])
                self.assertTrue(info['previous_soft_limit_unlimited'])


if __name__ == '__main__':
    unittest.main()
