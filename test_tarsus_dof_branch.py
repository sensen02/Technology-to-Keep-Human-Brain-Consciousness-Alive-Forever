"""Saved-artifact checks for the tarsus-DOF closed-loop branch."""
import json
import unittest
import numpy as np
import run_tarsus_dof_branch as run

class TarsusDofBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'tarsus_dof_results.json').read_text())
        cls.z = np.load(run.OUT / 'tarsus_dof_traces.npz')

    def test_extension_is_connected_to_real_pools(self):
        e = self.doc['extension']
        self.assertEqual(len(e['new_actuators']), 12)
        self.assertEqual(len(e['tarsus_joint_angles']), 6)
        for actuator, detail in e['pool_detail'].items():
            self.assertGreater(detail['cells'], 0, actuator)
            exported = json.loads((run.OUT / 'tarsus_motor_pools.json').read_text())['totals']
            self.assertEqual(detail['cells'], exported[actuator])
        self.assertTrue(self.doc['checks']['new_actuators_have_pools'])
        self.assertGreater(e['tarsus_command_max'], 0.0)

    def test_controls_hold(self):
        for key in ('replay_exact', 'silence_zero_drive', 'all_finite', 'sources_unchanged'):
            self.assertTrue(self.doc['checks'][key], key)

    def test_same_clock_and_pose_protocol_as_baseline(self):
        self.assertEqual(self.doc['episodes']['tarsus_normal']['actual_neural_steps'], 400)
        for key in ('tarsus_normal', 'tarsus_replay', 'tarsus_spike_output_silence'):
            self.assertEqual(len(self.doc['episodes'][key]['command_valid_per_5ms']), 40)
            self.assertTrue(all(self.doc['episodes'][key]['command_valid_per_5ms']), key)

    def test_negative_outcome_is_reported_not_hidden(self):
        b, base = self.doc['branch'], self.doc['published_25us_baseline']
        self.assertLess(b['escape_time_s'], base['escape_time_s'])
        self.assertLess(self.doc['comparison']['escape_time_delta_s'], 0)
        self.assertLess(b['endpoint_heights_mm'][2], base['endpoint_heights_mm'][2])
        self.assertIn('worse', self.doc['limitations'][2])

    def test_no_behaviour_claim_in_label(self):
        self.assertIn('not a cure', self.doc['label'])
        self.assertIn('not walking', self.doc['label'])

if __name__ == '__main__':
    unittest.main()
