"""Saved-artifact checks for the joint torque deficiency audit; no integration."""
import json
import unittest
from pathlib import Path
import numpy as np
import run_joint_torque_deficiency as run

class TorqueDeficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'torque_deficiency_results.json').read_text())
        cls.z = np.load(run.OUT / 'torque_deficiency_traces.npz')

    def test_balance_identity_verified(self):
        self.assertTrue(self.doc['balance_residual']['within_tolerance'])
        self.assertLess(self.doc['balance_residual']['max_abs'], self.doc['balance_residual']['tolerance'])

    def test_deficit_is_required_minus_actual(self):
        self.assertTrue(np.allclose(self.z['deficit'], self.z['required'] - self.z['actual'], atol=0, rtol=0))

    def test_snapshot_numbers_match_arrays(self):
        names = [str(x) for x in self.z['joint_names']]
        for ms, s in self.doc['snapshots_ms'].items():
            i = s['sample']
            self.assertEqual(s['max_required_abs'], float(np.abs(self.z['required'][i]).max()))
            self.assertEqual(s['max_actual_abs'], float(np.abs(self.z['actual'][i]).max()))
            self.assertEqual(s['max_deficit_abs'], float(np.abs(self.z['deficit'][i]).max()))
            for w in s['worst_joints']:
                j = names.index(w['joint'])
                self.assertEqual(w['deficit'], float(self.z['deficit'][i][j]))

    def test_tibia_joints_unsupported_initially_and_pattern_not_force_limited(self):
        names = [str(x) for x in self.z['joint_names']]
        req0, act0 = self.z['required'][0], self.z['actual'][0]
        for leg in ('LF', 'LH', 'RF', 'RH', 'LM'):
            j = names.index(f'joint_{leg}Tibia_pitch')
            self.assertLess(req0[j], -1.0)
            self.assertLess(abs(act0[j]), 0.05)
        d = self.doc['whole_episode']
        self.assertLessEqual(d['max_deficit_abs'], d['max_actual_abs'])
        self.assertEqual(d['joints_exceeding_weight_at_any_time'], 0)

if __name__ == '__main__':
    unittest.main()
