"""Frozen identity checks; no neural or physical simulation."""
import json
from pathlib import Path
import tempfile
import unittest

from engine.embodied.motor_cadence_identity import require_frozen_inputs


ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BATCH = Path('/run/media/sensen/Data2/engine修复验收/pool-gain-quarter-resume2-20261018T164139Z')
POSE = Path('/home/sensen/Desktop/engine修复验收/六足内带显式约束-20261018T031500Z/candidate_qpos.npz')


class FrozenIdentityTests(unittest.TestCase):
    def test_actual_frozen_weight_and_pose_but_changed_entry(self):
        result = require_frozen_inputs(BATCH/'baseline_weights.json', POSE,
                                       BATCH/'source_hashes_before.json',
                                       ROOT/'run_neural_muscle_closure.py')
        self.assertTrue(result['fresh_5ms_baseline_required'])
        self.assertFalse(result['historical_entry_equal'])
        self.assertEqual(result['historical_entry_sha256'],
                         'a15ed4077279cd43c5a3ca2d05d6eaafa51a4c63518f3673d8200695c24ce1fc')

    def test_rejects_wrong_weight_and_initial_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'bad'; bad.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'weight identity mismatch'):
                require_frozen_inputs(bad, POSE, BATCH/'source_hashes_before.json',
                                      ROOT/'run_neural_muscle_closure.py')
            with self.assertRaisesRegex(ValueError,'initial pose identity mismatch'):
                require_frozen_inputs(BATCH/'baseline_weights.json',bad,
                                      BATCH/'source_hashes_before.json',
                                      ROOT/'run_neural_muscle_closure.py')


if __name__=='__main__':
    unittest.main()
