"""Pure tests: exact initial state, no hidden pose repair or body integration."""
import unittest
from unittest.mock import patch
import numpy as np
from engine.embodied.initial_pose_contract import validate_explicit_qpos
from run_motor_body_integration import require_legal_initial_pose


class ExplicitInitialTests(unittest.TestCase):
    def test_exact_copy_and_no_mutation(self):
        q = np.zeros(49); q[3] = 1; q[8] = .123
        got = validate_explicit_qpos(q, 49)
        np.testing.assert_array_equal(got, q)
        got[8] = 2
        self.assertEqual(q[8], .123)

    def test_explicit_prepare_never_calls_legacy_pose_or_seating(self):
        from pathlib import Path
        import run_neural_muscle_closure as loop
        path = Path('/home/sensen/Desktop/engine修复验收/联合初姿约束修复-20261018T0230/slot_08_pose.npz')
        if not path.exists(): self.skipTest('optional saved research fixture not available')
        with np.load(path, allow_pickle=False) as saved:
            q = saved['qpos'].copy()
        with patch.object(loop.r39, 'apply_round37_pose', side_effect=AssertionError('legacy pose used')), \
             patch.object(loop.cp, 'seat_root', side_effect=AssertionError('seating used')):
            model, loaded, gate, _, _ = loop.prepare_body(initial_qpos=q)
        np.testing.assert_array_equal(loaded, q)
        self.assertTrue(gate['initial_scalar_joint_ranges']['within_declared_scalar_ranges'])
        self.assertTrue(gate['tarsus_only'])

    def test_rejects_bad_saved_states(self):
        q = np.zeros(49); q[3] = 1
        for invalid in (q[:48], q.astype(complex), q.astype(str), q*0, np.full(49, np.nan)):
            with self.assertRaises(ValueError): validate_explicit_qpos(invalid, 49)

    def test_hard_gate_missing_or_illegal(self):
        for gate in ({}, {'initial_scalar_joint_ranges': {'within_declared_scalar_ranges': False}, 'tarsus_only': True},
                     {'initial_scalar_joint_ranges': {'within_declared_scalar_ranges': True}, 'tarsus_only': False}):
            with self.assertRaises(ValueError): require_legal_initial_pose(gate)


if __name__ == '__main__': unittest.main()
