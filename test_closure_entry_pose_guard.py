"""The new closure CLI cannot run the historically illegal round37 pose."""
from pathlib import Path
from unittest.mock import patch
import sys
import unittest
import numpy as np
import run_neural_muscle_closure as closure

CANDIDATE=Path('/home/sensen/Desktop/engine修复验收/六足内带显式约束-20261018T031500Z/candidate_qpos.npz')


class ClosurePoseEntryGuardTests(unittest.TestCase):
    def test_missing_initial_state_stops_before_output_or_neural_build(self):
        with patch.object(sys, 'argv', ['closure']), \
             patch.object(closure, 'OUT', Path('/nonexistent/dsh-closure-test-must-not-write')), \
             patch.object(closure, 'build_neural_tier', side_effect=AssertionError('neural build started')):
            with self.assertRaisesRegex(ValueError, 'INITIAL_POSE_REJECTED'):
                closure.main()

    def test_saved_candidate_passes_entry_pose_gate_without_mutation(self):
        if not CANDIDATE.exists(): self.skipTest('optional research candidate not available')
        with np.load(CANDIDATE,allow_pickle=False) as z: q=z['qpos'].copy()
        m,base,gate,_,geom_leg=closure.prepare_body(initial_qpos=q)
        from run_motor_body_integration import require_legal_initial_pose
        require_legal_initial_pose(gate)
        np.testing.assert_array_equal(base,q)
        self.assertEqual(m.nq,len(q))
        self.assertEqual(closure.require_six_foot_initial_contacts(m,base,geom_leg), sorted(closure.LEGS))

    def test_partial_foot_contacts_fail_new_entry_but_existing_gate_allows(self):
        if not CANDIDATE.exists(): self.skipTest('optional research candidate not available')
        with np.load(CANDIDATE,allow_pickle=False) as z: q=z['qpos'].copy()
        q[2] += .005  # keep original joints and contact identities; no clipping
        m,base,gate,_,geom_leg=closure.prepare_body(initial_qpos=q)
        from run_motor_body_integration import require_legal_initial_pose
        require_legal_initial_pose(gate)  # measured old gate accepted four-foot posture
        np.testing.assert_array_equal(base,q)
        self.assertEqual(len(closure.initial_foot_contact_legs(m,base,geom_leg)),4)
        with self.assertRaisesRegex(ValueError,'six foot-plane contacts required'):
            closure.require_six_foot_initial_contacts(m,base,geom_leg)
        with patch.object(sys,'argv',['closure','--initial-state',str(CANDIDATE)]), \
             patch.object(closure,'OUT',Path('/nonexistent/dsh-closure-test-must-not-write')), \
             patch.object(closure,'build_neural_tier',side_effect=AssertionError('neural build started')), \
             patch.object(closure.np,'load') as fake_load:
            fake_load.return_value.__enter__.return_value={'qpos':q}
            with self.assertRaisesRegex(ValueError,'six foot-plane contacts required'):
                closure.main()

    def test_illegal_explicit_state_rejected_before_neural_build_or_output(self):
        if not CANDIDATE.exists(): self.skipTest('optional research candidate not available')
        with np.load(CANDIDATE,allow_pickle=False) as z: q=z['qpos'].copy()
        q[36]=.7795  # documented RH coxa pitch native upper .7375
        with patch.object(sys,'argv',['closure','--initial-state',str(CANDIDATE)]), \
             patch.object(closure,'OUT',Path('/nonexistent/dsh-closure-test-must-not-write')), \
             patch.object(closure,'build_neural_tier',side_effect=AssertionError('neural build started')), \
             patch.object(closure.np,'load') as fake_load:
            fake_load.return_value.__enter__.return_value={'qpos':q}
            with self.assertRaisesRegex(ValueError,'INITIAL_POSE_REJECTED'):
                closure.main()


if __name__=='__main__':unittest.main()
