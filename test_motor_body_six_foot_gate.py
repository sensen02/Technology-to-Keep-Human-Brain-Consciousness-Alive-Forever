import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as closure
import run_motor_body_integration as integration

CANDIDATE=Path('/home/sensen/Desktop/engine修复验收/六足内带显式约束-20261018T031500Z/candidate_qpos.npz')

class MotorBodySixFootGateTests(unittest.TestCase):
    def test_explicit_candidate_passes_six_foot_gate(self):
        with np.load(CANDIDATE,allow_pickle=False) as z:q=z['qpos'].copy()
        m,b,g,_,gl=closure.prepare_body(initial_qpos=q)
        integration.require_legal_initial_pose(g)
        self.assertEqual(closure.require_six_foot_initial_contacts(m,b,gl),sorted(closure.LEGS))
        np.testing.assert_array_equal(b,q)

    def test_four_foot_explicit_pose_is_rejected_before_tier_build(self):
        with np.load(CANDIDATE,allow_pickle=False) as z:q=z['qpos'].copy()
        q[2]+=.005
        m,b,g,_,gl=closure.prepare_body(initial_qpos=q)
        integration.require_legal_initial_pose(g)
        self.assertEqual(len(closure.initial_foot_contact_legs(m,b,gl)),4)
        with self.assertRaisesRegex(ValueError,'six foot-plane contacts required'):
            closure.require_six_foot_initial_contacts(m,b,gl)

if __name__=='__main__':unittest.main()
