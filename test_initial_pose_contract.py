import unittest
import numpy as np
from engine.embodied.initial_pose_contract import scalar_limit_report
class PoseContractTests(unittest.TestCase):
    def test_no_silent_clip_and_separate_limits(self):
        q=np.array([.2,1.1]);old=q.copy()
        d=scalar_limit_report(['a','b'],[0,1],[3,3],[1,1],[[-1,1],[-1,1]],q)
        self.assertFalse(d['within_declared_scalar_ranges']);self.assertEqual(d['violations'][0]['joint'],'b');np.testing.assert_array_equal(q,old)
    def test_boundary_and_unlimited(self):
        self.assertTrue(scalar_limit_report(['a','b'],[0,1],[3,3],[1,0],[[-1,1],[-1,1]],[1,2])['within_declared_scalar_ranges'])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):scalar_limit_report(['a'],[0],[3],[1],[[-1,1]],[float('nan')])
if __name__=='__main__':unittest.main(verbosity=2)
