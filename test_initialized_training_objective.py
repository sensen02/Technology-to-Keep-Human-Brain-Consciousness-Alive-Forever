import unittest
import numpy as np
from initialized_training_objective import score


class FrozenProxy:
    WEIGHT_NATIVE=2.
    @staticmethod
    def proxy(a, scales):
        return dict(total=7.,parts={})


class ObjectiveTests(unittest.TestCase):
    def arrays(self):
        return dict(qpos=np.zeros((4001,49)),foot_force=np.zeros((4001,6,3)),limit_excess=np.zeros(4001))
    def test_startup_penalties_distinct_from_formal(self):
        a=self.arrays(); a['limit_excess'][:101]=.001001
        r=score(a,2.,{},FrozenProxy)
        self.assertAlmostEqual(r['formal_proxy'],7.)
        self.assertAlmostEqual(r['early_load'],1.)
        self.assertAlmostEqual(r['early_limit'],1.)
        self.assertAlmostEqual(r['total'],9.)
    def test_early_window_only(self):
        a=self.arrays(); a['foot_force'][:101,0,2]=1.6; a['limit_excess'][101:]=100
        r=score(a,2.,{},FrozenProxy)
        self.assertAlmostEqual(r['total'],7.)
    def test_boundary_half_weight(self):
        a=self.arrays(); a['foot_force'][:100,0,2]=1.6
        self.assertAlmostEqual(score(a,2.,{},FrozenProxy)['early_load'],.005)
    def test_wrong_weight_rejected(self):
        with self.assertRaises(ValueError):score(self.arrays(),3.,{},FrozenProxy)
    def test_short_trajectory_rejected(self):
        a=self.arrays();a['qpos']=a['qpos'][:101]
        with self.assertRaises(ValueError):score(a,2.,{},FrozenProxy)


if __name__=='__main__': unittest.main()
