import unittest
import numpy as np
from engine.embodied.pool_gain_quarter_search import PoolGainQuarterSearch
from engine.embodied.pool_gain_search import PoolGainCoordinateSearch

class QuarterTests(unittest.TestCase):
    def test_step_and_historical_default(self):
        q=PoolGainQuarterSearch((4453,.9));p,m=q.probes()
        self.assertEqual(p[0],1.25);self.assertEqual(m[0],.75)
        self.assertEqual(PoolGainCoordinateSearch((4453,.9)).probes()[0][0],1.5)
        np.testing.assert_array_equal(p[1:],np.ones(65))
    def test_pair_same_center_and_exact_ties(self):
        q=PoolGainQuarterSearch((4453,.9));p,m=q.probes()
        q.record(p,(4454,.8),'plus')
        with self.assertRaises(ValueError):q.update()
        np.testing.assert_array_equal(q.gains,np.ones(66))
        q.record(m,(4454,.8),'minus');q.update()
        self.assertEqual(q.label,'plus');self.assertEqual(q.gains[0],1.25)
    def test_once_all66_no_dense_selection(self):
        q=PoolGainQuarterSearch((4453,.9))
        for k in range(66):
            self.assertEqual(q.coordinate,k)
            for s,g in zip(('plus','minus'),q.probes()):q.record(g,(4453,.9),f'{k}_{s}')
            q.update()
        self.assertEqual(q.label,'original_cached_baseline')
        with self.assertRaises(ValueError):q.probes()
        np.testing.assert_array_equal(q.gains,np.ones(66))
if __name__=='__main__':unittest.main()
