"""Pure completed-score fixtures; no model, neuron execution or physics."""
import unittest
import numpy as np
from engine.embodied.pool_gain_search import PoolGainCoordinateSearch


class PoolGainSearchTests(unittest.TestCase):
    def test_all66_both_sides_one_pass_and_hard_stop(self):
        s=PoolGainCoordinateSearch((4453,.9374))
        for i in range(66):
            before=s.gains;plus,minus=s.probes()
            for sign,g in ((1,plus),(-1,minus)):
                expected=before.copy();expected[i]+=sign*.5
                np.testing.assert_array_equal(g,expected)
                s.record(g,(4453,.9374),f'{i}_{sign}')
            s.update();self.assertEqual(s.coordinate,i+1)
            np.testing.assert_array_equal(s.gains,np.ones(66))
        self.assertEqual(s.label,'original_cached_baseline')
        with self.assertRaises(ValueError):s.probes()
        with self.assertRaises(ValueError):s.update()

    def test_strict_primary_secondary_tie_and_same_center(self):
        s=PoolGainCoordinateSearch((10,1.));p,m=s.probes()
        s.record(p,(11,.1),'p');s.record(m,(11,.1),'m');s.update()
        self.assertEqual(s.label,'p');self.assertEqual(s.gains[0],1.5)
        p,m=s.probes();self.assertEqual(p[0],1.5);self.assertEqual(m[0],1.5)
        s.record(p,(11,.2),'p2');s.record(m,(10,100.),'m2');s.update()
        self.assertEqual(s.label,'p2');self.assertEqual(s.gains[1],1.5)

    def test_missing_probe_no_partial_update_and_duplicate_cache(self):
        s=PoolGainCoordinateSearch((10,1.));p,m=s.probes()
        s.record(p,(20,1.),'p')
        with self.assertRaises(ValueError):s.update()
        self.assertEqual(s.score,(10,1.));self.assertEqual(s.coordinate,0)
        with self.assertRaises(ValueError):s.record(p,(30,1.),'again')
        self.assertEqual(s.cached(p),((20,1.),'p'))
        s.record(m,(10,2.),'m');s.update();self.assertEqual(s.score,(20,1.))

    def test_reject_unmeasured_direction_bad_scores_and_mutable_aliases(self):
        s=PoolGainCoordinateSearch((10,1.));p,m=s.probes();bad=p.copy();bad[1]=1.5
        with self.assertRaises(ValueError):s.record(bad,(20,1.),'bad')
        for score in ((1.5,1.),(True,1.),(-1,1.),(88001,1.),(10,np.nan),(10,-1.)):
            with self.assertRaises(ValueError):PoolGainCoordinateSearch(score)
        exposed=s.gains;exposed[:]=0;np.testing.assert_array_equal(s.gains,np.ones(66))
        p[:]=0;np.testing.assert_array_equal(s.probes()[0][1:],np.ones(65))
        with self.assertRaises(ValueError):s.cached(np.full(66,2.1))


if __name__=='__main__':unittest.main()
