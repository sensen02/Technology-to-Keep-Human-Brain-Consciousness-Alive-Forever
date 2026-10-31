import unittest
import numpy as np
from engine.embodied.command_tier import downstream_union

class TierTests(unittest.TestCase):
    def setUp(self):
        self.d={'root_ids':np.array([11,22,33,44,55]),'pre':np.array([0,1,2,3]),
                'post':np.array([1,2,3,4]),'ann_nt_verified':np.array(['acetylcholine','gaba','unknown','acetylcholine','gaba'])}
    def test_real_edges_and_unchanged_base(self):
        base=np.array([4]);before=base.copy()
        t=downstream_union(self.d,base,['11'],hops=2)
        np.testing.assert_array_equal(t['global_index'],[0,1,2,4])
        np.testing.assert_array_equal(base,before)
        np.testing.assert_array_equal(t['inside_rows'],[1,1,0,0])
        self.assertEqual(t['local_of_global'][3],-1)
        t=downstream_union(self.d,base,['11'],hops=4)
        np.testing.assert_array_equal(t['global_index'],[0,1,2,4])
    def test_fail_closed(self):
        for opts in ({'target_root_ids':['99']},{'target_root_ids':['11','11']},
                     {'max_neurons':2},{'hops':True},{'hops':5}):
            kw=dict(target_root_ids=['11'],hops=2,max_neurons=10);kw.update(opts)
            with self.assertRaises(ValueError):downstream_union(self.d,np.array([4]),**kw)
    def test_bad_indices(self):
        for base in (np.array([-1]),np.array([5]),np.array([1,1]),np.array([1.])):
            with self.assertRaises(ValueError):downstream_union(self.d,base,['11'])
if __name__=='__main__':unittest.main()
