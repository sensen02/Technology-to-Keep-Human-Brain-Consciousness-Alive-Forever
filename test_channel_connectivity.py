import unittest
import numpy as np
from scipy import sparse
from engine.embodied.channel_connectivity import channel_connectivity
class ConnectivityTests(unittest.TestCase):
    def test_post_row_pre_column_and_unreachable(self):
        w=sparse.csr_matrix(([1.,1.],([1,2],[0,1])),shape=(4,4));zero=w*0
        d=channel_connectivity(w,zero,{'lf':{'touch':[0],'isolated':[3],'empty':[]}},[2])['channels']['lf']
        self.assertEqual(d['touch']['minimum_motor_hops'],2)
        self.assertEqual(d['isolated']['status'],'NO_MODELED_MOTOR_PATH')
        self.assertEqual(d['empty']['status'],'NO_TIER_CELLS')
    def test_logical_union_not_cancellation(self):
        w=sparse.csr_matrix(([1.],([1],[0])),shape=(2,2))
        d=channel_connectivity(w,-w,{'lf':{'touch':[0,0]}},[1,1])['channels']['lf']['touch']
        self.assertEqual(d['reachable_motor_cells'],1);self.assertEqual(d['cells'],1)
    def test_invalid_indices(self):
        w=sparse.csr_matrix((2,2))
        with self.assertRaises(ValueError):channel_connectivity(w,w,{'lf':{'touch':[2]}},[1])
if __name__=='__main__':unittest.main(verbosity=2)
