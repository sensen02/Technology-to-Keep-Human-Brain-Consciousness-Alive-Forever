import unittest
import numpy as np
from tools_s2_dynamic_residual import entries,scalar,stats

class ResidualTests(unittest.TestCase):
    def test_dense_sparse_padded(self):
        J=np.array([[2.,0.,-3.],[0.,4.,0.]])
        dense=entries(J.ravel(),3,2,nJ=6)
        sparse=entries([99,2,-3,99,4,99],3,2,True,[2,1],[1,4],[99,0,2,99,1,99],3)
        for rows in (dense,sparse):
            z=scalar(rows,[1,2,3],[2,3,4],[1,-2],[2,3],[16,-30],[1,1],J.T@np.array([16,-30]),J@np.array([2,3,4])-np.array([1,-2]))
            np.testing.assert_array_equal(z['r'],[-8,10]);np.testing.assert_array_equal(z['e'],[0,0]);np.testing.assert_array_equal(z['mapping'],[0,0,0]);np.testing.assert_array_equal(z['b_error'],[0,0])
    def test_zero_denominator(self):
        s=stats([0.,1.],[0.,2.]);self.assertEqual(s['relative_undefined'],1);self.assertEqual(s['relative_absmax'],.5)
        self.assertIsNone(stats([0.],[0.])['relative_absmax'])
    def test_nonquadratic_coverage(self):
        z=scalar(entries([1,1,1,1],1,4),[0],[0],[0]*4,[1]*4,[0,7,8,9],[1,0,2,4],[24],[0]*4)
        self.assertEqual(np.sum(z['quadratic']),1);self.assertEqual(np.sum(~z['quadratic']),3)
        self.assertEqual(stats(z['e'][z['quadratic']])['absmax'],0)
    def test_invalid_layout(self):
        for args in (([1],2,1),([1,2],2,1,True,[2],[0],[0,0],2),([1],2,1,True,[1],[1],[0],1),([1],2,1,True,[1],[0],[2],1)):
            with self.assertRaises(ValueError): entries(*args)

if __name__=='__main__':unittest.main()
