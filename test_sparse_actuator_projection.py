"""Synthetic storage/projection tests, no MuJoCo imports or model loads."""
import unittest
import numpy as np


def independent_decode(raw, rownnz, rowadr, colind, nv):
    out=np.zeros((len(rownnz),nv))
    for i,(start,n) in enumerate(zip(rowadr,rownnz)):
        cols=colind[start:start+n]
        if len(set(cols.tolist()))!=n: raise ValueError('duplicate columns')
        out[i,cols]=raw[start:start+n]
    return out


class ProjectionTests(unittest.TestCase):
    def test_noncontiguous_sparse_rows_signed_projection(self):
        # Sentinel unused slots make blind reshape / contiguous assumptions fail.
        raw=np.array([777.,2.,-3.,888.,4.,999.]); cols=np.array([0,2,0,0,1,0])
        dense=independent_decode(raw,np.array([2,1]),np.array([1,4]),cols,3)
        np.testing.assert_array_equal(dense,[[ -3,0,2],[0,4,0]])
        np.testing.assert_array_equal(dense.T@[-2.,3.],[6,12,-4])
    def test_positive_coordinate_power_identity(self):
        m=np.array([[-.2,0,.8],[0,-.5,.1]]); f=np.array([-12.,-7.]);v=np.array([3.,-2.,4.])
        self.assertAlmostEqual(float(v@(m.T@f)),float((m@v)@f))
    def test_duplicate_column_rejected(self):
        with self.assertRaises(ValueError):independent_decode(np.ones(2),np.array([2]),np.array([0]),np.array([1,1]),3)


if __name__=='__main__':unittest.main()
