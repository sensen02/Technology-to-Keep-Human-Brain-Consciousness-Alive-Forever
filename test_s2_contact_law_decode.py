import unittest
import numpy as np
from tools_s2_contact_law_decode import decode_local6


class DecodeTests(unittest.TestCase):
    def test_dim1(self):
        np.testing.assert_array_equal(decode_local6([99, 2], 1, 1, []), [2,0,0,0,0,0])

    def test_dim3_positive_negative_tangential(self):
        np.testing.assert_array_equal(decode_local6([4,1,2,6],0,3,[.5,.25]),[13,1.5,-1,0,0,0])
        np.testing.assert_array_equal(decode_local6([1,4,6,2],0,3,[.5,.25]),[13,-1.5,1,0,0,0])

    def test_inactive_bounds(self):
        np.testing.assert_array_equal(decode_local6([], -1, 3, []), np.zeros(6))
        for adr in (-2, 1, 4):
            with self.assertRaises(ValueError): decode_local6([1,2,3,4],adr,3,[1,1])

    def test_friction_dimensions_finite(self):
        for mu in ([1], [[1,1]], [1,np.nan], [1,-1]):
            with self.assertRaises(ValueError): decode_local6([1,2,3,4],0,3,mu)
        with self.assertRaises(ValueError): decode_local6([1,np.inf,3,4],0,3,[1,1])
        with self.assertRaises(ValueError): decode_local6([1],0,2,[1])
        with self.assertRaises(ValueError): decode_local6([1],0,1,[],None)

    def test_dim4_dim6_and_elliptic(self):
        for dim in (4,6):
            p = np.arange(1,2*(dim-1)+1,dtype=float)
            mu = np.arange(1,dim,dtype=float)
            expect = np.zeros(6); expect[0]=p.sum(); expect[1:dim]=-mu
            np.testing.assert_array_equal(decode_local6(p,0,dim,mu),expect)
        np.testing.assert_array_equal(decode_local6([1,2,3],0,3,[1,1],1),[1,2,3,0,0,0])

    def test_independent_matrix_oracle(self):
        # Do not rely on Python binding dimension inference; explicit C formula.
        for dim in (3,4,6):
            n = 2*(dim-1)
            p = np.arange(1,n+1,dtype=float)
            mu = np.array([.1,.2,.3,.4,.5])
            matrix = np.zeros((6,n)); matrix[0,:] = 1
            for k in range(dim-1):
                matrix[k+1,2*k] = mu[k]; matrix[k+1,2*k+1] = -mu[k]
            np.testing.assert_allclose(decode_local6(p,0,dim,mu),matrix @ p,rtol=0,atol=1e-14)


if __name__ == '__main__': unittest.main()
