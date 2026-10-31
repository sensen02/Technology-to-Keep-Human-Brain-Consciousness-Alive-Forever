"""Offline finite-difference stencil semantics for S113; no simulator."""
import unittest
import numpy as np
class FDStencilTests(unittest.TestCase):
    def test_interior_first_is_centered_not_endpoint(self):
        t=np.arange(31)*1e-4;p=t**3
        fd=np.gradient(p,t,edge_order=2)
        self.assertAlmostEqual(fd[1],(p[2]-p[0])/(t[2]-t[0]),places=18)
    def test_downsample_changes_derivative_not_integrated_trajectory(self):
        t=np.arange(121)*.000025;p=t**3
        fine=np.gradient(p,t,edge_order=2)[4]
        coarse=np.gradient(p[::4],t[::4],edge_order=2)[1]
        self.assertNotEqual(float(fine),float(coarse))
        self.assertEqual(float(p[4]),float(p[::4][1]))
if __name__=='__main__':unittest.main()
