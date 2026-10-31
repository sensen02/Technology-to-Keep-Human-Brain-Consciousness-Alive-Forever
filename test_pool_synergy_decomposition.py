"""Fixed-reference torque difference algebra, not an intervention simulator."""
import unittest
import numpy as np


class DecompositionTests(unittest.TestCase):
    def test_force_and_geometry_terms_recover_actual_difference(self):
        m0=np.array([.016,.017,.018]);m1=np.array([.015,.018,.019])
        f0=np.array([-1.,-4.,-2.]);f1=np.array([-2.,-5.,-1.])
        delta=f1*m1-f0*m0
        fixed_reference_force=(f1-f0)*m0
        changed_geometry=f1*(m1-m0)
        np.testing.assert_allclose(delta,fixed_reference_force+changed_geometry,atol=1e-16,rtol=0)
    def test_force_alone_cannot_explain_torque_when_geometry_changes(self):
        m0=.02;m1=.01;f0=-5.;f1=-6.
        self.assertLess((f1-f0)*m0,0)
        self.assertGreater(f1*m1-f0*m0,0)
    def test_no_generalized_dof_sum(self):
        m=np.array([[.02,-.01,.03],[.01,.04,-.02]])
        f=np.array([-5.,-3.]);actual=m*f[:,None]
        np.testing.assert_allclose(actual.sum(0),m.T@f,atol=1e-16,rtol=0)


if __name__=='__main__':unittest.main()
