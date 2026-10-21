import unittest
import numpy as np
from tools_s2_contact_state_audit import classify


class StateAuditTests(unittest.TestCase):
    def test_stale_labels_and_boundary(self):
        r,s,error,e = classify(np.eye(3), [-2,0,3], [0]*3, [2]*3, [4,0,0], [1]*3)
        np.testing.assert_array_equal(s,[1,0,0])
        np.testing.assert_array_equal(error,[0,0,0])
        np.testing.assert_array_equal(e,[0,0,6])

    def test_scalar_oracle(self):
        J=np.array([[1.,2.],[-2.,3.]])
        q=np.array([3.,-4.]);D=np.array([.2,.4]);a=np.array([2.,-1.])
        r=np.array([sum(float(v)*float(w) for v,w in zip(row,q))-x for row,x in zip(J,a)])
        f=np.array([-d*x if x<0 else 0 for d,x in zip(D,r)])
        actual=classify(J,q,a,D,f,[0,1])
        np.testing.assert_allclose(actual[0],r,rtol=0,atol=0)
        np.testing.assert_allclose(actual[2],0,rtol=0,atol=0)

    def test_reject_bad_inputs(self):
        for J,q,a,D,f,s in [([1],[1],[0],[1],[0],[0]), ([[1]],[1],[0],[0],[0],[0]), ([[1]],[float('nan')],[0],[1],[0],[0]), ([[1]],[1],[0],[1],[0],[])]:
            with self.assertRaises(ValueError): classify(J,q,a,D,f,s)


if __name__=='__main__': unittest.main()
