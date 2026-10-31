import unittest
import numpy as np
from run_neural_muscle_closure import resolve_sensory_reference
class ReferenceTests(unittest.TestCase):
 def test_default_compatible_owned(self):
  q=np.arange(49,dtype=float);r=resolve_sensory_reference(q);self.assertEqual(q.tobytes(),r.tobytes());q[8]=99;self.assertEqual(r[8],8)
 def test_physical_change_never_moves_explicit_reference(self):
  ref=np.arange(49,dtype=float);q=ref.copy();a=resolve_sensory_reference(q,ref);q[8]+=.3;b=resolve_sensory_reference(q,ref);np.testing.assert_array_equal(a,b);self.assertAlmostEqual(q[8]-b[8],.3);ref[8]=999;self.assertEqual(b[8],8)
 def test_invalid(self):
  for r in (np.zeros(48),np.full(49,np.nan)):
   with self.assertRaises(ValueError):resolve_sensory_reference(np.zeros(49),r)
if __name__=='__main__':unittest.main()
