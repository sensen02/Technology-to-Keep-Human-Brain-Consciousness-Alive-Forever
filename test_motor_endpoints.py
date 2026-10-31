import unittest
import numpy as np
from engine.embodied.motor_endpoints import retain_motor_endpoints
class Endpoints(unittest.TestCase):
 def setUp(self):
  self.d=dict(root_ids=np.array([11,22,33]),ann_super_class=np.array(['motor','motor','sensory']),pre=np.array([2,1]),post=np.array([0,0]))
  self.t=dict(global_index=np.array([2,0]))
 def test_append_preserves_order_and_identity(self):
  r=retain_motor_endpoints(self.d,self.t,['22']);np.testing.assert_array_equal(r['global_index'],[2,0,1]);np.testing.assert_array_equal(r['local_of_global'],[1,2,0]);self.assertEqual(r['added_root_ids'],['22'])
 def test_existing_idempotent(self):self.assertEqual(retain_motor_endpoints(self.d,self.t,['11'])['n'],2)
 def test_reject_bad_target_and_cap(self):
  for ids in (['44'],['33'],['22','22']):
   with self.assertRaises(ValueError):retain_motor_endpoints(self.d,self.t,ids)
  with self.assertRaises(ValueError):retain_motor_endpoints(self.d,self.t,['22'],max_neurons=2)
if __name__=='__main__':unittest.main()
