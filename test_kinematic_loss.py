"""Semantic tests for research loss masks, frame gauges and time boundaries."""
import unittest
import numpy as np
from engine.embodied.kinematic_loss import masked_cartesian_huber,segment_direction_loss,masked_velocity_huber,geometric_interior_angles,interior_angle_loss
class Tests(unittest.TestCase):
 def setUp(self):
  self.p=np.array([[[[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]]],[[[1.,0.,0.],[2.,0.,0.],[2.,1.,0.]]]])
  self.m=np.ones(self.p.shape[:-1],bool)
 def test_missing_not_success(self):
  p=np.full_like(self.p,np.nan);m=np.zeros_like(self.m)
  for result in [masked_cartesian_huber(p,p,m,normalizer=1.),segment_direction_loss(p,p,m)]:self.assertFalse(result['evaluable']);self.assertIsNone(result['loss'])
 def test_bad_prediction_not_silently_masked(self):
  p=self.p.copy();p[0,0,1,0]=np.nan
  with self.assertRaises(ValueError):masked_cartesian_huber(p,self.p,self.m,normalizer=1.)
 def test_direction_gauge_and_angle(self):
  r=segment_direction_loss(self.p*3+np.array([2,4,8]),self.p,self.m);self.assertEqual(r['loss'],0.)
  turned=self.p@np.array([[0,1,0],[-1,0,0],[0,0,1.]])
  self.assertAlmostEqual(segment_direction_loss(turned,self.p,self.m)['loss'],(np.pi/2)**2)
 def test_weighted_cartesian(self):
  q=self.p.copy();q[:,0,0,0]+=2
  out=masked_cartesian_huber(self.p,q,self.m,normalizer=2.,delta=1.,weights=[1.,0.,0.]);self.assertEqual(out['loss'],.5)
 def test_degenerate_bone_rejected(self):
  p=self.p.copy();p[:,:,1]=p[:,:,0]
  with self.assertRaises(ValueError):segment_direction_loss(p,self.p,self.m)
 def test_time_reset_only_between_trials(self):
  p=np.concatenate([self.p,self.p],axis=0);m=np.ones(p.shape[:-1],bool)
  out=masked_velocity_huber(p,p,m,[0,.01,0,.01],normalizer_per_second=1.,max_gap_s=.02,trial_ids=['a','a','b','b'])
  self.assertEqual(out['eligible_time_intervals'],2);self.assertEqual(out['loss'],0.)
  with self.assertRaises(ValueError):masked_velocity_huber(p,p,m,[0,.01,0,.01],normalizer_per_second=1.,max_gap_s=.02)
 def test_gap_exclusion_and_no_zeros_success(self):
  out=masked_velocity_huber(self.p,self.p,self.m,[0,1],normalizer_per_second=1.,max_gap_s=.02)
  self.assertFalse(out['evaluable']);self.assertEqual(out['excluded_gap_or_trial_intervals'],1)
 def test_no_valid_endpoint_pair(self):
  m=np.zeros_like(self.m);m[0]=True
  out=masked_velocity_huber(self.p,self.p,m,[0,.01],normalizer_per_second=1.,max_gap_s=.02)
  self.assertFalse(out['evaluable'])
 def test_interior_invariant_under_frame_and_scale(self):
  rotation=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
  r=interior_angle_loss(self.p@rotation*3+np.array([2.,-1,7]),self.p,self.m)
  self.assertEqual(r['loss'],0.)
  angle,mask=geometric_interior_angles(self.p,self.m);self.assertTrue(mask.all());np.testing.assert_allclose(angle,np.pi/2)
 def test_interior_detects_bending_change(self):
  straight=self.p.copy();straight[:,:,2]=2*self.p[:,:,1]-self.p[:,:,0]
  self.assertAlmostEqual(interior_angle_loss(straight,self.p,self.m)['loss'],(np.pi/2)**2)
 def test_interior_invalid_triplet_not_zero_success(self):
  m=self.m.copy();m[:,:,1]=False
  result=interior_angle_loss(self.p,self.p,m);self.assertFalse(result['evaluable']);self.assertIsNone(result['loss'])
 def test_interior_rejects_degenerate_or_bad_prediction(self):
  p=self.p.copy();p[:,:,1]=p[:,:,0]
  with self.assertRaises(ValueError):interior_angle_loss(p,self.p,self.m)
  p=self.p.copy();p[0,0,0,0]=np.nan
  with self.assertRaises(ValueError):interior_angle_loss(p,self.p,self.m)
if __name__=='__main__':unittest.main()
