"""Semantic tests: motion cannot buy off nonfoot support and failed spans stay failed."""
import unittest
import numpy as np
from engine.embodied.support_priority import evaluate_support,select_support_first
class Tests(unittest.TestCase):
 def evaluate(self,nf=None,foot=1.,assistance=None):
  return evaluate_support([.01,.02,.03],np.full((3,6),foot/6),[0,0,0] if nf is None else nf,[1,1,1],weight=1,h0=1,horizon_s=.03,assistance=assistance)
 def test_clean_support_is_censored_not_invented_event(self):
  r=self.evaluate();self.assertTrue(r['clean_horizon_feasible']);self.assertIsNone(r['first_nonfoot_s']);self.assertEqual(r['observed_clean_lower_bound_s'],.03)
 def test_failed_first_interval_is_not_clean_duration(self):
  r=self.evaluate([.1,0,0]);self.assertFalse(r['clean_horizon_feasible']);self.assertEqual(r['observed_clean_lower_bound_s'],0.)
 def test_average_motion_cannot_override_support(self):
  bad=dict(label='pretty-fallen',support=self.evaluate([0,.2,.3]),motion_loss=0.)
  good=dict(label='clean',support=self.evaluate(),motion_loss=10.)
  r=select_support_first([bad,good]);self.assertEqual(r['admitted_label'],'clean')
 def test_no_feasible_policy_stays_unadmitted(self):
  early=dict(label='early',support=self.evaluate([0,.2,.3]),motion_loss=0.)
  late=dict(label='later',support=self.evaluate([0,0,.3]),motion_loss=10.)
  r=select_support_first([early,late]);self.assertIsNone(r['admitted_label']);self.assertEqual(r['diagnostic_leader'],'later');self.assertFalse(r['walking_certified'])
 def test_assistance_rejected(self):
  with self.assertRaises(ValueError):self.evaluate(assistance=np.ones((3,6)))
 def test_incomplete_horizon_rejected(self):
  with self.assertRaises(ValueError):evaluate_support([.01,.02],np.ones((2,6)),[0,0],[1,1],weight=1,h0=1,horizon_s=.03)
 def test_bad_force_and_geometry_rejected(self):
  with self.assertRaises(ValueError):self.evaluate([0,-1,0])
  with self.assertRaises(ValueError):self.evaluate([0,float('nan'),0])
 def test_deadband_explicit(self):
  r=self.evaluate([0,1e-9,0]);self.assertTrue(r['clean_horizon_feasible']);self.assertEqual(r['nonfoot_force_epsilon_weight'],1e-8)
 def test_airborne_no_contact_not_support(self):
  r=self.evaluate(foot=0.);self.assertTrue(r['feasibility_gates']['no_positive_nonfoot']);self.assertFalse(r['clean_horizon_feasible'])
  choice=select_support_first([dict(label='airborne',support=r,motion_loss=0.)]);self.assertIsNone(choice['admitted_label'])
 def test_sunken_height_without_nonfoot_not_support(self):
  r=evaluate_support([.01,.02,.03],np.full((3,6),1/6),[0,0,0],[1,.7,.6],weight=1,h0=1,horizon_s=.03)
  self.assertFalse(r['clean_horizon_feasible']);self.assertFalse(r['feasibility_gates']['minimum_height'])
 def test_different_feasibility_settings_not_ranked(self):
  one=dict(label='a',support=self.evaluate(),motion_loss=1.)
  another=dict(label='b',support=dict(one['support'],nonfoot_force_epsilon_weight=1e-3),motion_loss=1.)
  with self.assertRaises(ValueError):select_support_first([one,another])
 def test_different_horizons_not_ranked(self):
  one=dict(label='a',support=self.evaluate(),motion_loss=1.)
  another=dict(label='b',support=dict(one['support'],declared_horizon_s=.05),motion_loss=1.)
  with self.assertRaises(ValueError):select_support_first([one,another])
if __name__=='__main__':unittest.main()
