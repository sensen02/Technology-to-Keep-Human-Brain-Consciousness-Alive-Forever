"""Regression of actual state ownership using fake physics, not behavior evidence."""
from unittest.mock import patch
import unittest,numpy as np
from test_continuous_handover import TestContinuousHandover
from engine.embodied.continuous_handover import run_neural_continuous
class WarmContinuity(unittest.TestCase):
 def test_retains_runtime_pending_and_nonzero_clock(self):
  helper=TestContinuousHandover();c=helper.setup_case();m,d,n,e,dec,physics,legacy,built,manifest,*_=c
  n.t_ms=10.;n.v=np.array([-53.]);dec.elapsed=.01;dec.elapsed_s=.01;dec.rates=np.array([31.]);pending=np.array([.73]);state=dict(runtime=(n,e,dec),pending=pending,body_time=d.time)
  with patch.dict('sys.modules',{'mujoco':physics,'run_neural_muscle_closure':legacy}),patch('engine.embodied.continuous_handover._make_runtime',side_effect=AssertionError('must not rebuild')):
   r=run_neural_continuous(m,d,built,manifest,{},.01,sensory_reference_qpos=np.array([.5]),joints={},geom_leg={},runtime_state=state)
  np.testing.assert_array_equal(r['commands'][0],pending);np.testing.assert_array_equal(r['runtime_boundary']['v'],[-53.]);np.testing.assert_array_equal(r['runtime_boundary']['decoder_rates'],[31.]);self.assertEqual(r['records'][0]['applied_source_interval'],1);self.assertAlmostEqual(n.t_ms,20);self.assertEqual(r['counts']['network_instances'],0);self.assertIs(r['runtime_state']['runtime'][0],n)
 def test_rejects_stale_runtime_before_withdrawal(self):
  h=TestContinuousHandover();c=h.setup_case();m,d,n,e,dec,physics,legacy,built,manifest,*_=c
  with patch.dict('sys.modules',{'mujoco':physics,'run_neural_muscle_closure':legacy}):
   with self.assertRaises(ValueError):run_neural_continuous(m,d,built,manifest,{},.01,sensory_reference_qpos=np.array([.5]),joints={},geom_leg={},runtime_state=dict(runtime=(n,e,dec),pending=np.array([.2]),body_time=0))
  np.testing.assert_array_equal(d.qfrc_applied,[12.])
if __name__=='__main__':unittest.main()
