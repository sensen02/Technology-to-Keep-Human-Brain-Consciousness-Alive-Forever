import unittest
import numpy as np
from engine.embodied.formal_training import score_formal,compare_scores

class FormalTrainingTests(unittest.TestCase):
 def fixture(self):
  q=np.zeros((4001,49));q[:,2]=1;q[:,3]=1
  return dict(qpos=q,qvel=np.zeros((4001,48)),contacts=np.ones((4001,6)),foot_force=np.tile([0.,0.,1.],(4001,6,1)),nonfoot_depth=np.zeros(4001),limit_excess=np.zeros(4001))
 def test_formal_only(self):
  a=self.fixture();a['foot_force'][:1500]=0;s=score_formal(a,6)
  self.assertEqual(s['loss'],0);self.assertGreater(s['preparation']['loss'],0);self.assertTrue(s['eligible'])
 def test_late_failure_counts(self):
  a=self.fixture();a['nonfoot_depth'][-1]=.01;s=score_formal(a,6)
  self.assertGreater(s['loss'],0);self.assertFalse(s['eligible']);self.assertAlmostEqual(s['formal_prefix_ms'],199.9)
 def test_acceptance_requires_support_not_loss(self):
  a=self.fixture();s=score_formal(a,6);self.assertFalse(compare_scores(s,s)['accepted'])
 def test_preparation_safety_cannot_trade_for_formal(self):
  a=self.fixture();a['contacts'][2000:]=0;b=score_formal(a,6)
  a=self.fixture();a['qpos'][1000,2]=.5;c=score_formal(a,6);result=compare_scores(b,c)
  self.assertTrue(result['formal_support_strict_improvement']);self.assertFalse(result['accepted'])
 def test_nonfinite_rejected(self):
  a=self.fixture();a['qvel'][3000,0]=np.nan
  with self.assertRaises(ValueError):score_formal(a,6)
if __name__=='__main__':unittest.main()
