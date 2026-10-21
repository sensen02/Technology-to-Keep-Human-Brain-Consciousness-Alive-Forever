"""Physics-free contracts for episode-return search."""
import unittest
import numpy as np
from engine.embodied.body_reward import body_return, AntitheticBodySearch

class BodyRewardTests(unittest.TestCase):
    def samples(self):
        return np.column_stack((np.arange(1,5501)*.0001, np.ones(5500), np.zeros(5500), np.ones(5500), np.zeros(5500)))
    def test_full_window_caps_and_collapse(self):
        s=self.samples(); self.assertAlmostEqual(body_return(s,1,1)['score'],1)
        s[:,1]=1e9; self.assertAlmostEqual(body_return(s,1,1)['score'],1)
        s[:,2]=1; self.assertLess(body_return(s,1,1)['score'],0)
        with self.assertRaises(ValueError):body_return(s[1:],1,1)
    def test_deterministic_bounds_actual_selected(self):
        w=np.tile([.4,.25,.2,.1,.05],(66,1)); a=AntitheticBodySearch(w); b=AntitheticBodySearch(w)
        d,p,m=a.probes(); self.assertTrue(np.array_equal(p,b.probes()[1])); self.assertTrue(np.allclose(p+m,2*w))
        r=lambda v:dict(score=v,nonfoot_duration_s=0.,nonfoot_count_integral=0.)
        out=a.update([dict(label='plus',weights=p,return_=None,**{'return':r(.2)}),dict(label='minus',weights=m,**{'return':r(.3)})])
        self.assertEqual(out['selected_label'],'minus'); self.assertTrue(np.array_equal(a.center,m))
    def test_tradeoff_warning_default_and_optional_stop(self):
        for policy in ('warn', 'stop'):
            a=AntitheticBodySearch(np.ones((66,5)),nonfoot_tradeoff=policy); _,p,m=a.probes()
            candidates=[dict(label='p',weights=p,**{'return':dict(score=1,nonfoot_duration_s=.1,nonfoot_count_integral=.1)}),dict(label='m',weights=m,**{'return':dict(score=0,nonfoot_duration_s=0,nonfoot_count_integral=0)})]
            if policy=='stop':
                with self.assertRaises(RuntimeError):a.update(candidates)
            else:
                out=a.update(candidates)
                self.assertEqual(out['selected_label'],'p')
                self.assertEqual(len(out['warnings']),1)
                self.assertEqual(out['warnings'][0]['worsened_metrics'],['nonfoot_duration_s','nonfoot_count_integral'])
                self.assertTrue(np.array_equal(a.center,p))
        self.assertEqual(AntitheticBodySearch(np.ones((66,5))).nonfoot_tradeoff,'warn')
        with self.assertRaises(ValueError):AntitheticBodySearch(np.ones((66,5)),nonfoot_tradeoff='ignore')
if __name__=='__main__':unittest.main()
