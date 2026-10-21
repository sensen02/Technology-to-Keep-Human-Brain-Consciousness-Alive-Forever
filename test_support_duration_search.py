import unittest
import numpy as np
from engine.embodied.support_duration_search import support_duration_score, SupportDurationSearch, weight_key

class TestSupportDuration(unittest.TestCase):
    def samples(self):
        a=np.zeros((8,5));a[:,0]=np.arange(1,9)*.001;a[:,1]=2;a[:,3]=1
        return a
    def test_prefix_and_all_good_secondary_recovery(self):
        a=self.samples();a[2,3]=.84;a[3:,1]=4
        m=support_duration_score(a,2,1,.001,.008)
        self.assertEqual(m['prefix_good_ticks'],2);self.assertEqual(m['cumulative_good_ticks'],7)
        self.assertEqual(m['foot_mean_while_good'],12/7);self.assertTrue(m['support_recovery'])
        self.assertEqual(m['good_segment_count'],2);self.assertEqual(m['first_failure_endpoint_s'],.003)
    def test_equal_threshold_and_initial_fail(self):
        a=self.samples();a[:,3]=.85;a[:,4]=.35
        self.assertEqual(support_duration_score(a,2,1,.001,.008)['prefix_good_ticks'],8)
        a[0,2]=1;m=support_duration_score(a,2,1,.001,.008)
        self.assertEqual(m['prefix_good_ticks'],0);self.assertTrue(m['support_recovery'])
    def test_invalid(self):
        a=self.samples();a[0,0]=float('nan')
        with self.assertRaises(ValueError):support_duration_score(a,2,1,.001,.008)
        with self.assertRaises(ValueError):support_duration_score(self.samples(),2,1,0,.008)
    def test_ties_and_strict_secondary(self):
        m=dict(prefix_good_ticks=2,foot_mean_while_good=1.)
        s=SupportDurationSearch(m,{})
        p,n=s.probes();s.remember(p,'plus',m,{});s.remember(n,'minus',m,{})
        self.assertFalse(s.update(p,n)['changed'])
        p,n=s.probes();better=dict(prefix_good_ticks=2,foot_mean_while_good=np.nextafter(1.,2.))
        s.remember(p,'plus2',better,{});s.remember(n,'minus2',better,{})
        self.assertTrue(s.update(p,n)['changed']);self.assertEqual(s.label,'plus2')
        self.assertEqual(s.center[0,4],.05)
    def test_two_fixed_passes_and_cache(self):
        m=dict(prefix_good_ticks=0,foot_mean_while_good=0.)
        s=SupportDurationSearch(m,{})
        for k in range(8):
            p,n=s.probes()
            for j,w in enumerate((p,n)):
                if weight_key(w) not in s.cache:s.remember(w,str((k,j)),m,{})
            s.update(p,n)
        with self.assertRaises(ValueError):s.probes()
if __name__=='__main__':unittest.main()
