import unittest
import numpy as np
from run_fixed_body_input_clock_replay import reconstruct_input,first_event_difference

class ReplayTests(unittest.TestCase):
    def record(self):
        return dict(actual_network_input_by_channel={'lf':{'claw':dict(count=2,min_nA=.125,max_nA=.125,sum_nA=.25)}},sensory_current_min_nA=0.,sensory_current_max_nA=.125)
    def test_exact_uniform_current(self):
        np.testing.assert_array_equal(reconstruct_input(self.record(),{'lf':{'claw':[1,3]}},5),[0,.125,0,.125,0])
    def test_nonuniform_refused(self):
        r=self.record();r['actual_network_input_by_channel']['lf']['claw']['max_nA']=.15
        with self.assertRaises(ValueError):reconstruct_input(r,{'lf':{'claw':[1,3]}},5)
    def test_duplicate_refused(self):
        with self.assertRaises(ValueError):reconstruct_input(self.record(),{'lf':{'claw':[1,1]}},5)
    def test_sum_refused(self):
        r=self.record();r['actual_network_input_by_channel']['lf']['claw']['sum_nA']=1.
        with self.assertRaises(ValueError):reconstruct_input(r,{'lf':{'claw':[1,3]}},5)
    def test_event_first_mismatch_and_missing(self):
        self.assertIsNone(first_event_difference([1.,2.],[1.,2.]))
        self.assertEqual(first_event_difference([1.,2.],[1.,1.5])['earliest_ms'],1.5)
        self.assertEqual(first_event_difference([1.],[1.,3.])['other_ms'],3.)
if __name__=='__main__':unittest.main()
