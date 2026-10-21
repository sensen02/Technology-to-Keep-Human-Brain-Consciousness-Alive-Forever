"""Synthetic force ledger arithmetic; no body simulation."""
import unittest
import numpy as np
from tools_s2_contact_peak_audit import metrics, select_peak


class ContactPeakAuditTest(unittest.TestCase):
    def test_multiple_pairs_and_missing_presence(self):
        r = metrics({'0,1':[3,4,0], '0,2':[1,0,0]}, {'0,1':[0,0,0], '0,3':[0,0,2]})
        self.assertEqual([p['pair'] for p in r['pairs']], ['0,1','0,3','0,2'])
        self.assertEqual(r['pairmax'], 5.)
        self.assertEqual(r['sum_pair_delta_norms'], 8.)
        np.testing.assert_array_equal(r['sum_vector_delta'], [4,4,-2])
        self.assertEqual(r['pairs'][1]['coarse_present'], False)
        self.assertEqual(r['pairs'][2]['fine_present'], False)
        self.assertEqual(r['pairs'][0]['norm_relative'], 1.)

    def test_cancellation_is_not_pairmax(self):
        r=metrics({'0,1':[1,0,0], '0,2':[-1,0,0]}, {})
        self.assertEqual(r['sum_vector_delta_norm'], 0.)
        self.assertEqual(r['pairmax'], 1.)
        self.assertEqual(r['sum_pair_delta_norms'], 2.)

    def test_zero_denominator_undefined(self):
        p=metrics({'0,1':[0,0,0]}, {'0,1':[0,0,0]})['pairs'][0]
        self.assertEqual(p['relative_denominator'],0.)
        self.assertIsNone(p['norm_relative'])
        self.assertEqual(p['delta_norm'],0.)
        self.assertEqual(metrics({}, {})['pairmax'],0.)

    def test_relative_uses_norm_not_component(self):
        p=metrics({'0,1':[3,4,0]}, {'0,1':[-3,-4,0]})['pairs'][0]
        self.assertEqual(p['norm_relative'],2.)
        self.assertEqual(p['delta_norm'],10.)
        np.testing.assert_array_equal(p['delta'],[6,8,0])

    def test_peak_window_and_earliest_tie(self):
        self.assertEqual(select_peak(np.array([1.,2.,3.,4.]), [99,5,5,99],2,3),1)
        with self.assertRaises(ValueError):
            select_peak(np.array([1.]),[1],2,3)

    def test_invalid_vector_rejected(self):
        for v in ([1,2], [0,0,float('nan')]):
            with self.assertRaises(ValueError):
                metrics({'0,1':v},{})


if __name__ == '__main__':
    unittest.main()
