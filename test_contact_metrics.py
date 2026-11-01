import unittest
from engine.embodied.contact_metrics import contact_episodes


class ContactMetricsTests(unittest.TestCase):
    def test_touchdown_anchor_and_swing_exclusion(self):
        # First stance drifts 3-4-5; enormous swing and new touchdown are not slip.
        r = contact_episodes(range(8), [0,1,1,0,0,1,1,0],
                             [(99,99),(0,0),(3,4),(500,500),(700,700),
                              (10,10),(10,12),(900,900)])
        self.assertEqual(r['complete_count'], 2)
        self.assertEqual([e['max_anchor_drift'] for e in r['episodes']], [5,2])
        self.assertEqual(r['mean_complete_duration'], 2)
        self.assertEqual(r['observed_contact_time'], 4)

    def test_edges_censored_and_last_sample_has_no_duration(self):
        r = contact_episodes([0,.1,.3,.8], [1,0,0,1], [(0,0)]*4)
        self.assertEqual(r['complete_count'], 0)
        self.assertIsNone(r['mean_complete_duration'])
        self.assertTrue(r['episodes'][0]['left_censored'])
        self.assertTrue(r['episodes'][1]['right_censored'])
        self.assertEqual(r['episodes'][1]['duration'], 0)
        self.assertAlmostEqual(r['observed_contact_time'], .1)

    def test_out_and_back_path_distinct_from_anchor_drift(self):
        r = contact_episodes(range(5), [0,1,1,1,0],
                             [(0,0),(0,0),(3,4),(0,0),(0,0)])
        self.assertEqual(r['episodes'][0]['contact_path'], 10)
        self.assertEqual(r['max_observed_anchor_drift'], 5)

    def test_no_contact_is_missing_slip_not_zero(self):
        r = contact_episodes([0,1], [0,0], [(0,0),(100,0)])
        self.assertIsNone(r['max_observed_anchor_drift'])
        self.assertEqual(r['episodes'], [])

    def test_short_contacts_are_not_silently_merged(self):
        r = contact_episodes([0,.001,.002,.003,.004], [0,1,0,1,0], [(0,0)]*5)
        self.assertEqual(r['complete_count'], 2)
        self.assertAlmostEqual(r['mean_complete_duration'], .001)

    def test_invalid_data(self):
        for t, f, xy in [([0,0],[0,1],[(0,0)]*2),
                         ([0,float('nan')],[0,1],[(0,0)]*2),
                         ([0,1],[0],[(0,0)]*2),
                         ([0,1],[0,2],[(0,0)]*2),
                         ([0,1],[0,1],[(0,0),(float('inf'),0)])]:
            with self.subTest(t=t, f=f, xy=xy):
                with self.assertRaises(ValueError):
                    contact_episodes(t,f,xy)


if __name__ == '__main__':
    unittest.main()
