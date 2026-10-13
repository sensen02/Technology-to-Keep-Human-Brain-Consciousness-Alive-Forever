"""Correction audit tests: passive-versus-actuated support framing."""
import json
import unittest
import numpy as np
import run_passive_vs_actuated_support as run

class PassiveVersusActuatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'passive_vs_actuated.json').read_text())
        cls.z = np.load(run.OUT / 'passive_vs_actuated.npz')

    def test_the_two_runs_are_the_same_integration(self):
        self.assertTrue(self.doc['cross_check_same_integration_as_branch'])

    def test_passive_figure_is_the_initial_state_only(self):
        t = self.doc['timeline']
        self.assertAlmostEqual(t['support_ratio_at_0'], 0.6378656668774672, places=12)
        self.assertEqual(t['mean_activation_at_0'], 0.0)

    def test_support_rises_above_weight_after_activation_starts(self):
        t = self.doc['timeline']
        self.assertIsNotNone(t['time_first_tibia_activation_s'])
        self.assertIsNotNone(t['time_first_support_above_weight_s'])
        self.assertLess(t['time_first_tibia_activation_s'], t['time_first_support_above_weight_s'])
        self.assertAlmostEqual(t['support_ratio_mean'], 0.9981815317602278, places=12)

    def test_correction_text_is_present_and_specific(self):
        c = self.doc['correction']
        for key in ('earlier_wording', 'corrected_wording', 'why'):
            self.assertIn(key, c)
            self.assertTrue(c[key])
        self.assertIn('passive', c['corrected_wording'].lower())

    def test_static_scans_are_reframed_not_overclaimed(self):
        r = self.doc['static_scan_reframing']
        self.assertIn('passive support only', r['meaning'])
        self.assertLess(r['two_parameter_scan_max_support_ratio'], 1.0)
        self.assertLess(r['symmetric_search_best_support_ratio'], 1.0)

if __name__ == '__main__':
    unittest.main()
