"""Saved-artifact checks for the injected-vs-propagable input audit."""
import json
import unittest
import run_propagable_input_audit as run

class PropagableInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'propagable_input.json').read_text())

    def test_totals_and_share(self):
        t = self.doc['totals']
        self.assertLess(t['propagable_current_nA'], t['injected_current_nA'])
        self.assertAlmostEqual(t['propagable_share'], t['propagable_current_nA'] / t['injected_current_nA'], places=12)
        self.assertLess(t['propagable_share'], 0.15)
        self.assertEqual(t['sensory_cells'], 3788)

    def test_cross_check_with_channel_audit_passes(self):
        cc = self.doc['cross_check_vs_channel_audit']
        self.assertTrue(all(v['agrees'] for v in cc.values()), cc)
        self.assertEqual(cc['bristle_touch']['channel_audit'], 0)
        self.assertEqual(cc['campaniform_strain']['channel_audit'], 12)

    def test_dead_end_channel(self):
        b = self.doc['per_channel']['bristle_touch']
        self.assertGreater(b['injected_current_nA'], 500)
        self.assertEqual(b['propagable_current_nA'], 0.0)
        self.assertEqual(b['verified_cells'], 0)

    def test_best_evidenced_channels_carry_most_propagable_current(self):
        ch = self.doc['per_channel']
        prop = {k: v['propagable_current_nA'] for k, v in ch.items()}
        self.assertGreater(prop['hook_direction'], prop['campaniform_strain'])
        self.assertGreater(prop['claw_position'], prop['campaniform_strain'])
        for name in ('claw_position', 'hook_direction'):
            self.assertGreater(ch[name]['propagable_fraction_of_channel'], 0.9)
        self.assertLess(ch['campaniform_strain']['propagable_fraction_of_channel'], 0.25)

    def test_no_label_was_filled(self):
        """The nan-as-label bug must not recur: bristle and multidendritic stay unverified."""
        for name in ('bristle_touch', 'multidendritic_noxious'):
            self.assertEqual(self.doc['per_channel'][name]['verified_cells'], 0)

if __name__ == '__main__':
    unittest.main()
