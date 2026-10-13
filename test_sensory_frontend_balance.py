"""Saved-artifact checks for the sensory front-end balance audit; no integration."""
import json
import unittest
import numpy as np
import run_sensory_frontend_balance as run

class FrontendBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'sensory_frontend_balance.json').read_text())
        cls.z = np.load(run.OUT / 'sensory_frontend_balance_traces.npz')

    def test_channel_shares_sum_to_one(self):
        total = sum(v['share_of_total_input'] or 0.0 for v in self.doc['per_channel'].values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_reference_pose_is_the_seated_pose(self):
        """Position/velocity channels must be exactly zero in their first sample."""
        for ch in ('claw_position', 'hook_direction', 'hairplate_angle'):
            v = self.z[f'value_{ch}']
            self.assertTrue(np.all(v[0] == 0.0), ch)
            self.assertGreater(np.abs(v[1:]).max(), 0.0, ch)

    def test_unmodelled_channels_emit_zero_everywhere(self):
        for ch in ('club_vibration', 'multidendritic_noxious'):
            self.assertTrue(np.all(self.z[f'value_{ch}'] == 0.0), ch)
            self.assertTrue(np.all(self.z[f'current_{ch}'] == 0.0), ch)

    def test_drive_dominated_by_channel_without_verified_transmitter(self):
        ch = self.doc['per_channel']
        self.assertGreater(ch['bristle_touch']['share_of_total_input'], 0.85)
        self.assertGreater(ch['campaniform_strain']['value_max'], 4.0)
        self.assertLess(ch['hairplate_angle']['value_max'], 1.0)
        self.assertEqual(ch['bristle_touch']['cells'], 2879)
        self.assertEqual(ch['campaniform_strain']['cells'], 60)

    def test_only_weak_evidence_channels_nonzero_at_t0(self):
        self.assertEqual(set(self.doc['channels_nonzero_at_t0']),
                         {'bristle_touch', 'campaniform_strain'})

if __name__ == '__main__':
    unittest.main()
