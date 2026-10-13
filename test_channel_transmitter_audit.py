"""Saved-artifact checks for the per-channel transmitter-evidence audit."""
import json
import unittest
from pathlib import Path
import run_channel_transmitter_audit as run

class ChannelTransmitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'channel_transmitter_audit.json').read_text())

    def test_all_seven_channels_and_total(self):
        ch = self.doc['per_channel']
        self.assertEqual(set(ch), {'claw_position', 'hook_direction', 'hairplate_angle',
                                   'campaniform_strain', 'club_vibration', 'bristle_touch',
                                   'multidendritic_noxious'})
        self.assertEqual(sum(v['tier_cells'] for v in ch.values()), 3788)

    def test_counts_partition_each_channel(self):
        for name, v in self.doc['per_channel'].items():
            self.assertEqual(v['verified_nonempty'] + v['verified_empty'] + v['absent_from_feather'],
                             v['tier_cells'], name)
            self.assertEqual(sum(v['verified_values'].values()), v['verified_nonempty'], name)
            self.assertLessEqual(v['predicted_only_nonempty'], v['verified_empty'], name)

    def test_weak_evidence_channels_are_the_contact_ones(self):
        ch = self.doc['per_channel']
        self.assertEqual(ch['bristle_touch']['verified_nonempty'], 0)
        self.assertEqual(ch['multidendritic_noxious']['verified_nonempty'], 0)
        self.assertLess(ch['campaniform_strain']['verified_fraction'], 0.25)
        self.assertLess(ch['hairplate_angle']['verified_fraction'], 0.35)
        for strong in ('claw_position', 'hook_direction'):
            self.assertGreater(ch[strong]['verified_fraction'], 0.95)

    def test_no_label_promotion(self):
        """Only 'acetylcholine' appears as verified; nothing was filled from predictions."""
        for name, v in self.doc['per_channel'].items():
            self.assertTrue(set(v['verified_values']) <= {'acetylcholine'}, name)

if __name__ == '__main__':
    unittest.main()
