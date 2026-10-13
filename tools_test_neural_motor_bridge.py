"""Deterministic causal tests for engineered spike decoder; no biological claim."""
import unittest
import numpy as np
from engine.embodied.neural_motor_bridge import SpikeIntentBridge


class BridgeTests(unittest.TestCase):
    def bridge(self):
        return SpikeIntentBridge(4, [0, 1], [2, 3], provenance='SYNTHETIC_TEST_NOT_IDENTIFIED_NEURONS')

    def test_swap_and_silence(self):
        s = np.zeros((100, 4), dtype=bool)
        s[::10, :2] = True
        a = self.bridge().advance(s, spike_dt_s=.001, locomotion_permission=.8)
        b = self.bridge().advance(s[:, ::-1], spike_dt_s=.001, locomotion_permission=.8)
        self.assertGreater(a['turn'], 0)
        self.assertAlmostEqual(a['turn'], -b['turn'])
        z = self.bridge().advance(np.zeros_like(s), spike_dt_s=.001, locomotion_permission=.8)
        self.assertEqual(z['turn'], 0)
        self.assertEqual(z['forward'], .8)  # external drive remains explicitly external

    def test_cut_clears_memory_and_drive(self):
        d = self.bridge()
        d.advance(np.ones((10,4)), spike_dt_s=.001, locomotion_permission=1)
        out = d.advance(np.ones((10,4)), spike_dt_s=.001, locomotion_permission=1, connected=False)
        self.assertEqual(out['turn'], 0)
        self.assertEqual(out['forward'], 0)
        np.testing.assert_array_equal(d.rates, [0, 0])

    def test_chunking_and_causality(self):
        s = np.random.default_rng(4).random((100,4)) < .1
        a,b = self.bridge(),self.bridge()
        a.advance(s, spike_dt_s=.001)
        b.advance(s[:37], spike_dt_s=.001)
        before = b.rates.copy()
        c = self.bridge()
        c.advance(s[:37], spike_dt_s=.001)
        np.testing.assert_array_equal(before, c.rates)
        b.advance(s[37:], spike_dt_s=.001)
        np.testing.assert_array_equal(a.rates, b.rates)

    def test_permission_and_dose(self):
        s = np.zeros((100,4), dtype=bool)
        s[::20, :2] = True
        lo = self.bridge().advance(s, spike_dt_s=.001, locomotion_permission=1)
        s[::10,:2] = True
        hi = self.bridge().advance(s, spike_dt_s=.001, locomotion_permission=1)
        self.assertGreater(hi['turn'], lo['turn'])
        stop = self.bridge().advance(s, spike_dt_s=.001)
        self.assertEqual(stop['forward'], 0)
        self.assertEqual(stop['turn'], 0)

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            SpikeIntentBridge(4, [], [2], provenance='test')
        with self.assertRaises(ValueError):
            SpikeIntentBridge(4, [1], [1], provenance='test')
        with self.assertRaises(ValueError):
            self.bridge().advance(np.full((2,4), np.nan), spike_dt_s=.001)


if __name__ == '__main__':
    unittest.main()
