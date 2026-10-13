"""Focused synthetic interface tests; not evidence of biological control."""
import copy
import unittest
import numpy as np
from engine.embodied.motor_output import build_motor_manifest, MotorOutput


def fixture():
    src = {'measured': {'resolved': [
        dict(actuator='LF_a', leg='LF', side='left', muscle_target='a', pool_root_ids=[11, 11, 12]),
        dict(actuator='LF_b', leg='LF', side='left', muscle_target='a', pool_root_ids=[11, 12]),
        dict(actuator='RF_a', leg='RF', side='right', muscle_target='a', pool_root_ids=[21]),
        dict(actuator='LH_a', leg='LH', side='left', muscle_target='a', pool_root_ids=[31])],
        'unresolved': [dict(actuator='RM_missing', reason='no annotated own-leg pool')]}}
    return src, [11, 21, 99], ['LF_a', 'LF_b', 'RF_a', 'LH_a', 'RM_missing']


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.src, self.roots, self.names = fixture()
        self.m = build_motor_manifest(self.src, self.roots, self.names)

    def test_coverage_and_shared_pool(self):
        self.assertEqual(self.m['usable_actuators'], 3)
        self.assertEqual(self.m['fully_covered_actuators'], 1)
        self.assertEqual(self.m['rows'][0]['duplicates_removed'], 1)
        self.assertEqual(len(self.m['shared_pools']), 1)
        self.assertEqual(self.m['rows'][3]['status'], 'absent_from_tier')

    def test_zero_and_identity(self):
        out = MotorOutput(self.m)
        z = np.zeros((10, 3))
        ctrl, _ = out.consume(z, spike_dt_s=.0005, start_s=0, end_s=.005)
        np.testing.assert_array_equal(ctrl, np.zeros(5))
        z[:, 0] = 1
        ctrl, rec = out.consume(z, spike_dt_s=.0005, start_s=.005, end_s=.01)
        self.assertGreater(ctrl[0], 0)
        self.assertEqual(ctrl[0], ctrl[1])
        np.testing.assert_array_equal(ctrl[2:], np.zeros(3))
        self.assertEqual(rec['valid_from_s'], .01)
        self.assertTrue(np.all(ctrl <= 1))

    def test_cut_and_decay(self):
        out = MotorOutput(self.m)
        out.consume(np.ones((10, 3)), spike_dt_s=.0005, start_s=0, end_s=.005)
        ctrl, _ = out.consume(np.zeros((10, 3)), spike_dt_s=.0005, start_s=.005, end_s=.01)
        self.assertGreater(ctrl[0], 0)  # causal filter retains previous spikes
        ctrl, _ = out.consume(np.ones((10, 3)), spike_dt_s=.0005, start_s=.01, end_s=.015, connected=False)
        np.testing.assert_array_equal(ctrl, np.zeros(5))

    def test_chunking_and_reset(self):
        sp = np.random.default_rng(1).integers(0, 2, (20, 3))
        a, b = MotorOutput(self.m), MotorOutput(self.m)
        c, _ = a.consume(sp, spike_dt_s=.0005, start_s=0, end_s=.01)
        b.consume(sp[:10], spike_dt_s=.0005, start_s=0, end_s=.005)
        d, _ = b.consume(sp[10:], spike_dt_s=.0005, start_s=.005, end_s=.01)
        np.testing.assert_array_equal(c, d)
        b.reset()
        np.testing.assert_array_equal(b.rates, np.zeros(5))

    def test_invalid_inputs(self):
        out = MotorOutput(self.m)
        for kwargs in [dict(spikes=np.ones((10, 3))*np.nan, start_s=0, end_s=.005),
                       dict(spikes=np.ones((10, 3)), start_s=.001, end_s=.006),
                       dict(spikes=np.ones((10, 3)), start_s=0, end_s=.01),
                       dict(spikes=np.ones((10, 2)), start_s=0, end_s=.005)]:
            with self.assertRaises(ValueError):
                out.consume(spike_dt_s=.0005, **kwargs)
        bad = copy.deepcopy(self.m)
        bad['rows'][0]['local_indices'] = [100]
        with self.assertRaises(ValueError):
            MotorOutput(bad)
        bad = copy.deepcopy(self.src)
        bad['measured']['resolved'][2]['pool_root_ids'] = [11]
        with self.assertRaises(ValueError):
            build_motor_manifest(bad, self.roots, self.names)
        with self.assertRaises(ValueError):
            build_motor_manifest(self.src, [11, 11], self.names)
        with self.assertRaises(ValueError):
            build_motor_manifest(self.src, self.roots, self.names[:-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
