"""Zero-body, zero-network checks for independently scheduled motor updates."""
import unittest

import numpy as np

from engine.embodied.motor_cadence_schedule import motor_windows
from engine.embodied.motor_translation import FiniteMotorTranslator


MANIFEST = dict(n_neurons=3, n_actuators=2, rows=[
    dict(actuator_index=0, local_indices=[0]),
    dict(actuator_index=1, local_indices=[]),
])


def decode_schedule(spikes, motor_dt_s, *, cut_from_tick=None):
    decoder = FiniteMotorTranslator(MANIFEST)
    windows = motor_windows(len(spikes) * .0005, motor_dt_s=motor_dt_s)
    pending = np.zeros(2)
    applied, features, source_ticks, sensory_indices = [], {}, [], []
    for w in windows:
        # The real body loop must apply this pending value BEFORE consuming
        # this window's not-yet-completed neural samples.
        applied.append(pending.copy())
        source_ticks.append(w.apply_source_end_tick)
        sensory_indices.append(w.sensory_index)
        connected = cut_from_tick is None or w.end_tick < cut_from_tick
        pending, trace = decoder.consume(
            spikes[w.start_tick:w.end_tick], spike_dt_s=.0005,
            start_s=w.start_tick*.0005, end_s=w.end_tick*.0005,
            connected=connected)
        features[w.end_tick] = np.asarray(trace['pool_features'])
        assert w.apply_source_end_tick is None or w.apply_source_end_tick <= w.start_tick
    return np.asarray(applied), features, source_ticks, sensory_indices


class MotorCadenceScheduleTests(unittest.TestCase):
    def test_sensory_hold_neural_grid_and_previous_source(self):
        for motor, per_sensory, total in ((.005, 1, 110), (.001, 5, 550)):
            with self.subTest(motor=motor):
                windows = motor_windows(.55, motor_dt_s=motor)
                self.assertEqual(len(windows), total)
                self.assertEqual([w.sensory_index for w in windows[:per_sensory]],
                                 [0]*per_sensory)
                self.assertEqual([w.sensory_index for w in windows[per_sensory:2*per_sensory]],
                                 [1]*per_sensory)
                self.assertIsNone(windows[0].apply_source_end_tick)
                for old, new in zip(windows, windows[1:]):
                    self.assertEqual(old.end_tick, new.start_tick)
                    self.assertEqual(new.apply_source_end_tick, old.end_tick)
                self.assertEqual(windows[-1].end_tick, 1100)

    def test_same_completed_endpoint_features_and_no_premature_application(self):
        spikes = np.zeros((100, 3), dtype=bool)
        spikes[0, 0] = True
        spikes[39, 0] = True
        slow, sfeatures, ssources, _ = decode_schedule(spikes, .005)
        fast, ffeatures, fsources, _ = decode_schedule(spikes, .001)
        for endpoint in range(10, 101, 10):
            np.testing.assert_array_equal(sfeatures[endpoint], ffeatures[endpoint])
        np.testing.assert_array_equal(slow[0], [0, 0])
        np.testing.assert_array_equal(fast[0], [0, 0])
        self.assertIsNone(ssources[0]); self.assertIsNone(fsources[0])
        self.assertGreater(fast[1, 0], 0)
        self.assertGreater(slow[1, 0], 0)
        self.assertEqual(fsources[1], 2)
        self.assertEqual(ssources[1], 10)
        np.testing.assert_array_equal(fast[:, 1], 0)

    def test_zero_and_disconnection(self):
        zero = np.zeros((100, 3), dtype=bool)
        for motor in (.005, .001):
            applied, _, _, _ = decode_schedule(zero, motor)
            np.testing.assert_array_equal(applied, np.zeros_like(applied))
            spikes = zero.copy(); spikes[0, 0] = True
            applied, _, _, _ = decode_schedule(spikes, motor, cut_from_tick=10)
            per_block = round(motor/.0005)
            # At tick 10 disconnection clears both output and FIR history.
            np.testing.assert_array_equal(applied[10//per_block+1:], 0)

    def test_rejects_mixed_or_non_divisible_grids(self):
        for duration, kwargs in ((.05, dict(motor_dt_s=.003)),
                                 (.05, dict(motor_dt_s=.00075)),
                                 (.05, dict(sensory_dt_s=.0045, motor_dt_s=.001)),
                                 (.051, dict(sensory_dt_s=.005))):
            with self.subTest(duration=duration, kwargs=kwargs), self.assertRaises(ValueError):
                motor_windows(duration, **kwargs)


if __name__ == '__main__':
    unittest.main()
