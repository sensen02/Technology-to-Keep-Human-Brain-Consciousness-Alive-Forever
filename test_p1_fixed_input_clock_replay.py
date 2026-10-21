"""Synthetic P2 contracts; no real neural network, bundle or body required."""
import unittest
from types import SimpleNamespace
import numpy as np
import run_p1_fixed_input_clock_replay as replay
from engine.embodied.motor_output import MotorOutputConfig


def manifest(n=3):
    return dict(n_neurons=n, n_actuators=2, rows=[
        dict(actuator_index=0, local_indices=[0, 1]),
        dict(actuator_index=1, local_indices=[])])


class FakeNetwork:
    def __init__(self, n, params, dt_ms):
        self.n = n; self.dt = dt_ms; self.t_ms = 0.; self.steps = 0
    def set_connectivity(self, we, wi):
        assert np.array_equal(we, np.eye(self.n)*.2)
        assert np.array_equal(wi, np.eye(self.n)*.4)
    def step(self, current_nA):
        self.steps += 1; self.t_ms += self.dt
        return np.asarray(current_nA) > 0 if self.steps % 3 == 0 else np.zeros(self.n, bool)


class ReplayTests(unittest.TestCase):
    def test_exact_selected_nonuniform_columns_no_statistics_inference(self):
        saved = dict(sensory_local_indices=np.array([0, 2]),
                     normal_sensory_current_nA=np.array([[.1, .27], [.5, -.11]]),
                     normal_outside_sensory_input_max_abs_nA=np.zeros(2))
        actual = replay.reconstruct_input(saved, 4)
        np.testing.assert_array_equal(actual[:, [0, 2]], saved['normal_sensory_current_nA'])
        self.assertFalse(actual[:, [1, 3]].any())
        saved['sensory_local_indices'] = np.array([0, 0])
        with self.assertRaises(ValueError): replay.reconstruct_input(saved, 4)

    def test_reject_outside_nonfinite_invalid_columns(self):
        for ids, current, outside in [(np.array([3]), [[1.]], [0.]),
                                      (np.array([0]), [[np.nan]], [0.]),
                                      (np.array([0]), [[1.]], [1.])]:
            with self.assertRaises(ValueError):
                replay.reconstruct_input(dict(sensory_local_indices=ids,
                    normal_sensory_current_nA=np.asarray(current),
                    normal_outside_sensory_input_max_abs_nA=np.asarray(outside)), 3)

    def test_fixed_raster_partitions_preserve_spike_recurrence(self):
        raster = np.zeros((100, 3), bool)
        raster[[0, 9, 10, 19, 20, 49, 99], 0] = True
        raster[[9, 10, 19, 99], 1] = True
        result, arrays = replay.partition_diagnostic(raster, manifest(), MotorOutputConfig())
        self.assertTrue(all(v['rates_exact'] and v['commands_exact'] for v in result.values()))
        self.assertEqual(arrays['10ms'][0].shape, (5, 2))
        self.assertFalse(arrays['10ms'][0][:, 1].any())
        with self.assertRaises(ValueError):
            replay.partition_diagnostic(raster, manifest(), MotorOutputConfig(), partitions_ms=(3,))

    def test_counts_all_sparse_events_lag_and_deterministic_duplicate(self):
        built = dict(n=3, edges=dict(We=np.eye(3), Wi=np.eye(3)*2))
        currents = np.ones((10, 3))
        module = SimpleNamespace(ConductanceNetwork=FakeNetwork, ConductanceParams=lambda: None)
        data = replay.simulate(built, currents, manifest(), MotorOutputConfig(), .5, module)
        duplicate = replay.simulate(built, currents, manifest(), MotorOutputConfig(), .5, module)
        for key in data: np.testing.assert_array_equal(data[key], duplicate[key])
        self.assertFalse(data['applied_controls'][0].any())
        np.testing.assert_array_equal(data['applied_controls'][1:], data['decoded_next_controls'][:-1])
        np.testing.assert_array_equal(data['applied_source_interval'], np.arange(10)-1)
        recovered = np.zeros((100, 3), bool)
        recovered[data['event_step'], data['event_neuron']] = True
        np.testing.assert_array_equal(recovered.reshape(10, 10, 3).sum(axis=1), data['spike_counts'])
        np.testing.assert_array_equal(data['event_time_ms'], (data['event_step']+1)*.5)

    def test_guard_only_standalone_entry(self):
        # Importing this module above did not import memory_guard or execute any run.
        self.assertEqual(replay.DEFAULT_BUNDLE, replay.Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d'))


if __name__ == '__main__':
    unittest.main()
