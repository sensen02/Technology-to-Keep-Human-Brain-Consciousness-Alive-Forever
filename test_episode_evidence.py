"""Data-free P1 instrumentation tests: no real body/connectome experiment."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as loop
import run_experiment_platform as platform


class FakeData:
    def __init__(self, model):
        self.qpos = np.zeros(1)
        self.qvel = np.zeros(1)
        self.act = np.zeros(1)
        self.ctrl = np.zeros(1)
        self.xpos = np.array([[0., 0., 1.]])
        self.xfrc_applied = np.zeros((1, 6))
        self.actuator_force = np.zeros(1)
        self.warning = []
        self.time = 0.


def body_step(model, data):
    data.time += loop.DT_BODY
    data.act[:] = data.ctrl
    data.qpos[:] += data.ctrl * loop.DT_BODY


class Encoder:
    def __init__(self, groups, n, config):
        self.n = n

    def encode(self, quantities):
        return np.array([.1, .2, .4, .5, 0., 0.]), {}


class Network:
    seen = []

    def __init__(self, n, params, dt_ms):
        self.n, self.dt, self.t_ms = n, dt_ms, 0.
        self.steps = 0
        type(self).seen = []

    def set_connectivity(self, exc, inh):
        pass

    def step(self, current_nA):
        type(self).seen.append(current_nA.copy())
        self.steps += 1
        self.t_ms = self.steps * self.dt
        return np.zeros(self.n, dtype=bool)


class Decoder:
    def __init__(self, manifest, config):
        self.count = 0

    def consume(self, spikes, **kwargs):
        self.count += 1
        return np.array([self.count / 10.]), {}


class EvidenceTests(unittest.TestCase):
    def run_episode(self, arm='normal', disabled=(), capture=True, replay=None, body_observer=None):
        model = SimpleNamespace(opt=SimpleNamespace(timestep=loop.DT_BODY), nu=1,
                                actuator_ctrlrange=np.array([[0., 1.]]),
                                actuator_ctrllimited=np.array([True]),
                                body=lambda name: SimpleNamespace(id=0))
        groups = {leg: {'bristle_touch': [0], 'claw': [1], 'disabled': [2, 3]}
                  for leg in loop.LEGS}
        # A non-LF sensory index must also survive compaction.
        groups['rh']['claw'] = [4]
        built = {'n': 6, 'edges': {'We': np.zeros((6, 6)), 'Wi': np.zeros((6, 6))}}
        manifest = {'rows': [{'local_indices': [5]}]}
        evidence = {} if capture else None
        fake = SimpleNamespace(MjData=FakeData, mj_step=body_step, mj_forward=lambda m, d: None)
        with patch.dict(sys.modules, mujoco=fake), \
             patch.object(loop, 'ChannelSensoryEncoder', Encoder), \
             patch.object(loop, 'ConductanceNetwork', Network), \
             patch.object(loop, 'MotorOutput', Decoder), \
             patch.object(loop, 'observe_quantities', return_value={}):
            result = loop.episode(model, np.zeros(1), {}, {}, built, manifest, groups,
                                  arm, .02, disabled_channels=disabled,
                                  replay=replay, evidence=evidence, body_observer=body_observer)
        return result, evidence, groups, np.array(Network.seen)

    def test_body_observer_captures_each_substep_without_changing_outputs(self):
        times = []
        observed, _, _, _ = self.run_episode(body_observer=lambda model, data: times.append(data.time))
        plain, _, _, _ = self.run_episode()
        self.assertEqual(len(times), 200)
        np.testing.assert_allclose(times, np.arange(1, 201)*loop.DT_BODY, atol=1e-14)
        for index in (1, 2, 3):
            np.testing.assert_array_equal(observed[index], plain[index])

    def test_exact_post_intervention_current_and_held_timing(self):
        result, evidence, groups, seen = self.run_episode(disabled=('disabled',))
        self.assertEqual(len(result), 4)
        expected = np.tile([.1, .2, 0., 0., 0., 0.], (4, 1))
        expected[1:3, 0] += .3
        np.testing.assert_array_equal(evidence['sensory_current_nA'], expected)
        np.testing.assert_array_equal(seen, np.repeat(expected, 10, axis=0))
        np.testing.assert_allclose(evidence['interval_start_s'], [0, .005, .01, .015], atol=1e-15)
        np.testing.assert_allclose(evidence['interval_end_s'], [.005, .01, .015, .02], atol=1e-15)
        np.testing.assert_array_equal(evidence['applied_source_interval'], [-1, 0, 1, 2])
        np.testing.assert_allclose(result[2][:, 0], [0., .1, .2, .3])
        np.testing.assert_allclose(evidence['decoded_next_controls'][:, 0], [.1, .2, .3, .4])
        np.testing.assert_array_equal(evidence['activation'], result[1][:, -1:])
        ids, arrays = platform.compact_evidence(evidence, groups)
        np.testing.assert_array_equal(ids, [0, 1, 2, 3, 4])
        self.assertFalse(arrays['outside_sensory_input_max_abs_nA'].any())
        np.testing.assert_array_equal(arrays['sensory_current_nA'], expected[:, ids])

    def test_sensory_cut_removes_touch_and_all_current(self):
        _, evidence, _, seen = self.run_episode('sensory_cut')
        self.assertFalse(evidence['sensory_current_nA'].any())
        self.assertFalse(seen.any())

    def test_motor_cut_applied_distinct_from_decoded_next(self):
        result, evidence, _, _ = self.run_episode('motor_cut')
        self.assertEqual(result[2][3, 0], 0.)
        self.assertEqual(evidence['decoded_next_controls'][3, 0], .4)

    def test_replay_preserves_actual_applied_and_source_identity(self):
        replay = np.array([[.8], [.7], [.6], [.5]])
        result, evidence, _, _ = self.run_episode('replay', replay=replay)
        np.testing.assert_array_equal(result[2], replay)
        np.testing.assert_array_equal(evidence['applied_source_interval'], [-1, 0, 1, 2])
        self.assertTrue(all(r['command_origin'] == 'offline_normal_replay' for r in result[0]['records']))

    def test_default_four_returns_and_values_unchanged(self):
        captured, _, _, _ = self.run_episode()
        default, evidence, _, _ = self.run_episode(capture=False)
        self.assertIsNone(evidence)
        self.assertEqual(len(default), 4)
        self.assertEqual(default[0], captured[0])
        for a, b in zip(default[1:], captured[1:]):
            np.testing.assert_array_equal(a, b)

    def test_compaction_rejects_unretained_nonzero_current(self):
        _, evidence, groups, _ = self.run_episode()
        evidence['sensory_current_nA'][0, 5] = .001
        with self.assertRaisesRegex(ValueError, 'Undeclared sensory input'):
            platform.compact_evidence(evidence, groups)

    def test_environment_snapshot_json_safe(self):
        import json
        snapshot = platform.environment_snapshot()
        json.dumps(snapshot, allow_nan=False)
        self.assertIn('numpy', snapshot['packages'])


if __name__ == '__main__':
    unittest.main()
