"""Manual prototype wiring tests; stub network is NOT biological evidence."""
import unittest
from unittest.mock import patch
import numpy as np
import run_neural_muscle_closure as loop
from test_closure_episode_clock import SilentNetwork


class CapturingNetwork(SilentNetwork):
    inputs = []

    def step(self, current_nA):
        self.inputs.append(current_nA.copy())
        return super().step(current_nA)


class FixedEncoder:
    def __init__(self, groups, n, config):
        self.n = n

    def encode(self, quantities):
        return np.arange(1, self.n+1, dtype=float)*.01, {}


class PairedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = loop.mg.guarded_run('test_canonical_body_paired_input', **loop.mg.BUDGETS['neural_muscle_closure'])
        cls.guard.__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.model, cls.pose, _, cls.joints, cls.geoms = loop.prepare_body()
        cls.groups = {leg: {'claw_position': [2*k], 'bristle_touch': [2*k+1]} for k, leg in enumerate(loop.LEGS)}
        cls.built = {'n': 12, 'edges': {'We': np.zeros((12, 12)), 'Wi': np.zeros((12, 12))}}
        cls.manifest = {'n_neurons': 12, 'n_actuators': cls.model.nu,
                        'rows': [{'actuator_index': i, 'local_indices': []} for i in range(cls.model.nu)]}

    def run_arm(self, arm='normal', touch=False, omitted=()):
        CapturingNetwork.inputs = []
        with patch.object(loop, 'ConductanceNetwork', CapturingNetwork), patch.object(loop, 'ChannelSensoryEncoder', FixedEncoder):
            result = loop.episode(self.model, self.pose, self.joints, self.geoms, self.built,
                                  self.manifest, self.groups, arm, .015,
                                  external_touch=touch, disabled_channels=omitted)
        return result[0], np.asarray(CapturingNetwork.inputs)

    def test_no_external_touch_does_not_inject(self):
        summary, inputs = self.run_arm()
        np.testing.assert_array_equal(inputs, np.tile(np.arange(1, 13)*.01, (30, 1)))
        self.assertFalse(any(r['external_lf_touch_stimulus'] for r in summary['records']))

    def test_channel_omission_reaches_actual_network_and_record(self):
        summary, inputs = self.run_arm(omitted=('bristle_touch',))
        self.assertTrue(np.all(inputs[:, 1::2] == 0))
        self.assertTrue(np.all(inputs[:, ::2] > 0))
        for r in summary['records']:
            for leg in loop.LEGS:
                self.assertEqual(r['actual_network_input_by_channel'][leg]['bristle_touch']['sum_nA'], 0)

    def test_sensory_cut_removes_body_and_declared_touch(self):
        summary, inputs = self.run_arm('sensory_cut', touch=True)
        self.assertTrue(np.all(inputs == 0))
        self.assertTrue(all(r['sensory_current_max_nA'] == 0 for r in summary['records']))

    def test_omission_applies_after_external_injection(self):
        _, inputs = self.run_arm(touch=True, omitted=('bristle_touch',))
        self.assertTrue(np.all(inputs[:, 1::2] == 0))

    def test_unknown_channel_fails_before_network(self):
        with patch.object(loop, 'ConductanceNetwork') as network:
            with self.assertRaises(ValueError):
                self.run_arm(omitted=('invented',))
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
