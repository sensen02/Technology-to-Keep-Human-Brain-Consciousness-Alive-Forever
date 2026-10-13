"""Real canonical MuJoCo timing with a synthetic silent one-cell neural stub.

Manual prototype integration tests only; no biological network/behaviour claim.
The shared heavy-job guard serializes even this bounded physics-only test.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import run_neural_muscle_closure as clo
from engine import memory_guard as mg


class SilentNetwork:
    def __init__(self, n, params, dt_ms):
        self.n, self.dt, self.steps, self.t_ms = n, dt_ms, 0, 0.0

    def set_connectivity(self, exc, inh):
        pass

    def step(self, current_nA):
        self.steps += 1
        self.t_ms = self.steps * self.dt
        return np.zeros(self.n, dtype=bool)


class SilentEncoder:
    def __init__(self, groups, n, config):
        self.n = n

    def encode(self, quantities):
        return np.zeros(self.n), {}


class EpisodeClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = mg.guarded_run("test_closure_episode_clock",
                                   **mg.BUDGETS["neural_muscle_closure"])
        cls.guard.__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.model, cls.seated, _, cls.joints, cls.geoms = clo.prepare_body()
        cls.manifest = {"n_neurons": 1, "n_actuators": cls.model.nu,
                        "rows": [{"actuator_index": i, "local_indices": []}
                                 for i in range(cls.model.nu)]}
        cls.built = {"n": 1, "edges": {"We": np.zeros((1, 1)), "Wi": np.zeros((1, 1))}}
        cls.groups = {"lf": {"bristle_touch": []}}

    def run_episode(self, start=None, end=None, duration=.010, replay=None):
        push = None if start is None else {"start_s": start, "end_s": end,
                                          "force_model": [.01, 0.0, 0.0]}
        forces = []
        real_step = mujoco.mj_step
        thorax = self.model.body("Thorax").id

        def witnessed_step(model, data):
            forces.append(data.xfrc_applied[thorax, :3].copy())
            real_step(model, data)

        with patch.object(clo, "ConductanceNetwork", SilentNetwork), \
             patch.object(clo, "ChannelSensoryEncoder", SilentEncoder), \
             patch.object(clo, "observe_quantities", return_value={}), \
             patch.object(mujoco, "mj_step", side_effect=witnessed_step):
            result = clo.episode(self.model, self.seated, self.joints, self.geoms,
                                 self.built, self.manifest, self.groups,
                                 "replay" if replay is not None else "normal", duration,
                                 replay=replay, push=push)
        return result, np.asarray(forces)

    def assert_window(self, start, end, duration, expected_counts):
        (summary, _, controls, _), forces = self.run_episode(start, end, duration)
        records = summary["records"]
        self.assertEqual([r["external_push_body_steps"] for r in records], expected_counts)
        self.assertEqual(np.count_nonzero(forces[:, 0]), sum(expected_counts))
        np.testing.assert_allclose(forces.sum(axis=0)*clo.DT_BODY,
                                   [.01*(end-start), 0, 0], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(np.sum([r["external_push_impulse_model"] for r in records], axis=0),
                                   forces.sum(axis=0)*clo.DT_BODY, atol=1e-15)
        expected = np.zeros_like(forces)
        expected[round(start/clo.DT_BODY):round(end/clo.DT_BODY), 0] = .01
        np.testing.assert_array_equal(forces, expected)
        self.assertAlmostEqual(records[-1]["time_end_s"], duration, places=12)
        self.assertFalse(controls.any())

    def test_one_to_two_ms_force_reaches_actual_body_steps(self):
        self.assert_window(.001, .002, .010, [10, 0])

    def test_one_to_six_ms_crosses_command_boundary(self):
        self.assert_window(.001, .006, .010, [40, 10])

    def test_legacy_grid_window_retains_fifty_steps(self):
        self.assert_window(.015, .020, .025, [0, 0, 0, 50, 0])

    def test_no_push_zero_replay_is_identical(self):
        (normal, trajectory, controls, _), forces = self.run_episode()
        (replay, replay_trajectory, _, _), replay_forces = self.run_episode(replay=controls)
        np.testing.assert_array_equal(trajectory, replay_trajectory)
        self.assertFalse(forces.any())
        self.assertFalse(replay_forces.any())
        self.assertTrue(all(not r["external_push_applied"] for r in normal["records"]))

    def test_wrong_body_clock_rejected_before_network(self):
        previous = self.model.opt.timestep
        try:
            self.model.opt.timestep = .0002
            with patch.object(clo, 'ConductanceNetwork') as network:
                with self.assertRaises(ValueError):
                    clo.episode(self.model, self.seated, self.joints, self.geoms,
                                self.built, self.manifest, self.groups, 'normal', .01)
                network.assert_not_called()
        finally:
            self.model.opt.timestep = previous

    def test_invalid_duration_rejected_before_neural_initialization(self):
        with patch.object(clo, "ConductanceNetwork") as network:
            for duration in (.012, .002, 0, -1, float("nan"), float("inf")):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    clo.episode(self.model, self.seated, self.joints, self.geoms,
                                self.built, self.manifest, self.groups, "normal", duration)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
