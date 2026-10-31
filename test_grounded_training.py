"""Synthetic scoring/optimizer tests; no neural or MuJoCo execution."""
import unittest
import numpy as np
from engine.embodied.grounded_training import score_trajectory, propose_weights


class GroundedTrainingTests(unittest.TestCase):
    def fixture(self):
        q = np.zeros((4001, 49)); q[:, 2] = 1.; q[:, 3] = 1.
        return dict(qpos=q, qvel=np.zeros((4001, 48)),
                    contacts=np.ones((4001, 6)), foot_force=np.tile([0., 0., 1.], (4001, 6, 1)),
                    nonfoot_depth=np.zeros(4001), limit_excess=np.zeros(4001),
                    reference_qpos=q[0].copy(), weight_native=6.)

    def test_grounded_support_passes(self):
        result = score_trajectory(**self.fixture())
        self.assertTrue(result['eligible']); self.assertEqual(result['prefix_ms'], 200.)
        self.assertEqual(result['loss'], 0.)

    def test_contact_without_load_fails(self):
        data = self.fixture(); data['foot_force'][:] = 0
        result = score_trajectory(**data)
        self.assertFalse(result['eligible']); self.assertGreater(result['loss'], 0)

    def test_suspended_force_without_contacts_fails(self):
        data = self.fixture(); data['contacts'][:] = 0
        self.assertFalse(score_trajectory(**data)['eligible'])

    def test_terminal_preparation_is_required(self):
        data = self.fixture(); data['contacts'][1999] = 0
        result = score_trajectory(**data)
        self.assertFalse(result['preparation_pass']); self.assertTrue(result['evaluation_pass'])
        self.assertFalse(result['eligible'])

    def test_late_failure_not_hidden_by_prefix(self):
        data = self.fixture(); data['nonfoot_depth'][-1] = .02
        result = score_trajectory(**data)
        self.assertFalse(result['eligible']); self.assertAlmostEqual(result['prefix_ms'], 199.9)

    def test_incomplete_and_nonfinite_rejected(self):
        data = self.fixture(); data['qpos'] = data['qpos'][:-1]
        with self.assertRaises(ValueError): score_trajectory(**data)
        data = self.fixture(); data['qvel'][1, 0] = np.nan
        with self.assertRaises(ValueError): score_trajectory(**data)

    def test_weight_step_bounded_and_mask_retained(self):
        w = np.ones((66, 5)); w[0] = 0
        plus = np.where(w > 0, w+.02, 0); minus = np.where(w > 0, w-.02, 0)
        new = propose_weights(w, 1e6, 1., plus, minus, 0)
        self.assertLessEqual(np.abs(new-w).max(), .02+1e-14)
        np.testing.assert_array_equal(new[0], w[0])
        self.assertTrue(np.all(new[1:] < w[1:]))

    def test_bad_probe_rejected(self):
        w = np.ones((2, 5))
        with self.assertRaises(ValueError): propose_weights(w, 2., 1., w, w, 0)
        w[0] = 0; plus = w+.01; minus = np.maximum(w-.01, 0)
        with self.assertRaises(ValueError): propose_weights(w, 2., 1., plus, minus, 0)


if __name__ == '__main__':
    unittest.main()
