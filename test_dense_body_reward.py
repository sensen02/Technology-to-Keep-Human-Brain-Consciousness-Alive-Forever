"""Pure saved/synthetic evidence tests; no physics episodes or neuron input."""
import unittest
import numpy as np
from engine.embodied.body_reward import body_return
from engine.embodied.dense_body_reward import dense_body_return


class DenseBodyRewardTests(unittest.TestCase):
    def samples(self):
        return np.column_stack((np.arange(1, 9)*6.25e-6, np.ones(8),
                                np.zeros(8), np.ones(8), np.zeros(8)))

    def score(self, s, **kw):
        return dense_body_return(s, 1, 1, dt=6.25e-6, duration=50e-6, **kw)

    def test_exact_cosine_removes_observed_deadzone(self):
        s = self.samples(); s[:, 4] = .8
        self.assertAlmostEqual(self.score(s)['score'], np.cos(.8))
        self.assertEqual(body_return(s, 1, 1, dt=6.25e-6, duration=50e-6)['score'], 0)
        s[:, 4] = .81
        self.assertLess(self.score(s)['score'], np.cos(.8))
        s[:, 4] = 2
        self.assertEqual(self.score(s)['score'], 0)

    def test_caps_zero_body_and_no_count_objective(self):
        s = self.samples(); s[:, 1] = 20; s[:, 3] = 20
        self.assertEqual(self.score(s)['score'], 1)
        s[:, 2] = 1; one = self.score(s)
        s[:, 2] = 200; many = self.score(s)
        self.assertEqual(one['score'], many['score'])
        self.assertEqual(one['nonfoot_duration_s'], many['nonfoot_duration_s'])
        self.assertGreater(many['nonfoot_count_integral'], one['nonfoot_count_integral'])
        s[:, 1] = 0; s[:, 2] = 0
        self.assertEqual(self.score(s)['score'], 0)
        s[:, 1] = 1; s[:, 3] = 0
        self.assertEqual(self.score(s)['score'], 0)

    def test_complete_finite_shape_and_time_validation(self):
        s = self.samples()
        for bad in (s[1:], s[:, :4], s.copy(), s.copy(), s.copy()):
            if bad.shape == s.shape:
                if np.array_equal(bad, s): bad[0, 0] = 0
            with self.assertRaises(ValueError): self.score(bad)
        for col, val in ((1, -1), (2, -1), (3, np.nan), (4, np.inf)):
            bad = s.copy(); bad[0, col] = val
            with self.assertRaises(ValueError): self.score(bad)
        for weight, height in ((0, 1), (1, 0), (np.inf, 1)):
            with self.assertRaises(ValueError): dense_body_return(s, weight, height, dt=6.25e-6, duration=50e-6)

    def test_legacy_diagnostics_and_input_unmodified(self):
        s = self.samples(); s[::2, 2] = 3; original = s.copy()
        old = body_return(s, 1, 1, dt=6.25e-6, duration=50e-6)
        new = self.score(s)
        for k in ('nonfoot_duration_s', 'nonfoot_count_integral', 'interval_count', 'duration_s'):
            self.assertEqual(old[k], new[k])
        np.testing.assert_array_equal(s, original)


if __name__ == '__main__': unittest.main()
