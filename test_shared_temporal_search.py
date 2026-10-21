"""Pure array-only search tests: no neuron execution or physics imports."""
import unittest
import numpy as np
from engine.embodied.body_reward import AntitheticBodySearch
from engine.embodied.shared_temporal_search import SharedTemporalBodySearch


class SharedTemporalTests(unittest.TestCase):
    def initial(self):
        return np.tile([.4, .25, .2, .1, .05], (66, 1))

    def test_five_private_rng_signs_shared_shapes_and_bounds(self):
        np.random.seed(7)
        before = np.random.get_state()
        search = SharedTemporalBodySearch(self.initial())
        rng = np.random.default_rng(20261018)
        for _ in range(6):
            direction, plus, minus = search.probes()
            np.testing.assert_array_equal(direction, np.tile(rng.choice([-1., 1.], size=5), (66, 1)))
            for a in (direction, plus, minus):
                self.assertEqual(a.shape, (66, 5))
                np.testing.assert_array_equal(a, np.tile(a[0], (66, 1)))
            np.testing.assert_array_equal(plus, np.clip(search.center+.025*direction, 0, 4))
            np.testing.assert_array_equal(minus, np.clip(search.center-.025*direction, 0, 4))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]);np.testing.assert_array_equal(before[1], after[1]);self.assertEqual(before[2:], after[2:])
        edge = self.initial();edge[:, 0] = 0;edge[:, 1] = 4
        for a in SharedTemporalBodySearch(edge).probes()[1:]:
            self.assertTrue(np.all((a >= 0) & (a <= 4)))

    def test_inherited_global_winner_exact_ties(self):
        shared = SharedTemporalBodySearch(self.initial())
        old = AntitheticBodySearch(self.initial())
        for scores in [(1., 1.), (.5, .6), (2., 3.)]:
            _, plus, minus = shared.probes()
            pair = [dict(label=f'{shared.round}_{sign}', weights=w,
                         **{'return':dict(score=score, nonfoot_duration_s=.5, nonfoot_count_integral=1.)})
                    for sign, w, score in [('plus', plus, scores[0]), ('minus', minus, scores[1])]]
            self.assertEqual(shared.update(pair), old.update(pair))
            np.testing.assert_array_equal(shared.center, old.center)
        self.assertEqual(shared.best_label, '2_minus')

    def test_reject_nonshared_initial_and_update(self):
        bad = self.initial();bad[1, 0] += .01
        with self.assertRaises(ValueError):SharedTemporalBodySearch(bad)
        shared = SharedTemporalBodySearch(self.initial())
        with self.assertRaises(ValueError):shared.update([dict(weights=bad)])


if __name__ == '__main__':
    unittest.main()
