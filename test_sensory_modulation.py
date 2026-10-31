"""Pure-array regression; no neural or body simulation imports."""
import unittest

import numpy as np

from engine.embodied.sensory_modulation import BodyCurrentModulator


class ModulationTests(unittest.TestCase):
    def setUp(self):
        self.groups = {"lf": {"claw_position": [1], "bristle_touch": [3]},
                       "rh": {"claw_position": [4]}}
        self.neutral = np.array([9., -0., 8., 7., 6., 5.])
        self.live = np.array([-0., 2., -3., 4., -5., 0.])

    def make(self, gain=1., **kwargs):
        opts = dict(channels=("claw_position",), legs=("lf",), gain=gain)
        opts.update(kwargs)
        return BodyCurrentModulator(self.groups, 6, self.neutral, **opts)

    def assertBytes(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_exact_endpoints_and_formula(self):
        for gain in (0., 1., .5, 1.5, 3.):
            with self.subTest(gain=gain):
                before = self.live.copy()
                result = self.make(gain).apply(self.live)
                expected = before.copy()
                expected[1] = (self.neutral[1] if gain == 0 else before[1]
                               if gain == 1 else self.neutral[1] + gain *
                               (before[1] - self.neutral[1]))
                self.assertBytes(result, expected)
                self.assertBytes(self.live, before)
                self.assertFalse(np.shares_memory(result, self.live))

    def test_all_legs_and_empty_tier_groups(self):
        result = self.make(0, legs=("lf", "rh")).apply(self.live)
        self.assertBytes(result[[1, 4]], self.neutral[[1, 4]])
        self.assertBytes(self.make(2, channels=("hook_direction",)).apply(self.live), self.live)
        empty = BodyCurrentModulator({}, 0, np.array([], dtype=float),
                                     channels=("claw_position",))
        self.assertEqual(empty.apply(np.array([], dtype=float)).size, 0)

    def test_owned_snapshots_and_metadata(self):
        mod = self.make(0)
        self.groups["lf"]["claw_position"][0] = 2
        self.neutral[1] = 123
        result = mod.apply(self.live)
        self.assertEqual(result[1].tobytes(), np.float64(-0.).tobytes())
        self.assertEqual(result[2], self.live[2])
        record = mod.describe()
        record["legs"].clear()
        self.assertEqual(mod.describe()["legs"], ["lf"])

    def test_readonly_noncontiguous_and_repeated(self):
        live = np.arange(12., dtype=float)[::2]
        live.flags.writeable = False
        mod = self.make(1.5)
        self.assertBytes(mod.apply(live), mod.apply(live))
        self.assertBytes(mod.apply(live)[[0, 2, 3, 4, 5]], live[[0, 2, 3, 4, 5]])

    def test_external_stimulus_composed_afterwards(self):
        external = np.zeros(6)
        external[1] = .3
        for gain in (0., 1., 1.5):
            body = self.make(gain).apply(self.live)
            total = body + external
            self.assertBytes(total, body + external)
            self.assertAlmostEqual(total[1] - body[1], .3)
            self.assertBytes(external, np.array([0., .3, 0., 0., 0., 0.]))

    def test_invalid_gains(self):
        for gain in (True, np.bool_(False), -1, float("nan"), float("inf"),
                     1j, "1", [1], np.array([1.])):
            with self.subTest(gain=repr(gain)), self.assertRaises(ValueError):
                self.make(gain)

    def test_invalid_selection(self):
        for kwargs in ({"channels": "claw_position"}, {"channels": ()},
                       {"channels": ("typo",)}, {"legs": ("LF",)},
                       {"legs": ("lf", "lf")}, {"legs": None},
                       {"channels": ("claw_position", "claw_position")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_invalid_vectors_even_at_endpoints(self):
        for gain in (0, 1, 1.5):
            for bad in (np.zeros((6, 1)), np.zeros(5), np.ones(6, dtype=int),
                        np.ones(6, dtype=np.float32), np.ones(6, dtype=complex),
                        np.array([0., 0., np.nan, 0., 0., 0.]),
                        np.array(["0"] * 6), np.ones(6, dtype=object)):
                with self.subTest(gain=gain, dtype=bad.dtype):
                    with self.assertRaises(ValueError):
                        self.make(gain).apply(bad)
                    with self.assertRaises(ValueError):
                        BodyCurrentModulator(self.groups, 6, bad,
                                             channels=("claw_position",), gain=gain)

    def test_invalid_partition(self):
        for groups in ({"lf": {"claw_position": [1], "hook_direction": [1]}},
                       {"lf": {"claw_position": [6]}},
                       {"lf": {"claw_position": [True]}},
                       {"lf": {"typo": [1]}}, {"LF": {"claw_position": [1]}}):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                BodyCurrentModulator(groups, 6, self.neutral,
                                     channels=("claw_position",))

    def test_overflow_rejected_without_clipping_but_endpoints_safe(self):
        self.live[1] = np.finfo(float).max
        self.neutral[1] = -np.finfo(float).max
        for gain in (0, 1):
            self.assertTrue(np.isfinite(self.make(gain).apply(self.live)).all())
        with self.assertRaises(ValueError):
            self.make(1.5).apply(self.live)


if __name__ == "__main__":
    unittest.main()
