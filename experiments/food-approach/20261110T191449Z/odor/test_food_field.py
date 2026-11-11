"""Numerical/causal contracts, independent analytic expectations, no simulation."""
import dataclasses
import math
import unittest
import numpy as np
from food_field import FoodSource, BilateralOdorSensory, bilateral_positions


def source(**kwargs):
    config = dict(center_xy=(0, 0), patch_radius=0.2, amplitude=2.0,
                  sigma=1.0, onset_s=0.0)
    config.update(kwargs)
    return FoodSource(**config)


class FieldTests(unittest.TestCase):
    def test_gaussian_and_patch_metadata(self):
        points = [[0, 0], [1, 0]]
        expected = [2, 2 * math.exp(-0.5)]
        np.testing.assert_allclose(source().concentration(points, 1), expected, rtol=1e-15)
        np.testing.assert_array_equal(source(patch_radius=100).concentration(points, 1), source().concentration(points, 1))

    def test_smooth_onset_and_causality(self):
        field = source(onset_s=1)
        for t, expected in ((0, 0), (1, 0), (1.1, 1), (1.2, 2), (2, 2)):
            np.testing.assert_allclose(field.concentration(np.zeros((2, 2)), t), expected, atol=2e-15)
        sensory = BilateralOdorSensory(field)
        # The interval crosses onset; START sample is still exactly zero.
        np.testing.assert_array_equal(sensory.sample(np.zeros((2, 2)), 0.99, 0.02), [0, 0])
        self.assertTrue((sensory.sample(np.zeros((2, 2)), 1.01) > 0).all())

    def test_symmetry(self):
        sensor = BilateralOdorSensory(source(center_xy=(2, 0)))
        points = bilateral_positions([0, 0, 1], np.eye(3), .1, .2)
        for i in range(40):
            output = sensor.sample(points, i * .005)
            self.assertEqual(output[0], output[1])

    def test_mirror_swaps(self):
        points = bilateral_positions([0, 0], np.eye(2), .1, .3)
        left = BilateralOdorSensory(source(center_xy=(.8, .6)))
        right = BilateralOdorSensory(source(center_xy=(.8, -.6)))
        for i in range(100):
            a, b = left.sample(points, i * .005), right.sample(points, i * .005)
            np.testing.assert_array_equal(a, b[::-1])
        self.assertGreater(a[0], a[1])

    def test_pose_transform_and_projection(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(bilateral_positions([2, 3, 7], rotation, .5, .25), [[1.75, 3.5], [2.25, 3.5]])
        # Head pitch projects forward displacement onto world z before XY.
        pitch = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        np.testing.assert_array_equal(bilateral_positions([2, 3, 7], pitch, .5, .25), [[2, 3.25], [2, 2.75]])

    def test_immutable_and_defensive_center(self):
        center = np.array([1., 2.])
        field = source(center_xy=center)
        center[:] = 100
        self.assertEqual(field.center_xy, (1., 2.))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            field.sigma = 2

    def test_finite_extreme_concentrations(self):
        field = source(center_xy=(1e308, -1e308), sigma=1e-308, amplitude=1e308)
        np.testing.assert_array_equal(field.concentration([[-1e308, 1e308], [1e308, -1e308]], 1), [0, 1e308])
        sensor = BilateralOdorSensory(field, K=1e308)
        out = sensor.sample([[1e308, -1e308]] * 2, 1)
        np.testing.assert_allclose(out, .5 * (1 - math.exp(-.005/.05)), rtol=1e-14)
        self.assertTrue(np.isfinite(out).all())


class ReceptorTests(unittest.TestCase):
    def test_exact_analytic_constant_stimulus(self):
        sensor = BilateralOdorSensory(source(), K=2)
        output = sensor.sample(np.zeros((2, 2)), 1, .005)
        np.testing.assert_allclose(output, .5 * (1 - math.exp(-.1)), atol=1e-16, rtol=0)

    def test_exact_timestep_composition(self):
        points = [[.3, .1], [.2, -.4]]
        single = BilateralOdorSensory(source()).sample(points, 1, .1)
        split = BilateralOdorSensory(source())
        for i in range(20):
            output = split.sample(points, 1 + i * .005)
        np.testing.assert_allclose(output, single, atol=1e-14, rtol=0)

    def test_decay_is_causal_memory(self):
        sensory = BilateralOdorSensory(source())
        old = sensory.sample(np.zeros((2, 2)), 1)
        output = sensory.sample([[1e6, 0]] * 2, 1.005)
        np.testing.assert_allclose(output, old * math.exp(-.1), atol=1e-16)
        self.assertTrue((output > 0).all())

    def test_seeded_noise_range_and_reproducibility(self):
        a = BilateralOdorSensory(source(), noise_std=3, seed=42)
        b = BilateralOdorSensory(source(), noise_std=3, seed=42)
        c = BilateralOdorSensory(source(), noise_std=3, seed=43)
        different = False
        for i in range(80):
            first = a.sample([[0, 0]] * 2, i * .005)
            second = b.sample([[0, 0]] * 2, i * .005)
            third = c.sample([[0, 0]] * 2, i * .005)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all() and (first >= 0).all() and (first <= 1).all())
            different |= not np.array_equal(first, third)
        self.assertTrue(different)

    def test_reject_does_not_advance_state_or_rng(self):
        a = BilateralOdorSensory(source(), noise_std=.1, seed=5)
        b = BilateralOdorSensory(source(), noise_std=.1, seed=5)
        a.sample([[0, 0]] * 2, 1)
        b.sample([[0, 0]] * 2, 1)
        for points, time, dt in (([[0, 0]] * 2, 1, .005), ([[0, 0]] * 2, 2, .005), ([[math.nan, 0]] * 2, 1.005, .005), ([[0, 0]] * 2, 1.005, 0)):
            with self.assertRaises(ValueError):
                a.sample(points, time, dt)
        np.testing.assert_array_equal(a.sample([[0, 0]] * 2, 1.005), b.sample([[0, 0]] * 2, 1.005))

    def test_output_does_not_alias_latent_state(self):
        a, b = BilateralOdorSensory(source()), BilateralOdorSensory(source())
        out = a.sample([[0, 0]] * 2, 1)
        b.sample([[0, 0]] * 2, 1)
        self.assertEqual(out.shape, (2,))
        out[:] = -1e10
        np.testing.assert_array_equal(a.sample([[0, 0]] * 2, 1.005), b.sample([[0, 0]] * 2, 1.005))


class ValidationTests(unittest.TestCase):
    def test_source_rejections(self):
        for name, values in dict(patch_radius=[0, -1, math.inf], sigma=[0, -1, math.nan], amplitude=[-1, math.inf], onset_s=[-1, math.nan], rise_s=[0, -1, math.inf], center_xy=[(0,), (0, math.nan)]).items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    source(**{name: value})

    def test_receptor_rejections(self):
        for config in (dict(K=0), dict(K=math.inf), dict(tau_s=0), dict(tau_s=math.nan), dict(noise_std=-1), dict(noise_std=math.inf), dict(noise_std=.1), dict(seed=-1), dict(seed=1.5), dict(seed=True)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                BilateralOdorSensory(source(), **config)
        for time, dt in ((-1, .005), (math.nan, .005), (math.inf, .005), (0, -1), (0, math.nan), (0, math.inf), (1e308, 1e308), (1e308, .005)):
            with self.subTest(time=time, dt=dt), self.assertRaises(ValueError):
                BilateralOdorSensory(source()).sample([[0, 0]] * 2, time, dt)
        for points in ([[0, 0]], [[0, 0, 0]] * 2, [[0, math.inf]] * 2):
            with self.assertRaises(ValueError):
                BilateralOdorSensory(source()).sample(points, 0)

    def test_pose_rejections(self):
        for position, rotation, forward, lateral in (([0, 0], np.zeros((2, 2)), 1, 1), ([0, 0], np.diag([1, -1]), 1, 1), ([0, math.nan], np.eye(2), 1, 1), ([0, 0], np.eye(2), math.inf, 1), ([0, 0], np.eye(2), 1, -1), ([0, 0], np.eye(3), 1, 1), ([1e308, 0], np.eye(2), 1e308, 1)):
            with self.assertRaises(ValueError):
                bilateral_positions(position, rotation, forward, lateral)


if __name__ == '__main__':
    unittest.main()
