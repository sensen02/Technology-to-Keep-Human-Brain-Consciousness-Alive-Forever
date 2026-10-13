"""Small synthetic API contract tests; NumPy only, no assets/network/MuJoCo.

These are ordinary regression tests, NOT expectedFailure-decorated tests: failures
on the pre-fix encoder are measured validation gaps, not implemented repairs.
Cross-channel/cross-leg index sharing remains intentional; duplicates *within*
a single channel group are rejected. Finite signed quantities are not clipped.
Run: python -m unittest -v test_sensory_channel_validation
"""
import unittest
from dataclasses import fields

import numpy as np

from engine.embodied.sensory_channels import (
    CHANNEL_SPEC, LEGS, ChannelConfig, ChannelQuantities, ChannelSensoryEncoder,
)


class ValidInputTests(unittest.TestCase):
    def test_signed_values_normalisation_records_and_declared_zeros(self):
        groups = {'lf': {ch: [i] for i, ch in enumerate(CHANNEL_SPEC)}}
        cfg = ChannelConfig(current_nA_per_unit=.5, reference_angle_rad=.25,
                            reference_velocity_rad_s=2., reference_force=4.)
        encoder = ChannelSensoryEncoder(groups, 8, cfg)
        current, record = encoder.encode({'lf': ChannelQuantities(
            tibia_joint_dev_rad=-.5, tibia_joint_vel_rad_s=6.,
            multi_joint_dev_rad=.125, contact_force_norm=8., foot_contact=1.)})
        np.testing.assert_allclose(current, [-1., 1.5, .25, 1., 0., .5, 0., 0.])
        expected = [-2., 3., .5, 2., 0., 1., 0.]
        for ch, value in zip(CHANNEL_SPEC, expected):
            self.assertEqual(record['per_leg']['lf'][ch]['value'], value)
            self.assertEqual(record['per_leg']['lf'][ch]['n_cells'], 1)
        self.assertEqual(len(record['unmodelled_channels']), 2)
        self.assertEqual(record['scale_nA_per_unit'], .5)

    def test_shared_indices_across_channels_and_legs_add(self):
        groups = {'lf': {'claw_position': [0], 'bristle_touch': [0]},
                  'rf': {'bristle_touch': [0]}}
        current, _ = ChannelSensoryEncoder(groups, 2).encode({
            'lf': ChannelQuantities(tibia_joint_dev_rad=.3, foot_contact=1.),
            'rf': ChannelQuantities(foot_contact=1.)})
        np.testing.assert_allclose(current, [.9, 0.])

    def test_integer_lists_tuples_arrays_and_empty_groups(self):
        for indices in ([0, 2], (0, 2), np.array([0, 2], dtype=np.int32),
                        np.array([0, 2], dtype=np.uint64)):
            with self.subTest(indices=repr(indices)):
                encoder = ChannelSensoryEncoder({'lf': {'bristle_touch': indices}}, 3)
                current, _ = encoder.encode({'lf': ChannelQuantities(foot_contact=1.)})
                np.testing.assert_allclose(current, [.3, 0., .3])
        encoder = ChannelSensoryEncoder({}, 3)
        current, record = encoder.encode({})
        np.testing.assert_array_equal(current, np.zeros(3))
        self.assertEqual(record['per_leg'], {})
        self.assertEqual(set(encoder.coverage()), set(LEGS))

    def test_zero_gain_is_valid_and_large_finite_touch_is_not_clipped(self):
        groups = {'lf': {'bristle_touch': [0]}}
        q = {'lf': ChannelQuantities(foot_contact=10.)}
        current, _ = ChannelSensoryEncoder(groups, 1).encode(q)
        np.testing.assert_allclose(current, [3.])
        current, _ = ChannelSensoryEncoder(
            groups, 1, ChannelConfig(current_nA_per_unit=0.)).encode(q)
        np.testing.assert_array_equal(current, [0.])


class IndexValidationTests(unittest.TestCase):
    def assert_invalid_indices(self, indices):
        with self.assertRaises((ValueError, TypeError)):
            ChannelSensoryEncoder({'lf': {'bristle_touch': indices}}, 3)

    def test_float_indices_are_not_silently_coerced(self):
        for indices in ([.5], [1.9], [1.0], np.array([0., 2.])):
            with self.subTest(indices=repr(indices)):
                self.assert_invalid_indices(indices)

    def test_boolean_indices_are_not_integer_cell_ids(self):
        for indices in ([True], [False], [0, True], np.array([True, False])):
            with self.subTest(indices=repr(indices)):
                self.assert_invalid_indices(indices)

    def test_indices_must_be_one_dimensional(self):
        for indices in (1, np.array(1), [[0, 1]], np.array([[0], [1]]), [[]]):
            with self.subTest(indices=repr(indices)):
                self.assert_invalid_indices(indices)

    def test_duplicate_indices_within_a_channel_are_rejected(self):
        self.assert_invalid_indices([1, 1])

    def test_out_of_range_indices_are_rejected(self):
        for indices in ([-1], [3]):
            with self.subTest(indices=indices):
                self.assert_invalid_indices(indices)


class IndexOwnershipTests(unittest.TestCase):
    def test_caller_mutation_cannot_reroute_validated_sensory_group(self):
        indices = np.array([0], dtype=np.int64)
        encoder = ChannelSensoryEncoder({'lf': {'bristle_touch': indices}}, 2)
        indices[:] = 1
        current, _ = encoder.encode({'lf': ChannelQuantities(foot_contact=1.)})
        np.testing.assert_array_equal(current, [.3, 0.])


class FiniteValidationTests(unittest.TestCase):
    def test_reference_denominators_must_be_positive_finite(self):
        for name in ('reference_force', 'reference_angle_rad', 'reference_velocity_rad_s'):
            for value in (0., -1., np.nan, np.inf, -np.inf):
                with self.subTest(field=name, value=value):
                    # Either construction or encode may reject, but not an incidental
                    # ZeroDivisionError or silent NaN/Inf/negative normalisation.
                    with self.assertRaises((ValueError, TypeError)):
                        cfg = ChannelConfig(**{name: value})
                        encoder = ChannelSensoryEncoder({}, 1, cfg)
                        encoder.encode({'lf': ChannelQuantities()})

    def test_current_gain_must_be_finite_and_nonnegative(self):
        for value in (-1., np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    cfg = ChannelConfig(current_nA_per_unit=value)
                    encoder = ChannelSensoryEncoder({}, 1, cfg)
                    encoder.encode({'lf': ChannelQuantities()})

    def test_finite_inputs_that_overflow_are_rejected_not_clipped(self):
        cases = (
            (ChannelConfig(reference_angle_rad=1e-300),
             {'lf': ChannelQuantities(tibia_joint_dev_rad=1e300)}, {}),
            (ChannelConfig(current_nA_per_unit=1e300),
             {'lf': ChannelQuantities(foot_contact=1e300)},
             {'lf': {'bristle_touch': [0]}}),
            (ChannelConfig(current_nA_per_unit=1e308),
             {'lf': ChannelQuantities(foot_contact=1.),
              'rf': ChannelQuantities(foot_contact=1.)},
             {'lf': {'bristle_touch': [0]}, 'rf': {'bristle_touch': [0]}}),
        )
        for cfg, quantities, groups in cases:
            with self.subTest(cfg=cfg, quantities=quantities):
                with np.errstate(over='ignore', invalid='ignore'):
                    with self.assertRaises(ValueError):
                        ChannelSensoryEncoder(groups, 1, cfg).encode(quantities)

    def test_mutated_config_is_checked_on_encode(self):
        cfg = ChannelConfig()
        encoder = ChannelSensoryEncoder({}, 1, cfg)
        cfg.reference_force = 0.
        with self.assertRaises((ValueError, TypeError)):
            encoder.encode({})

    def test_finite_negative_quantities_are_not_newly_restricted(self):
        encoder = ChannelSensoryEncoder({'lf': {'bristle_touch': [0]}}, 1)
        current, record = encoder.encode({'lf': ChannelQuantities(
            contact_force_norm=-2., foot_contact=-1.)})
        np.testing.assert_array_equal(current, [-.3])
        self.assertEqual(record['per_leg']['lf']['campaniform_strain']['value'], -2.)

    def test_every_quantity_must_be_finite_even_for_empty_groups(self):
        for field in fields(ChannelQuantities):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field.name, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        quantity = ChannelQuantities(**{field.name: value})
                        ChannelSensoryEncoder({}, 1).encode({'lf': quantity})


class NeuronCountValidationTests(unittest.TestCase):
    def test_neuron_count_requires_nonnegative_integer_not_bool(self):
        for count in (1.5, 2.0, True, False, -1):
            with self.subTest(count=count):
                with self.assertRaises((ValueError, TypeError)):
                    ChannelSensoryEncoder({}, count)

    def test_integer_and_zero_neuron_counts_remain_valid(self):
        for count in (0, 2, np.int64(2)):
            with self.subTest(count=count):
                current, _ = ChannelSensoryEncoder({}, count).encode({})
                self.assertEqual(current.shape, (int(count),))


if __name__ == '__main__':
    unittest.main()
