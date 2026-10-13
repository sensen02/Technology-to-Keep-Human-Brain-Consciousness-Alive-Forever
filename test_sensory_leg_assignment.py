"""Synthetic row-assignment contracts: NumPy only, no assets/model/neural run.

Run: python -m unittest -v test_sensory_leg_assignment
Failures before the repair expose real defects, not expectedFailure placeholders.
"""
import unittest

import numpy as np

from engine.embodied.sensory_channels import ChannelSensoryEncoder, LEGS


class SensoryLegAssignmentTests(unittest.TestCase):
    def encoder(self, size=66):
        encoder = ChannelSensoryEncoder({}, 0)
        encoder.set_reference_pose(np.arange(size, dtype=float) / 10)
        return encoder

    def quantities(self, encoder, leg='lf', q=None, w=None):
        ref = encoder.reference_angles
        return encoder.quantities(leg, ref + 1 if q is None else q,
                                  np.ones(ref.size) if w is None else w, 2., 1.)

    def test_two_legs_independent_in_both_assignment_orders(self):
        for order in (('lf', 'rf'), ('rf', 'lf')):
            with self.subTest(order=order):
                encoder = self.encoder()
                rows = {'lf': (1, 2, [1, 2]), 'rf': (4, 5, [4, 5])}
                for leg in order:
                    encoder.assign(leg, *rows[leg])
                dev = np.zeros(66)
                dev[[1, 2, 4, 5]] = [-2., -4., 5., 7.]
                w = np.zeros(66)
                w[[1, 4]] = [-9., 11.]
                for leg, expected in [('lf', (-2., -9., -3.)),
                                      ('rf', (5., 11., 6.))]:
                    actual = self.quantities(encoder, leg, encoder.reference_angles + dev, w)
                    self.assertEqual((actual.tibia_joint_dev_rad,
                                      actual.tibia_joint_vel_rad_s,
                                      actual.multi_joint_dev_rad), expected)
                    self.assertEqual((actual.contact_force_norm, actual.foot_contact), (2., 1.))

    def test_small_nonempty_reference_and_signed_values(self):
        for size in (1, 6, 67):
            with self.subTest(size=size):
                encoder = self.encoder(size)
                encoder.assign('lf', 0, 0, [0])
                q = self.quantities(encoder, q=encoder.reference_angles - 2,
                                    w=np.full(size, -3.))
                self.assertEqual((q.tibia_joint_dev_rad, q.tibia_joint_vel_rad_s,
                                  q.multi_joint_dev_rad), (-2., -3., -2.))

    def test_case_aliases_address_same_mapping(self):
        encoder = self.encoder()
        encoder.assign('LF', 2, 3, [2, 3])
        self.assertEqual(self.quantities(encoder, 'LF'), self.quantities(encoder, 'lf'))
        encoder.assign('lf', 4, 5, [4])
        q = self.quantities(encoder, 'LF', q=encoder.reference_angles + np.arange(66))
        self.assertEqual(q.tibia_joint_dev_rad, 4.)

    def test_explicit_shared_rows_across_all_legs_allowed(self):
        encoder = self.encoder()
        for leg in LEGS:
            encoder.assign(leg, np.int64(2), np.int32(3), (2, 3))
        self.assertTrue(all(self.quantities(encoder, leg) == self.quantities(encoder)
                            for leg in LEGS))

    def test_missing_mapping_never_falls_back(self):
        encoder = self.encoder()
        with self.assertRaises(ValueError):
            self.quantities(encoder)
        encoder.assign('lf', 0, 1, [0, 1])
        with self.assertRaises(ValueError):
            self.quantities(encoder, 'rf')

    def test_unknown_legs_rejected_for_assign_and_quantities(self):
        encoder = self.encoder()
        encoder.assign('lf', 0, 1, [0, 1])
        for leg in ('unknown', '', None, 1):
            with self.subTest(leg=leg), self.assertRaises(ValueError):
                encoder.assign(leg, 0, 1, [0, 1])
            with self.subTest(leg=leg), self.assertRaises(ValueError):
                self.quantities(encoder, leg)

    def test_reference_reset_invalidates_every_mapping(self):
        encoder = self.encoder()
        for leg in LEGS:
            encoder.assign(leg, 0, 1, [0, 1])
        encoder.set_reference_pose(np.ones(66))
        for leg in LEGS:
            with self.subTest(leg=leg), self.assertRaises(ValueError):
                self.quantities(encoder, leg)
        encoder.assign('lf', 0, 1, [0, 1])
        self.assertEqual(self.quantities(encoder).tibia_joint_dev_rad, 1.)

    def test_reference_requires_nonempty_finite_1d(self):
        for ref in ([], np.zeros((66, 1)), np.full(66, np.nan),
                    np.full(66, np.inf), np.full(66, -np.inf)):
            with self.subTest(ref=repr(ref)), self.assertRaises(ValueError):
                ChannelSensoryEncoder({}, 0).set_reference_pose(ref)

    def test_reference_is_copied(self):
        ref = np.zeros(66)
        encoder = ChannelSensoryEncoder({}, 0)
        encoder.set_reference_pose(ref)
        ref[:] = 9
        np.testing.assert_array_equal(encoder.reference_angles, np.zeros(66))

    def test_assignment_requires_reference(self):
        with self.assertRaises(ValueError):
            ChannelSensoryEncoder({}, 0).assign('lf', 0, 1, [0, 1])

    def test_scalar_rows_require_integer_type_and_bounds(self):
        for invalid in (-1, 66, 1.5, 1.0, True, np.bool_(False), '1', np.nan, np.inf):
            for field in (0, 1):
                encoder = self.encoder()
                args = [0, 1, [0, 1]]
                args[field] = invalid
                with self.subTest(field=field, value=repr(invalid)), self.assertRaises(ValueError):
                    encoder.assign('lf', *args)

    def test_multi_rows_require_nonempty_unique_1d_integer_bounds(self):
        for invalid in ([], [1, 1], [-1, 0], [0, 66], [0, 1.5], [0, 1.0],
                        [0, True], ['1'], [[0, 1]], 1, [np.nan], [np.inf]):
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                self.encoder().assign('lf', 0, 1, invalid)

    def test_assigned_rows_are_copied(self):
        encoder = self.encoder()
        rows = np.array([0, 1])
        encoder.assign('lf', 0, 1, rows)
        rows[:] = 4
        result = self.quantities(encoder, q=encoder.reference_angles + np.arange(66))
        self.assertEqual(result.multi_joint_dev_rad, .5)

    def test_q_and_w_require_exact_reference_shape(self):
        for field in ('q', 'w'):
            for invalid in (np.zeros(65), np.zeros(67), np.zeros((66, 1)), 0., []):
                encoder = self.encoder()
                encoder.assign('lf', 0, 1, [0, 1])
                with self.subTest(field=field, value=repr(invalid)), self.assertRaises(ValueError):
                    self.quantities(encoder, **{field: invalid})

    def test_q_and_w_require_all_finite_even_outside_assigned_rows(self):
        for field in ('q', 'w'):
            for value in (np.nan, np.inf, -np.inf):
                encoder = self.encoder()
                encoder.assign('lf', 0, 1, [0, 1])
                invalid = np.zeros(66)
                invalid[65] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.quantities(encoder, **{field: invalid})

    def test_contact_inputs_are_finite_scalars(self):
        encoder = self.encoder()
        encoder.assign('lf', 0, 1, [0, 1])
        for field in (0, 1):
            for invalid in (np.nan, np.inf, -np.inf, [1.]):
                args = [2., 1.]
                args[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    encoder.quantities('lf', np.zeros(66), np.zeros(66), *args)


if __name__ == '__main__':
    unittest.main()
