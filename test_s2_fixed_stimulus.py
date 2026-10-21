"""Synthetic helper tests only; no canonical CLI simulation."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
import run_s2_fixed_stimulus as s2


class MomentTests(unittest.TestCase):
    def setUp(self):
        self.model = NS(nu=2, nv=3)
        self.expected = np.array([[0., 2., -3.], [4., 0., 0.]])

    def test_dense_matrix_and_copy(self):
        data = NS(actuator_moment=self.expected)
        got = s2.dense_moment(self.model, data)
        np.testing.assert_array_equal(got, self.expected)
        got[0, 0] = 99
        self.assertEqual(self.expected[0, 0], 0)

    def test_legacy_flat_dense(self):
        np.testing.assert_array_equal(s2.dense_moment(self.model,
            NS(actuator_moment=self.expected.ravel())), self.expected)

    def test_sparse_with_gaps_and_unsorted_columns(self):
        data = NS(actuator_moment=np.array([-3., 2., 77., 4., 99., 99.]),
            moment_rownnz=np.array([2, 1]), moment_rowadr=np.array([0, 3]),
            moment_colind=np.array([2, 1, 0, 0, 1, 1]))
        # Buffer size equals dense size: still sparse because metadata exists.
        np.testing.assert_array_equal(s2.dense_moment(self.model, data), self.expected)

    def test_sparse_zero_rows(self):
        got = s2.dense_moment(self.model, NS(actuator_moment=np.array([]),
            moment_rownnz=np.array([0, 0]), moment_rowadr=np.array([0, 0]),
            moment_colind=np.array([], dtype=int)))
        np.testing.assert_array_equal(got, np.zeros((2, 3)))

    def test_invalid_sparse_rejected(self):
        base = dict(actuator_moment=np.array([1., 2.]), moment_rownnz=np.array([1, 1]),
            moment_rowadr=np.array([0, 1]), moment_colind=np.array([0, 1]))
        variants = [dict(moment_rowadr=np.array([-1, 1])),
            dict(moment_colind=np.array([3, 1])), dict(moment_rownnz=np.array([3, 1])),
            dict(moment_colind=np.array([0., 1.])),
            dict(moment_rownnz=np.array([2, 0]), moment_colind=np.array([0, 0]))]
        for patch in variants:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                s2.dense_moment(self.model, NS(**(base | patch)))
        with self.assertRaises(ValueError):
            s2.dense_moment(self.model, NS(actuator_moment=np.zeros(6), moment_rowadr=np.zeros(2)))

    def test_factorization_and_tolerance(self):
        force = np.array([5., -2.])
        qfrc = self.expected.T @ force
        calculated, residual, ok = s2.factorization(self.expected, force, qfrc)
        self.assertTrue(ok)
        np.testing.assert_array_equal(calculated, [-8., 10., -15.])
        np.testing.assert_array_equal(residual, 0.)
        self.assertFalse(s2.factorization(self.expected, force, qfrc+1e-5)[2])
        self.assertTrue(s2.factorization(self.expected, force, qfrc+1e-11)[2])


class MappingTests(unittest.TestCase):
    def test_activation_nonidentity_and_multistate(self):
        model = NS(nu=3, na=3, actuator_actadr=np.array([2, -1, 0]),
                   actuator_actnum=np.array([1, 0, 2]))
        self.assertEqual(s2.activation_slices(model), [[2], [], [0, 1]])

    def test_activation_bad_layouts(self):
        for adr, num, na in [([0, 0], [1, 1], 2), ([0, 2], [1, 1], 2),
                             ([-1, 0], [1, 1], 2), ([0, 1], [0, 1], 2),
                             ([0, -1], [1, 0], 2)]:
            with self.subTest(adr=adr), self.assertRaises(ValueError):
                s2.activation_slices(NS(nu=2, na=na, actuator_actadr=np.array(adr), actuator_actnum=np.array(num)))

    def test_exact_first_difference_and_empty_activation(self):
        times = np.array([.0001, .0002, .0003])
        a = np.zeros((3, 2)); b = a.copy(); b[1, 0] = 1e-200
        self.assertEqual(s2.first_difference(a, b, times), .0002)
        self.assertIsNone(s2.first_difference(a, a, times))
        self.assertIsNone(s2.first_difference(np.zeros((3, 0)), np.zeros((3, 0)), times))
        with self.assertRaises(ValueError):
            s2.first_difference(a, b[:2], times)

    def test_control_validation_refuses_adjustment(self):
        model = NS(nu=2, actuator_ctrlrange=np.array([[.0001, 1.], [0., 1.]]),
                   actuator_ctrllimited=np.ones(2))
        s2.validate_controls(model)
        model.actuator_ctrlrange[1, 1] = .05
        with self.assertRaises(ValueError):
            s2.validate_controls(model)


class FakeData:
    def __init__(self, model):
        for key, size in [('qpos', 1), ('qvel', 1), ('act', 2), ('actuator_force', 2),
                          ('qfrc_actuator', 1), ('qfrc_passive', 1), ('qfrc_bias', 1),
                          ('qfrc_constraint', 1), ('ctrl', 2)]:
            setattr(self, key, np.zeros(size))
        self.actuator_moment = np.array([[1.], [2.]])
        self.warning = NS(number=np.zeros(8, int))
        self.time = 0.


class FakeMuJoCo:
    MjData = FakeData
    @staticmethod
    def mj_forward(model, data):
        data.actuator_force[:] = data.act
        data.qfrc_actuator[:] = data.actuator_moment.T @ data.actuator_force
    @staticmethod
    def mj_step(model, data):
        data.act[:] += data.ctrl * s2.DT
        data.qvel[:] += data.act.sum()*s2.DT
        data.qpos[:] += data.qvel*s2.DT
        data.time += s2.DT


class CaseTests(unittest.TestCase):
    def test_fresh_cases_fixed_ctrl_natural_activation_and_endstep(self):
        model = NS(nu=2, nv=1)
        zero, error = s2.run_case(model, np.array([7.]), None, FakeMuJoCo)
        self.assertIsNone(error)
        repeat, error = s2.run_case(model, np.array([7.]), None, FakeMuJoCo)
        for k in zero:
            np.testing.assert_array_equal(zero[k], repeat[k])
        stimulated, error = s2.run_case(model, np.array([7.]), 1, FakeMuJoCo)
        self.assertIsNone(error)
        self.assertEqual(stimulated['qpos'].shape, (100, 1))
        self.assertEqual(stimulated['moment'].shape, (100, 2, 1))
        np.testing.assert_array_equal(stimulated['initial_ctrl'], [0., 0.])
        np.testing.assert_array_equal(stimulated['initial_act'], [0., 0.])
        self.assertTrue(np.all(stimulated['ctrl'][:, 1] == .1))
        self.assertAlmostEqual(stimulated['act'][-1, 1], .001)
        self.assertGreater(stimulated['qpos'][-1, 0], 7.)
        np.testing.assert_array_equal(stimulated['actuator_force'], stimulated['act'])
        self.assertAlmostEqual(stimulated['time_s'][-1], .01)
        summary = s2.summarize_case(stimulated, zero, 1, [[0], [1]])
        self.assertEqual(summary['first_activation_difference_s'], .0001)
        self.assertTrue(summary['factorization_allclose'])

    def test_warning_preserves_partial_failure_arrays(self):
        class Bad(FakeMuJoCo):
            @staticmethod
            def mj_step(model, data):
                FakeMuJoCo.mj_step(model, data)
                data.warning.number[0] += 1
        arrays, error = s2.run_case(NS(nu=2, nv=1), np.array([7.]), 0, Bad)
        self.assertIn('warning', error)
        self.assertEqual(len(arrays['time_s']), 1)
        self.assertEqual(arrays['warning_counts'][0, 0], 1)


if __name__ == '__main__':
    unittest.main()
