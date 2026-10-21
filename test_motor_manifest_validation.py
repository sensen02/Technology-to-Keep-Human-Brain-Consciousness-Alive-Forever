"""NumPy-only motor constructor contracts; no simulation or biological claims.

Exact integer identity matters before conversion; empty and shared pools remain
valid. Consume/default connectivity/filtering/gains are deliberately unchanged.
"""
import copy
import unittest

import numpy as np

from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from test_neural_muscle_output import fixture
from engine.embodied.motor_output import build_motor_manifest


def manifest():
    return dict(n_neurons=3, n_actuators=2, rows=[
        dict(actuator_index=0, local_indices=[0, 2]),
        dict(actuator_index=1, local_indices=[])])


class MotorManifestValidationTests(unittest.TestCase):
    def test_counts_require_nonnegative_exact_integers(self):
        for field in ('n_neurons', 'n_actuators'):
            for value in (True, False, np.bool_(True), 2.5, 3.0, np.float64(2),
                          '3', None, -1, np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    m = manifest()
                    # Empty pools isolate count validation from index bounds.
                    for row in m['rows']:
                        row['local_indices'] = []
                    if field == 'n_actuators' and isinstance(value, (bool, np.bool_)):
                        m['rows'] = m['rows'][:int(value)]
                    m[field] = value
                    with self.assertRaises((ValueError, TypeError)):
                        MotorOutput(m)

    def test_actuator_order_requires_exact_integers(self):
        for value in (False, np.bool_(False), 0.0, np.float64(0), .5,
                      '0', None, np.nan, np.inf, -1, 1, [0]):
            with self.subTest(value=value):
                m = manifest()
                m['rows'][0]['actuator_index'] = value
                with self.assertRaises((ValueError, TypeError)):
                    MotorOutput(m)

    def test_indices_require_exact_integers_before_array_coercion(self):
        for indices in ([.5], [1.9], [1.0], [True], [False], [0, True],
                        [0, np.bool_(True)], [0, 1.0], ['1'], [None],
                        [np.nan], [np.inf], [-np.inf], [1+0j],
                        np.array([0., 2.]), np.array([True, False]),
                        np.array([0, True], dtype=object)):
            with self.subTest(indices=repr(indices)):
                m = manifest()
                m['rows'][0]['local_indices'] = indices
                with self.assertRaises((ValueError, TypeError)):
                    MotorOutput(m)

    def test_indices_require_one_dimension_unique_and_in_range(self):
        for indices in (1, np.array(1), [[0]], [[]], [0, 0], [-1], [3],
                        [2**100], np.array([2**64-1], dtype=np.uint64)):
            with self.subTest(indices=repr(indices)):
                m = manifest()
                m['rows'][0]['local_indices'] = indices
                with self.assertRaises((ValueError, TypeError)):
                    MotorOutput(m)

    def test_count_mismatch_and_reordered_rows_rejected(self):
        m = manifest()
        m['n_actuators'] = 1
        with self.assertRaises(ValueError):
            MotorOutput(m)
        m = manifest()
        m['rows'].reverse()
        with self.assertRaises(ValueError):
            MotorOutput(m)

    def test_unresolved_source_identity_retained_without_wiring(self):
        src, roots, names = fixture()
        unresolved = next(r for r in src['measured']['unresolved']
                          if r['actuator'] == names[-1])
        unresolved.update(leg='LF', side='left', muscle_target='test_muscle')
        m = build_motor_manifest(src, roots, names)
        row = m['rows'][-1]
        self.assertEqual({key: row[key] for key in ('leg', 'side', 'muscle_target')},
                         dict(leg='LF', side='left', muscle_target='test_muscle'))
        self.assertEqual(row['status'], 'unresolved')
        self.assertEqual(row['local_indices'], [])
        self.assertEqual(row['pool_root_ids'], [])
        out = MotorOutput(m)
        spikes = np.ones((1, len(roots)), dtype=bool)
        ctrl, _ = out.consume(spikes, spike_dt_s=.0005, start_s=0, end_s=.0005)
        self.assertEqual(ctrl[-1], 0)

    def test_numpy_integer_counts_order_and_collections_valid(self):
        for indices in ([0, np.int64(2)], (0, 2), np.array([0, 2], np.int32),
                        np.array([0, 2], np.uint64)):
            with self.subTest(indices=repr(indices)):
                m = manifest()
                m['n_neurons'], m['n_actuators'] = np.int64(3), np.uint32(2)
                m['rows'][0].update(actuator_index=np.int32(0), local_indices=indices)
                out = MotorOutput(m)
                np.testing.assert_array_equal(out.groups[0], [0, 2])
                self.assertEqual(out.groups[0].dtype, np.dtype(int))

    def test_caller_array_mutation_cannot_reroute_validated_groups(self):
        m = manifest()
        indices = np.array([0, 2], dtype=int)
        m['rows'][0]['local_indices'] = indices
        out = MotorOutput(m)
        indices[:] = [1, 1]
        np.testing.assert_array_equal(out.groups[0], [0, 2])

    def test_zero_counts_and_empty_groups_remain_valid(self):
        for n, nu in ((0, 0), (3, 0), (0, 2)):
            m = dict(n_neurons=n, n_actuators=nu, rows=[
                dict(actuator_index=i, local_indices=[]) for i in range(nu)])
            out = MotorOutput(m)
            ctrl, record = out.consume(np.zeros((2, n)), spike_dt_s=.001,
                                       start_s=0., end_s=.002)
            np.testing.assert_array_equal(ctrl, np.zeros(nu))
            self.assertEqual(record['pool_spike_counts'], [0]*nu)
        m['rows'][0]['local_indices'] = [0]
        with self.assertRaises(ValueError):
            MotorOutput(m)

    def test_builder_shared_and_missing_pools_keep_exact_filter_behavior(self):
        src, roots, names = fixture()
        m = build_motor_manifest(src, roots, names)
        original = copy.deepcopy(m)
        out = MotorOutput(m)
        self.assertEqual(m, original)
        self.assertEqual(out.config, MotorOutputConfig())
        dt = .0005
        spikes = np.zeros((1, len(roots)), bool)
        spikes[0, 0] = True
        ctrl, record = out.consume(spikes, spike_dt_s=dt, start_s=0., end_s=dt)
        rate = (1-np.exp(-dt/.005))/dt
        np.testing.assert_allclose(out.rates, [rate, rate, 0, 0, 0], rtol=1e-14)
        np.testing.assert_array_equal(ctrl, [1, 1, 0, 0, 0])
        self.assertTrue(record['connected'])
        self.assertEqual(record['pool_spike_counts'], [1, 1, 0, 0, 0])
        self.assertEqual(m['rows'][3]['status'], 'absent_from_tier')
        self.assertEqual(m['rows'][4]['status'], 'unresolved')
        ctrl, _ = out.consume(spikes, spike_dt_s=dt, start_s=dt, end_s=2*dt,
                              connected=False)
        np.testing.assert_array_equal(ctrl, np.zeros(len(names)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
