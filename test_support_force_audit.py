"""Focused serial guarded force audit regressions; no canonical modifications."""
import run_support_force_audit as audit  # establishes thread limits before NumPy import
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np


class SupportForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = audit.cp.sha256_file(audit.cp.CANONICAL_XML)
        cls.doc = json.loads(audit.SOURCE.read_text())
        cls.result = json.loads(audit.OUT.read_text())
        cls.model = audit.cp.load_model()

    @classmethod
    def tearDownClass(cls):
        assert cls.before == audit.cp.sha256_file(audit.cp.CANONICAL_XML)

    def test_contact_sign_independent_sphere_plane(self):
        probe = audit.convention_probe()
        self.assertTrue(probe['upward_force'])
        self.assertTrue(probe['force_balance']['within_tolerance'], probe)
        self.assertLess(probe['force_balance']['max_abs_error'], 1e-10)

    def test_contact_sign_both_geom_orders_and_rotated_frame(self):
        frame = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        local = np.array([4., 2., 3.])
        np.testing.assert_array_equal(audit.force_on_nonfloor(frame, local, 0, 2, 0), [2., 3., 4.])
        np.testing.assert_array_equal(audit.force_on_nonfloor(frame, local, 2, 0, 0), [-2., -3., -4.])
        with self.assertRaises(ValueError):
            audit.force_on_nonfloor(frame, local, 2, 3, 0)

    def test_dense_sparse_constraint_reconstruction(self):
        dense = np.array([[0., 2., 0.], [3., 0., 4.]])
        d = SimpleNamespace(nefc=2, efc_J=dense.ravel())
        m = SimpleNamespace(nv=3)
        with patch.object(audit.mujoco, 'mj_isSparse', return_value=False):
            np.testing.assert_array_equal(audit.constraint_jacobian(m, d), dense)
        d = SimpleNamespace(nefc=2, efc_J=np.array([2., 3., 4.]), efc_J_rowadr=[0, 1],
                            efc_J_rownnz=[1, 2], efc_J_colind=np.array([1, 0, 2]))
        with patch.object(audit.mujoco, 'mj_isSparse', return_value=True):
            np.testing.assert_array_equal(audit.constraint_jacobian(m, d), dense)

    def test_same_pose_and_saved_commands_not_different_best(self):
        rows = audit.selected_rows(self.doc)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]['qpos'], rows[1]['qpos'])
        for row, pose in zip(rows, self.result['poses']):
            self.assertEqual(row['label'], pose['label'])
            self.assertEqual(len(pose['snapshots']), 3)
            for s in pose['snapshots']:
                np.testing.assert_array_equal(s['qpos'], row['qpos'])
                self.assertEqual(s['qpos_sha256'], pose['qpos_sha256'])
                expected = self.model.actuator_ctrlrange[:, 0] if s['condition'] == 'native_floor' else row[s['condition']]['activation']
                np.testing.assert_array_equal(s['activation'], expected)

    def test_force_balance_and_live_snapshot(self):
        for pose in self.result['poses']:
            for s in pose['snapshots']:
                self.assertTrue(s['qvel_zero'])
                for name, check in s['checks'].items():
                    self.assertTrue(check['within_tolerance'], (pose['label'], s['condition'], name, check))
                np.testing.assert_allclose(s['mass_times_COM_acceleration'], s['gravity_contact_other_sum'], atol=3e-7, rtol=1e-9)
                self.assertEqual(s['contact']['distinct_named_foot_geom_count'], len(set(s['contact']['distinct_named_foot_geoms'])))
                for name, terms in s['constraint_by_type'].items():
                    if 'LIMIT_JOINT' in name:
                        np.testing.assert_allclose(terms['root_force_world'], 0, atol=1e-12)
        saved = self.result['poses'][0]['snapshots'][0]
        live = audit.snapshot(self.model, saved['qpos'], saved['activation'], 'native_floor')
        np.testing.assert_array_equal(live['contact']['force_world'], saved['contact']['force_world'])
        np.testing.assert_array_equal(live['COM_acceleration_mm_s2'], saved['COM_acceleration_mm_s2'])

    def test_frozen_xml_and_source_hashes(self):
        p = self.result['provenance']
        for key in ('xml_sha256_before', 'xml_sha256_after', 'expected_frozen_sha256'):
            self.assertEqual(p[key], audit.cp.CANONICAL_XML_SHA256)
        self.assertEqual(self.before, audit.cp.CANONICAL_XML_SHA256)
        self.assertEqual(p['source_sha256'], audit.cp.sha256_file(audit.SOURCE))
        self.assertEqual(p['source_sha256_after'], p['source_sha256'])
        self.assertEqual(p['code_sha256'], audit.cp.sha256_file(audit.Path(audit.__file__)))


if __name__ == '__main__':
    with audit.mg.guarded_run('support_force_tests', **audit.mg.BUDGETS['neural_muscle_closure']):
        unittest.main(verbosity=2)
