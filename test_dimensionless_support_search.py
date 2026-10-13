"""Focused numerical/protocol regression tests; canonical model read-only, serial guard."""
import unittest
import numpy as np
import run_dimensionless_support_search as search


class DimensionlessTests(unittest.TestCase):
    def test_unit_rescaling_mm_to_m(self):
        q = np.array([120., -20., -9801., 2., -3., 20., 71.])
        in_m = q.copy()
        in_m[:3] *= .001
        np.testing.assert_allclose(search.dimensionless(q, 9801., 1.),
                                   search.dimensionless(in_m, 9.801, .001), rtol=1e-15)

    def test_scale_sensitivity_and_signed_vertical(self):
        q = np.arange(48, dtype=float) - 24
        a = search.dimensionless(q, 9801, .5)
        b = search.dimensionless(q, 9801, 2.)
        np.testing.assert_array_equal(a[:3], b[:3])
        np.testing.assert_allclose(b[3:], 4*a[3:])
        self.assertLess(search.metrics(q, 9801)['signed_root_z_mm_s2'], 0)


class CanonicalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.mujoco = mujoco
        cls.hash_before = search.cp.sha256_file(search.cp.CANONICAL_XML)
        cls.model = search.cp.load_model()

    @classmethod
    def tearDownClass(cls):
        assert cls.hash_before == search.cp.sha256_file(search.cp.CANONICAL_XML)

    def test_initialization_identity_and_relative_offsets(self):
        m = self.model
        expected = self.mujoco.MjData(m)
        self.mujoco.mj_resetDataKeyframe(m, expected, 0)
        actual = search.initialize(m)
        for field in ('qpos', 'qvel', 'act', 'ctrl'):
            np.testing.assert_array_equal(getattr(expected, field), getattr(actual, field))
        name = 'joint_LFCoxa_yaw'
        actual = search.initialize(m, {name: .001})
        expected.qpos[m.jnt_qposadr[m.joint(name).id]] += .001
        np.testing.assert_array_equal(expected.qpos, actual.qpos)

    def test_excluded_floor_and_runtime_mapping(self):
        m = self.model
        covered, report, _ = search.coverage(m)
        ev = search.FixedPose(m, search.initialize(m).qpos)
        a = ev.command(covered, np.full(len(covered), .2))
        excluded = np.setdiff1d(ev.muscles, covered)
        self.assertEqual(len(excluded), 16)
        np.testing.assert_array_equal(a[excluded], ev.floor[excluded])
        ev.acceleration(a)
        np.testing.assert_array_equal(ev.data.act[ev.addresses], a[ev.muscles])
        np.testing.assert_array_equal(ev.data.ctrl, a)
        if search.MANIFEST.exists():
            self.assertTrue(report['runtime_validated'])

    def test_residual_independent_of_history(self):
        m = self.model
        ev = search.FixedPose(m, search.initialize(m).qpos)
        a = np.full(m.nu, .2)
        q1 = ev.acceleration(a)
        ev.acceleration(np.full(m.nu, .8))
        ev.data.qacc_warmstart[:] = 1e8
        ev.data.qvel[:] = 123
        ev.data.qfrc_applied[:] = 14
        ev.data.xfrc_applied[:] = 2
        q2 = ev.acceleration(a)
        np.testing.assert_array_equal(q1, q2)

    def test_actual_keyframe_limit_violation_is_not_silently_repaired(self):
        m = self.model
        data = search.initialize(m)
        np.testing.assert_array_equal(data.qpos, m.key_qpos[0])
        with self.assertRaisesRegex(ValueError, 'joint_LHCoxa_pitch'):
            search.validate_limits(m, data.qpos)

    def test_outside_joint_limits_rejected(self):
        m = self.model
        j = next(j for j in range(m.njnt) if m.jnt_limited[j])
        q = m.key_qpos[0].copy()
        q[m.jnt_qposadr[j]] = m.jnt_range[j, 1] + 1
        with self.assertRaises(ValueError):
            search.validate_limits(m, q)


if __name__ == '__main__':
    with search.mg.guarded_run('dimensionless_support_tests',
                               **search.mg.BUDGETS['neural_muscle_closure']):
        unittest.main(verbosity=2)
