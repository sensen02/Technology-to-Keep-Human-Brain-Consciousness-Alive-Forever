import unittest
from types import SimpleNamespace
import numpy as np
from engine.embodied.declared_pose_projection import project_scalar_ranges


class ProjectionTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(nq=4, njnt=4, jnt_limited=np.array([False, True, True, True]),
            jnt_type=np.array([0, 3, 3, 2]), jnt_qposadr=np.arange(4),
            jnt_range=np.array([[0, 0], [-1, 1], [-2, 2], [0, 1]]),
            joint=lambda i: SimpleNamespace(name='joint_'+str(i)))

    def test_only_outside_scalar_coordinates_change_without_mutation(self):
        original = np.array([7., 1.2, -.5, -.1])
        saved = original.copy()
        q, report = project_scalar_ranges(self.model(), original)
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(q, [7., 1., -.5, 0.])
        self.assertEqual(len(report['changes']), 2)
        self.assertTrue(report['scalar_limit_report']['within_declared_scalar_ranges'])
        self.assertFalse(report['frozen_pose_modified'])

    def test_idempotent(self):
        q, _ = project_scalar_ranges(self.model(), [7., 1.2, -.5, -.1])
        qq, report = project_scalar_ranges(self.model(), q)
        np.testing.assert_array_equal(q, qq)
        self.assertFalse(report['changes'])

    def test_bad_qpos_rejected(self):
        for q in ([1, 2], [1, 2, 3, float('nan')]):
            with self.assertRaises(ValueError):
                project_scalar_ranges(self.model(), q)


if __name__ == '__main__':
    unittest.main()
