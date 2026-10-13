"""Saved-artifact checks for the commensurate-normaliser branch."""
import json
import unittest
import numpy as np
import run_commensurate_normaliser_branch as run

class CommensurateNormaliserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'commensurate_normaliser_results.json').read_text())
        cls.z = np.load(run.OUT / 'commensurate_normaliser_traces.npz')

    def test_only_normalisers_were_declared_changed(self):
        a = self.doc['declared_assumptions']
        self.assertIn('UNCHANGED', a['reference_pose'])
        for untouched in ('current_nA_per_unit', 'reference pose', 'network parameters', 'connectivity',
                          'transmitter labels', 'motor mapping'):
            self.assertIn(untouched, a['not_changed'])
        self.assertGreater(a['A1_angle_scale_rad'], 0)
        self.assertGreater(a['A3_force_scale'], 0)

    def test_controls_hold(self):
        c = self.doc['checks']
        for key in ('replay_exact', 'silence_zero_drive', 'all_finite', 'sources_unchanged',
                    'reference_left_at_start_pose'):
            self.assertTrue(c[key], key)

    def test_scale_is_the_median_half_range_not_a_fitted_number(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(run.XML))
        ranges = []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            if name.startswith('joint_') and model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                ranges.append(0.5 * float(hi - lo))
        self.assertEqual(len(ranges), 42)
        import statistics
        self.assertAlmostEqual(self.doc['declared_assumptions']['A1_angle_scale_rad'],
                               statistics.median(ranges), places=12)

    def test_negative_result_reported_honestly(self):
        b, base = self.doc['branch'], self.doc['published_25us_baseline']
        self.assertLess(b['escape_time_s'], base['escape_time_s'])
        self.assertLess(b['endpoint_heights_mm'][2], base['endpoint_heights_mm'][2])
        self.assertLess(b['endpoint_heights_mm'][2], 1.0)

if __name__ == '__main__':
    unittest.main()
