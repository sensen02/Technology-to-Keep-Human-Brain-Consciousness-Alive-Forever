"""Saved-artifact checks for the anatomical joint-axis audit."""
import json
import unittest
import run_anatomical_joint_axis_audit as run

class AnatomicalAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'anatomical_joint_axis.json').read_text())

    def test_all_six_legs_are_covered(self):
        self.assertEqual(set(self.doc['dataset_per_leg']), {'LF', 'LM', 'LH', 'RF', 'RM', 'RH'})
        for leg, v in self.doc['dataset_per_leg'].items():
            self.assertGreater(v['condyle_separation_mm'], 0, leg)
            self.assertAlmostEqual(sum(x * x for x in v['axis_unit']), 1.0, places=5, msg=leg)

    def test_anatomical_axis_is_oblique_and_ours_is_perpendicular(self):
        c = self.doc['comparison']
        self.assertLess(c['dataset_angle_axis_to_tibia_deg']['mean'], 85.0)
        self.assertGreater(c['our_angle_axis_to_tibia_deg']['mean'], 88.0)
        self.assertGreater(c['difference_deg'], 5.0)
        self.assertLess(c['dataset_angle_axis_to_tibia_deg']['min'], 70.0)

    def test_femur_axis_also_available(self):
        for leg, v in self.doc['dataset_per_leg'].items():
            self.assertIn('angle_femur_tibia_axis_deg', v)

    def test_source_is_hashed(self):
        import hashlib
        from pathlib import Path
        p = Path(self.doc['source']['csv'])
        self.assertTrue(p.is_file())
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), self.doc['source']['csv_sha256'])

    def test_limits_are_stated(self):
        lim = ' '.join(self.doc['limitations'])
        self.assertIn('frame mapping is not solved', lim)
        self.assertIn('kinematic model of one female fly', lim)
        self.assertTrue(any('range' in x for x in self.doc['not_licensed']))

if __name__ == '__main__':
    unittest.main()
