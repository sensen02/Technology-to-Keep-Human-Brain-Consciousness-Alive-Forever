"""Saved-artifact checks for the FlyBody anatomy extraction."""
import json
import unittest
import run_flybody_anatomy_audit as run

class FlyBodyAnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'flybody_anatomy.json').read_text())

    def test_tarsus_and_femur_joints_are_documented(self):
        s = self.doc['tarsus_and_femur_summary']
        self.assertIn('tarsus', s['tarsus_axis'].lower())
        self.assertIn('-0.7..1.2', s['tarsus_axis'])
        self.assertIn('twist', s['femur_joints'])
        self.assertEqual(s['actuators_per_leg'], 8)

    def test_what_it_does_not_license_is_explicit(self):
        no = ' '.join(self.doc['what_this_does_not_license'])
        self.assertIn('axis number directly', no)
        self.assertIn('muscle attachment geometry', no)
        for key in ('what_this_licenses', 'what_this_does_not_license'):
            self.assertGreaterEqual(len(self.doc[key]), 3)

    def test_source_is_hashed_and_in_package(self):
        src = self.doc['source']
        import hashlib
        from pathlib import Path
        p = Path(src['file'])
        self.assertTrue(p.is_file())
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), src['sha256'])

    def test_no_model_change_or_behaviour_claim(self):
        self.assertEqual(self.doc['status'], 'MEASURED_FLYBODY_JOINT_DEFINITIONS')
        self.assertIn('not a behaviour result', self.doc['label'])
        self.assertIn('No model was modified', ' '.join(self.doc['limitations']))

if __name__ == '__main__':
    unittest.main()
