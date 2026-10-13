"""Saved-artifact checks for the declared-reference sensory branch."""
import json
import unittest
import numpy as np
import run_declared_reference_branch as run

class DeclaredReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'declared_reference_results.json').read_text())
        cls.z = np.load(run.OUT / 'declared_reference_traces.npz')

    def test_artifact_identity_and_hashes(self):
        self.assertEqual(run.sha(run.OUT / 'declared_reference_traces.npz'), self.doc['provenance']['npz_sha256'])
        for path, digest in self.doc['provenance']['source_hashes'].items():
            self.assertEqual(run.sha(path), digest, path)

    def test_declared_parameters_are_the_only_changes(self):
        a = self.doc['declared_assumptions']
        self.assertGreater(a['A1_joints_changed'], 0)
        self.assertGreater(a['A1_limited_joints'], 0)
        self.assertIn('weight share', a['A2_contact_force_normaliser'])
        for untouched in ('current_nA_per_unit', 'reference_angle_rad', 'reference_velocity_rad_s'):
            self.assertIn(untouched, a['not_changed'])

    def test_reference_differs_from_the_starting_posture(self):
        """A1 must actually move the zero away from the episode start pose."""
        self.assertFalse(np.array_equal(self.z['seated_qpos'], self.z['declared_reference_qpos']))
        self.assertTrue(np.array_equal(self.z['seated_qpos'][:3], self.z['declared_reference_qpos'][:3]),
                        'root free-joint coordinates must not be altered')

    def test_controls_hold(self):
        c = self.doc['checks']
        self.assertTrue(c['declared_normal_and_replay_exact'])
        self.assertTrue(c['declared_silence_zero_drive'])
        self.assertTrue(c['all_finite'])
        self.assertTrue(c['sources_unchanged'])

    def test_negative_result_is_reported_honestly(self):
        """The declared branch must not be presented as an improvement: it was worse."""
        d = self.doc['declared_branch']
        b = self.doc['published_25us_baseline']
        self.assertLess(d['escape_time_s'], b_escape(self.doc, b))
        self.assertGreater(d['com_drop_mm'], b['com_drop_mm'])
        self.assertLess(d['endpoint_heights_mm'][2], b['endpoint_heights_mm'][2])


def b_escape(doc, baseline):
    """The published 25 us baseline escape time, from the fine-clock artifact."""
    import json as _json
    from pathlib import Path
    p = Path(run.HERE) / 'outputs/model_branches/fine_clock/fine_clock_results.json'
    return _json.loads(p.read_text())['episodes']['25us_normal']['escape_time_s']


if __name__ == '__main__':
    unittest.main()
