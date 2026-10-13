"""Saved-artifact checks for the tarsus-DOF structural extension."""
import json
import unittest
from pathlib import Path
import numpy as np
import run_tarsus_dof_structure as run

class TarsusDofStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'tarsus_dof_structure.json').read_text())

    def test_dimensions_add_exactly_six_dof_and_twelve_actuators(self):
        d = self.doc['dimensions']
        self.assertEqual(d['added_dof'], 6)
        self.assertEqual(d['added_actuators'], 12)
        self.assertEqual(d['new_nq'] - d['base_nq'], 6)
        self.assertEqual(d['new_nv'] - d['base_nv'], 6)
        self.assertEqual(d['new_nu'] - d['base_nu'], 12)

    def test_canonical_is_untouched(self):
        self.assertEqual(run.sha(run.closure.cp.CANONICAL_XML), self.doc['provenance']['canonical_sha256'])
        self.assertEqual(self.doc['provenance']['canonical_sha256'],
                         '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6')

    def test_branch_xml_exists_and_hashes_match(self):
        xml = Path(self.doc['provenance']['branch_xml'])
        self.assertTrue(xml.is_file())
        self.assertEqual(run.sha(xml), self.doc['provenance']['branch_sha256'])
        self.assertTrue(self.doc['provenance']['geometry_content_changes'] is False)
        # built on the zero-floor branch, whose asset paths are already absolute
        self.assertEqual(self.doc['provenance']['asset_paths_resolved'], 0)
        self.assertEqual(run.sha(self.doc['provenance']['base_branch_xml']),
                         self.doc['provenance']['base_branch_sha256'])

    def test_assumptions_are_numeric_and_labelled(self):
        a = self.doc['declared_assumptions']
        # v2: the joint is sourced from flybody; the attachments remain a declared assumption
        self.assertIn('flybody', a['A1_joint'].lower())
        self.assertIn('flybody', a['A1_range_source'].lower())
        self.assertEqual(set(a['A1_axes_local']), {'LF', 'LM', 'LH', 'RF', 'RM', 'RH'})
        self.assertIn('0.020', a['A3_attachments'])
        self.assertIn('no new tuning', a['A2_actuators'])
        for key in ('existing joints', 'existing actuators', 'geometry', 'masses', 'inertias'):
            self.assertIn(key, a['not_changed'])

    def test_no_behaviour_claim(self):
        self.assertEqual(self.doc['status'], 'STRUCTURAL_EXTENSION_BUILT')
        self.assertIn('not a behaviour result', self.doc['label'])

if __name__ == '__main__':
    unittest.main()
