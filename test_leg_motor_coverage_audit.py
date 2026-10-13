"""Saved-artifact checks for the leg motor coverage vocabulary audit."""
import json
import unittest
import run_leg_motor_coverage_audit as run

class LegMotorCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'leg_motor_coverage.json').read_text())

    def test_leg_nerve_populations(self):
        a = self.doc['annotation']
        seg = a['efferent_on_leg_nerves_by_segment']
        self.assertEqual(sum(seg.values()), a['efferent_on_leg_nerves'])
        for key in ('prothoracic', 'mesothoracic', 'metathoracic'):
            self.assertGreater(seg[key], 0)
        self.assertGreater(a['distinct_cell_types_on_leg_nerves'], 40)

    def test_promotor_absent_on_middle_and_hind_nerves(self):
        w = self.doc['whole_vocabulary_search']
        self.assertEqual(set(w['by_nerve']),
                         {'left_dorsal_prothoracic_nerve', 'right_dorsal_prothoracic_nerve'})
        self.assertEqual(self.doc['per_missing_target']['pleural_promotor_muscle']['cells_on_leg_nerves'], 0)
        self.assertEqual(self.doc['per_missing_target']['tergopleural_promotor_muscle']['cells_on_leg_nerves'], 0)

    def test_conclusion_is_data_gap_not_naming(self):
        self.assertIn('annotation coverage', self.doc['conclusion'])
        self.assertIn('14 of 16', self.doc['conclusion'])
        self.assertEqual(self.doc['unresolved_actuators'], 16)

    def test_compound_group_recorded_but_not_wired(self):
        c = self.doc['annotation_named_compound_group']
        self.assertIn('NOT wired', c['proposal_status'])
        self.assertIn('pleural_promotor', c['cell_type'])
        manifest = run.RUN / 'motor_manifest.json'
        before = json.loads(manifest.read_text())['measured']
        self.assertEqual(before['usable_actuators'], 74)
        self.assertEqual(sum(1 for r in before['rows'] if r['status'] == 'unresolved'), 16)

if __name__ == '__main__':
    unittest.main()
