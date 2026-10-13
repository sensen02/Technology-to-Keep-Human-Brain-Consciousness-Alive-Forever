"""Saved-artifact checks for the model provenance structure audit."""
import json
import unittest
import run_model_provenance_structure_audit as run

class ModelProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'model_provenance_structure.json').read_text())

    def test_upstream_has_one_leg_of_muscles(self):
        up = self.doc['upstream_open_sim']
        self.assertEqual(up['muscles'], 15)
        self.assertEqual(set(up['muscles_per_leg']), {'LF'})
        self.assertEqual(up['joints_per_leg']['LF'], 3)
        self.assertEqual(up['joints_per_leg']['RF'], 3)

    def test_upstream_has_all_six_legs_as_geometry_only(self):
        up = self.doc['upstream_open_sim']
        self.assertEqual(set(up['legs_with_geometry']), {'LF', 'LM', 'LH', 'RF', 'RM', 'RH'})
        segs = up['segment_kinds_per_leg']['LF']
        for required in ('Coxa', 'Femur', 'Tibia', 'Trochanter'):
            self.assertIn(required, segs)

    def test_canonical_replicates_one_leg_to_six(self):
        mj = self.doc['canonical_mjcf']
        self.assertEqual(mj['legs'], 6)
        self.assertEqual(set(mj['actuators_per_leg'].values()), {15})
        self.assertEqual(mj['actuators'], 90)
        self.assertEqual(set(mj['joints_per_leg'].values()), {7})

    def test_gap_decomposition_matches_the_manifest(self):
        per_leg = self.doc['per_leg_targets']
        counts = {leg: len(v['modelled_but_not_annotated']) for leg, v in per_leg.items()}
        self.assertEqual(counts['LF'], 1)
        self.assertEqual(counts['LM'], 2)
        # 2 front-leg actuators (a single target each) + middle/hind legs with two targets each,
        # tergopleural promotor having two bundles per leg -> 16 total with the two single cells.
        self.assertEqual(self.doc['canonical_mjcf']['actuators'], 90)
        self.assertIn('replicating a front-leg muscle set', self.doc['conclusion'])

if __name__ == '__main__':
    unittest.main()
