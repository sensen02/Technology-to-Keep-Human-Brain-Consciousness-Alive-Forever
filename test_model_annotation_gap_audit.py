"""Saved-artifact checks for the model-versus-annotation gap audit."""
import json
import unittest
import run_model_annotation_gap_audit as run

class GapAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'model_annotation_gap.json').read_text())

    def test_model_has_no_femur_or_tarsus_joint(self):
        kinds = self.doc['model']['leg_joint_kinds']
        self.assertEqual(self.doc['model']['dof_per_leg'], 7)
        self.assertNotIn('Femur_pitch', kinds)
        self.assertNotIn('Tarsus_pitch', kinds)
        for required in ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                         'Trochanter_roll', 'Tibia_pitch'):
            self.assertEqual(kinds[required], 6, required)

    def test_reverse_gap_is_exactly_the_unmodelled_muscles(self):
        self.assertEqual(sorted(self.doc['reverse_gap']['targets']),
                         ['accessory_tibia_flexor_muscle', 'femur_reductor_muscle', 'long_tendon_muscle',
                          'tarsus_depressor_muscle', 'tarsus_levator_muscle'])

    def test_forward_gap_partitions_to_sixteen(self):
        fg = self.doc['forward_gap']
        self.assertEqual(fg['unresolved_actuators'], 16)
        self.assertEqual(sum(fg['by_reason'].values()), 16)
        self.assertEqual(fg['by_reason']['target absent from the dataset entirely'], 6)

    def test_export_is_fresh_and_all_actuators_parsed(self):
        self.assertEqual(self.doc['model']['unparsed_actuator_names'], [])
        self.assertEqual(self.doc['model']['muscles_without_alias'], [])
        import hashlib
        from pathlib import Path
        export = Path(self.doc['annotation']['coverage_export'])
        self.assertEqual(hashlib.sha256(export.read_bytes()).hexdigest(),
                         self.doc['annotation']['coverage_export_sha256'])
        self.assertEqual(hashlib.sha256(Path(self.doc['annotation']['feather']).read_bytes()).hexdigest(),
                         self.doc['annotation']['feather_sha256'])

    def test_no_substitution_is_proposed(self):
        """No field may offer a substitute/borrowed pool; prohibitions in prose are fine."""
        banned_keys = {'substitute', 'substitutes', 'substitution', 'borrowed_pool', 'mirrored_pool',
                       'would_be_wired_to', 'alternative_pool', 'fallback_pool'}

        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k.lower()
                    yield from keys(v)
            elif isinstance(node, list):
                for item in node:
                    yield from keys(item)

        offenders = banned_keys & set(keys(self.doc))
        self.assertEqual(offenders, set())
        for entry in self.doc['forward_gap']['detail']:
            self.assertEqual(set(entry) & banned_keys, set())

    def test_forward_gap_detail_count_matches(self):
        fg = self.doc['forward_gap']
        self.assertEqual(len(fg['detail']), fg['unresolved_actuators'])

if __name__ == '__main__':
    unittest.main()
