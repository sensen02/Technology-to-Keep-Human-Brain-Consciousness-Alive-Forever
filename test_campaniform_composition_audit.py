"""Saved-artifact checks for the campaniform composition audit."""
import json
import unittest
import run_campaniform_composition_audit as run

class CampaniformCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'campaniform_composition.json').read_text())

    def test_channel_size_and_counts(self):
        s = self.doc['scope']
        self.assertEqual(s['tier_campaniform_cells'], 60)
        self.assertEqual(sum(v['cells'] for v in s['by_cell_type'].values()), 60)
        self.assertEqual(s['verified_cells'], 12)
        self.assertEqual(sum(v['verified'] for v in s['by_cell_type'].values()), 12)

    def test_evidence_kinds_do_not_overlap(self):
        o = self.doc['overlap']
        self.assertEqual(o['segment_identified_and_verified'], 0)
        self.assertEqual(o['verified_without_segment_identity'], 12)
        self.assertEqual(self.doc['segment_identified']['verified'], 0)

    def test_segment_identity_is_trochanter_only_and_not_a_named_sensillum(self):
        rule = self.doc['segment_identified']['rule']
        self.assertIn('SNpp53', rule)
        self.assertIn('trochanter', rule)
        for forbidden in ('Sc3', 'TrFa', 'TrFp', 'TrG'):
            self.assertNotIn(forbidden, rule)
        self.assertGreaterEqual(len(self.doc['segment_identified']['nerves']), 5)

    def test_conflicting_labels_are_recorded(self):
        c = self.doc['label_conflicts']
        self.assertEqual(c['cells'], 8)
        self.assertEqual(c['verified'], 2)
        self.assertEqual(set(c['types']), {'SNpp39', 'SNpp43', 'SNpp45', 'SNxx30'})

    def test_nothing_is_licensed_beyond_the_evidence(self):
        for clause in ('named sensillum', 'femoral', 'load/force encoder', 'foot-contact channel'):
            self.assertTrue(any(clause in x for x in self.doc['not_licensed']), clause)
        for word in ('tarsal campaniform', 'load sensor'):
            self.assertNotIn(word, self.doc['conclusion'].lower().replace('cannot be relabelled', ''))

if __name__ == '__main__':
    unittest.main()
