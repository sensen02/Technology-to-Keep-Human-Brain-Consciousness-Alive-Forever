"""Saved-artifact checks for the cross-audit consistency report."""
import json
import unittest
import run_cross_audit_consistency as run

class CrossAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.B / 'cross_audit_consistency/cross_audit_consistency.json').read_text())

    def test_every_check_passed(self):
        failed = {k: v for k, v in self.doc['checks'].items() if not v}
        self.assertEqual(failed, {})
        self.assertGreaterEqual(len(self.doc['checks']), 12)

    def test_no_check_is_vacuous(self):
        """A check must carry evidence, not a hardcoded pass."""
        for name, ev in self.doc['evidence'].items():
            self.assertIsInstance(ev, dict, name)
            self.assertTrue(ev, f'{name} has no evidence')

    def test_shared_quantities_agree(self):
        ev = self.doc['evidence']
        self.assertEqual(ev['weight_matches_existing_audit']['support_audit'],
                         ev['weight_matches_existing_audit']['earlier_support_force_audit'])
        counts = ev['sensory_cell_total_consistent_across_audits']
        self.assertEqual(len(set(counts.values())), 1)

    def test_uncommandable_count_is_cross_confirmed(self):
        ev = self.doc['evidence']['census_uncommandable_equals_manifest_unresolved']
        self.assertEqual(ev['census'], ev['manifest_unresolved'])
        self.assertEqual(ev['census'], ev['rows_without_pool'])

if __name__ == '__main__':
    unittest.main()
