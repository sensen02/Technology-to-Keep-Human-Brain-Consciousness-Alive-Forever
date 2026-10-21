"""Source-consistency tests; do not validate receptor physiology."""
import json
from pathlib import Path
import unittest
from tools_active_edge_transmitter_audit import sign


class ActiveSource(unittest.TestCase):
    def test_no_predicted_or_mixed_fallback(self):
        self.assertEqual(sign('acetylcholine'), 'excitatory')
        for value in ['gaba', 'glutamate']:
            self.assertEqual(sign(value), 'inhibitory')
        for value in ['', 'histamine', 'acetylcholine,gaba', 'octopamine', 'ACH']:
            self.assertEqual(sign(value), 'excluded')

    def test_artifact_consistency(self):
        p = Path(__file__).resolve().parent/'outputs/model_branches/flybody_loop/active_edge_transmitter_audit.json'
        r = json.loads(p.read_text())
        self.assertGreater(r['unique_fired_roots'], 0)
        self.assertEqual(len(r['roots']), r['roots_audited'])
        self.assertEqual(len({x['root_id'] for x in r['roots']}), r['roots_audited'])
        self.assertEqual(sum(x['fired'] for x in r['roots']), r['unique_fired_roots'])
        self.assertEqual(len(r['edge_checks']), r['motor_edge_records_checked'])
        self.assertTrue(all(x['csv_cache_exact_agree'] for x in r['roots']))
        self.assertTrue(all(x['matches_runtime_rule'] for x in r['edge_checks']))


if __name__ == '__main__':
    unittest.main()
