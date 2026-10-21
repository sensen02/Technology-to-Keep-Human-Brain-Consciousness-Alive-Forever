import json
import unittest
from pathlib import Path
from tools_missing_nt_evidence_audit import evidence_state

class EvidenceTests(unittest.TestCase):
    def test_type_never_supplies_transmitter(self):
        self.assertEqual(evidence_state(None, None, None, 'cholinergic_named_cell'), 'no_transmitter_evidence_in_audited_fields')
        self.assertEqual(evidence_state(None, 'acetylcholine', None, None), 'prediction_only')
        self.assertEqual(evidence_state('gaba', 'acetylcholine', None, None), 'verified_present')

    def test_artifact(self):
        p = Path(__file__).resolve().parent/'outputs/model_branches/flybody_loop/missing_nt_evidence_audit.json'
        r = json.loads(p.read_text())
        self.assertEqual(len(r['roots']), 34)
        self.assertEqual(len({x['root_id'] for x in r['roots']}), 34)
        edges = [e for x in r['roots'] for e in x['direct_motor_targets']]
        self.assertEqual(len(edges), 114)
        self.assertEqual(len({e['target_root_id'] for e in edges}), 28)
        self.assertTrue(all(e['sign'] is None and not e['modelled'] for e in edges))
        self.assertTrue(all(e['target_annotation']['super_class'] == 'motor' for e in edges))
        self.assertTrue(all(x['feather_verified'] is None for x in r['roots']))
        self.assertTrue(all(x['evidence_state'] != 'verified_present' for x in r['roots']))

if __name__ == '__main__': unittest.main()
