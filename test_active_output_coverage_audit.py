"""Raw partition accounting, not functional recovery validation."""
import json
from pathlib import Path
import unittest


class CoverageAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((Path(__file__).resolve().parent/'outputs/model_branches/flybody_loop/active_output_coverage_audit.json').read_text())

    def test_exact_sources(self):
        r = self.audit
        self.assertEqual(r['missing_verified_fired_roots'], 34)
        self.assertEqual(len({x['root_id'] for x in r['per_root']}), 34)
        self.assertEqual(r['tier_neurons'], 12000)
        self.assertEqual(r['motor_cells'], 202)

    def test_partitions(self):
        for r in self.audit['per_root']+[self.audit['summary']]:
            for key in ('rows', 'synapses'):
                self.assertAlmostEqual(r['all_outputs'][key], r['outside_tier'][key]+r['inside_tier_excluded_missing_verified'][key])
                self.assertAlmostEqual(r['inside_tier_excluded_missing_verified'][key], r['inside_to_motor'][key]+r['inside_to_nonmotor'][key])
        for category, counts in self.audit['summary'].items():
            for key, count in counts.items():
                self.assertAlmostEqual(count, sum(r[category][key] for r in self.audit['per_root']))

    def test_direct_motor_rows_remain_unsigned(self):
        r = self.audit
        edges = r['raw_direct_motor_edges']
        self.assertEqual(len(edges), r['summary']['inside_to_motor']['rows'])
        self.assertAlmostEqual(sum(e['raw_synapses'] for e in edges), r['summary']['inside_to_motor']['synapses'])
        self.assertEqual(len({e['target_root_id'] for e in edges}), r['unique_direct_motor_targets'])
        self.assertTrue(all(e['sign'] is None and not e['modelled'] for e in edges))


if __name__ == '__main__':
    unittest.main()
