"""Direction and accounting checks, not scientific closure validation."""
import json
from pathlib import Path
import unittest
import numpy as np
from tools_connectome_path_diagnostics import distances_from_sources


class DirectedPaths(unittest.TestCase):
    def test_target_source_orientation(self):
        w = np.zeros((4, 4))
        w[1, 0] = 1
        w[2, 1] = 1
        self.assertEqual(distances_from_sources(w, [0]).tolist(), [0, 1, 2, -1])
        self.assertEqual(distances_from_sources(w, [2]).tolist(), [-1, -1, 0, -1])
        self.assertEqual(distances_from_sources(w, []).tolist(), [-1]*4)

    def test_artifact(self):
        p = Path(__file__).resolve().parent/'outputs/model_branches/flybody_loop/flybody_connectome_path_audit.json'
        r = json.loads(p.read_text())
        self.assertEqual(len(r['arms']), 3)
        for arm in r['arms']:
            d = arm['diagnostics']
            self.assertEqual(len(d['motor_cells']), r['motor_pool_cells'])
            self.assertEqual(sum(x['spikes'] for x in d['fired_nonsensory_cells']), arm['spike_totals']['nonsensory'])
            self.assertEqual(sum(x['spikes'] for x in d['motor_cells']), arm['spike_totals']['motor'])
            self.assertTrue(all(not x['is_directly_stimulated'] for x in d['motor_cells']))
            self.assertTrue(all(x['peak_post_decay_ge_uS'] >= 0 and x['peak_post_decay_gi_uS'] >= 0 for x in d['motor_cells']))
            if arm['arm'] == 'zero_input':
                self.assertTrue(all(x['shortest_unsigned_hops'] is None for x in d['motor_cells']))


if __name__ == '__main__':
    unittest.main()
