"""Validate event timing from source emission + existing 2ms delay."""
import json
import unittest
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent/'outputs/model_branches/flybody_loop'


class EventAccounting(unittest.TestCase):
    def test_arrival_reconstruction(self):
        r = json.loads((OUT/'flybody_connectome_event_audit.json').read_text())
        for arm in r['arms']:
            self.assertEqual(len(arm['spike_events']), arm['spike_totals']['all'])
            for f in arm['focus_motor_cells']:
                predicted = {}
                for edge in f['incoming_fired_edges']:
                    for event in arm['spike_events']:
                        if event['root_id'] == edge['source_root_id']:
                            t = event['emission_end_ms']+2.0
                            key = 'arriving_exc_uS' if edge['kind'] == 'excitatory' else 'arriving_inh_uS'
                            predicted[t, key] = predicted.get((t, key), 0.)+edge['conductance_increment_uS']
                for row in f['trace']:
                    for key in ['arriving_exc_uS', 'arriving_inh_uS']:
                        self.assertAlmostEqual(row[key], predicted.get((row['time_start_ms'], key), 0.), places=12)
                diag = next(x for x in arm['diagnostics']['motor_cells'] if x['root_id'] == f['root_id'])
                self.assertAlmostEqual(max(-65., max(x['voltage_mV'] for x in f['trace'])), diag['peak_post_step_voltage_mV'])

    def test_annotation_nonempty_unique_join(self):
        r = json.loads((OUT/'flybody_connectome_event_annotations.json').read_text())
        self.assertEqual(r['join_key'], 'banc_888_id')
        self.assertEqual(r['joined_unique_roots'], len(r['annotations']))
        self.assertGreater(r['joined_unique_roots'], 0)
        for arm in r['arms']:
            for focus in arm['focus']:
                self.assertIn(focus['root_id'], r['annotations'])
                self.assertEqual(focus['annotation']['super_class'], 'motor')
                self.assertTrue(np.isfinite(focus['peak_voltage_mV']))


if __name__ == '__main__':
    unittest.main()
