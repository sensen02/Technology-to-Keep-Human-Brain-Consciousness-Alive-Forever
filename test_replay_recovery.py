"""Synthetic serialization fixtures only; no neural or body integration."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from engine.replay_recovery import ARRAYS, inspect_recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.rows = [dict(arm='factual', tick=i, total=i+1) for i in range(2)]
        self.state = dict(arm='factual', completed_tick=1, total_calls=2,
                          neural_scalars_json=json.dumps(dict(n=3, dt=.5,
                          delay_steps=4, _ring_i=2, steps=402, t_ms=201.,
                          voltage_method='backward_euler')),
                          FIR_history=np.zeros((80, 2)), FIR_elapsed_s=.201,
                          FIR_interval=2, weights=np.zeros((2, 5)),
                          actuator_pool=np.zeros(4, dtype=int))
        for k in ARRAYS:
            self.state['neural_'+k] = np.zeros((5, 3) if k == '_ring' else (3,))
        for k in ('current', 'spikes'):
            self.state[k] = np.zeros((2, 3))
        for k in ('pool_activity', 'features', 'raw_command', 'command'):
            self.state[k] = np.zeros((2, 2))

    def save(self):
        (self.d/'CALLS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        np.savez(self.d/'factual_RECOVERY.npz', **self.state)

    def test_aligned_is_not_execution_authorization(self):
        self.save()
        r = inspect_recovery(self.d)
        self.assertTrue(r['safe_to_resume'])
        self.assertIn('REQUIRES_IDENTITY', r['verdict'])

    def test_gap_blocks_without_reclaim(self):
        self.rows.append(dict(arm='factual', tick=2, total=3))
        self.save()
        r = inspect_recovery(self.d)
        self.assertFalse(r['safe_to_resume'])
        self.assertEqual(r['uncommitted_reservations'], 1)
        self.assertEqual(r['remaining_unreserved_calls'], 797)

    def test_malformed_journal(self):
        self.rows[-1]['tick'] = 7
        self.save()
        with self.assertRaises(ValueError):
            inspect_recovery(self.d)

    def test_missing_ring(self):
        del self.state['neural__ring']
        self.save()
        with self.assertRaises(ValueError):
            inspect_recovery(self.d)

    def test_bad_clock(self):
        scalar = json.loads(self.state['neural_scalars_json'])
        scalar['t_ms'] = 0
        self.state['neural_scalars_json'] = json.dumps(scalar)
        self.save()
        with self.assertRaises(ValueError):
            inspect_recovery(self.d)

    def test_nonfinite(self):
        self.state['neural_v'][0] = np.nan
        self.save()
        with self.assertRaises(ValueError):
            inspect_recovery(self.d)

    def test_checkpoint_ahead(self):
        self.rows.pop()
        self.save()
        with self.assertRaises(ValueError):
            inspect_recovery(self.d)

    def test_read_only(self):
        self.save()
        before = {p.name:p.read_bytes() for p in self.d.iterdir()}
        inspect_recovery(self.d)
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.d.iterdir()})


if __name__ == '__main__':
    unittest.main()
