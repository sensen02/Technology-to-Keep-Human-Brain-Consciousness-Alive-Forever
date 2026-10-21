"""Artifact accounting, not biological validation."""
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class TransferAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((HERE/'outputs/model_branches/flybody_loop/flybody_connectome_transfer_audit.json').read_text())

    def test_identity_and_protocol(self):
        r = self.audit
        self.assertEqual(r['n_neurons'], 12000)
        self.assertEqual(r['sensory_cells'], 3788)
        self.assertGreater(r['motor_pool_cells'], 0)
        self.assertEqual(r['pulse_ms'], 20)
        self.assertEqual(r['duration_ms'], 100)
        self.assertEqual(len(r['arms']), 3)

    def test_count_accounting(self):
        for arm in self.audit['arms']:
            self.assertEqual(len(arm['trace']), 200)
            for row in arm['trace']:
                self.assertEqual(row['all'], row['sensory']+row['nonsensory'])
                self.assertLessEqual(row['motor'], row['all'])
                self.assertLessEqual(row['target'], row['sensory'])
            for key, total in arm['spike_totals'].items():
                self.assertEqual(total, sum(x[key] for x in arm['trace']))

    def test_control_zero_input(self):
        arm = self.audit['arms'][0]
        self.assertEqual(arm['arm'], 'zero_input')
        self.assertEqual(arm['current_nA_per_cell'], 0)
        self.assertEqual(arm['target_cells'], 0)
        # Do not require pulse arms to spike; null propagation is a valid result.


if __name__ == '__main__':
    unittest.main()
