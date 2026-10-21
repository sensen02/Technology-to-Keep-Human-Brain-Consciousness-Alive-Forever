"""Pure synthetic tests for saved-artifact decoder diagnostics."""
import copy
import unittest
import numpy as np
from tools_decoder_saturation_audit import audit, saturation_summary


def fixture(connected=True):
    rate = (1 - np.exp(-0.0005 / 0.005)) / 0.0005 if connected else 0.0
    rows = [dict(actuator=name, actuator_index=j, local_indices=[0],
                 pool_root_ids=[100, 101], status='partial') for j, name in enumerate(('a', 'b'))]
    record = dict(time_start_s=0.0,
                  decoder=dict(valid_from_s=0.0005, source_end_s=0.0005,
                               rates_hz=[rate, rate], pool_spike_counts=[1, 1], connected=connected),
                  probe_emitted_spike_events_step_column=[[0, 0]])
    episode = dict(records=[record], probe_local_indices=[0], probe_spike_dt_s=0.0005)
    results = dict(provenance=dict(motor_config=dict(filter_tau_s=.005,
                     rate_for_full_control_hz=100., max_control=1.)),
                   episodes={'zero_floor_body': episode})
    identity = dict(tier_root_ids=[100, 200], motor_manifest=dict(rows=rows))
    arrays = dict(zero_floor_body_controls=np.zeros((1, 2)),
                  zero_floor_body_spike_counts=np.array([[1, 5]]),
                  zero_floor_body_trajectory=np.zeros((1, 1)))
    return results, identity, arrays


class DecoderAuditTests(unittest.TestCase):
    def test_threshold_inclusive_and_first_time(self):
        summary = saturation_summary([99., 100., 110.], [.005, .010, .015], 100.)
        self.assertEqual(summary['saturated_count'], 2)
        self.assertEqual(summary['first_saturation_s'], .010)
        self.assertIsNone(saturation_summary([0.], [.005], 100.)['first_saturation_s'])

    def test_alias_counts_partial_normalization_and_lag(self):
        result = audit(*fixture())
        self.assertEqual(result['unique_motor_cells'], 1)
        self.assertEqual(result['unique_nonempty_pools'], 1)
        episode = result['episodes']['zero_floor_body']
        self.assertEqual(episode['network_spike_total'], 6)
        self.assertEqual(episode['unique_motor_spike_total'], 1)
        self.assertEqual(episode['actuator_pool_spike_total'], 2)
        self.assertEqual(episode['decoder_saturated_samples'], 2)
        self.assertEqual(episode['unique_pool_saturated_samples'], 1)
        self.assertEqual(episode['applied_saturated_samples'], 0)
        self.assertEqual(episode['rate_reconstruction_max_error_hz'], 0.)
        row = episode['actuators'][0]
        self.assertEqual(row['aliases'], ['a', 'b'])
        self.assertEqual(row['present_cell_count'], 1)
        self.assertEqual(row['declared_cell_count'], 2)
        self.assertGreater(row['unclipped_normalized_demand'][0], 1.)
        self.assertTrue(row['alias_rates_exact'])
        self.assertEqual(row['substep']['first_saturation_s'], .0005)

    def test_disconnected_retains_neural_evidence_but_clears_rate(self):
        episode = audit(*fixture(False))['episodes']['zero_floor_body']
        self.assertEqual(episode['unique_motor_spike_total'], 1)
        self.assertEqual(episode['decoder_saturated_samples'], 0)
        self.assertEqual(episode['rate_reconstruction_max_error_hz'], 0.)

    def test_global_difference_without_motor_difference(self):
        results, identity, arrays = fixture()
        results['episodes']['touch'] = copy.deepcopy(results['episodes']['zero_floor_body'])
        for suffix in ('controls', 'spike_counts', 'trajectory'):
            arrays['touch_' + suffix] = arrays['zero_floor_body_' + suffix].copy()
        arrays['touch_spike_counts'][0, 1] += 10
        comparison = audit(results, identity, arrays)['comparisons'][0]
        self.assertEqual(comparison['network_spike_delta'], 10)
        self.assertEqual(comparison['motor_spike_delta'], 0)
        self.assertTrue(comparison['motor_interval_counts_exact'])
        self.assertTrue(comparison['controls_exact'])

    def test_missing_motor_probe_rejected(self):
        results, identity, arrays = fixture()
        results['episodes']['zero_floor_body']['probe_local_indices'] = []
        with self.assertRaisesRegex(ValueError, 'motor raster incomplete'):
            audit(results, identity, arrays)

    def test_duplicate_event_rejected(self):
        results, identity, arrays = fixture()
        results['episodes']['zero_floor_body']['records'][0]['probe_emitted_spike_events_step_column'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate spike events'):
            audit(results, identity, arrays)


if __name__ == '__main__':
    unittest.main()
