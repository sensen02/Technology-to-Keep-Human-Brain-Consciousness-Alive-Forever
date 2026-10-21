"""Synthetic fixtures only; these tests do not run an archived/current simulation."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tools_verify_experiment_platform as audit


def synthetic():
    count, n, nu = 4, 3, 1
    dc, dn = .005, .0005
    result = dict(duration_s=count*dc,
                  clocks=dict(dt_body_s=.0001, dt_neural_s=dn, dt_command_s=dc),
                  evidence_schema=dict(version=1, sensory_local_indices_key='sensory_local_indices',
                      body_nq=1, body_nv=1, body_na=1, body_nu=1,
                      motor_translation=dict(filter_tau_s=.005, rate_for_full_control_hz=100., max_control=1.),
                      replay_source_arms=dict(replay_baseline='normal', replay_push='normal')),
                  tier_root_ids=['100', '200', '300'],
                  motor_manifest=dict(n_neurons=n, n_actuators=nu,
                      rows=[dict(actuator_index=0, local_indices=[0], tier_root_ids=[100])]),
                  sensory_groups=dict(lf=dict(claw=[1], disabled=[2])), episodes={})
    arrays = dict(sensory_local_indices=np.array([1, 2], dtype=np.int64),
                  projected_pose=np.zeros(1), native_control_ranges=np.array([[.02, 1.]]))
    decay = math.exp(-dn/.005)
    for arm in audit.ARMS:
        decoded = np.zeros((count, nu))
        counts = np.zeros((count, n), dtype=np.int64)
        records = []
        rates = np.zeros(nu)
        currents = np.tile([.2, 0.], (count, 1))
        if arm == 'sensory_cut':
            currents[:] = 0
        for k in range(count):
            raw = [[0, 0]] if k == 0 and arm != 'sensory_cut' else []
            emitted = [] if arm == 'output_silence' else raw
            counts[k, 0] = len(emitted)
            connected = not (arm == 'motor_cut' and (k+1)*dc >= .015-1e-12)
            if connected:
                for step in range(10):
                    rates = decay*rates + (1-decay)*(1/dn if [step, 0] in emitted else 0)
            else:
                rates[:] = 0
            decoded[k] = np.clip(rates/100, 0, 1)
            trace = dict(source_interval=k, source_start_s=k*dc, source_end_s=(k+1)*dc,
                         valid_from_s=(k+1)*dc, pool_spike_counts=[len(emitted)],
                         rates_hz=rates.tolist(), connected=connected,
                         saturated=(decoded[k] >= 1).tolist())
            records.append(dict(time_start_s=k*dc, time_end_s=(k+1)*dc,
                applied_source_interval=k-1, source_completed_by_s=k*dc,
                command_origin='offline_normal_replay' if arm.startswith('replay_') else 'previous_completed_neural_interval',
                decoder=trace, probe_raw_spike_events_step_column=raw,
                probe_emitted_spike_events_step_column=emitted,
                actual_network_input_by_channel={'lf': {name: dict(count=1, sum_nA=float(value), min_nA=float(value), max_nA=float(value))
                                                       for name, value in zip(('claw', 'disabled'), currents[k])}},
                disabled_channels=['disabled'], total_network_spikes=int(counts[k].sum()),
                motor_spikes=len(emitted)))
        controls = np.vstack([np.zeros((1, nu)), decoded[:-1]])
        if arm == 'motor_cut':
            controls[3:] = 0
        if arm.startswith('replay_'):
            controls = arrays['normal_controls'].copy()
        trajectory = np.column_stack([np.ones(count), np.zeros(count), controls[:, 0]])
        result['episodes'][arm] = dict(records=records, probe_local_indices=[0], probe_spike_dt_s=dn)
        for kind, value in dict(controls=controls, decoded_next_controls=decoded,
                trajectory=trajectory, activation=controls.copy(), spike_counts=counts,
                sensory_current_nA=currents, outside_sensory_input_max_abs_nA=np.zeros(count),
                interval_start_s=np.arange(count)*dc, interval_end_s=(np.arange(count)+1)*dc,
                applied_source_interval=np.arange(count, dtype=np.int64)-1).items():
            arrays[arm+'_'+kind] = value
    return result, arrays


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.result, self.arrays = synthetic()

    def fail(self, pattern):
        with self.assertRaisesRegex(audit.EvidenceError, pattern):
            audit.validate_arrays(self.result, self.arrays)

    def test_good_synthetic_nonzero_causal_motor_evidence(self):
        report = audit.validate_arrays(self.result, self.arrays)
        self.assertTrue(report['decoder_reconstructed'])
        self.assertEqual(report['intervals_per_arm'], 4)
        self.assertGreater(self.arrays['normal_controls'].max(), 0)

    def test_future_command(self):
        self.arrays['normal_controls'][0] = .1
        self.fail('command causality')

    def test_clock_tamper(self):
        self.arrays['normal_interval_end_s'][1] += .0005
        self.fail('end clocks')

    def test_missing_arm(self):
        del self.result['episodes']['replay_push']
        self.fail('eight expected')

    def test_current_tamper_with_counts_unchanged(self):
        self.arrays['normal_sensory_current_nA'][0, 0] += .001
        self.fail('current lf/claw')

    def test_outside_sensory_current(self):
        self.arrays['normal_outside_sensory_input_max_abs_nA'][0] = .1
        self.fail('outside sensory')

    def test_nonfinite(self):
        self.arrays['normal_trajectory'][0, 0] = np.nan
        self.fail('nonfinite')

    def test_bad_counts(self):
        self.arrays['normal_spike_counts'][0, 0] = 11
        self.fail('invalid spike counts')

    def test_raster_tamper(self):
        self.result['episodes']['normal']['records'][0]['probe_emitted_spike_events_step_column'] = []
        self.fail('raw/emitted')

    def test_duplicate_raster(self):
        events = self.result['episodes']['normal']['records'][0]['probe_raw_spike_events_step_column']
        events.append([0, 0])
        self.fail('duplicate raster')

    def test_decoder_rate_tamper(self):
        self.result['episodes']['normal']['records'][0]['decoder']['rates_hz'][0] += .1
        self.fail('reconstructed rates')

    def test_decoded_next_tamper(self):
        self.arrays['normal_decoded_next_controls'][-1] += .01
        self.fail('reconstructed control')

    def test_repeat_input_tamper_consistent_stats(self):
        self.arrays['normal_repeat_sensory_current_nA'][0, 0] = .3
        stat = self.result['episodes']['normal_repeat']['records'][0]['actual_network_input_by_channel']['lf']['claw']
        stat.update(sum_nA=.3, min_nA=.3, max_nA=.3)
        self.fail('repeat exact')

    def test_baseline_replay_body_tamper(self):
        self.arrays['replay_baseline_trajectory'][0, 0] += .1
        self.fail('replay baseline: trajectory')

    def test_motor_cut_tamper(self):
        self.arrays['motor_cut_controls'][3] = .1
        self.fail('command causality')

    def test_silence_emission_tamper(self):
        self.result['episodes']['output_silence']['records'][0]['probe_emitted_spike_events_step_column'] = [[0, 0]]
        self.fail('raw/emitted')

    def test_wrong_motor_identity(self):
        self.result['motor_manifest']['rows'][0]['tier_root_ids'] = [200]
        self.fail('motor root identity')

    def bundle(self, root):
        bundle = root/'bundle'
        bundle.mkdir()
        # Bytes are intentionally non-executable evidence; importing them would raise.
        source = root/'archived_source.py'
        source.write_text('raise RuntimeError("ARCHIVED CODE MUST NOT RUN")\n')
        self.result['source_hashes'] = {str(source): audit.sha256(source)}
        self.result['input_hashes'] = {str(source): audit.sha256(source)}
        audit.snapshot_utility().build_source_snapshot(self.result['source_hashes'], bundle/'source_snapshot.zip')
        self.result['source_snapshot_sha256'] = audit.sha256(bundle/'source_snapshot.zip')
        np.savez_compressed(bundle/'traces.npz', **self.arrays)
        self.result['traces_sha256'] = audit.sha256(bundle/'traces.npz')
        (bundle/'result.json').write_text(json.dumps(self.result))
        return bundle, source

    def test_bundle_no_heavy_or_archived_imports_and_live_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, source = self.bundle(Path(tmp))
            self.assertEqual(audit.verify_bundle(bundle)['status'], 'PASS')
            source.write_text('changed')
            self.assertEqual(audit.verify_bundle(bundle)['status'], 'PASS')
            with self.assertRaisesRegex(audit.EvidenceError, 'live hash mismatch'):
                audit.verify_bundle(bundle, check_live_sources=True)
            with self.assertRaisesRegex(audit.EvidenceError, 'live hash mismatch'):
                audit.verify_bundle(bundle, check_live_inputs=True)

    def test_trace_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self.bundle(Path(tmp))
            with (bundle/'traces.npz').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(audit.EvidenceError, 'traces SHA'):
                audit.verify_bundle(bundle)

    def test_zip_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self.bundle(Path(tmp))
            with (bundle/'source_snapshot.zip').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(audit.EvidenceError, 'source ZIP SHA'):
                audit.verify_bundle(bundle)

    def test_object_arrays_not_unpickled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.arrays['danger'] = np.array([{'object': True}], dtype=object)
            bundle, _ = self.bundle(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'allow_pickle=False'):
                audit.verify_bundle(bundle)

    def test_exclusive_report_outside_bundle_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, source = self.bundle(root)
            report = audit.verify_bundle(bundle)
            with self.assertRaisesRegex(audit.EvidenceError, 'outside'):
                audit.write_report(bundle/'report.json', report, bundle, self.result)
            with self.assertRaisesRegex(audit.EvidenceError, 'source/input'):
                audit.write_report(source, report, bundle, self.result)
            dest = root/'report.json'
            audit.write_report(dest, report, bundle, self.result)
            with self.assertRaises(FileExistsError):
                audit.write_report(dest, report, bundle, self.result)
            self.assertEqual(json.loads(dest.read_text())['status'], 'PASS')

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, _ = self.bundle(root)
            with patch('builtins.print'):
                self.assertEqual(audit.main([str(bundle), '--report', str(root/'report.json')]), 0)
                self.assertEqual(audit.main([str(bundle), '--report', str(root/'report.json')]), 1)


if __name__ == '__main__':
    unittest.main()
