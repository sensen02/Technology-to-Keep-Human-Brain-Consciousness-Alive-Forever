"""Saved-artifact regressions; no model compilation or extra physics episodes."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from engine.embodied.audit_provenance import inspect_audit

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/neural_muscle_closure'
PRIOR = OUT / 'runs/f10c9354-8b21-4faa-b48c-80fee6a5c157'


class PoseComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((OUT/'neural_pose_comparison.json').read_text())
        cls.arr = np.load(OUT/'neural_pose_comparison.npz')

    @classmethod
    def tearDownClass(cls):
        cls.arr.close()

    def test_exact_pose_sources_and_restricted_differences(self):
        a = self.arr
        self.assertEqual(np.flatnonzero(a['original_qpos'] != a['repaired_qpos']).tolist(), [2, 8, 11, 23, 36, 42])
        source = json.loads((HERE/'outputs/workspace_recheck/pose_support_search.json').read_text())
        row, = [r for r in source['rows'] if r['label'] == 'baseline_exact_source']
        np.testing.assert_array_equal(a['repaired_qpos'], row['qpos'])
        with np.load(PRIOR/'closure_traces.npz') as prior:
            np.testing.assert_array_equal(a['original_qpos'], prior['seated_qpos'])
        self.assertFalse(self.doc['provenance']['source_activation_used'])

    def test_zero_initial_activation_control_velocity_and_ranges(self):
        for label, episode in self.doc['episodes'].items():
            for field in ('act', 'ctrl', 'qvel'):
                self.assertFalse(np.any(episode['initial'][field]), (label, field))
            self.assertEqual(len(episode['initial']['ranges']['violations']), 5 if label.startswith('original') else 0)
            self.assertTrue(episode['initial']['contact_gate']['tarsus_only'])
            self.assertEqual(episode['initial']['contact_gate']['n_body_floor_contact'], 0)

    def test_original_prior_arrays_exact(self):
        with np.load(PRIOR/'closure_traces.npz') as prior:
            for arm in ('normal', 'spike_output_silence'):
                for field in ('trajectory', 'controls', 'spike_counts'):
                    np.testing.assert_array_equal(self.arr['original_'+arm+'_'+field], prior[arm+'_'+field])

    def test_mapping_native_and_frozen_hashes(self):
        p = self.doc['provenance']
        evidence = inspect_audit(HERE, 'neural_pose_comparison')
        self.assertEqual(evidence['source_bundle']['path'], str(PRIOR))
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertFalse(evidence['original_source_authenticated'])
        self.assertEqual(evidence['run_sources'][str(HERE/'run_neural_muscle_closure.py')]['status'], 'drift')
        xml = HERE/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
        self.assertEqual(hashlib.sha256(xml.read_bytes()).hexdigest(), p['canonical_xml_sha256'])
        native = p['native_model_and_mapping']
        mapping = dict(joints=native['joints'], geom_leg={int(k): v for k, v in native['geom_leg'].items()})
        self.assertEqual(hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(), native['mapping_sha256'])
        self.assertTrue(all(self.doc['checks'].values()))
        self.assertEqual(hashlib.sha256((OUT/'neural_pose_comparison.npz').read_bytes()).hexdigest(), p['npz_sha256'])

    def test_all_steps_including_endpoint_health_and_silence(self):
        a = self.arr
        np.testing.assert_array_equal(a['body_step'], np.arange(501))
        self.assertEqual(a['body_time_s'][-1], .05)
        for label, episode in self.doc['episodes'].items():
            self.assertEqual(a[label+'_body_qpos'].shape, (501, 49))
            self.assertEqual(a[label+'_heights_mm'].shape, (501, 3))
            self.assertEqual(a[label+'_body_metrics'].shape, (501, 3))
            self.assertEqual(episode['endpoint_step'], 500)
            self.assertTrue(a[label+'_body_finite'].all())
            self.assertFalse(a[label+'_body_warning_counts'].any())
            self.assertTrue(episode['health']['all_live_states_finite'])
            np.testing.assert_array_equal(a[label+'_body_qpos'][-1], a[label+'_trajectory'][-1, :49])
            np.testing.assert_array_equal(a[label+'_body_qpos'][0], a[episode['pose_label']+'_qpos'])
            if label.endswith('spike_output_silence'):
                self.assertFalse(a[label+'_controls'].any())
                self.assertFalse(a[label+'_spike_counts'].any())
                self.assertIsNone(episode['first_emitted_motor_spike_s'])
                self.assertIsNone(episode['first_applied_nonzero_control_s'])

    def test_passive_constraint_force_components_not_mixed(self):
        for episode in self.doc['episodes'].values():
            passive = episode['initial']['passive_constraints']
            self.assertEqual(sum(r['row_count'] for r in passive['by_type']), passive['nefc'])
            for row in passive['by_type']:
                self.assertEqual(len(row['row_force_model']), row['row_count'])
                self.assertEqual(len(row['generalized_force_model']), 48)
                self.assertTrue(np.isfinite(row['generalized_force_model']).all())


if __name__ == '__main__':
    unittest.main()
