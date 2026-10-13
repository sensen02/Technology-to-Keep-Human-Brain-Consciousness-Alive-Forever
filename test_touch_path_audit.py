"""Artifact-only regression checks; never builds a tier or runs physics."""
import json
import unittest
import numpy as np
from scipy import sparse
import run_touch_path_audit as audit
from engine.embodied.audit_provenance import inspect_audit, validate_saved_bundle


class TouchPathAuditTests(unittest.TestCase):
    def test_bfs_post_rows_pre_columns(self):
        we = sparse.csr_matrix(([1.,1.],([1,2],[0,1])),shape=(5,5))
        wi = sparse.csr_matrix(([2.],([3],[2])),shape=(5,5))
        np.testing.assert_array_equal(audit.directed_hops(we,wi,[0]),[0,1,2,3,-1])
        np.testing.assert_array_equal(audit.directed_hops(we,wi,[3]),[-1,-1,-1,0,-1])

    def test_missing_records_rejected(self):
        with self.assertRaises(ValueError):
            audit.reconstruct({'detail':'trimmed'},audit.ChannelConfig())

    def test_saved_trajectory_aggregate_equality_and_finite(self):
        result = json.loads((audit.OUT/'touch_path_audit.json').read_text())
        bundle = validate_saved_bundle(result['source_bundle'])
        self.assertEqual(bundle['manifest'],result['source_bundle']['manifest'])
        self.assertEqual(audit.sha(result['spike_npz']),result['spike_npz_sha256'])
        with np.load(result['spike_npz'],allow_pickle=False) as data, np.load(audit.Path(bundle['path'])/'closure_traces.npz',allow_pickle=False) as saved:
            for key in data.files:
                self.assertTrue(np.isfinite(data[key]).all(),key)
            for arm in audit.ARMS:
                spikes = data[arm+'_spikes']
                self.assertEqual(spikes.dtype,np.dtype(bool))
                np.testing.assert_array_equal(audit.aggregate(spikes),saved[arm+'_spike_counts'])
            delta = data['normal_currents_nA']-data['no_external_touch_currents_nA']
            expected = np.zeros_like(delta)
            expected[1:3,data['lf_bristle_indices']] = .3
            np.testing.assert_allclose(delta,expected,rtol=0,atol=1e-14)
            self.assertEqual(len(data['lf_bristle_indices']),450)

    def test_source_checksums(self):
        result = json.loads((audit.OUT/'touch_path_audit.json').read_text())
        evidence = inspect_audit(audit.Path(__file__).resolve().parent, 'touch_path_audit')
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertFalse(evidence['execution_equivalence_claimed'])
        closure = str(audit.Path(__file__).resolve().parent/'run_neural_muscle_closure.py')
        self.assertEqual(evidence['run_sources'][closure]['status'], 'drift')
        self.assertTrue(all(result['validation']['saved_per_5ms_per_neuron_spike_aggregate_exact'].values()))
        self.assertLess(result['wall_s'],60)


if __name__ == '__main__':
    unittest.main()
