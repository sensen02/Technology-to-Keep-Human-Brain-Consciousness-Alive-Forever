"""Saved-artifact checks for the load-support audit; never reruns an episode."""
import json
import unittest
from pathlib import Path
import numpy as np
import run_support_audit as run

class SupportAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'support_audit_results.json').read_text())
        cls.data = np.load(run.OUT / 'support_audit_traces.npz')

    def test_artifact_identity(self):
        self.assertEqual(run.hashlib_sha(run.OUT / 'support_audit_traces.npz'), self.doc['provenance']['npz_sha256'])
        self.assertEqual(set(self.doc['cases']), {'100us_normal', '25us_normal'})
        for case in self.doc['cases'].values():
            self.assertTrue(case['finite'])
            self.assertEqual(case['neural_steps'], 400)

    def test_support_ratios_match_saved_arrays(self):
        weight = self.doc['weight_and_mass']['weight_model_units']
        for us in (100, 25):
            key = f'{us}us_normal'
            grf = self.data[f'{us}us_grf_z_B']
            case = self.doc['cases'][key]
            self.assertAlmostEqual(float(grf[0] / weight), case['initial_grf_z_over_weight'], places=12)
            self.assertAlmostEqual(float(grf.max() / weight), case['max_grf_z_over_weight'], places=12)
            self.assertAlmostEqual(float(grf.mean() / weight), case['mean_grf_z_over_weight'], places=12)
            self.assertAlmostEqual(float(grf[-1] / weight), case['final_grf_z_over_weight'], places=12)
            self.assertLess(case['initial_grf_z_over_weight'], 1.0)

    def test_convention_and_physical_sign(self):
        """Support must be upward (positive along -gravity) and convention B must be the positive one."""
        for us in (100, 25):
            self.assertGreater(float(self.data[f'{us}us_grf_z_B'].mean()), 0.0)
            self.assertLess(float(self.data[f'{us}us_grf_z_A'].mean()), 0.0)

    def test_per_leg_detail_is_consistent(self):
        detail = self.doc['initial_leg_detail']['canonical_100us']
        total = sum(v['vertical_force'] for v in detail.values())
        self.assertAlmostEqual(total, float(self.data['100us_grf_z_B'][0]), places=9)
        unloaded = [leg for leg, v in detail.items() if v['contacts'] == 0]
        self.assertEqual(unloaded, ['rm'])
        for leg, v in detail.items():
            self.assertFalse(v['contacts'] and v['vertical_force'] < 0)

    def test_100us_reproduces_published_zero_floor_trace(self):
        with np.load(run.zero.OUT / 'zero_floor_traces.npz') as old:
            self.assertTrue(np.array_equal(self.data['100us_root_z'], old['normal_heights_mm'][:, 0]))

if __name__ == '__main__':
    unittest.main()
