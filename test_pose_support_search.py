"""Regression checks for the saved finite mechanical pose experiment."""
import json
import unittest
import numpy as np
import run_pose_support_search as p


class PoseSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(p.OUT.read_text())
        cls.source = json.loads(p.audit.SOURCE.read_text())
        cls.model = p.cp.load_model()

    def test_fixed_objective(self):
        q = np.arange(48, dtype=float)
        j = np.zeros((3, 48)); j[:, :3] = np.eye(3); j[:, 3:] = .01
        r = p.residual(q, j, 9801.)
        self.assertAlmostEqual(float(r@r), float(np.sum(p.s.dimensionless(q, 9801.)**2)+((j@q)[2]/9801.)**2))
        d = self.doc
        self.assertEqual(d['method']['weight'], 1.)
        for row in [d['baseline_original']] + [r for r in d['rows'] if r['status']=='EVALUATED']:
            q = np.array(row['force_audit']['raw_vectors']['qacc'])
            cost = float(np.sum(p.s.dimensionless(q, d['gravity_mm_s2'])**2)+row['COM_z_over_g']**2)
            self.assertAlmostEqual(row['objective'], cost, places=12)

    def test_baseline_identity_and_fixed_reference(self):
        d = self.doc; src = self.source['best']['covered74']
        b = d['baseline_original']
        self.assertEqual(b['qpos'], src['qpos'])
        self.assertEqual(b['activation'], src['activation'])
        row = next(r for r in d['rows'] if r['label']=='baseline_exact_source')
        self.assertEqual(row['qpos'], src['qpos'])
        self.assertEqual(d['method']['angular_reference_rad_s2'], b['angular_rms_rad_s2'])
        for r in d['rows']:
            if r['status']!='EVALUATED': continue
            self.assertEqual(r['seed_activation_sha256'], p.audit.digest_array(src['activation']))
            self.assertEqual(r['angular_screen_pass'], r['angular_rms_rad_s2']<=1.1*b['angular_rms_rad_s2'])
            self.assertEqual(r['screen_pass'], r['angular_screen_pass'] and abs(r['COM_z_over_g'])<=.01)

    def test_declared_poses_and_no_clipping(self):
        d = self.doc; base = np.array(d['baseline_original']['qpos']); m = self.model
        self.assertEqual(len(d['rows']), 17)
        for row, (label, delta, dz) in zip(d['rows'], p.pose_specs()):
            self.assertEqual(row['label'], label)
            np.testing.assert_array_equal(row['qpos'], p.candidate_pose(m, base, delta, dz))
            self.assertFalse(row['seated'])
            self.assertLessEqual(abs(row['root_height_delta_mm']), .01+1e-12)
            for j in range(1, m.njnt):
                name = m.joint(j).name; a = m.jnt_qposadr[j]
                actual = row['qpos'][a]-base[a]
                self.assertEqual(row['actual_joint_deltas_rad'][name], actual)
                self.assertAlmostEqual(actual, delta.get(name, 0.), places=14)
                self.assertLessEqual(abs(actual), .03+1e-12)
            if row['status']=='EVALUATED':
                p.s.validate_limits(m, row['qpos'])
                self.assertTrue(row['gate']['tarsus_only'])
                self.assertEqual(row['gate']['n_body_floor_contact'], 0)
                self.assertLessEqual(row['max_abs_foot_penetration_mm'], .06+1e-12)
            else:
                self.assertEqual(row['status'], 'REJECTED_GATE_OR_LIMIT')
                self.assertTrue(row['reject_reasons'])
        with self.assertRaises(ValueError):
            p.candidate_pose(m, base, {}, .011)

    def test_floors_hashes_force_checks(self):
        d = self.doc; excluded = d['excluded_indices']; m = self.model
        self.assertTrue(d['mapping']['runtime_validated'])
        self.assertEqual(len(d['covered_indices']), 74)
        self.assertEqual(d['provenance']['xml_sha256_before'], p.cp.CANONICAL_XML_SHA256)
        self.assertEqual(d['provenance']['xml_sha256_after'], p.cp.sha256_file(p.cp.CANONICAL_XML))
        self.assertEqual(d['provenance']['source_sha256'], p.cp.sha256_file(p.audit.SOURCE))
        self.assertEqual(d['provenance']['source_sha256_after'], d['provenance']['source_sha256'])
        self.assertEqual(d['mapping']['manifest_sha256'], p.cp.sha256_file(p.s.MANIFEST))
        for r in [d['baseline_original']] + [r for r in d['rows'] if r['status']=='EVALUATED']:
            a = np.array(r['activation'])
            np.testing.assert_array_equal(a[excluded], m.actuator_ctrlrange[excluded, 0])
            self.assertTrue(np.all(a>=m.actuator_ctrlrange[:, 0]) and np.all(a<=m.actuator_ctrlrange[:, 1]))
            self.assertEqual(r['qpos_sha256'], p.audit.digest_array(r['qpos']))
            self.assertEqual(r['activation_sha256'], p.audit.digest_array(a))
            self.assertEqual(r['force_audit']['qpos'], r['qpos'])
            self.assertTrue(all(c['within_tolerance'] for c in r['force_audit']['checks'].values()))
            if 'solver' in r:
                self.assertLessEqual(r['objective'], r['solver']['seed_objective']+1e-12)
                self.assertAlmostEqual(r['objective'], r['solver']['best_evaluated_objective'], places=12)
                self.assertTrue(r['solver']['retained_best_including_seed'])
                if r['solver']['nfev'] is not None: self.assertLessEqual(r['solver']['nfev'], 60)

    def test_ranking_integrations_and_budget(self):
        d = self.doc; ranked = sorted([r for r in d['rows'] if r['status']=='EVALUATED'], key=lambda r:r['objective'])
        self.assertEqual(d['ranked_labels'], [r['label'] for r in ranked])
        self.assertEqual(d['best_objective_label'], ranked[0]['label'])
        angular = [r for r in ranked if r['angular_screen_pass']]
        self.assertEqual(d['best_angular_screen_label'], angular[0]['label'] if angular else None)
        expected = ['baseline_original_saved_activation', d['best_objective_label']]
        if 'baseline_exact_source' not in expected: expected.append('baseline_exact_source')
        self.assertEqual([i['label'] for i in d['integrations']], expected)
        self.assertLessEqual(len(d['integrations']), 3)
        self.assertLessEqual(d['wall_s'], 80.)
        for item in d['integrations']:
            result = item['result']
            self.assertEqual(result['requested_duration_s'], .05)
            self.assertTrue(result['finite'])
            self.assertFalse(result['automatic_reset_detected'])
            self.assertEqual(result['warning_count'], 0)
            self.assertIn('no act writes', result['activation_initialization'])


if __name__ == '__main__':
    with p.s.mg.guarded_run('test_pose_support_search', **p.s.mg.BUDGETS['neural_muscle_closure']):
        unittest.main(verbosity=2)
