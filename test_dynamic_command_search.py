"""Fast unit/provenance/prefix regression tests; no search or hidden tuning."""
import copy
import json
import unittest
import numpy as np
import run_dynamic_command_search as d


class DynamicCommandSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = d.cp.load_model()
        cls.covered, _, _ = d.s.coverage(cls.m)
        cls.doc = json.loads(d.dyn.SOURCE.read_text())
        cls.a, cls.b = d.source_rows(cls.doc, cls.m, cls.covered)

    def test_score_unit_scaling_and_all_directions(self):
        p = np.array([[0., 0., 0.], [1., 2., -3.], [-2., 1., 4.]])
        v = np.array([[0., 0., 0.], [1., 0., 0.], [2., 4., 5.]])
        q = [[1., 0., 0., 0.]]*3
        j = [[0., 0.], [.1, .2], [.2, .4]]
        terms, score = d.score_terms(p, v, q, j, 1., 9810.)
        changed, other = d.score_terms(p/1000, v/1000, q, j, .001, 9.81)
        np.testing.assert_allclose(list(terms.values()), list(changed.values()), rtol=1e-14)
        self.assertAlmostEqual(score, other)
        self.assertAlmostEqual(terms['COM_position_RMS_over_L']**2, (14+21)/2)
        self.assertAlmostEqual(score, d.score_terms(-p, -v, q, j)[1])
        self.assertAlmostEqual(score, sum(x*x for x in terms.values()))
        self.assertAlmostEqual(d.orientation_angle([0, 1, 0, 0], [1, 0, 0, 0]), np.pi)
        self.assertEqual(d.orientation_angle([-2, 0, 0, 0], [1, 0, 0, 0]), 0.)

    def test_source_rejections(self):
        for mutation in ('array', 'xml', 'floors', 'covered'):
            doc = copy.deepcopy(self.doc)
            a = next(x for x in doc['rows'] if x['label'] == 'baseline_exact_source')
            if mutation == 'array':
                a['qpos'][2] -= .01
            elif mutation == 'xml':
                doc['provenance']['xml_sha256_after'] = 'wrong'
            elif mutation == 'floors':
                excluded = np.setdiff1d(np.arange(self.m.nu), self.covered)
                a['activation'][excluded[0]] += .01
                a['activation_sha256'] = d.dyn.audit.digest_array(a['activation'])
            else:
                doc['covered_indices'] = []
            with self.assertRaises(ValueError):
                d.source_rows(doc, self.m, self.covered)

    def test_candidate_bounds_determinism_and_clipping_ledger(self):
        args = (self.a['activation'], self.b['activation'], self.covered, self.m.actuator_ctrlrange)
        rows = d.candidates(*args)
        self.assertEqual(rows, d.candidates(*args))
        self.assertEqual(len(rows), 11)
        excluded = np.setdiff1d(np.arange(self.m.nu), self.covered)
        for row in rows:
            a = np.asarray(row['activation']); raw = np.asarray(row['requested_activation'])
            np.testing.assert_array_equal(a[excluded], self.m.actuator_ctrlrange[excluded, 0])
            self.assertTrue(np.all(a >= self.m.actuator_ctrlrange[:, 0]))
            self.assertTrue(np.all(a <= self.m.actuator_ctrlrange[:, 1]))
            self.assertEqual(set(np.flatnonzero(a != raw)), {x['index'] for x in row['clipping_changes']})
            self.assertEqual(row['activation_sha256'], d.dyn.audit.digest_array(a))
            if row['recipe']['kind'] == 'bounded_perturbation':
                self.assertLessEqual(np.max(np.abs(np.asarray(row['actual_delta_from_A']))), .020000000000001)
        for minus, plus in zip(rows[5::2], rows[6::2]):
            np.testing.assert_allclose(minus['requested_delta_from_A'], -np.asarray(plus['requested_delta_from_A']), atol=1e-16)

    def test_selection_excludes_global_min_and_empty(self):
        rows = [dict(label='invalid', eligible=False, metrics=dict(score=0)),
                dict(label='valid', eligible=True, metrics=dict(score=1))]
        self.assertEqual(d.choose(rows)['label'], 'valid')
        self.assertIsNone(d.choose(rows[:1]))

    def test_repeat_and_prefix_native_pipeline(self):
        args = (self.m, self.a['qpos'], self.a['activation'])
        first = d.integrate(*args, duration=.002)
        repeat = d.integrate(*args, duration=.002)
        longer = d.integrate(*args, duration=.003)
        self.assertEqual(first, repeat)
        for group in ('trace', 'native_trace'):
            for key, values in first[group].items():
                self.assertTrue(d.dyn.bit_identical(values, longer[group][key][:len(values)]))
        self.assertGreater(np.linalg.norm(first['trace']['COM_velocity_mm_s'][-1]), 0)
        self.assertEqual(first['activation_initialization_writes'], 1)

    @unittest.skipUnless(d.OUT.exists(), 'artifact not produced yet')
    def test_artifact_hashes_constraints_and_holdout(self):
        out = json.loads(d.OUT.read_text())
        p = out['provenance']
        self.assertEqual(p['xml_sha256_before'], d.cp.CANONICAL_XML_SHA256)
        self.assertEqual(p['xml_sha256_after'], d.cp.sha256_file(d.cp.CANONICAL_XML))
        self.assertEqual(p['code_sha256'], d.cp.sha256_file(d.Path(d.__file__)))
        self.assertEqual(p['test_code_sha256'], d.cp.sha256_file(d.Path(__file__)))
        for path, digest in p['source_hashes'].items():
            self.assertEqual(digest, d.cp.sha256_file(d.Path(path)))
        self.assertEqual(out['initial_pose']['qpos'], self.a['qpos'])
        self.assertTrue(out['baseline_crossed_AA_identity']['all_bit_identical'])
        self.assertLess(out['wall_s'], 60)
        winner = d.choose(out['candidates'])
        self.assertEqual(out['selection']['winner_label'], winner['label'] if winner else None)
        for row in out['candidates']:
            self.assertEqual(row['steps'], 500)
            self.assertEqual(len(row['trace']['time_s']), 51)
            self.assertEqual(len(row['native_trace']['time_s']), 501)
            self.assertEqual(row['eligible'], not row['ineligibility_reasons'])
            self.assertEqual(row['metrics']['score'], sum(x*x for x in row['metrics']['dimensionless_terms'].values()))
        for row in out['validation100ms']:
            self.assertTrue(row['first50ms_identity']['all_bit_identical'])
            self.assertEqual(row['steps'], 1000)
            self.assertEqual(row['loses_validity_by100ms'], row['validity_at50ms'] and not row['validity_at100ms'])


if __name__ == '__main__':
    with d.mg.guarded_run('test_dynamic_command_search', **d.mg.BUDGETS['neural_muscle_closure']):
        unittest.main()
