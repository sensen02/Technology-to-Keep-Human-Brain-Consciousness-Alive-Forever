"""Focused dynamic audit regressions, guarded and serial; canonical XML never edited."""
import run_dynamic_support_audit as dynamic
import ast
import copy
import inspect
import json
import unittest
import numpy as np
import mujoco


class DynamicSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = dynamic.cp.sha256_file(dynamic.cp.CANONICAL_XML)
        cls.doc = json.loads(dynamic.SOURCE.read_text())
        cls.result = json.loads(dynamic.OUT.read_text())
        cls.model = dynamic.cp.load_model()

    @classmethod
    def tearDownClass(cls):
        assert cls.before == dynamic.cp.sha256_file(dynamic.cp.CANONICAL_XML)

    def test_same_source_exact_optimized_rows(self):
        rows = dynamic.select_rows(self.doc)
        self.assertEqual([r['label'] for r in rows], list(dynamic.LABELS))
        for row, trajectory in zip(rows, self.result['trajectories']):
            for field in ('qpos', 'activation'):
                np.testing.assert_array_equal(row[field], trajectory['initial_'+field])
                self.assertEqual(row[field+'_sha256'], trajectory[field+'_sha256'])
            self.assertNotEqual(row['activation_sha256'], self.doc['baseline_original']['activation_sha256'])
        broken = copy.deepcopy(self.doc)
        broken['best_objective_label'] = 'baseline_exact_source'
        with self.assertRaises(ValueError):
            dynamic.select_rows(broken)
        broken = copy.deepcopy(self.doc)
        next(r for r in broken['rows'] if r['label'] == dynamic.LABELS[0])['qpos'][2] += .001
        with self.assertRaises(ValueError):
            dynamic.select_rows(broken)

    def test_saved_observation_identity_and_health(self):
        self.assertEqual(self.result['status'], 'MEASURED_WITHIN_DECLARED_TOLERANCES')
        for t in self.result['trajectories']:
            self.assertTrue(all(t['observation_identity'].values()))
            self.assertEqual(t['steps'], 500)
            self.assertEqual(len(t['samples']), 51)
            self.assertTrue(t['finite'])
            self.assertFalse(t['automatic_reset_detected'])
            self.assertEqual(t['warning_count'], 0)
            self.assertEqual(sum(t['final_warning_counts']), 0)
            self.assertTrue(t['command_unchanged'])
            self.assertEqual(t['activation_initialization_writes'], 1)
            for key in ('final_qpos', 'final_activation'):
                self.assertTrue(dynamic.bit_identical(t[key], t['reference'][key]))
            for key in t['trace']:
                self.assertTrue(dynamic.bit_identical(t['trace'][key], t['reference']['trace'][key]))
            self.assertEqual(len(t['summary_at_ms']), 4)

    def test_all_sample_newton_convergence_and_force_decomposition(self):
        for t in self.result['trajectories']:
            for sample in t['samples']:
                for name, check in sample['checks'].items():
                    self.assertTrue(check['within_tolerance'], (t['label'], sample['time_s'], name, check))
                muscle = sample['muscles']
                for name, check in muscle['checks'].items():
                    self.assertTrue(check['within_tolerance'], (name, check))
                np.testing.assert_allclose(np.array(muscle['active_force'])+muscle['passive_force'], muscle['total_force'], atol=1e-10)
                np.testing.assert_allclose(np.sum([v['force_world'] for v in sample['named_feet'].values()], axis=0),
                    np.sum([r['force_world'] for r in sample['contact']['rows'] if r['is_named_foot']], axis=0), atol=1e-10)
            self.assertGreater(max(np.linalg.norm(x['Jdot_qvel_mm_s2']) for x in t['samples']), .01)

    def moving_state(self):
        row = dynamic.select_rows(self.doc)[1]
        ev = dynamic.s.FixedPose(self.model, row['qpos'])
        ev.acceleration(np.array(row['activation']))
        for _ in range(100):
            mujoco.mj_step(self.model, ev.data)
            mujoco.mj_forward(self.model, ev.data)
        return ev.data

    def test_live_moving_com_fd_and_independent_jacdot(self):
        m, d = self.model, self.moving_state()
        self.assertGreater(np.linalg.norm(d.qvel), 0)
        com = dynamic.moving_com(m, d)
        half = dynamic.moving_com(m, d, com['epsilon_s']/2)
        np.testing.assert_allclose(com['acceleration'], half['acceleration'], atol=.002, rtol=1e-7)
        scratch = mujoco.MjData(m)
        mujoco.mj_copyData(scratch, m, d)
        mujoco.mj_comVel(m, scratch)
        jdot = np.zeros((3, m.nv))
        jac = np.zeros_like(jdot)
        for b in range(1, m.nbody):
            mujoco.mj_jacDot(m, scratch, jac, None, scratch.xipos[b], b)
            jdot += m.body_mass[b]*jac
        jdot /= m.body_mass[1:].sum()
        np.testing.assert_allclose(com['Jdot_qvel'], jdot @ d.qvel, atol=.002, rtol=1e-6)
        # Independent directional position difference verifies COM velocity.
        h = com['epsilon_s']
        positions = []
        for sign in (-1, 1):
            scratch.qpos[:] = d.qpos
            mujoco.mj_integratePos(m, scratch.qpos, d.qvel, sign*h)
            mujoco.mj_kinematics(m, scratch)
            positions.append((m.body_mass[1:, None]*scratch.xipos[1:]).sum(axis=0)/m.body_mass[1:].sum())
        np.testing.assert_allclose(com['velocity'], (positions[1]-positions[0])/(2*h), atol=1e-5, rtol=1e-6)
        obs = dynamic.observe(m, d)
        self.assertTrue(obs['checks']['COM_Newton_balance']['within_tolerance'], obs['checks'])

    def test_live_observation_does_not_mutate_integration_state(self):
        m, d = self.model, self.moving_state()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(m, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(m, d, before, spec)
        arrays = {key: getattr(d, key).copy() for key in ('qacc', 'actuator_force', 'qfrc_constraint', 'qacc_warmstart')}
        obs = dynamic.observe(m, d)
        mujoco.mj_getState(m, d, after, spec)
        self.assertTrue(dynamic.bit_identical(before, after))
        for key, value in arrays.items():
            self.assertTrue(dynamic.bit_identical(value, getattr(d, key)))
        self.assertTrue(obs['muscles']['same_physical_state'])
        self.assertTrue(all(c['within_tolerance'] for c in obs['muscles']['checks'].values()))

    def test_no_command_or_act_write_in_integration_loop(self):
        tree = ast.parse(inspect.getsource(dynamic.integrate_observed))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'acceleration']
        self.assertEqual(len(calls), 1)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for part in ast.walk(target):
                        if isinstance(part, ast.Attribute):
                            self.assertNotIn(part.attr, ('ctrl', 'act'))

    def test_frozen_provenance(self):
        p = self.result['provenance']
        for key in ('xml_sha256_before', 'xml_sha256_after', 'expected_frozen_sha256'):
            self.assertEqual(p[key], self.before)
        self.assertEqual(p['source_sha256'], dynamic.cp.sha256_file(dynamic.SOURCE))
        self.assertEqual(p['source_sha256'], p['source_sha256_after'])
        self.assertEqual(p['code_sha256'], dynamic.cp.sha256_file(dynamic.Path(dynamic.__file__)))
        for path, digest in p['dependency_code_sha256'].items():
            self.assertEqual(digest, dynamic.cp.sha256_file(dynamic.Path(path)))


if __name__ == '__main__':
    with dynamic.mg.guarded_run('dynamic_support_tests', **dynamic.mg.BUDGETS['neural_muscle_closure']):
        unittest.main(verbosity=2)
