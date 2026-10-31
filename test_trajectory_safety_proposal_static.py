"""Static-only S137 contract checks: never import/execute proposer or optimizer."""
import ast
from pathlib import Path
import unittest

SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/propose_trajectory_safety_step.py')


class StaticProposalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.tree = ast.parse(cls.text, filename=str(SOURCE))
        cls.functions = {n.name: n for n in cls.tree.body if isinstance(n, ast.FunctionDef)}

    def test_syntax_and_inert_module(self):
        for node in self.tree.body:
            self.assertIsInstance(node, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.If))
        imports = [n for n in self.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            self.assertFalse(any(n.startswith(('scipy', 'verify_', 'run_', 'torch', 'mujoco')) for n in names))

    def test_exactly_one_optimizer_call_and_separate_certificate(self):
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual(sum(n.func.id == 'minimize' for n in calls), 1)
        self.assertEqual(sum(n.func.id == 'qp_certificate' for n in calls), 1)
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == 'subprocess' for a in n.names) for n in ast.walk(self.tree)))
        self.assertIn("H = .1 * np.eye(16); g = np.r_[np.zeros(15), 1.]", self.text)

    def test_verification_before_output_and_optimization(self):
        main = ast.get_source_segment(self.text, self.functions['main'])
        self.assertLess(main.index('inputs(source, verification)'), main.index('out.mkdir'))
        self.assertLess(main.index("save(out / 'PREREG.json'"), main.index('result = minimize('))
        inputs = ast.get_source_segment(self.text, self.functions['inputs'])
        for token in ("r['passed'] is True", "r['evidence_passed'] is True", "check_hashes(original_inputs)", "check_hashes(outputs)"):
            self.assertIn(token, inputs)

    def test_raw_constraints_and_slack_units(self):
        p = ast.get_source_segment(self.text, self.functions['problem'])
        for token in ('margin > 0.', 'lower_rows[:, 15] = SLACK_SCALE',
                      'upper_rows[:, 15] = -SLACK_SCALE', "bounds[15] = [0., margin / SLACK_SCALE]",
                      "bounds[~np.r_[j['valid'], True]] = 0.", "'all121_delta_load_nonworsen'",
                      "'all121_formal_height'", "'all121_formal_angle'", "'all121_formal_COM_vz'",
                      "'all121_formal_root_vz'"):
            self.assertIn(token, p)
        self.assertNotIn('qacc', p)
        self.assertEqual(p.count('[:, 15] ='), 2)

    def test_sixteen_postgates_before_candidate(self):
        gates = next(n.value for n in ast.walk(self.functions['main']) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'gates' for t in n.targets))
        self.assertIsInstance(gates, ast.Call)
        self.assertEqual(len(gates.keywords), 16)
        self.assertIn('if accepted: np.savez_compressed', self.text)
        self.assertIn("p['initial_margin']-slack >= MIN_IMPROVEMENT", self.text)
        self.assertIn('MIN_IMPROVEMENT = 1e-8', self.text)
        self.assertIn('LIMITED_LOCAL_PROPOSAL_NOT_TRUTH', self.text)


if __name__ == '__main__':
    unittest.main()
