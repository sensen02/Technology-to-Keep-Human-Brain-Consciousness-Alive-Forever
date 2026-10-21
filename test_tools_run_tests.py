import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "tools_run_tests.py"


class TestRunner(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args], cwd=HERE,
            capture_output=True, text=True, check=False,
        )

    def test_selected_module_and_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "result.json"
            module = root / "test_ok.py"
            module.write_text("import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n")
            proc = subprocess.run(
                [sys.executable, str(RUNNER), "--modules", str(module), "--timeout", "30", "--output", str(output)],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            document = json.loads(output.read_text())
            self.assertEqual(document["summary"]["total"], 1)
            self.assertTrue(document["modules"][0]["passed"])
            self.assertIn("stdout", document["modules"][0])
            self.assertIn("stderr", document["modules"][0])

    def test_absolute_path_does_not_run_same_named_local_module(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            other = root / 'other'
            other.mkdir()
            (root / 'test_collision.py').write_text(
                'import unittest\nclass T(unittest.TestCase):\n def test_wrong(self): self.fail("wrong file")\n')
            selected = other / 'test_collision.py'
            selected.write_text(
                'import unittest\nclass T(unittest.TestCase):\n def test_selected(self): pass\n')
            proc = subprocess.run(
                [sys.executable, str(RUNNER), '--modules', str(selected)],
                cwd=root, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_empty_module_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'test_empty.py').write_text('value = 1\n')
            output = root / 'result.json'
            proc = subprocess.run([sys.executable, str(RUNNER), '--output', str(output)],
                                  cwd=root, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['summary']['no_tests'], 1)
            self.assertFalse(result['modules'][0]['passed'])

    def test_no_modules_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run([sys.executable, str(RUNNER)], cwd=td,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('no test modules found', proc.stderr)

    def test_nonfinite_timeout_is_rejected(self):
        for value in ('nan', 'inf'):
            with self.subTest(value=value):
                proc = self.invoke('--timeout', value)
                self.assertEqual(proc.returncode, 2)
                self.assertIn('finite and positive', proc.stderr)

    def test_missing_interpreter_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'test_probe.py').write_text('value = 1\n')
            output = root / 'result.json'
            proc = subprocess.run(
                [sys.executable, str(RUNNER), '--python', str(root / 'missing-python'),
                 '--output', str(output)], cwd=root, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result['summary']['launch_errors'], 1)
            self.assertTrue(result['modules'][0]['launch_error'])

    def test_globs_are_relative_to_declared_root(self):
        import tools_run_tests as runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            selected = root / 'test_probe.py'
            selected.write_text('value = 1\n')
            self.assertEqual(runner._module_paths(['test_*.py'], root), [selected])
            self.assertEqual(runner._module_paths([], root), [selected])

    def test_timeout_is_recorded_and_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module = root / "test_slow.py"
            module.write_text("import time\ntime.sleep(2)\n")
            proc = subprocess.run(
                [sys.executable, str(RUNNER), "--modules", str(module), "--timeout", "0.05"],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("TIMEOUT", proc.stdout)


if __name__ == "__main__":
    unittest.main()
