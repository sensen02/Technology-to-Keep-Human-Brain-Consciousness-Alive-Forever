"""Isolated CLI regressions; never rewrite the real sourced sensory artifacts."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools_apply_sourced_sensory_config as gate


class SourcedSensoryConfigCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / 'tools_apply_sourced_sensory_config.py'
        shutil.copyfile(gate.__file__, self.script)
        self.source = self.root / 'data/literature_sensory_parameters.json'
        self.output = self.root / 'outputs/model_branches/flybody_loop/sourced_sensory_config.json'
        self.source.parent.mkdir()
        self.source.write_text(json.dumps({
            'angle_normaliser_channel': 'claw_position',
            'club_quantity': 'zero',
            'channels': {'claw_position': {
                'quantity': 'femur-tibia joint angle',
                'published_range': {'low': -60, 'high': 60, 'unit': 'deg'},
                'source': {'citation': 'Test fixture', 'url': 'https://example.org/paper'},
            }},
        }))

    def run_cli(self):
        return subprocess.run(
            [sys.executable, str(self.script)], cwd=self.root,
            capture_output=True, text=True, timeout=10)

    def test_successful_cli_exits_zero_without_dict_on_stderr(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        self.assertIn(str(self.output), result.stdout)
        output = json.loads(self.output.read_text())
        self.assertEqual(output['status'], 'DERIVED_FROM_PUBLISHED_MEASUREMENTS')
        self.assertAlmostEqual(output['reference_angle_rad'], 1.0471975511965976)

    def test_missing_source_cli_still_refuses_with_nonzero_exit(self):
        self.source.unlink()
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('REFUSED:', result.stderr)
        self.assertFalse(self.output.exists())

    def test_main_keeps_returning_the_written_dictionary(self):
        with patch.object(gate, 'SOURCE', self.source), patch.object(gate, 'OUT', self.output):
            with contextlib.redirect_stdout(io.StringIO()):
                output = gate.main()
        self.assertIsInstance(output, dict)
        self.assertEqual(output, json.loads(self.output.read_text()))


if __name__ == '__main__':
    unittest.main()
