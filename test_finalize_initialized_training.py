"""Offline S1/P1 checkpoint-decision regression tests; no physics is imported/run."""
import builtins
import contextlib
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PHASES = ('early', 'preparation', 'formal')
SAFETY_METRICS = (
    'min_height_ratio', 'max_downward_speed', 'max_abs_roll_pitch',
    'max_nonfoot_penetration', 'max_limit_excess',
)
SUPPORT_METRICS = ('support_ms', 'longest_all_gates_true_endpoint_span_ms')
_REAL_IMPORT = builtins.__import__


def offline_import(name, *args, **kwargs):
    """Fail loudly if finalization ever starts importing simulation code."""
    if name.split('.')[0] in {'engine', 'mujoco', 'flygym', 'brian2', 'dm_control'}:
        raise AssertionError('Physics/simulation imports forbidden in this test: ' + name)
    return _REAL_IMPORT(name, *args, **kwargs)


class FinalizeInitializedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch('builtins.__import__', side_effect=offline_import):
            cls.finalizer = importlib.import_module('finalize_initialized_training')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='finalize-initialized-test-')
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.baseline = np.array([[0.0, 0.125, -0.5], [1.0, 2.0, 3.0]], dtype=np.float64)
        self.updated = self.baseline + 0.03125
        self.protocol = {
            'require_composite_strict_decrease': True,
            'require_safety_nonworsening_in': list(PHASES),
            'tolerance': 1e-12,
            'safety_metrics': list(SAFETY_METRICS),
        }
        baseline = {'label': 'baseline', 'optimization_loss': 10.0}
        for phase in PHASES:
            baseline[phase] = dict(zip(SAFETY_METRICS, (0.9, 2.0, 0.2, 0.01, 0.001)))
        baseline['formal'].update(dict(zip(SUPPORT_METRICS, (2.0, 1.0))))
        candidate = copy.deepcopy(baseline)
        candidate.update(label='updated', optimization_loss=9.0)
        candidate['formal']['support_ms'] = 3.0
        self.evidence = {
            'passed': True,
            'checks': [baseline, candidate],
            'comparisons': {'updated': {'physical_joint_acceptance': True}},
        }
        self.source = {'path': 'synthetic-source-weights.npz', 'sha256': 'synthetic-source-hash'}
        self.save_inputs()

    @property
    def candidate(self):
        return self.evidence['checks'][1]

    def save_inputs(self):
        for name, value in (
            ('INDEPENDENT_CHECK.json', self.evidence),
            ('ACCEPTANCE_PROTOCOL.json', self.protocol),
            ('PREREG.json', {'same_weights': self.source}),
        ):
            (self.out / name).write_text(json.dumps(value), encoding='utf-8')
        np.savez(self.out / 'DIRECTION.npz', weights=self.baseline)
        np.savez(self.out / 'UPDATE.npz', updated=self.updated)

    def invoke(self):
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['finalize_initialized_training.py', '--output', str(self.out)]):
            with mock.patch('builtins.__import__', side_effect=offline_import):
                with contextlib.redirect_stdout(output):
                    self.finalizer.main()
        return json.loads(output.getvalue())

    def assert_checkpoint(self, accepted):
        printed = self.invoke()
        record = json.loads((self.out / 'CHECKPOINT.json').read_text(encoding='utf-8'))
        self.assertEqual(printed, record)
        self.assertIs(record['accepted'], accepted)
        self.assertEqual(record['selected_arm'], 'updated' if accepted else 'baseline')
        self.assertEqual(record['decision'], 'RETAIN_VERIFIED_UPDATE_FOR_LATER_VALIDATION'
                         if accepted else 'REJECT_UPDATE_RETAIN_SOURCE_WEIGHTS')
        with np.load(self.out / 'CHECKPOINT.npz', allow_pickle=False) as saved:
            expected = self.updated if accepted else self.baseline
            self.assertEqual(saved.files, ['weights'])
            self.assertEqual(saved['weights'].dtype, expected.dtype)
            np.testing.assert_array_equal(saved['weights'], expected)
        self.assertEqual(record['source_weights'], self.source)
        self.assertEqual(record['checkpoint'], str(self.out / 'CHECKPOINT.npz'))
        for field, name in (
            ('checkpoint_sha256', 'CHECKPOINT.npz'),
            ('independent_check_sha256', 'INDEPENDENT_CHECK.json'),
            ('acceptance_protocol_sha256', 'ACCEPTANCE_PROTOCOL.json'),
        ):
            self.assertEqual(record[field], hashlib.sha256((self.out / name).read_bytes()).hexdigest())
        self.assertEqual(record['comparison'], self.evidence['comparisons']['updated'])
        self.assertIs(record['deployment'], False)
        self.assertEqual(record['behavior_claim'], 'NONE')
        return record

    def assert_no_checkpoint(self):
        self.assertFalse((self.out / 'CHECKPOINT.npz').exists())
        self.assertFalse((self.out / 'CHECKPOINT.json').exists())

    def test_loss_decline_with_safety_regression_preserves_baseline(self):
        original = copy.deepcopy(self.evidence)
        for phase in PHASES:
            for metric in SAFETY_METRICS:
                with self.subTest(phase=phase, metric=metric):
                    self.evidence = copy.deepcopy(original)
                    change = -0.01 if metric == 'min_height_ratio' else 0.01
                    self.candidate[phase][metric] += change
                    self.evidence['comparisons']['updated']['physical_joint_acceptance'] = False
                    self.save_inputs()
                    self.assert_checkpoint(False)
                    (self.out / 'CHECKPOINT.json').unlink()
                    (self.out / 'CHECKPOINT.npz').unlink()

    def test_no_formal_improvement_rejected(self):
        self.candidate['formal']['support_ms'] = 2.0
        # An early-window improvement is not formal support improvement.
        self.candidate['early']['support_ms'] = 100.0
        self.evidence['comparisons']['updated']['physical_joint_acceptance'] = False
        self.save_inputs()
        self.assert_checkpoint(False)

    def test_formal_metric_regression_cannot_trade_for_other_improvement(self):
        self.candidate['formal'][SUPPORT_METRICS[1]] = 0.5
        self.evidence['comparisons']['updated']['physical_joint_acceptance'] = False
        self.save_inputs()
        self.assert_checkpoint(False)

    def test_true_joint_acceptance_retains_updated(self):
        self.assert_checkpoint(True)

    def test_longest_span_improvement_alone_can_satisfy_support(self):
        self.candidate['formal'].update(dict(zip(SUPPORT_METRICS, (2.0, 1.5))))
        self.save_inputs()
        self.assert_checkpoint(True)

    def test_loss_must_strictly_decline(self):
        for loss in (10.0, 11.0):
            with self.subTest(loss=loss):
                self.candidate['optimization_loss'] = loss
                self.evidence['comparisons']['updated']['physical_joint_acceptance'] = False
                self.save_inputs()
                self.assert_checkpoint(False)
                (self.out / 'CHECKPOINT.json').unlink()
                (self.out / 'CHECKPOINT.npz').unlink()

    def test_rerun_does_not_overwrite_checkpoint(self):
        self.assert_checkpoint(True)
        paths = [self.out / 'CHECKPOINT.json', self.out / 'CHECKPOINT.npz']
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        # A second invocation would choose completely different weights if allowed.
        self.updated.fill(99.0)
        self.save_inputs()
        with self.assertRaises(AssertionError):
            self.invoke()
        for path in paths:
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before[path])

    def test_evidence_acceptance_mismatch_fails_before_checkpoint_writing(self):
        for computed_acceptance in (True, False):
            with self.subTest(computed_acceptance=computed_acceptance):
                self.candidate['optimization_loss'] = 9.0 if computed_acceptance else 11.0
                self.evidence['comparisons']['updated']['physical_joint_acceptance'] = not computed_acceptance
                self.save_inputs()
                # Missing arrays prove mismatch is rejected before checkpoint input reads too.
                (self.out / 'DIRECTION.npz').unlink()
                (self.out / 'UPDATE.npz').unlink()
                with mock.patch.object(self.finalizer.np, 'savez', side_effect=AssertionError('unexpected write')) as save:
                    with self.assertRaises(AssertionError):
                        self.invoke()
                    save.assert_not_called()
                self.assert_no_checkpoint()

    def test_failed_independent_check_fails_before_checkpoint_writing(self):
        self.evidence['passed'] = False
        self.save_inputs()
        with self.assertRaises(AssertionError):
            self.invoke()
        self.assert_no_checkpoint()


if __name__ == '__main__':
    unittest.main()
