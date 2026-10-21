"""End-to-end entry gates and frozen 5ms recomputation; synthetic only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_motor_training import manifest, samples
from test_motor_data_qualification import synthetic_schema_fixture
from engine.embodied.motor_training import recorded_features
from engine.embodied.motor_translation import FiniteMotorTranslator
from train_motor_translation import main


def dataset():
    spikes, _ = samples()
    _, features = recorded_features(manifest(), spikes, command_ticks=10)
    pool = np.clip(features.sum(axis=2)*.04, 0, 1)
    return dict(spikes=spikes, targets=np.column_stack([pool[:, 0], pool[:, 0], pool[:, 1], np.zeros(10)]),
                connected=np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
                source_start_s=np.arange(10)*.005, source_end_s=np.arange(1, 11)*.005,
                target_start_s=np.arange(1, 11)*.005, target_end_s=np.arange(2, 12)*.005)


def metadata(kind='INTERFACE_TEST'):
    return dict(target_kind=kind, command_ticks=10,
                target_alignment='next_interval_after_completed_source_block')


def support(**overrides):
    return {'support_'+k: v for k, v in synthetic_schema_fixture(**overrides).items()}


class TrainingEntryTests(unittest.TestCase):
    def invoke(self, root, arrays, meta):
        mp, dp, output = root/'manifest.json', root/'dataset.npz', root/'model.json'
        mp.write_text(json.dumps(manifest()))
        np.savez(dp, **arrays, metadata=json.dumps(meta))
        args = ['--manifest', str(mp), '--dataset', str(dp), '--output', str(output),
                '--steps', '10', '--learning-rate', '.1']
        return args, output

    def test_rejections_happen_before_fit_and_export(self):
        cases = []
        cases.append(('failed_support', dict(dataset(), **support(root_height=[1., 1.])), metadata('BEHAVIOR'), 'support qualification failed'))
        cases.append(('missing_support', dataset(), metadata('BEHAVIOR'), 'missing='))
        missing = dict(dataset(), **support()); del missing['support_foot_normal']
        cases.append(('missing_field', missing, metadata('BEHAVIOR'), 'foot_normal'))
        cases.append(('passed_support_no_provenance', dict(dataset(), **support()), metadata('BEHAVIOR'), 'provenance NOT_ESTABLISHED'))
        for field in ('source_start_s', 'source_end_s', 'target_start_s', 'target_end_s'):
            arrays = dataset(); arrays[field] = arrays[field].copy(); arrays[field][4] += .0005
            cases.append((field, arrays, metadata(), 'misaligned'))
        arrays = dataset(); del arrays['target_start_s']
        cases.append(('missing_timestamp', arrays, metadata(), 'missing command timestamp'))
        arrays = dataset(); arrays['source_end_s'][2] = np.nan
        cases.append(('nonfinite_timestamp', arrays, metadata(), 'finite'))
        cases.append(('unknown_use', dataset(), metadata('UNKNOWN'), 'target_kind'))
        arrays = dataset(); arrays['targets'] = arrays['targets'].astype(complex)+1j
        cases.append(('complex_targets', arrays, metadata(), 'real numeric'))
        arrays = dataset(); arrays['targets'] = arrays['targets'].astype(str)
        cases.append(('string_targets', arrays, metadata(), 'real numeric'))
        for bad in (np.nan, np.inf, -np.inf):
            arrays = dataset(); arrays['targets'][0, 0] = bad
            cases.append(('nonfinite_targets', arrays, metadata(), 'finite'))
        for bad in (np.nan, np.inf, -np.inf):
            meta = metadata(); meta['extra'] = {'nested': bad}
            cases.append(('nonfinite_metadata', dataset(), meta, 'finite JSON'))
        cases.append(('support_failure_even_interface', dict(dataset(), **support(effective_nonfoot=[1, 0])), metadata(), 'support qualification failed'))
        for name, arrays, meta, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); args, output = self.invoke(root, arrays, meta)
                with patch('train_motor_translation.fit') as fit, contextlib.redirect_stderr(io.StringIO()) as err:
                    with self.assertRaises(SystemExit):
                        main(args)
                    fit.assert_not_called()
                self.assertIn(message, err.getvalue())
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output)+'.training_receipt.json').exists())

    def test_actual_cli_rejects_complex_and_overflow_metadata(self):
        import os
        import subprocess
        import sys
        for name in ('complex', 'metadata_overflow'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); arrays = dataset()
                if name == 'complex': arrays['targets'] = arrays['targets'].astype(complex)+1j
                args, output = self.invoke(root, arrays, metadata())
                if name == 'metadata_overflow':
                    raw = json.dumps(metadata())[:-1]+', "extra": {"nested": [1e999]}}'
                    np.savez(root/'dataset.npz', **arrays, metadata=raw)
                    with patch('train_motor_translation.fit') as fit, contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit): main(args)
                        fit.assert_not_called()
                command = [sys.executable, '-B', str(Path(__file__).with_name('train_motor_translation.py')), *args]
                run = subprocess.run(command, capture_output=True, text=True,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn('real numeric' if name == 'complex' else 'finite JSON', run.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output)+'.training_receipt.json').exists())

    def test_direct_fit_rejects_invalid_labels_before_objective(self):
        from engine.embodied.motor_training import fit
        arrays = dataset()
        invalid = [arrays['targets'].astype(complex)+1j, arrays['targets'].astype(str),
                   np.full_like(arrays['targets'], np.nan),
                   np.full_like(arrays['targets'], np.inf),
                   np.full_like(arrays['targets'], -np.inf),
                   np.full(arrays['targets'].shape, None, dtype=object)]
        for labels in invalid:
            with self.subTest(dtype=labels.dtype), patch('engine.embodied.motor_training.loss_and_gradient') as objective:
                with self.assertRaises(ValueError):
                    fit(manifest(), arrays['spikes'], labels, command_ticks=10, steps=1)
                objective.assert_not_called()

    def test_pair_serialization_and_publication_failures_leave_no_half_pair(self):
        import os
        from train_motor_translation import publish_artifacts
        translator = FiniteMotorTranslator(manifest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root/'model.json'; receipt = root/'receipt.json'
            with self.assertRaises(ValueError):
                publish_artifacts(translator, {'nested': {'bad': np.nan}}, model, receipt)
            self.assertFalse(model.exists()); self.assertFalse(receipt.exists())
            with patch.object(translator, 'save', side_effect=lambda path: path.write_text('{"bad":NaN}')):
                with self.assertRaises(ValueError):
                    publish_artifacts(translator, {}, model, receipt)
            self.assertFalse(model.exists()); self.assertFalse(receipt.exists())
            link = os.link
            def fail_second(source, destination):
                if Path(destination) == receipt: raise OSError('injected receipt publication failure')
                return link(source, destination)
            with patch('train_motor_translation.os.link', side_effect=fail_second):
                with self.assertRaises(OSError):
                    publish_artifacts(translator, {}, model, receipt)
            self.assertFalse(model.exists()); self.assertFalse(receipt.exists())
            self.assertEqual(list(root.iterdir()), [])
            # Simulate an existing receipt appearing after the CLI precheck.
            receipt.write_text('existing receipt')
            with self.assertRaises(FileExistsError):
                publish_artifacts(translator, {}, model, receipt)
            self.assertFalse(model.exists()); self.assertEqual(receipt.read_text(), 'existing receipt')
            receipt.unlink(); model.write_text('existing model')
            with self.assertRaises(FileExistsError):
                publish_artifacts(translator, {}, model, receipt)
            self.assertEqual(model.read_text(), 'existing model'); self.assertFalse(receipt.exists())

    def test_entry_receipt_serialization_failure_leaves_no_artifacts(self):
        from engine.embodied.motor_training import fit
        arrays = dataset()
        result = fit(manifest(), arrays['spikes'], arrays['targets'], command_ticks=10, steps=1)
        result.receipt['injected_nonfinite'] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            args, output = self.invoke(Path(directory), arrays, metadata())
            with patch('train_motor_translation.fit', return_value=result):
                with self.assertRaises(ValueError): main(args)
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output)+'.training_receipt.json').exists())
            self.assertFalse(list(Path(directory).glob('.motor-training-*')))

    def test_success_load_loss_and_causal_prefix(self):
        arrays = dataset()
        with tempfile.TemporaryDirectory() as directory:
            args, output = self.invoke(Path(directory), arrays, metadata())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            receipt = json.loads(Path(str(output)+'.training_receipt.json').read_text())
            self.assertEqual(receipt['data_qualification']['clock'], '5ms_next_interval')
            self.assertFalse(receipt['data_qualification']['behavior_success'])
            loaded = FiniteMotorTranslator.load(manifest(), output)
            fresh = FiniteMotorTranslator(manifest(), receipt['final_weights'])
            changed = arrays['spikes'].copy(); changed[60:, :2] = 1-changed[60:, :2]
            future = FiniteMotorTranslator(manifest(), receipt['final_weights'])
            outputs = []
            for b in range(10):
                kw = dict(spike_dt_s=.0005, start_s=b*.005, end_s=(b+1)*.005,
                          connected=bool(arrays['connected'][b]))
                sp = arrays['spikes'][b*10:(b+1)*10]
                out, _ = loaded.consume(sp, **kw)
                np.testing.assert_array_equal(out, fresh.consume(sp, **kw)[0])
                altered = future.consume(changed[b*10:(b+1)*10], **kw)[0]
                if b < 6: np.testing.assert_array_equal(out, altered)
                if b == 3: np.testing.assert_array_equal(out, 0)
                self.assertEqual(out[0], out[1]); self.assertEqual(out[3], 0)
                outputs.append(out)
            mapped = loaded.actuator_pool >= 0
            loss = np.mean((np.array(outputs)[:, mapped]-arrays['targets'][:, mapped])**2)
            self.assertAlmostEqual(loss, receipt['losses'][-1], places=14)
            self.assertLess(receipt['losses'][-1], receipt['losses'][0])


if __name__ == '__main__':
    unittest.main()
