"""Small bounded S4 engineering tests; synthetic labels are INTERFACE_TEST only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from engine.embodied.motor_training import fit, recorded_features, loss_and_gradient
from engine.embodied.motor_translation import FiniteMotorTranslator


def manifest():
    return dict(n_neurons=4, n_actuators=4, rows=[
        dict(actuator_index=0, local_indices=[0]),
        dict(actuator_index=1, local_indices=[0]),
        dict(actuator_index=2, local_indices=[1]),
        dict(actuator_index=3, local_indices=[])])


def samples():
    spikes = np.zeros((100, 4), dtype=np.uint8)
    spikes[[0, 31, 79], 0] = 1
    spikes[[11, 57], 1] = 1
    spikes[:, 2:] = 1  # outside MN whitelist: must be irrelevant
    _, features = recorded_features(manifest(), spikes)
    pool_targets = np.clip(np.einsum('tpk,pk->tp', features,
                                    np.full((2, 5), .04)), 0, 1)
    return spikes, np.column_stack([pool_targets[:, 0], pool_targets[:, 0],
                                   pool_targets[:, 1], np.zeros(len(spikes))])


class MotorTrainingTests(unittest.TestCase):
    def test_gradient_matches_finite_difference_with_shared_multiplicity(self):
        runtime = FiniteMotorTranslator(manifest())
        rng = np.random.default_rng(2)
        features = rng.uniform(.01, .1, (7, 2, 5))
        weights = rng.uniform(.1, .3, (2, 5))
        targets = np.tile([.3, .3, .6, 0.], (7, 1))
        _, gradient = loss_and_gradient(weights, features, targets, runtime.actuator_pool)
        numeric = np.zeros_like(weights)
        for p, k in np.ndindex(weights.shape):
            plus, minus = weights.copy(), weights.copy()
            plus[p, k] += 1e-6
            minus[p, k] -= 1e-6
            numeric[p, k] = (loss_and_gradient(plus, features, targets, runtime.actuator_pool)[0]
                             - loss_and_gradient(minus, features, targets, runtime.actuator_pool)[0])/2e-6
        np.testing.assert_allclose(gradient, numeric, atol=1e-10, rtol=1e-7)
        self.assertTrue(np.isfinite(gradient).all())

    def test_clip_derivative_saturated_and_boundary_zero(self):
        runtime = FiniteMotorTranslator(manifest())
        targets = np.tile([.2, .2, .4, 0.], (3, 1))
        for weight in (0., .2, 1., -.2):
            weights = np.full((2, 5), weight)
            features = np.ones((3, 2, 5))
            _, gradient = loss_and_gradient(weights, features, targets, runtime.actuator_pool)
            np.testing.assert_array_equal(gradient, 0)
        # Strictly saturated, not just a selected kink subgradient.
        weights = np.ones((2, 5))
        for delta in (-1e-6, 1e-6):
            loss, _ = loss_and_gradient(weights+delta, features, targets, runtime.actuator_pool)
            self.assertEqual(loss, loss_and_gradient(weights, features, targets, runtime.actuator_pool)[0])

    def test_parameter_updates_reduce_loss_and_frozen_invariants(self):
        spikes, targets = samples()
        before = spikes.copy()
        result = fit(manifest(), spikes, targets, steps=40, lr=.1)
        self.assertEqual(len(result.losses), 41)
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertGreater(result.receipt['changed_parameters'], 0)
        self.assertTrue(np.all((result.translator.weights >= 0) & (result.translator.weights <= 4)))
        self.assertEqual(result.translator.elapsed_s, 0)
        np.testing.assert_array_equal(spikes, before)
        for t in range(len(spikes)):
            ctrl, _ = result.translator.consume(spikes[t:t+1], spike_dt_s=.0005,
                                               start_s=t*.0005, end_s=(t+1)*.0005)
            self.assertEqual(ctrl[0], ctrl[1])
            self.assertEqual(ctrl[3], 0)
            self.assertTrue(np.isfinite(ctrl).all())

    def test_features_reuse_runtime_and_disconnect_clears(self):
        spikes, _ = samples()
        connected = np.ones(len(spikes), bool)
        connected[1] = False
        _, features = recorded_features(manifest(), spikes, connected)
        np.testing.assert_array_equal(features[1], 0)
        np.testing.assert_array_equal(features[2], 0)
        changed = spikes.copy()
        changed[:, 2:] = 0
        np.testing.assert_array_equal(features, recorded_features(manifest(), changed, connected)[1])

    def test_5ms_blocks_match_runtime_next_interval_and_prefix(self):
        spikes, _ = samples()
        cuts = np.ones(10, bool)
        cuts[3] = False
        runtime, features = recorded_features(manifest(), spikes, cuts, command_ticks=10)
        direct = FiniteMotorTranslator(manifest())
        for block in range(10):
            ctrl, trace = direct.consume(spikes[block*10:(block+1)*10], spike_dt_s=.0005,
                                         start_s=block*.005, end_s=(block+1)*.005,
                                         connected=bool(cuts[block]))
            np.testing.assert_array_equal(features[block], trace['pool_features'])
            self.assertAlmostEqual(trace['valid_from_s'], (block+1)*.005)
        np.testing.assert_array_equal(features[3], 0)
        changed = spikes.copy()
        changed[60:, :2] = 1-changed[60:, :2]
        np.testing.assert_array_equal(features[:6], recorded_features(
            manifest(), changed, cuts, command_ticks=10)[1][:6])
        pooled = np.clip(np.einsum('tpk,pk->tp', features, np.full((2, 5), .04)), 0, 1)
        targets = np.column_stack([pooled[:, 0], pooled[:, 0], pooled[:, 1], np.zeros(10)])
        result = fit(manifest(), spikes, targets, cuts, steps=5, lr=.1, command_ticks=10)
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertEqual(result.receipt['ticks'], 100)
        self.assertEqual(result.receipt['command_blocks'], 10)
        np.testing.assert_allclose(result.receipt['target_valid_from_s'], np.arange(1, 11)*.005)
        np.testing.assert_allclose(result.receipt['target_valid_until_s'], np.arange(2, 12)*.005)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen.json'
            result.translator.save(path)
            loaded = FiniteMotorTranslator.load(manifest(), path)
            for block in range(10):
                kw = dict(spike_dt_s=.0005, start_s=block*.005,
                          end_s=(block+1)*.005, connected=bool(cuts[block]))
                data = spikes[block*10:(block+1)*10]
                np.testing.assert_array_equal(result.translator.consume(data, **kw)[0],
                                              loaded.consume(data, **kw)[0])

    def test_command_block_validation(self):
        spikes, _ = samples()
        for ticks in (True, 10., 0, 2, 11):
            with self.assertRaises(ValueError):
                recorded_features(manifest(), spikes, command_ticks=ticks)
        with self.assertRaises(ValueError):
            recorded_features(manifest(), spikes[:-1], command_ticks=10)
        with self.assertRaises(ValueError):
            recorded_features(manifest(), spikes, np.ones(100), command_ticks=10)

    def test_5ms_cli_requires_alignment_and_matches_fit(self):
        from train_motor_translation import main
        spikes, _ = samples()
        _, features = recorded_features(manifest(), spikes, command_ticks=10)
        pooled = np.clip(features.sum(axis=2)*.04, 0, 1)
        targets = np.column_stack([pooled[:, 0], pooled[:, 0], pooled[:, 1], np.zeros(10)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp, dp, output = root/'manifest.json', root/'ticks.npz', root/'frozen.json'
            mp.write_text(json.dumps(manifest()))
            metadata = dict(target_kind='INTERFACE_TEST', command_ticks=10)
            np.savez(dp, spikes=spikes, targets=targets, metadata=json.dumps(metadata))
            args = ['--manifest', str(mp), '--dataset', str(dp), '--output', str(output), '--steps', '2']
            with self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(output.exists())
            metadata['target_alignment'] = 'next_interval_after_completed_source_block'
            np.savez(dp, spikes=spikes, targets=targets, metadata=json.dumps(metadata),
                     source_start_s=np.arange(10)*.005, source_end_s=np.arange(1, 11)*.005,
                     target_start_s=np.arange(1, 11)*.005, target_end_s=np.arange(2, 12)*.005)
            self.assertEqual(main(args), 0)
            expected = fit(manifest(), spikes, targets, steps=2, command_ticks=10)
            np.testing.assert_array_equal(FiniteMotorTranslator.load(manifest(), output).weights,
                                          expected.translator.weights)

    def test_validation(self):
        spikes, targets = samples()
        with self.assertRaises(ValueError):
            fit(manifest(), spikes, targets.astype(complex)+1j, steps=1)
        for value in (np.nan, np.inf, -.1, 1.1):
            invalid = targets.copy()
            invalid[0, 0] = value
            with self.assertRaises(ValueError):
                fit(manifest(), spikes, invalid, steps=1)
        for column in (1, 3):
            invalid = targets.copy()
            invalid[0, column] = .7
            with self.assertRaises(ValueError):
                fit(manifest(), spikes, invalid, steps=1)
        for steps in (0, 101, True, 1.5):
            with self.assertRaises(ValueError):
                fit(manifest(), spikes, targets, steps=steps)
        for lr in (0, -1, np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                fit(manifest(), spikes, targets, steps=1, lr=lr)
        for connected in (np.ones(2), np.full(len(spikes), np.nan), np.full(len(spikes), 2)):
            with self.assertRaises(ValueError):
                fit(manifest(), spikes, targets, connected, steps=1)
        for invalid in (np.empty((0, 4)), np.full((2, 4), np.nan), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                fit(manifest(), invalid, targets, steps=1)

    def test_closure_factory_loads_frozen_and_rejects_mapping_drift(self):
        from run_neural_muscle_closure import make_motor_decoder
        from engine.embodied.motor_output import MotorOutput
        self.assertIsInstance(make_motor_decoder(manifest()), MotorOutput)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frozen.json'
            FiniteMotorTranslator(manifest(), np.full((2, 5), .07)).save(path)
            loaded = make_motor_decoder(manifest(), path)
            self.assertIsInstance(loaded, FiniteMotorTranslator)
            np.testing.assert_array_equal(loaded.weights, .07)
            drift = manifest()
            drift['rows'][2]['local_indices'] = [2]
            with self.assertRaises(ValueError):
                make_motor_decoder(drift, path)

    def test_cli_accepts_existing_result_manifest_envelope(self):
        from train_motor_translation import main
        spikes, targets = samples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp, dp, output = root/'manifest.json', root/'ticks.npz', root/'frozen.json'
            mp.write_text(json.dumps({'measured': manifest(), 'status': 'PASS'}))
            np.savez(dp, spikes=spikes, targets=targets,
                     metadata=json.dumps({'target_kind': 'INTERFACE_TEST'}))
            self.assertEqual(main(['--manifest', str(mp), '--dataset', str(dp),
                                   '--output', str(output), '--steps', '1']), 0)
            FiniteMotorTranslator.load(manifest(), output)

    def test_cli_export_load_receipt_and_no_overwrite(self):
        spikes, targets = samples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp, dp, output = root/'manifest.json', root/'ticks.npz', root/'frozen.json'
            mp.write_text(json.dumps(manifest()))
            np.savez(dp, spikes=spikes, targets=targets,
                     metadata=json.dumps({'target_kind': 'INTERFACE_TEST', 'source': 'unit synthetic'}))
            command = [sys.executable, '-B', str(Path(__file__).with_name('train_motor_translation.py')),
                       '--manifest', str(mp), '--dataset', str(dp), '--output', str(output),
                       '--steps', '4', '--learning-rate', '.1']
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            completed = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            loaded = FiniteMotorTranslator.load(manifest(), output)
            expected = fit(manifest(), spikes, targets, steps=4, lr=.1)
            np.testing.assert_array_equal(loaded.weights, expected.translator.weights)
            receipt = json.loads(Path(str(output)+'.training_receipt.json').read_text())
            self.assertEqual(receipt['target_kind'], 'INTERFACE_TEST')
            self.assertEqual(receipt['mapping_digest'], loaded.mapping_digest)
            self.assertEqual(receipt['behavior_claim'], 'NONE')
            retry = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn('already exists', retry.stderr)
            np.savez(dp, spikes=spikes, targets=targets,
                     metadata=json.dumps({'target_kind': 'behavior'}))
            command[command.index(str(output))] = str(root/'other.json')
            rejected = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('INTERFACE_TEST', rejected.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
