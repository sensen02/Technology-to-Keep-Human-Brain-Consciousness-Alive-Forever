"""Fail-stop software lifecycle tests; no transactional rollback is claimed."""
import unittest

from engine.layers import ConstantSource, Integrator, Layer, Pipeline, WiringError
from engine.units import Quantity


class Stateful(Layer):
    name = 'stateful'
    provides = {'x': '1'}

    def initialize(self, inputs):
        self.state['initializations'] = self.state.get('initializations', 0) + 1

    def step(self, dt, inputs):
        self.state['advances'] = self.state.get('advances', 0) + 1
        return {'x': self.state['advances']}


class PipelineFailureTests(unittest.TestCase):
    def assert_rerun_denied(self, pipeline, layer):
        before = dict(layer.state)
        count = layer.n_steps
        reason = pipeline.failure_reason
        self.assertTrue(pipeline.failed)
        self.assertIsInstance(reason, str)
        # Deliberately invalid declarations/clocks: denial must precede even wire.
        layer.provides = {'x': 'invalid'}
        with self.assertRaisesRegex(RuntimeError, 'NEW Pipeline.*NEW layer instances'):
            pipeline.run(Quantity(-1, 's'))
        with self.assertRaises(RuntimeError):
            pipeline.run(Quantity(0, 's'))
        self.assertEqual(layer.state, before)
        self.assertEqual(layer.n_steps, count)
        self.assertEqual(pipeline.failure_reason, reason)

    def test_failed_step_retains_partial_state_and_denies_retry(self):
        layer = Stateful()
        layer.target_dt = Quantity(.5, 's')
        pipeline = Pipeline().add(layer)
        step = layer.step

        def fail_second(dt, inputs):
            result = step(dt, inputs)
            if layer.state['advances'] == 2:
                raise ValueError('step broke')
            return result

        layer.step = fail_second
        with self.assertRaisesRegex(ValueError, 'step broke'):
            pipeline.run(Quantity(1, 's'))
        self.assertEqual(layer.state['advances'], 2)
        self.assertEqual(layer.n_steps, 1)
        self.assertIn('stateful.step: ValueError: step broke', pipeline.failure_reason)
        self.assert_rerun_denied(pipeline, layer)

    def test_failed_initialization_denies_retry(self):
        layer = Stateful()
        pipeline = Pipeline().add(layer)

        def initialize(inputs):
            layer.state['partly_initialized'] = True
            raise ValueError('init broke')

        layer.initialize = initialize
        with self.assertRaisesRegex(ValueError, 'init broke'):
            pipeline.run(Quantity(1, 's'))
        self.assertEqual(layer.n_steps, 0)
        self.assertIn('stateful.initialize', pipeline.failure_reason)
        self.assert_rerun_denied(pipeline, layer)

    def test_failed_callback_denies_retry_without_rollback(self):
        layer = Stateful()
        pipeline = Pipeline().add(layer)
        observed = []

        def callback(index, state):
            observed.append(state['x'])
            layer.state['callback_mutation'] = True
            raise ValueError('callback broke')

        with self.assertRaisesRegex(ValueError, 'callback broke'):
            pipeline.run(Quantity(1, 's'), on_step=callback)
        self.assertEqual(observed, [1])
        self.assertEqual(layer.n_steps, 1)
        self.assertIn('on_step callback', pipeline.failure_reason)
        self.assert_rerun_denied(pipeline, layer)

    def test_runtime_baseexceptions_also_fail_stop(self):
        for error in (KeyboardInterrupt, SystemExit):
            with self.subTest(error=error):
                layer = Stateful()
                pipeline = Pipeline().add(layer)

                def step(dt, inputs):
                    layer.state['touched'] = True
                    raise error('interrupted')

                layer.step = step
                with self.assertRaises(error):
                    pipeline.run(Quantity(1, 's'))
                self.assert_rerun_denied(pipeline, layer)

    def test_output_contract_failure_also_denies_retry(self):
        layer = Stateful()
        layer.step = lambda dt, inputs: {'rogue': 1}
        pipeline = Pipeline().add(layer)
        with self.assertRaises(WiringError):
            pipeline.run(Quantity(1, 's'))
        self.assert_rerun_denied(pipeline, layer)

    def test_wiring_preflight_is_recoverable(self):
        source = ConstantSource('x', 's', 2)
        sink = Integrator('sink', 'x', '1', 'y', '1')
        pipeline = Pipeline().add(source).add(sink)
        with self.assertRaises(WiringError):
            pipeline.run(Quantity(1, 's'))
        self.assertFalse(pipeline.failed)
        self.assertIsNone(pipeline.failure_reason)
        source.provides['x'] = '1'
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertEqual(state['y'], 2)

    def test_clock_preflight_is_recoverable(self):
        layer = Stateful()
        pipeline = Pipeline().add(layer)
        layer.target_dt = Quantity(0, 's')
        with self.assertRaises(ValueError):
            pipeline.run(Quantity(1, 's'))
        self.assertEqual(layer.state, {})
        self.assertFalse(pipeline.failed)
        layer.target_dt = Quantity(1, 's')
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertEqual(state['x'], 1)

    def test_zero_noop_does_not_fail_or_initialize(self):
        layer = Stateful()
        pipeline = Pipeline().add(layer)
        state, trace = pipeline.run(Quantity(0, 's'))
        self.assertEqual((state, trace), ({}, []))
        self.assertFalse(pipeline.failed)
        self.assertEqual(layer.state, {})
        pipeline.run(Quantity(1, 's'))
        self.assertEqual(layer.state['advances'], 1)

    def test_fresh_pipeline_and_fresh_layers_recover(self):
        damaged = Stateful()
        damaged.step = lambda dt, inputs: (_ for _ in ()).throw(ValueError('broken'))
        pipeline = Pipeline().add(damaged)
        with self.assertRaises(ValueError):
            pipeline.run(Quantity(1, 's'))
        fresh = Stateful()
        recovered = Pipeline().add(fresh)
        state, _ = recovered.run(Quantity(1, 's'))
        self.assertEqual(state['x'], 1)
        self.assertFalse(recovered.failed)


if __name__ == '__main__':
    unittest.main()
