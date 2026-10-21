"""Data-free pipeline contracts; these are software tests, not biology validation."""
import unittest

import numpy as np

from engine.layers import ConstantSource, Integrator, Layer, Pipeline, WiringError
from engine.units import Quantity


class FunctionLayer(Layer):
    def __init__(self, name, requires, provides, function):
        super().__init__()
        self.name = name
        self.requires = dict(requires)
        self.provides = dict(provides)
        self.function = function

    def step(self, dt, inputs):
        return self.function(inputs)


class PipelineContractTests(unittest.TestCase):
    def pipeline(self):
        source = ConstantSource('x', '1', 2.0)
        sink = Integrator('sink', 'x', '1', 'y', '1')
        return Pipeline().add(source).add(sink), source, sink

    def test_undeclared_output_cannot_overwrite_upstream(self):
        observed = []
        bad = FunctionLayer('bad', {'x': '1'}, {'y': '1'},
                            lambda inputs: {'x': 99, 'y': 3})
        pipeline = Pipeline().add(ConstantSource('x', '1', 2)).add(bad)
        pipeline.add(FunctionLayer('observer', {'x': '1'}, {},
                                   lambda inputs: observed.append(inputs['x'])))
        with self.assertRaisesRegex(WiringError, 'bad.*undeclared.*x'):
            pipeline.run(Quantity(1, 's'))
        self.assertEqual(observed, [])
        self.assertEqual(bad.n_steps, 0)

    def test_every_substep_output_is_validated(self):
        calls = []

        def output(inputs):
            calls.append(1)
            return {'rogue': 8} if len(calls) == 1 else {'y': 3}

        bad = FunctionLayer('bad', {}, {'y': '1'}, output)
        bad.target_dt = Quantity(0.5, 's')
        with self.assertRaises(WiringError):
            Pipeline().add(bad).run(Quantity(1, 's'))
        self.assertEqual(len(calls), 1)

    def test_nonmapping_outputs_rejected_even_when_falsy(self):
        for output in ([], [1], 0, False, np.array([1, 2])):
            with self.subTest(output=repr(output)):
                layer = FunctionLayer('bad', {}, {}, lambda inputs: output)
                with self.assertRaisesRegex(WiringError, 'mapping'):
                    Pipeline().add(layer).run(Quantity(1, 's'))

    def test_none_and_partial_outputs_keep_legacy_behavior(self):
        for output in (None, {}, {'x': 2.0}):
            with self.subTest(output=output):
                layer = FunctionLayer('optional', {}, {'x': 'mmHg'},
                                      lambda inputs: output)
                state, _ = Pipeline().add(layer).run(
                    Quantity(1, 's'), state={'x': 1.0})
                self.assertEqual(state['x'], 2.0 if output else 1.0)
                self.assertIsInstance(state['x'], float)

    def test_changed_enabled_requires_provides_and_units_revalidate(self):
        for change in ('enabled', 'requires', 'provides', 'units', 'strict_units'):
            with self.subTest(change=change):
                pipeline, source, sink = self.pipeline()
                if change == 'strict_units':
                    pipeline.strict_units = False
                    sink.requires['x'] = 's'
                pipeline.wire()
                if change == 'enabled':
                    source.enabled = False
                elif change == 'requires':
                    sink.requires['missing'] = '1'
                elif change == 'provides':
                    source.provides.clear()
                elif change == 'units':
                    source.provides['x'] = 's'
                else:
                    pipeline.strict_units = True
                with self.assertRaises(WiringError):
                    pipeline.run(Quantity(1, 's'), state={'x': 77})
                self.assertEqual(source.n_steps, 0)
                self.assertEqual(sink.state, {})
                self.assertFalse(pipeline.wired)
                self.assertEqual(pipeline.ports(), {})

    def test_valid_contract_mutation_refreshes_provider_table(self):
        source = ConstantSource('x', 'mmHg', 20.0)
        pipeline = Pipeline().add(source).wire()
        source.provides = {'pressure': 'mmHg'}
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertEqual(state, {'pressure': 20.0})
        self.assertEqual(set(pipeline.ports()), {'pressure'})
        self.assertIsInstance(state['pressure'], float)

    def test_failed_explicit_rewire_clears_wired_and_partial_providers(self):
        pipeline, source, sink = self.pipeline()
        pipeline.wire()
        sink.requires['missing'] = '1'
        with self.assertRaises(WiringError):
            pipeline.wire()
        self.assertFalse(pipeline.wired)
        self.assertEqual(pipeline.ports(), {})

    def test_duplicate_mutated_layer_names_rejected(self):
        pipeline, source, sink = self.pipeline()
        pipeline.wire()
        sink.name = source.name
        with self.assertRaisesRegex(WiringError, 'duplicate layer name'):
            pipeline.run(Quantity(1, 's'))

    def test_initialize_and_step_cannot_mutate_shared_arrays(self):
        original = np.array([2.0, 4.0])
        source = ConstantSource('x', '1', original)
        seen = []

        def mutate(inputs):
            seen.append(inputs['x'].copy())
            inputs['x'][:] = -1
            return {'y': inputs['x']}

        consumer = FunctionLayer('mutator', {'x': '1'}, {'y': '1'}, mutate)
        consumer.target_dt = Quantity(0.5, 's')
        consumer.initialize = lambda inputs: inputs['x'].fill(99)
        pipeline = Pipeline().add(source).add(consumer)
        state, _ = pipeline.run(Quantity(1, 's'), state={'x': original})
        np.testing.assert_array_equal(original, [2, 4])
        np.testing.assert_array_equal(state['x'], [2, 4])
        np.testing.assert_array_equal(state['y'], [-1, -1])
        for value in seen:
            np.testing.assert_array_equal(value, [2, 4])

    def test_nested_input_arrays_are_isolated_and_ports_are_filtered(self):
        original = {'values': [np.array([3.0])]}
        seen = []

        def mutate(inputs):
            seen.append(set(inputs))
            inputs['x']['values'][0][0] = 9
            return {}

        layer = FunctionLayer('mutator', {'x': '1'}, {}, mutate)
        pipeline = Pipeline().add(ConstantSource('x', '1', original)).add(layer)
        pipeline.run(Quantity(1, 's'), state={'secret': 42})
        self.assertEqual(seen, [{'x'}])
        np.testing.assert_array_equal(original['values'][0], [3])

    def test_same_unit_raw_arrays_are_not_coerced_or_rescaled(self):
        source = ConstantSource('pressure', 'mmHg', np.array([20.0, 40.0]))
        sink = Integrator('sum', 'pressure', 'mmHg', 'integral', '1')
        state, _ = Pipeline().add(source).add(sink).run(Quantity(1, 's'))
        self.assertIsInstance(state['pressure'], np.ndarray)
        np.testing.assert_array_equal(state['pressure'], [20, 40])
        self.assertEqual(state['integral'], 30.0)


if __name__ == '__main__':
    unittest.main()
