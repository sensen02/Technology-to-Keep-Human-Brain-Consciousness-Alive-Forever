"""Data-free software unit-boundary tests, not biological validation."""
import unittest

import numpy as np

from engine.layers import ConstantSource, Integrator, Layer, Pipeline, WiringError
from engine.units import Quantity, UnitError


class Reader(Layer):
    name = 'reader'
    provides = {}

    def __init__(self, unit='Pa'):
        super().__init__()
        self.requires = {'pressure': unit}
        self.initial = []
        self.values = []

    def initialize(self, inputs):
        self.initial.append(inputs['pressure'])

    def step(self, dt, inputs):
        self.values.append(inputs['pressure'])
        return {}


class PipelineUnitTests(unittest.TestCase):
    def build(self, value, provider='mmHg', consumer='Pa', strict=True):
        source = ConstantSource('pressure', provider, value)
        reader = Reader(consumer)
        return Pipeline(strict_units=strict).add(source).add(reader), reader

    def test_scalar_conversion_initialize_step_and_provider_state(self):
        pipeline, reader = self.build(20)
        state, trace = pipeline.run(Quantity(1, 's'), state={'pressure': 10})
        self.assertAlmostEqual(reader.initial[0], 1333.22)
        self.assertAlmostEqual(reader.values[0], 2666.44)
        self.assertEqual(state['pressure'], 20)
        self.assertEqual(trace[0]['pressure'], 20)

    def test_actual_numeric_consumer_works(self):
        pipeline = Pipeline().add(ConstantSource('pressure', 'mmHg', 20))
        pipeline.add(Integrator('sink', 'pressure', 'Pa', 'integral', '1'))
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertAlmostEqual(state['integral'], 2666.44)

    def test_arrays_lists_and_scalar_numpy_inputs(self):
        for value in (np.array([20, 40]), [20, 40], np.float64(20)):
            with self.subTest(value=value):
                pipeline, reader = self.build(value)
                state, _ = pipeline.run(Quantity(1, 's'), state={'pressure': value})
                np.testing.assert_allclose(reader.initial[0], np.asarray(value) * 133.322)
                np.testing.assert_allclose(reader.values[0], np.asarray(value) * 133.322)
                self.assertIs(state['pressure'], value)

    def test_converted_arrays_are_private_across_substeps(self):
        original = np.array([20., 40.])
        pipeline, reader = self.build(original)
        reader.target_dt = Quantity(.5, 's')
        seen = []

        def mutate(dt, inputs):
            seen.append(inputs['pressure'].copy())
            inputs['pressure'][:] = -1
            return {}

        reader.step = mutate
        reader.initialize = lambda inputs: inputs['pressure'].fill(-2)
        state, _ = pipeline.run(Quantity(1, 's'), state={'pressure': original})
        for value in seen:
            np.testing.assert_allclose(value, [2666.44, 5332.88])
        np.testing.assert_array_equal(original, [20, 40])
        np.testing.assert_array_equal(state['pressure'], original)

    def test_factor_one_preserves_raw_type_and_nested_payload(self):
        for value in (20, [20, 40], np.array([20, 40]), {'nested': [np.array([20])] } ):
            for consumer in ('mmHg', 'torr'):
                with self.subTest(value=value, consumer=consumer):
                    pipeline, reader = self.build(value, consumer=consumer)
                    pipeline.run(Quantity(1, 's'), state={'pressure': value})
                    self.assertIsInstance(reader.initial[0], type(value))
                    self.assertIsInstance(reader.values[0], type(value))

    def test_none_preserved_at_both_boundaries(self):
        pipeline, reader = self.build(None)
        pipeline.run(Quantity(1, 's'))
        self.assertEqual(reader.initial, [None])
        self.assertEqual(reader.values, [None])

    def test_scaled_nonnumeric_payload_rejected(self):
        for value in ({'nested': [np.array([20])]}, ['20'], [20, {}], np.array([20], dtype=object)):
            with self.subTest(value=value):
                pipeline, reader = self.build(value)
                with self.assertRaisesRegex(WiringError, 'nonnumeric.*pressure'):
                    pipeline.run(Quantity(1, 's'))
                self.assertEqual(reader.values, [])

    def test_disabled_strict_bypasses_dimensions_scale_and_quantity_normalization(self):
        for value in (20, [20], {'nested': [20]}, Quantity(20, 's')):
            with self.subTest(value=value):
                pipeline, reader = self.build(value, consumer='Pa', strict=False)
                state, _ = pipeline.run(Quantity(1, 's'), state={'pressure': value})
                self.assertIsInstance(reader.initial[0], type(value))
                self.assertIsInstance(reader.values[0], type(value))
                self.assertIs(state['pressure'], value)

    def test_quantity_output_normalized_before_numeric_consumer(self):
        pipeline = Pipeline().add(ConstantSource('pressure', 'mmHg', Quantity(2.66644, 'kPa')))
        pipeline.add(Integrator('sink', 'pressure', 'Pa', 'integral', '1'))
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertIsInstance(state['pressure'], float)
        self.assertAlmostEqual(state['pressure'], 20)
        self.assertAlmostEqual(state['integral'], 2666.44)

    def test_quantity_seed_normalized_at_initialize(self):
        pipeline, reader = self.build(20)
        pipeline.run(Quantity(1, 's'), state={'pressure': Quantity(2.66644, 'kPa')})
        self.assertAlmostEqual(reader.initial[0], 2666.44)

    def test_wrong_quantity_dimensions_rejected_before_publication(self):
        pipeline, reader = self.build(Quantity(20, 's'))
        published = {'pressure': 7}
        with self.assertRaises(UnitError):
            pipeline.run(Quantity(1, 's'), state=published)
        self.assertEqual(published, {'pressure': 7})
        self.assertEqual(reader.values, [])
        self.assertTrue(pipeline.failed)

    def test_isolated_root_unknown_units_rejected_preflight(self):
        pipeline = Pipeline().add(ConstantSource('pressure', 'made_up', 20))
        with self.assertRaisesRegex(WiringError, 'invalid unit'):
            pipeline.run(Quantity(1, 's'))
        self.assertFalse(pipeline.failed)

    def test_dimensionless_empty_unit_and_percent_scale(self):
        pipeline, reader = self.build(20, provider='%', consumer='')
        pipeline.run(Quantity(1, 's'), state={'pressure': 10})
        self.assertAlmostEqual(reader.initial[0], .1)
        self.assertAlmostEqual(reader.values[0], .2)
        pipeline, _ = self.build(20, provider='mmHg', consumer='')
        with self.assertRaises(WiringError):
            pipeline.run(Quantity(1, 's'))

    def test_quantity_dimensions_validated_on_every_substep(self):
        source = ConstantSource('pressure', 'mmHg', 20)
        source.target_dt = Quantity(.5, 's')
        calls = []

        def step(dt, inputs):
            calls.append(1)
            return {'pressure': Quantity(1, 's') if len(calls) == 1 else 20}

        source.step = step
        pipeline = Pipeline().add(source)
        with self.assertRaises(UnitError):
            pipeline.run(Quantity(1, 's'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(source.n_steps, 0)

    def test_isolated_quantity_output_normalized_or_rejected(self):
        pipeline = Pipeline().add(ConstantSource('pressure', 'mmHg', Quantity(2666.44, 'Pa')))
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertAlmostEqual(state['pressure'], 20)
        pipeline = Pipeline().add(ConstantSource('pressure', 'mmHg', Quantity(1, 's')))
        with self.assertRaises(UnitError):
            pipeline.run(Quantity(1, 's'))

    def test_nonfinite_or_underflow_unit_scale_and_factors_are_preflight_errors(self):
        for provider, consumer in (('km^400', 'm^400'), ('nm^400', 'm^400'),
                                   ('km^30', 'nm^30'), ('nm^30', 'km^30')):
            with self.subTest(provider=provider, consumer=consumer):
                pipeline, _ = self.build(20, provider=provider, consumer=consumer)
                with self.assertRaises(WiringError):
                    pipeline.run(Quantity(1, 's'))
                self.assertFalse(pipeline.failed)

    def test_strict_and_unit_contracts_snapshotted_during_hooks(self):
        pipeline, reader = self.build(Quantity(20, 'mmHg'))

        def initialize(inputs):
            pipeline.strict_units = False
            reader.requires['pressure'] = 's'
            pipeline.layers[0].provides['pressure'] = 's'

        reader.initialize = initialize
        state, _ = pipeline.run(Quantity(1, 's'))
        self.assertAlmostEqual(state['pressure'], 20)
        self.assertAlmostEqual(reader.values[0], 2666.44)


if __name__ == '__main__':
    unittest.main()
