"""Data-free regressions for pipeline clocks, not biological validation."""
import unittest

from engine.layers import ConstantSource, Integrator, Pipeline
from engine.units import Quantity, UnitError


class PipelineTimeTests(unittest.TestCase):
    def pipeline(self):
        source = ConstantSource('x', '1', 2)
        consumer = Integrator('consumer', 'x', '1', 'y', '1')
        return Pipeline().add(source).add(consumer), source, consumer

    def test_invalid_duration_rejected_before_step(self):
        for value in [-1, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value):
                pipeline, source, consumer = self.pipeline()
                with self.assertRaises(ValueError):
                    pipeline.run(Quantity(value, 's'))
                self.assertEqual(source.n_steps, 0)
                self.assertEqual(consumer.state, {})

    def test_wrong_dimensions_rejected(self):
        for parameter in ['duration', 'dt_outer', 'target_dt']:
            with self.subTest(parameter=parameter):
                pipeline, source, _ = self.pipeline()
                kwargs = {'duration': Quantity(1, 's')}
                if parameter == 'target_dt':
                    source.target_dt = Quantity(1, 'm')
                else:
                    kwargs[parameter] = Quantity(1, 'm')
                with self.assertRaises(UnitError):
                    pipeline.run(**kwargs)
                self.assertEqual(source.n_steps, 0)

    def test_invalid_steps_rejected_before_initialize(self):
        for parameter in ['dt_outer', 'target_dt']:
            for value in [0, -1, float('nan'), float('inf')]:
                with self.subTest(parameter=parameter, value=value):
                    pipeline, source, consumer = self.pipeline()
                    kwargs = {}
                    if parameter == 'target_dt':
                        consumer.target_dt = Quantity(value, 's')
                    else:
                        kwargs[parameter] = Quantity(value, 's')
                    with self.assertRaises(ValueError):
                        pipeline.run(Quantity(1, 's'), **kwargs)
                    self.assertEqual(source.n_steps, 0)
                    self.assertEqual(consumer.state, {})

    def test_zero_duration_is_noop(self):
        pipeline, source, consumer = self.pipeline()
        original = {'existing': 3}
        state, trace = pipeline.run(Quantity(0, 's'), state=original)
        self.assertEqual(state, original)
        self.assertIsNot(state, original)
        self.assertEqual(trace, [])
        self.assertEqual(source.n_steps, 0)
        self.assertEqual(consumer.state, {})

    def test_millisecond_clock_and_partial_final_interval(self):
        pipeline, _, consumer = self.pipeline()
        consumer.target_dt = Quantity(200, 'ms')
        state, trace = pipeline.run(Quantity(1500, 'ms'), Quantity(400, 'ms'))
        self.assertAlmostEqual(state['y'], 3)
        self.assertEqual(len(trace), 4)
        self.assertEqual(consumer.n_steps, 8)

    def test_bare_number_is_not_a_clock(self):
        pipeline, _, _ = self.pipeline()
        with self.assertRaises(TypeError):
            pipeline.run(1)


if __name__ == '__main__':
    unittest.main()
