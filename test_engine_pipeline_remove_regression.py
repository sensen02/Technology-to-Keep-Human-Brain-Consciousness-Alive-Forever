"""Independent, data-free regression for stale Pipeline wiring after removal."""
import unittest

from engine.layers import ConstantSource, Integrator, Pipeline, WiringError
from engine.units import Quantity


class PipelineRemovalRegression(unittest.TestCase):
    def test_removing_wired_provider_cannot_silently_run_dependent_layer(self):
        pipeline = Pipeline().add(ConstantSource('x', '1', 2)).add(
            Integrator('sink', 'x', '1', 'y', '1'))
        pipeline.wire()
        pipeline.remove('constant_source')
        with self.assertRaises(WiringError):
            pipeline.run(Quantity(1, 's'))


if __name__ == '__main__':
    unittest.main()
