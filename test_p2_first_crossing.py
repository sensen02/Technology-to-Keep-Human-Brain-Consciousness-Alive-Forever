import math
import unittest
from tools_p2_first_crossing import analyze, first_crossing


class CrossingTests(unittest.TestCase):
    def test_exact_frozen_crossing_is_first_end_grid_after_analytic(self):
        result = analyze()
        analytic = result['analytic_first_crossing_ms']
        for row in result['rows']:
            if row['method'] == 'frozen_exact':
                self.assertAlmostEqual(row['first_spike_ms'], math.ceil(analytic/row['dt_ms'])*row['dt_ms'])
                self.assertGreaterEqual(row['analytic_error_ms'], 0)
                self.assertLess(row['analytic_error_ms'], row['dt_ms'])

    def test_no_crossing_below_threshold_equilibrium(self):
        self.assertIsNone(first_crossing(.001, .5, 'backward_euler'))
        self.assertIsNone(analyze(.001)['analytic_first_crossing_ms'])

    def test_backward_euler_fine_error_reduces(self):
        rows = [r for r in analyze()['rows'] if r['method'] == 'backward_euler']
        self.assertGreater(rows[0]['analytic_error_ms'], rows[-1]['analytic_error_ms'])


if __name__ == '__main__':
    unittest.main()
