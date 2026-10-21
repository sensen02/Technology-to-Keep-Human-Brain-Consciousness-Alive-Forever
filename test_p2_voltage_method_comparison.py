"""Scoped numerical/recording regressions, not biological acceptance."""
import unittest
from dataclasses import replace
import numpy as np
from scipy import sparse
import run_p2_voltage_method_comparison as diagnostic
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.motor_output import MotorOutputConfig


class VoltageMethodComparisonTests(unittest.TestCase):
    def fixture(self):
        built = dict(n=2, edges=dict(We=sparse.csr_matrix((2, 2)), Wi=sparse.csr_matrix((2, 2))))
        manifest = dict(n_neurons=2, n_actuators=1, rows=[dict(actuator_index=0, local_indices=[0])])
        return built, manifest

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            ConductanceNetwork(1, voltage_method='unknown')
        built, manifest = self.fixture()
        with self.assertRaises(ValueError):
            diagnostic.simulate(built, np.zeros((1, 2)), manifest, MotorOutputConfig(), .5, 'unknown')

    def test_zero_synaptic_analytic(self):
        p = replace(ConductanceParams(), threshold_mV=1000.)
        for dt in (.5, .25, .125):
            for method in diagnostic.METHODS:
                net = ConductanceNetwork(1, p, dt, voltage_method=method)
                current = .01
                for _ in range(round(50/dt)):
                    self.assertFalse(net.step(current).any())
                factor = np.exp(-50*p.leak_uS/p.capacitance_nF) if method == 'frozen_exact' else (
                    1+dt*p.leak_uS/p.capacitance_nF)**(-round(50/dt))
                expected = p.rest_mV+current/p.leak_uS*(1-factor)
                self.assertAlmostEqual(net.v[0], expected, places=11)
                self.assertFalse(net.ge.any() or net.gi.any() or net.adaptation.any())

    def test_default_exact_backward_euler(self):
        default = ConductanceNetwork(2, dt_ms=.5)
        explicit = ConductanceNetwork(2, dt_ms=.5, voltage_method='backward_euler')
        for _ in range(100):
            np.testing.assert_array_equal(default.step([.03, .01]), explicit.step([.03, .01]))
            for key in ('v', 'ge', 'gi', 'adaptation', 'refractory_until_ms'):
                np.testing.assert_array_equal(getattr(default, key), getattr(explicit, key))

    def test_frozen_step_coefficients(self):
        p = replace(ConductanceParams(), threshold_mV=1000.)
        net = ConductanceNetwork(1, p, .5, voltage_method='frozen_exact')
        net.ge[:] = .03; net.gi[:] = .02
        G = p.leak_uS+.03+.02
        equilibrium = (p.leak_uS*p.rest_mV+.02*p.inh_reversal_mV+.025)/G
        expected = equilibrium+(p.rest_mV-equilibrium)*np.exp(-.5*G/p.capacitance_nF)
        net.step(.025)
        self.assertAlmostEqual(net.v[0], expected, places=12)
        self.assertAlmostEqual(net.ge[0], .03*np.exp(-.5/p.exc_tau_ms), places=15)

    def test_duplicate_sparse_counts_and_lag(self):
        built, manifest = self.fixture()
        for method in diagnostic.METHODS:
            for dt in (.5, .25, .125):
                a = diagnostic.simulate(built, np.full((10, 2), .03), manifest, MotorOutputConfig(), dt, method)
                b = diagnostic.simulate(built, np.full((10, 2), .03), manifest, MotorOutputConfig(), dt, method)
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertTrue(all(diagnostic.verify_data(a, dt, manifest, MotorOutputConfig()).values()))
                broken = {**a, 'applied_controls': a['applied_controls'].copy()}
                broken['applied_controls'][0, 0] = 1
                self.assertFalse(diagnostic.verify_data(broken, dt, manifest, MotorOutputConfig())['applied_lag_exact'])

    def test_comparison_pairs_no_tolerance(self):
        built, manifest = self.fixture()
        a = diagnostic.simulate(built, np.full((10, 2), .03), manifest, MotorOutputConfig(), .25, 'backward_euler')
        b = diagnostic.simulate(built, np.full((10, 2), .03), manifest, MotorOutputConfig(), .125, 'backward_euler')
        metrics = diagnostic.compare(a, b, np.asarray([0]))
        self.assertGreaterEqual(metrics['motor_rates_l1_hz'], metrics['motor_rates_max_abs_hz'])
        self.assertNotIn('converged', metrics)


if __name__ == '__main__':
    unittest.main()
