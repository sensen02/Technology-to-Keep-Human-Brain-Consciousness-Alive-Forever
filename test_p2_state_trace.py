"""Independent tiny fixtures for observational P2 transition, not physiology."""
import unittest
import numpy as np
from scipy import sparse
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from run_p2_state_trace import predict, observed_step, FIELDS


class StateTraceTests(unittest.TestCase):
    def pre(self, **kw):
        values = dict(v=-65., ge=0., gi=0., adaptation=0., refractory_until_ms=0.)
        values.update(kw)
        return {k: np.asarray([v]) for k, v in values.items()}

    def test_independent_formula_arrival_then_decay(self):
        p = ConductanceParams()
        pre = self.pre(v=-62., ge=.003, gi=.002, adaptation=.001)
        out = predict(pre, np.array([.025]), np.array([.004]), np.array([.001]), 1., .5, p)
        # Scalar hand derivation: C/dt=.04; gE=.007; gI=.003; leak=.001
        expected = (.04*-62 + .001*-65 + .003*-70 + .025-.001)/(.04+.011)
        self.assertAlmostEqual(out['predicted_pre_reset_v'][0], expected, places=13)
        self.assertAlmostEqual(out['ge'][0], .007*np.exp(-.5/3), places=15)
        self.assertAlmostEqual(out['gi'][0], .003*np.exp(-.5/8), places=15)
        self.assertAlmostEqual(out['adaptation'][0], .001*np.exp(-.5/100), places=15)

    def test_threshold_reset_and_adaptation_order(self):
        out = predict(self.pre(v=-51., adaptation=.001), np.array([.2]), np.array([0.]),
                      np.array([0.]), 3., .5, ConductanceParams())
        self.assertTrue(out['predicted_spike'][0])
        self.assertGreater(out['predicted_pre_reset_v'][0], -50)
        self.assertEqual(out['v'][0], -65)
        self.assertEqual(out['refractory_until_ms'][0], 5.5)
        self.assertAlmostEqual(out['adaptation'][0], .001*np.exp(-.005)+.002, places=15)

    def test_refractory_start_boundary(self):
        pre = self.pre(v=-51., refractory_until_ms=2.)
        inactive = predict(pre, np.array([10.]), np.array([.1]), np.array([0.]), 1.5, .5, ConductanceParams())
        self.assertFalse(inactive['active'][0]); self.assertFalse(inactive['predicted_spike'][0])
        self.assertEqual(inactive['v'][0], -65)
        active = predict(pre, np.array([10.]), np.array([0.]), np.array([0.]), 2., .5, ConductanceParams())
        self.assertTrue(active['active'][0]); self.assertTrue(active['predicted_spike'][0])

    def pair(self):
        w = sparse.csr_matrix(([.003, .004], ([1, 2], [0, 1])), shape=(3, 3))
        wi = sparse.csr_matrix(([.005], ([2], [0])), shape=(3, 3))
        return [ConductanceNetwork(3, dt_ms=.5).set_connectivity(w, wi) for _ in range(2)]

    def test_ring_exact_end_to_start_delay(self):
        net = self.pair()[0]
        traces = []
        for k in range(7):
            current = np.array([1. if k == 0 else 0., 0., 0.])
            _, trace, arrived, incoming = observed_step(net, current, np.arange(3))
            traces.append((trace, arrived, incoming))
        # emission end .5 -> arrival start 2.5, NOT 2 or 3
        for k in range(5):
            self.assertEqual(len(traces[k][1]), 0)
        self.assertEqual(traces[5][0]['start_ms'], 2.5)
        np.testing.assert_array_equal(traces[5][1], [0])
        self.assertEqual(traces[5][0]['incoming_e_uS'][1], .003)
        self.assertEqual(traces[5][0]['incoming_i_uS'][2], .005)
        self.assertEqual(traces[5][2], [(1, 0, 0, .003), (2, 0, 1, .005)])

    def test_observation_invariance_entire_state_and_events(self):
        observed, plain = self.pair()
        for k in range(100):
            current = np.array([.06 if k < 60 else 0., .025, .01])
            events, trace, _, _ = observed_step(observed, current, np.arange(3))
            expected = plain.step(current_nA=current)
            np.testing.assert_array_equal(events, expected)
            self.assertTrue(trace['threshold_check'].all())
            self.assertLessEqual(trace['state_residual'].max(), 1e-12)
            for key in (*FIELDS, '_ring', 'spike_count'):
                np.testing.assert_array_equal(getattr(observed, key), getattr(plain, key))
            self.assertEqual(observed._ring_i, plain._ring_i)
            self.assertEqual(observed.t_ms, plain.t_ms)


if __name__ == '__main__':
    unittest.main()
