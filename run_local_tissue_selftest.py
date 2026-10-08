"""Numerical selftests, not physiological validation. Run: venv/bin/python run_local_tissue_selftest.py"""
import unittest

import numpy as np

from engine.local_tissue import (
    AMOUNT_NM_UM3_TO_MOL, FinitePool, LocalTissue,
    NonPhysiologicalSupportProxy, PrescribedBath,
)


class LocalTissueTests(unittest.TestCase):
    def assert_balanced(self, model, atol=1e-9):
        self.assertAlmostEqual(model.mass_balance_error(), 0., delta=atol)
        self.assertTrue(np.all(model.free_nM >= 0))
        self.assertTrue(np.all(model.bound_nM >= 0))
        self.assertTrue(np.all(model.bound_nM <= model.capacity_nM))
        self.assertTrue(np.all(model.pool_nM >= 0))

    def test_units_uniform_and_analytic_source_clearance(self):
        self.assertEqual(AMOUNT_NM_UM3_TO_MOL, 1e-24)
        m = LocalTissue((3, 2, 4), spacing_um=(2, 3, 4), initial_nM=7.)
        m.step(1000.)
        np.testing.assert_allclose(m.free_nM, 7., rtol=1e-12)
        self.assert_balanced(m)
        # Uniform field follows exactly the discrete backward-Euler recurrence.
        m = LocalTissue((2, 3, 2), spacing_um=(2, 2, 2), initial_nM=2.,
                        clearance_s=.4, source_amount_s=8.*.3)
        for _ in range(10):
            m.step(.2)
        expected = .3/.4+(2.-.3/.4)/(1.+.2*.4)**10
        np.testing.assert_allclose(m.free_nM, expected, atol=1e-12)
        self.assert_balanced(m)

    def test_diffusion_conservation_positivity_large_dt(self):
        c = np.zeros((4, 3, 2))
        c[0, 0, 0] = 20.
        m = LocalTissue(c.shape, initial_nM=c, spacing_um=(1, 2, 3))
        for dt in (.001, .2, 10., 1000.):
            m.step(dt)
            self.assert_balanced(m)
        self.assertLess(np.max(m.free_nM)-np.min(m.free_nM), .02)

    def test_two_voxel_analytic_and_timestep_convergence(self):
        errors = []
        for dt in (.2, .1, .05):
            m = LocalTissue((2, 1, 1), initial_nM=np.array([2., 0.])[:, None, None])
            for _ in range(round(1./dt)):
                m.step(dt)
            exact = 1.+np.exp(-2.)
            errors.append(abs(m.free_nM[0, 0, 0]-exact))
            discrete = 1.+(1.+2.*dt)**(-round(1./dt))
            self.assertAlmostEqual(m.free_nM[0, 0, 0], discrete, places=12)
            self.assert_balanced(m)
        self.assertLess(errors[1], errors[0]*.6)
        self.assertLess(errors[2], errors[1]*.6)

    def test_neumann_cosine_mesh_convergence(self):
        errors = []
        # Same physical domain, exact continuum no-flux cosine mode.
        # Tiny dt isolates spatial convergence (BE temporal error has opposite sign).
        for n in (6, 12, 24):
            x = (np.arange(n)+.5)/n
            initial = (1.+.2*np.cos(np.pi*x))[:, None, None]
            m = LocalTissue((n, 1, 1), spacing_um=(1./n, 1, 1), initial_nM=initial)
            dt = 1e-5
            for _ in range(100):
                m.step(dt)
            exact = 1.+.2*np.exp(-np.pi**2*.001)*np.cos(np.pi*x)
            errors.append(np.max(np.abs(m.free_nM[:, 0, 0]-exact)))
            self.assert_balanced(m)
        self.assertLess(errors[1], errors[0]*.3)
        self.assertLess(errors[2], errors[1]*.3)

    def test_finite_binding_equilibrium_and_heterogeneous_knockout(self):
        capacity = np.array([0., 1., 4.])[:, None, None]
        m = LocalTissue((3, 1, 1), diffusion_um2_s=0., initial_nM=2.,
                        receptor_capacity_nM=capacity, kon_nM_inv_s=.5,
                        koff_s=.25, response_tau_s=2.)
        for _ in range(200):
            m.step(1.)
        # Kd=.5, total ligand=2: (T-B)(R-B)=Kd*B.
        r = capacity
        b = ((2.+r+.5)-np.sqrt((2.+r+.5)**2-8.*r))/2.
        np.testing.assert_allclose(m.bound_nM, b, atol=1e-12)
        np.testing.assert_allclose(m.response, m.occupancy, atol=1e-12)
        self.assertEqual(m.response[0, 0, 0], 0.)
        self.assertEqual(m.bound_nM[0, 0, 0], 0.)
        self.assertLess(m.free_nM[2, 0, 0], m.free_nM[1, 0, 0])
        self.assert_balanced(m)

    def test_binding_depletion_release_and_extreme_step(self):
        m = LocalTissue((1, 1, 1), initial_nM=.01, receptor_capacity_nM=100.,
                        kon_nM_inv_s=100., koff_s=0.)
        m.step(1e5)
        self.assertLess(float(m.free_nM.item()), 1e-9)
        self.assertLessEqual(float(m.bound_nM.item()), .01)
        self.assert_balanced(m)
        m = LocalTissue((1, 1, 1), initial_nM=0., receptor_capacity_nM=5.,
                        initial_bound_nM=3., kon_nM_inv_s=0., koff_s=.2)
        m.step(2.)
        self.assertAlmostEqual(m.bound_nM.item(), 3./1.4, places=12)
        self.assertAlmostEqual(m.free_nM.item(), 3.-3./1.4, places=12)
        self.assert_balanced(m)

    def test_binding_timestep_convergence(self):
        def solve(dt):
            m = LocalTissue((1, 1, 1), initial_nM=2., receptor_capacity_nM=3.,
                            kon_nM_inv_s=1., koff_s=.3)
            for _ in range(round(1./dt)):
                m.step(dt)
            return m.bound_nM.item()
        reference = solve(.001)
        errors = [abs(solve(dt)-reference) for dt in (.1, .05, .025)]
        self.assertLess(errors[1], .6*errors[0])
        self.assertLess(errors[2], .6*errors[1])

    def test_zero_barrier_isolation_and_nonzero_conductance(self):
        initial = np.array([4., 0., 0., 0.])[:, None, None]
        m = LocalTissue((4, 1, 1), initial_nM=initial, barrier_edges={(1, 2): 0.})
        m.step(100.)
        np.testing.assert_array_equal(m.free_nM[2:], 0.)
        self.assert_balanced(m)
        slow = LocalTissue((2, 1, 1), initial_nM=initial[:2], barrier_edges={(0, 1): .1})
        slow.step(2.)
        self.assertAlmostEqual(slow.free_nM[0, 0, 0], 2.+2./1.4, places=12)
        self.assert_balanced(slow)

    def test_finite_pool_depletion_and_release(self):
        m = LocalTissue((1, 1, 1), receptor_capacity_nM=4.,
                        finite_pools=[FinitePool(2., 3., 1.)])
        for _ in range(100):
            m.step(1.)
        self.assertLess(m.pool_nM[0], 3.)
        self.assertGreater(m.bound_nM.item(), 0.)
        self.assertAlmostEqual(m.total_amount(), 6., places=10)
        self.assertEqual(sum(m.ledger.values()), 0.)
        self.assert_balanced(m)
        m = LocalTissue((1, 1, 1), initial_bound_nM=2., receptor_capacity_nM=2.,
                        kon_nM_inv_s=0., koff_s=1., finite_pools=[FinitePool(2., 0., 1.)])
        for _ in range(30):
            m.step(1.)
        self.assertGreater(m.pool_nM[0], .6)
        self.assertLess(m.last_pool_to_grid_amount[0], 0.)
        self.assert_balanced(m)

    def test_prescribed_reservoir_accounting_in_out(self):
        m = LocalTissue((2, 1, 1), initial_nM=1., receptor_capacity_nM=2.,
                        baths=[PrescribedBath(4., 2.), PrescribedBath(0., .3)],
                        source_amount_s=.2, clearance_s=.5)
        for _ in range(100):
            m.step(.1)
            self.assert_balanced(m)
        self.assertGreater(m.ledger['boundary_in'], 0.)
        self.assertGreater(m.ledger['boundary_out'], 0.)
        self.assertGreater(m.ledger['clearance'], 0.)
        self.assertAlmostEqual(m.ledger['source'], 4., places=12)
        # One voxel, no binding: exact discrete Robin-exchange update.
        m = LocalTissue((1, 1, 1), baths=[PrescribedBath(5., 2.)])
        m.step(3.)
        self.assertAlmostEqual(m.free_nM.item(), 30./7., places=12)
        self.assertAlmostEqual(m.ledger['boundary_in'], 30./7., places=12)
        self.assert_balanced(m)

    def test_slow_response_and_explicit_proxy(self):
        m = LocalTissue((1, 1, 1), receptor_capacity_nM=2., initial_bound_nM=1.,
                        kon_nM_inv_s=0., koff_s=0., response_tau_s=5.)
        m.step(2.)
        self.assertAlmostEqual(m.response.item(), .5*(1.-np.exp(-2./5.)), places=12)
        p = NonPhysiologicalSupportProxy((2, 1, 1), initial=.8)
        p.step(2., supply_s=.3, demand_s=.7)
        np.testing.assert_allclose(p.availability, .3+.5*np.exp(-2.))
        p.step(1e6, demand_s=1.)
        np.testing.assert_array_equal(p.availability, 0.)

    def test_invalid_inputs_and_zero_dt(self):
        for kwargs in (dict(initial_nM=-1.), dict(spacing_um=(0, 1, 1)),
                       dict(response_tau_s=0.), dict(initial_bound_nM=1.),
                       dict(diffusion_um2_s=float('nan')),
                       dict(barrier_edges={(0, 26): 0.}),
                       dict(finite_pools=[FinitePool(0., 1., 1.)])):
            with self.assertRaises(ValueError):
                LocalTissue(**kwargs)
        m = LocalTissue(initial_nM=1.)
        m.step(0.)
        self.assertEqual(m.time_s, 0.)
        with self.assertRaises(ValueError):
            m.step(-1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
