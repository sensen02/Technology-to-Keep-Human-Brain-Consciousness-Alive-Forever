"""Bounded cable geometry/numerics tests; no repository outputs are written.

Run: venv/bin/python run_cable_geometry_selftest.py
Independent references: quadrature of tapered resistance, dense linear solve,
and matrix exponential of the continuous passive system (small fixtures only).
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from engine.cable import Morphology, CableNeuron, Synapse, swc_to_morphology


class CableGeometryTests(unittest.TestCase):
    def swc(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.swc"
            path.write_text(text)
            return swc_to_morphology(path, **kwargs)

    def test_swc_units_mapping_and_reduction(self):
        text = '# Meta: {"units": "1 nanometer"}\n30 3 2000 0 0 500 20\n10 1 0 0 0 1000 -1\n20 3 1000 0 0 750 10\n'
        m, ids = self.swc(text)
        self.assertEqual(ids, [30, 10, 20])
        np.testing.assert_array_equal(m.parent, [2, -1, 1])
        np.testing.assert_allclose(m.x, [2, 0, 1])
        np.testing.assert_allclose(m.d, [1, 2, 1.5])
        with self.assertRaises(ValueError):
            self.swc(text, units="um")
        with self.assertRaises(NotImplementedError):
            self.swc(text, keep=2)
        self.assertEqual(self.swc(text, keep=3)[1], ids)
        with self.assertRaises(ValueError):
            self.swc('1 1 0 0 0 1 -1\n')
        self.assertEqual(self.swc('1 1 0 0 0 1 -1\n', units="um")[0].n, 1)

    def test_invalid_swc(self):
        invalid = [
            '1 1 0 0 0 1 -1\n2 3 1 0 0 1 9',  # missing parent
            '1 1 0 0 0 1 -1\n1 3 1 0 0 1 1',  # duplicate
            '1 1 0 0 0 0 -1', '1 1 0 0 0 -1 -1',
            '1 1 nan 0 0 1 -1', '1 1 0 0 0 nan -1',
            '1 1 0 0 0 1 -1\n2 3 1 0 0 1 -1',  # forest
            '1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 2',
            '1 1 0 0 0 1 -1\n2 3 0 0 0 1 1', 'broken',
            '# units: 2 nanometers\n1 1 0 0 0 1 -1']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.swc(text, units="um")

    def test_morphology_and_parameter_validation(self):
        for parents in ([1, 0], [-1, 2], [-1, 0.5], [-1, -1]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                Morphology([0, 1], [0, 0], [0, 0], [1, 1], parents)
        m = Morphology.cylinder(10, 1, 2)
        for kw in ({"dt_ms": 0}, {"Ra_ohm_cm": -1}, {"Cm_uF_cm2": 0},
                   {"g_leak_S_cm2": -1}, {"E_leak_mV": np.nan}, {"siz_index": 0.5}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                CableNeuron(m, **kw)
        with self.assertRaises(NotImplementedError):
            CableNeuron(m, active=True)
        with self.assertRaises(ValueError):
            m.x[0] = 2

    def test_order_independent_branched_matrix(self):
        x, y, d, parent = np.array([0, 2, 4, 2]), np.array([0, 0, 0, 3]), np.array([2., 1.8, 1., .8]), np.array([-1, 0, 1, 1])
        a = CableNeuron(Morphology(x, y, np.zeros(4), d, parent))
        order = np.array([3, 1, 0, 2])
        inverse = np.argsort(order)
        newpar = np.array([-1 if parent[i] == -1 else inverse[parent[i]] for i in order])
        b = CableNeuron(Morphology(x[order], y[order], np.zeros(4), d[order], newpar))
        np.testing.assert_allclose(b.G.toarray(), a.G.toarray()[np.ix_(order, order)])
        current = np.array([.01, 0, -.02, .03])
        np.testing.assert_allclose(b.step(i_inject=current[order]), a.step(i_inject=current)[order])

    def test_taper_quadrature_reference(self):
        c = CableNeuron(Morphology([0, 10, 30], [0]*3, [0]*3, [4, 2, 1], [-1, 0, 1]))
        # Edge 2 connects parent center x=5 to child center x=20 through x=10.
        integrand1 = lambda x: c.Ra * 1e4 / (np.pi * (2 - x / 10) ** 2)
        integrand2 = lambda x: c.Ra * 1e4 / (np.pi * (1 - (x - 10) / 40) ** 2)
        reference = quad(integrand1, 5, 10)[0] + quad(integrand2, 10, 20)[0]
        self.assertAlmostEqual((1e6 / c.g_ax[2]) / reference, 1., places=12)
        self.assertAlmostEqual(c.area_um2[2], np.pi * 1.5 * np.hypot(20, .5))

    def test_sparse_large_tree_no_dense_allocation(self):
        m = Morphology.cylinder(10000, 1, 20000)
        original = np.zeros
        def guarded(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and min(shape) > 100:
                raise AssertionError("dense matrix allocation")
            return original(shape, *args, **kwargs)
        with patch("engine.cable.np.zeros", guarded):
            c = CableNeuron(m)
            self.assertLessEqual(c.G.nnz, 3 * m.n)
            self.assertLess(c.G.data.nbytes + c.G.indices.nbytes + c.G.indptr.nbytes, 1_000_000)
            np.testing.assert_allclose(c.step(), c.E_leak, atol=1e-6)

    def test_independent_dense_step_and_dt_invalidation(self):
        c = CableNeuron(Morphology.cylinder(30, 2, 3))
        # Independent incidence Laplacian, not the assembled sparse G.
        incidence = np.zeros((3, 4))
        for j in range(3):
            incidence[j, j], incidence[j, j+1] = -1, 1
        G = incidence.T @ np.diag(c.g_ax[1:]) @ incidence + np.diag(c.g_mem)
        np.testing.assert_allclose(c.G.toarray(), G)
        for dt in (.025, .1):
            c.dt = dt
            rhs = c.C / dt * c.v + c.g_mem * c.E_leak + .01
            expected = np.linalg.solve(np.diag(c.C / dt) + G, rhs)
            np.testing.assert_allclose(c.step(i_inject=.01), expected, rtol=1e-12)

    def test_matrix_exponential_convergence(self):
        m = Morphology.cylinder(100, 1, 4)
        errors = []
        for dt in (.04, .02, .01):
            c = CableNeuron(m, dt_ms=dt)
            initial = np.array([0., 1., -2., 3., 0.])
            c.v += initial
            exact = expm(-np.diag(1 / c.C) @ c.G.toarray() * .4) @ initial
            simulated = c.run(.4)[-1] - c.E_leak
            errors.append(np.linalg.norm(simulated - exact))
        self.assertLess(errors[1], .65 * errors[0])
        self.assertLess(errors[2], .65 * errors[1])

    def test_cut_edges_end_leak_and_conservation(self):
        c = CableNeuron(Morphology.cylinder(30, 1, 3), g_leak_S_cm2=0)
        c.v += [0, 1, 2, 3]
        charge = c.C @ c.v
        c.step()
        self.assertAlmostEqual(c.C @ c.v, charge, places=12)
        old_lu = c._lu
        c.cut_edges([2])
        self.assertIsNone(c._lu)
        self.assertEqual(c.G[1, 2], 0)
        c.v[:] = c.E_leak
        c.step(i_inject=[.1, 0, 0, 0])
        self.assertIsNot(old_lu, c._lu)
        np.testing.assert_allclose(c.v[2:], c.E_leak, atol=1e-9)
        c.cut_edges([2])  # idempotent
        c.set_end_leak([2], .1, -20)
        self.assertIsNone(c._lu)
        self.assertTrue(np.all(c.g_mem == 0))
        c.run(10)
        np.testing.assert_allclose(c.v[2:], -20, atol=1e-6)
        with self.assertRaises(ValueError):
            c.cut_edges([0])
        with self.assertRaises(ValueError):
            c.set_end_leak([2], -1, 0)

    def test_synapse_exact_decay_sign_and_validation(self):
        s = Synapse(.01, 0, 2)
        self.assertEqual(s.step(3, 0), .03)
        self.assertAlmostEqual(s.step(False, 10), .03 * np.exp(-5))
        self.assertLess(s.current(-60), 0)
        self.assertEqual(s.inward_current(-60), -s.current(-60))
        self.assertEqual(s.current(0), 0)
        self.assertGreater(s.current(20), 0)
        c = CableNeuron(Morphology.cylinder(10, 1, 2))
        self.assertTrue(np.all(c.step(i_syn=s.inward_current(c.v)) > c.E_leak))
        for kw in ({"tau_ms": 0}, {"g_max_uS": -1}, {"E_rev_mV": np.inf}):
            with self.assertRaises(ValueError):
                Synapse(**kw)
        for count, dt in ((-1, 1), (.5, 1), (1, -1), (1, np.nan)):
            with self.assertRaises(ValueError):
                s.step(count, dt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
