"""Numerical method tests; not biological validation."""
import unittest
import numpy as np
from dataclasses import replace
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.adapters import TierConfig

class VoltageTests(unittest.TestCase):
    def test_exact_frozen(self):
        p=replace(ConductanceParams(),threshold_mV=1000.)
        for dt in (.5,.25,.1):
            n=ConductanceNetwork(1,p,dt,voltage_method='frozen_exact')
            n.ge[:]=.03;n.gi[:]=.02
            G=.051;eq=(.001*-65+.02*-70+.025)/G
            expected=eq+(-65-eq)*np.exp(-G*dt/.02)
            n.step(.025)
            self.assertAlmostEqual(float(n.v[0]),expected,places=12)
    def test_default_unchanged(self):
        a=ConductanceNetwork(2,dt_ms=.5)
        b=ConductanceNetwork(2,dt_ms=.5,voltage_method='backward_euler')
        for k in range(100):
            np.testing.assert_array_equal(a.step(.05),b.step(.05))
            np.testing.assert_array_equal(a.v,b.v)
    def test_be_formula(self):
        n=ConductanceNetwork(1,replace(ConductanceParams(),threshold_mV=1000.),.5)
        n.ge[:]=.03;n.gi[:]=.02
        expected=(.02/.5*-65+.001*-65+.02*-70+.025)/(.02/.5+.051)
        n.step(.025);self.assertEqual(float(n.v[0]),expected)
    def test_invalid(self):
        with self.assertRaises(ValueError):ConductanceNetwork(1,voltage_method='magic')
        with self.assertRaises(ValueError):TierConfig(voltage_method='magic').validate()
    def test_ownership_zero(self):
        n=ConductanceNetwork(2,voltage_method='frozen_exact')
        for _ in range(100):
            self.assertFalse(n.step([0,1],enabled=np.array([True,False])).any())
        np.testing.assert_array_equal(n.v,[-65,-65])
    def test_refractory(self):
        n=ConductanceNetwork(1,dt_ms=.1,voltage_method='frozen_exact');times=[]
        for _ in range(500):
            if n.step(1)[0]:times.append(n.t_ms)
        self.assertGreater(len(times),2)
        self.assertTrue(np.all(np.diff(times)>=2.-1e-10))

if __name__=='__main__':unittest.main()
