import unittest
import numpy as np
from tools_s2_switched_activation import control,euler,reference
from tools_s2_activation_reference import rhs

class SwitchedTests(unittest.TestCase):
    def test_event_semantics(self):
        self.assertEqual(control(.000999,'pulse'),.1);self.assertEqual(control(.001,'pulse'),0)
        self.assertEqual(control(0,'decay'),0)
    def test_decay_derivative(self):self.assertAlmostEqual(rhs(.1,0,.0001,.0004),-162.5)
    def test_alignment_rejection(self):
        for dt in (0,-1,float('nan'),.00003):
            with self.assertRaises(ValueError):euler(dt,'pulse',0)
    def test_first_step_and_reference_monotonic_decay(self):
        t,a=euler(.0001,'decay',0);self.assertAlmostEqual(a[0],.08375)
        r=reference(t,'decay',0);self.assertTrue(np.all(np.diff(r)<0));self.assertTrue(np.all(r>=0))
    def test_reference_outside_interval_rejected(self):
        with self.assertRaises(ValueError):reference(np.array([np.nextafter(.01,1.)]),'pulse',0)
    def test_pulse_first_step(self):
        t,a=euler(.0001,'pulse',0);self.assertEqual(a[0],.2)
        self.assertEqual(len(t),100)

if __name__=='__main__':unittest.main()
