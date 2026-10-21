"""Scheduling arithmetic only; does not certify an actual coupled simulation."""
import unittest
from engine.embodied.rk4_closure_contract import RK4ClosureClock

class ClockContractTests(unittest.TestCase):
    def test_declared_ratios(self):
        c=RK4ClosureClock()
        self.assertEqual(c.body_per_command,200)
        self.assertEqual(c.neural_per_command,10)
        self.assertEqual(round(c.neural_dt_s/c.body_dt_s),20)
    def test_boundary_k4_keeps_left_command(self):
        c=RK4ClosureClock()
        self.assertEqual(c.command_row(199),0)
        self.assertAlmostEqual(c.stage_times(199)[-1],.005)
        self.assertEqual(c.command_row(200),1)
    def test_complete_source_schedule(self):
        c=RK4ClosureClock()
        for k in range(10):
            self.assertTrue(c.validate_source(k,k-1,k*.005))
    def test_reject_future_stale_and_early(self):
        c=RK4ClosureClock()
        for args in ((1,1,.005),(2,0,.010),(1,0,0.),(0,-1,.005)):
            with self.assertRaises(ValueError):c.validate_source(*args)
    def test_refined_ratio(self):
        self.assertEqual(RK4ClosureClock(body_dt_s=.0000125).body_per_command,400)
    def test_bad_grid(self):
        for kwargs in ({'body_dt_s':.00003},{'neural_dt_s':.0007},{'body_dt_s':0},{'body_dt_s':float('nan')}):
            with self.assertRaises(ValueError):RK4ClosureClock(**kwargs)
    def test_invalid_indices(self):
        c=RK4ClosureClock()
        for x in (True,-1,1.5):
            with self.assertRaises(ValueError):c.command_row(x)
        with self.assertRaises(ValueError):c.validate_source(True,0,.005)
if __name__=='__main__':unittest.main()
