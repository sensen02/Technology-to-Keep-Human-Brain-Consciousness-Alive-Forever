import unittest
from engine.embodied.closure_clock import grid_steps,push_steps
class ClockTests(unittest.TestCase):
    def test_duration_rejects_silent_rounding(self):
        for value in [.012,.002,float('nan'),float('inf'),0,-.005]:
            with self.assertRaises(ValueError):grid_steps(value,.005)
        self.assertEqual(grid_steps(.015,.005),3)
    def test_push_body_grid_not_command_grid(self):
        for start,end in [(.001,.002),(.001,.006),(.015,.020)]:
            window=push_steps(dict(start_s=start,end_s=end,force_model=[1,0,0]),.05,.0001)
            steps=[i for i in range(500) if window[0]<=i<window[1]]
            self.assertEqual(len(steps),round((end-start)/.0001))
            self.assertAlmostEqual(steps[0]*.0001,start)
            self.assertAlmostEqual((steps[-1]+1)*.0001,end)
    def test_bad_push_rejected(self):
        for start,end in [(.00015,.001),(.002,.001),(.049,.051)]:
            with self.assertRaises(ValueError):push_steps(dict(start_s=start,end_s=end,force_model=[1,0,0]),.05,.0001)
if __name__=='__main__':unittest.main(verbosity=2)
