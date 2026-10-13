"""Three lightweight tests: no MuJoCo compilation or episode."""
import unittest
from run_timestep_constraint_audit import effective_timeconst, static_xml, BASE, HERE, DTS

class ConstraintAuditTests(unittest.TestCase):
    def test_default_timeconstant_is_not_clamped(self):
        self.assertEqual([effective_timeconst(.02, dt) for dt in DTS], [.02]*3)
    def test_small_positive_and_direct_format(self):
        self.assertEqual([effective_timeconst(.0001, dt) for dt in DTS], [.0002,.0001,.0001])
        self.assertEqual(effective_timeconst(-100, .0001), -100)
        self.assertEqual(effective_timeconst(.0001, .0001, False), .0001)
    def test_static_sources_have_no_hidden_solver_override(self):
        for p in (BASE/'fruitfly_zero_floor.xml', HERE/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'):
            s=static_xml(p)
            self.assertEqual(s['explicit_solver_attributes'], [])
            self.assertEqual(s['includes'], [])
            self.assertEqual(s['flags'], [])
            self.assertEqual(float(s['option']['timestep']), .0001)

if __name__ == '__main__':
    unittest.main()
