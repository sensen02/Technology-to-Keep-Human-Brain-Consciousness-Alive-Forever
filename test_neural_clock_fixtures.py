import unittest
import numpy as np
from tools_neural_clock_fixtures import DTS,constant,refractory,delay,pulse
from tools_decoder_clock_response import compare


class ClockFixtureTests(unittest.TestCase):
    def test_frozen_constant_matches_analytic(self):
        for dt in DTS:
            self.assertAlmostEqual(constant(dt,'frozen_exact')['error_mV'],0.,places=11)

    def test_backward_euler_constant_error_decreases(self):
        errors=[abs(constant(dt,'backward_euler')['error_mV']) for dt in DTS]
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))

    def test_refractory_eligible_end_sample(self):
        for dt in DTS:
            row=refractory(dt)
            self.assertAlmostEqual(row['min_isi_ms'],row['expected_min_isi_ms'],places=10)

    def test_delay_end_to_start_exact(self):
        for dt in DTS:
            row=delay(dt)
            self.assertEqual(len(row['source_emission_end_ms']),1)
            self.assertEqual(len(row['target_arrival_start_ms']),1)
            self.assertAlmostEqual(row['observed_delay_ms'],2.,places=10)

    def test_pulse_conductance_exact_decay(self):
        for kind,tau in [('exc',3.),('inh',8.)]:
            for dt in DTS:
                row=pulse(dt,'backward_euler',kind)
                np.testing.assert_allclose(row['conductance_end_uS'],.002*np.exp(-np.array(row['time_end_ms'])/tau),rtol=1e-12)
                self.assertGreater(row['left_hold_area_ratio'],1.)

    def test_fixed_timestamp_decoder_convention(self):
        errors=[]
        for dt in (.0005,.00025,.000125,.0000625):
            row=compare(dt)
            expected=(1-np.exp(-dt/.005))/(dt/.005)-1
            self.assertAlmostEqual(row['relative_rate_error'],expected,places=11)
            errors.append(abs(row['relative_rate_error']))
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))


if __name__=='__main__':unittest.main()
