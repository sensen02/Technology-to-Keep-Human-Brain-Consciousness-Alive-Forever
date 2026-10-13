import unittest
from engine.embodied.feco_translation import assess_feco_channel
class FeCOTests(unittest.TestCase):
    def test_subtype_not_direction(self):
        for subtype in ('claw','hook'):
            r=assess_feco_channel({'subtype':subtype,'root_id_verified':True},.005)
            self.assertIn('direction_verified',r['missing'])
            self.assertEqual(r['status'],'UNREADY_NO_CURRENT_AUTHORIZED')
    def test_vibration_aliasing(self):
        r=assess_feco_channel({'subtype':'club','max_frequency_hz':2000},.005)
        self.assertIn('sampling_rate_above_nyquist_for_declared_band',r['missing'])
    def test_no_broad_name_guess(self):
        with self.assertRaises(ValueError):assess_feco_channel({'subtype':'claw_tpGRN'},.005)
    def test_fast_sampling_not_enough(self):
        r=assess_feco_channel({'subtype':'club','max_frequency_hz':2000},.0001)
        self.assertNotIn('sampling_rate_above_nyquist_for_declared_band',r['missing'])
        self.assertIn('frequency_tuning_verified',r['missing'])
if __name__=='__main__':unittest.main()
