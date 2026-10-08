"""Paired causal checks of hand-built IDEALIZED integration, not validation of flies.
Run: venv/bin/python -B run_isolation_selftest.py. Does not write output files.
"""
import unittest
from dataclasses import replace
import numpy as np
from engine.isolation_scenario import IsolationConfig, run_isolation_scenario


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = IsolationConfig()
        cls.r = {s: run_isolation_scenario(replace(cls.c, scenario=s))
                 for s in ('intact', 'sham', 'neck_cut', 'eye_loss', 'neck_cut_supported')}

    def test_common_baseline_and_sham(self):
        ref = self.r['intact']
        baseline = ref['times_ms'] <= self.c.phase_ms
        for result in self.r.values():
            for key, probe in ref['probes'].items():
                np.testing.assert_array_equal(probe[baseline], result['probes'][key][baseline])
        for key, probe in ref['probes'].items():
            np.testing.assert_array_equal(probe, self.r['sham']['probes'][key])

    def test_neck_cut_preserves_eye_brain_but_blocks_body(self):
        intact, cut = self.r['intact'], self.r['neck_cut']
        for key in ('eye_mV', 'brain_mV'):
            np.testing.assert_array_equal(intact['probes'][key], cut['probes'][key])
        np.testing.assert_array_equal(intact['inputs']['eye_input_nA'], cut['inputs']['eye_input_nA'])
        self.assertGreater(intact['metrics']['body_post_spikes'], 5)
        self.assertEqual(cut['metrics']['neck_post_spikes'], 0)
        self.assertEqual(cut['metrics']['body_post_spikes'], 0)
        self.assertEqual(cut['metrics']['cut_children'], [1])

    def test_eye_loss_zeros_mapping_not_visual_command(self):
        intact, loss = self.r['intact'], self.r['eye_loss']
        post = loss['input_times_ms'] >= self.c.phase_ms
        np.testing.assert_array_equal(intact['inputs']['visual_command_nA'], loss['inputs']['visual_command_nA'])
        self.assertTrue(np.all(loss['inputs']['eye_mapping'][post] == 0))
        self.assertEqual(loss['metrics']['cut_children'], [])
        self.assertGreater(intact['metrics']['brain_post_spikes'], loss['metrics']['brain_post_spikes'])

    def test_default_off_hypotheses_and_supported_proxy(self):
        cut, supported = self.r['neck_cut'], self.r['neck_cut_supported']
        self.assertEqual(self.c.chemical_target_gain_uS, 0)
        self.assertEqual(self.c.support_target_gain_nA, 0)
        self.assertTrue(np.all(cut['inputs']['chemical_increment_uS'] == 0))
        for key in ('eye_mV', 'brain_mV', 'neck_distal_mV', 'body_mV'):
            np.testing.assert_array_equal(cut['probes'][key], supported['probes'][key])
        self.assertLess(cut['probes']['support_proxy'][-1], supported['probes']['support_proxy'][-1])
        self.assertAlmostEqual(supported['probes']['support_proxy'][-1], .8)

    def test_clock_no_future_leakage_and_unique_events(self):
        for result in self.r.values():
            times, inp = result['times_ms'], result['inputs']
            self.assertAlmostEqual(result['metrics']['final_chemical_time_s'], self.c.duration_ms/1000)
            self.assertTrue(np.all(inp['chemical_clock_used_s'] <= result['input_times_ms']/1000 + 1e-12))
            np.testing.assert_array_equal(inp['chemical_response_used'], result['probes']['chemical_response'][:-1])
            self.assertTrue(np.all(inp['chemical_response_used'][:round(self.c.chemical_dt_ms/self.c.dt_ms)] == 0))
            np.testing.assert_allclose(np.unique(np.round(np.diff(result['probes']['chemical_clock_s']), 12)), [0, .001])
            phase = result['events']
            self.assertEqual(len(phase), 1)
            self.assertEqual(phase[0]['time_ms'], self.c.phase_ms)
            audit = phase + result['deliveries'] + result['emissions']
            self.assertEqual(len(audit), len({e['event_id'] for e in audit}))
            emission_pairs = {(e['region'], e['time_ms']) for e in result['emissions']}
            for e in result['deliveries']:
                self.assertAlmostEqual(e['arrival_ms']-e['emitted_ms'], 1.)
                self.assertIn((e['pre'], e['emitted_ms']), emission_pairs)
                self.assertLess(e['arrival_ms'], times[-1])
            self.assertLess(abs(result['metrics']['chemical_mass_balance_error_nM_um3']), 1e-7)
        later = run_isolation_scenario(replace(self.c, scenario='neck_cut', phase_ms=60.))
        baseline = later['times_ms'] <= 60
        for key in ('brain_mV', 'neck_distal_mV', 'body_mV'):
            np.testing.assert_array_equal(later['probes'][key][baseline], self.r['intact']['probes'][key][baseline])

    def test_optional_chemical_response_actually_reaches_brain(self):
        active = run_isolation_scenario(replace(self.c, chemical_target_gain_uS=.005))
        absent = run_isolation_scenario(replace(self.c, chemical_initial_nM=0., chemical_target_gain_uS=.005))
        for other in (active, absent):
            np.testing.assert_array_equal(other['probes']['eye_mV'], self.r['intact']['probes']['eye_mV'])
        self.assertGreater(active['inputs']['chemical_increment_uS'].max(), 0.)
        self.assertLess(absent['inputs']['chemical_increment_uS'].max(), 1e-20)
        self.assertGreater(np.max(np.abs(active['probes']['brain_mV']-absent['probes']['brain_mV'])), 1.)
        zero = run_isolation_scenario(replace(self.c, chemical_initial_nM=0.))
        np.testing.assert_array_equal(zero['probes']['brain_mV'], self.r['intact']['probes']['brain_mV'])

    def test_support_hypothesis_opt_in(self):
        a = run_isolation_scenario(replace(self.c, scenario='neck_cut', support_target_gain_nA=.08))
        b = run_isolation_scenario(replace(self.c, scenario='neck_cut_supported', support_target_gain_nA=.08))
        self.assertLess(a['inputs']['support_current_nA'][-1], 0.)
        self.assertGreater(np.max(np.abs(a['probes']['brain_mV']-b['probes']['brain_mV'])), 1.)
        self.assertEqual(a['metrics']['body_post_spikes'], 0)
        self.assertEqual(b['metrics']['body_post_spikes'], 0)  # proxy cannot reconnect cable

    def test_field_couples_and_artifact_is_not_vm(self):
        off = run_isolation_scenario(replace(self.c, stimulus_nA=0.))
        on = self.r['intact']
        before = on['times_ms'] <= self.c.stimulus_start_ms
        np.testing.assert_array_equal(off['probes']['neck_distal_mV'][before], on['probes']['neck_distal_mV'][before])
        self.assertGreater(np.max(np.abs(off['probes']['neck_distal_mV']-on['probes']['neck_distal_mV'])), 1e-5)
        self.assertTrue(np.all(off['inputs']['artifact_unfiltered_mV'] == 0))
        self.assertGreater(np.max(np.abs(on['inputs']['artifact_unfiltered_mV'])), .01)
        self.assertFalse(np.array_equal(on['inputs']['artifact_recorded_mV'], on['probes']['neck_distal_mV'][1:]))
        xyz = np.asarray(on['provenance']['segment_world_um'])
        self.assertTrue(np.all(np.diff(xyz[:, 1]) > 0))
        np.testing.assert_array_equal(xyz[:, 0], 100.)

    def test_fixed_initialization_and_seed(self):
        c = replace(self.c, duration_ms=10., phase_ms=4., stimulus_start_ms=8., recorder_noise_sd_mV=.01)
        a, b = run_isolation_scenario(c), run_isolation_scenario(c)
        other = run_isolation_scenario(replace(c, seed=c.seed+1))
        for key in a['probes']:
            np.testing.assert_array_equal(a['probes'][key], b['probes'][key])
            np.testing.assert_array_equal(a['probes'][key], other['probes'][key])
        np.testing.assert_array_equal(a['inputs']['artifact_recorded_mV'], b['inputs']['artifact_recorded_mV'])
        self.assertFalse(np.array_equal(a['inputs']['artifact_recorded_mV'], other['inputs']['artifact_recorded_mV']))

    def test_configuration_validation(self):
        for kwargs in ({'scenario':'fly'}, {'phase_ms':0}, {'chemical_dt_ms':.07},
                       {'phase_ms':40.01}, {'chemical_target_gain_uS':-1}, {'dt_ms':float('nan')}):
            with self.assertRaises(ValueError):
                replace(self.c, **kwargs).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
