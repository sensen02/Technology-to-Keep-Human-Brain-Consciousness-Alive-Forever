"""Saved-artifact checks for the motor output census."""
import json
import unittest
import numpy as np
import run_motor_output_census as run

class MotorOutputCensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((run.OUT / 'motor_output_census.json').read_text())
        cls.z = np.load(run.OUT / 'motor_output_census.npz')

    def test_counts_partition_the_actuators(self):
        s = self.doc['summary']
        self.assertEqual(s['actuators'], 90)
        self.assertEqual(s['uncommandable_no_pool'] + s['silent_pool_no_spikes'] + s['driven'], 90)
        self.assertEqual(len(self.doc['uncommandable']), s['uncommandable_no_pool'])
        self.assertEqual(len(self.doc['silent_pools']), s['silent_pool_no_spikes'])

    def test_uncommandable_are_coxa_bundles(self):
        for entry in self.doc['uncommandable']:
            self.assertIn(entry['pool_status'], ('unresolved', 'partial'))
            self.assertTrue(any(t in entry['muscle_target'] for t in
                                ('promotor', 'remotor', 'adductor', 'abductor')), entry)
        legs = sorted({e['leg'] for e in self.doc['uncommandable']})
        self.assertIn('LH', legs)
        self.assertIn('RH', legs)

    def test_silent_pools_have_cells(self):
        for entry in self.doc['silent_pools']:
            self.assertGreater(entry['pool_cells'], 0)

    def test_right_middle_trochanter_flexor_is_undriven(self):
        c = [x for x in self.doc['census'] if x['leg'] == 'RM' and x['joint_family'] == 'trochanter']
        self.assertTrue(c)
        self.assertEqual(sum(x['pool_spikes_total'] for x in c if x['direction'] == 'flexor'), 0)
        self.assertGreater(sum(x['pool_spikes_total'] for x in c if x['direction'] == 'extensor'), 0)

    def test_canonical_still_falls(self):
        """This census must not be presented as a capability result."""
        doc = json.loads((run.HERE / 'outputs/model_branches/fine_clock/fine_clock_results.json').read_text())
        self.assertLess(doc['episodes']['25us_normal']['endpoint_heights_mm'][2], 1.0)

if __name__ == '__main__':
    unittest.main()
