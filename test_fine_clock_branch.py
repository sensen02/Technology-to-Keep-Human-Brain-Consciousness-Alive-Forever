"""Focused source, strict mismatch, and saved fine-clock branch regression tests."""
import json
import unittest
import numpy as np
import run_fine_clock_branch as run

class FineClockTests(unittest.TestCase):
    def test_source_patch(self):
        self.assertIn('body_dt',run.branch.verify_source_copy()['diff'])
    def test_mismatch_not_bypassed(self):
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(run.zero.XML))
        with self.assertRaisesRegex(ValueError,'timestep differs'):
            run.branch.episode(model,None,None,None,None,None,None,'normal',.2,body_dt=.000025)
    def test_saved_contract(self):
        doc=json.loads((run.OUT/'fine_clock_results.json').read_text())
        self.assertTrue(all(doc['checks'].values()))
        self.assertEqual(len(doc['episodes']),5)
        for key,e in doc['episodes'].items():
            self.assertEqual(e['actual_neural_steps'],400)
            self.assertEqual(e['body_samples'],2001)
            self.assertEqual(e['completed_body_steps'],round(.2/e['body_dt_s']))
            self.assertEqual(len(e['command_valid_per_5ms']),40)
            self.assertTrue(all(e['command_valid_per_5ms']))
            self.assertFalse(any(e['health']['warning_max']))
        with np.load(run.OUT/'fine_clock_traces.npz') as data,np.load(run.zero.OUT/'zero_floor_traces.npz') as baseline:
            for k in ('trajectory','controls','spike_counts'):
                self.assertTrue(np.array_equal(data['100us_normal_'+k],baseline['normal_'+k]))
            for key in doc['episodes']:
                self.assertEqual(data[key+'_body_qpos'].shape[0],2001)
                self.assertEqual(data[key+'_controls'].shape,(40,90))
                self.assertTrue(np.isfinite(data[key+'_body_act']).all())
        for path,h in doc['provenance']['source_hashes_after'].items():self.assertEqual(run.sha(path),h,path)

if __name__=='__main__':unittest.main()
