"""Saved-artifact tests only: never rerun or add experimental cases."""
import json
import unittest
from pathlib import Path
import numpy as np
import run_clock_refinement as run

class ClockRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=json.loads((run.OUT/'clock_refinement_results.json').read_text())
    def test_frozen_history_and_archive(self):
        run.validate_history()
        p=self.doc['provenance']
        self.assertEqual(p['source_hashes_before'],p['source_hashes_after'])
        for name,h in p['source_hashes_after'].items():
            self.assertEqual(run.sha(name),h,name)
            if name.endswith('.py'):
                self.assertEqual(run.sha(run.OUT/'source_archive'/Path(name).relative_to(run.HERE)),h)
        self.assertEqual(run.sha(run.OUT/'clock_refinement_traces.npz'),p['npz_sha256'])
        self.assertIn('body_dt',run.branch.verify_source_copy()['diff'])
    def test_two_cases_health_clock(self):
        self.assertEqual(set(self.doc['episodes']),{'12.5us_normal','6.25us_normal'})
        for us,steps,stride in ((12.5,16000,8),(6.25,32000,16)):
            e=self.doc['episodes'][run.label(us)]
            self.assertEqual(e['completed_body_steps'],steps)
            self.assertEqual(e['actual_neural_steps'],400)
            self.assertEqual(e['sample_integer_stride'],stride)
            self.assertEqual(e['body_samples'],2001)
            self.assertEqual(len(e['command_valid_per_5ms']),40)
            self.assertTrue(all(e['command_valid_per_5ms']))
            self.assertTrue(e['health']['all_fine_body_steps_finite'])
            self.assertTrue(e['health']['all_neural_steps_finite'])
            self.assertTrue(e['health']['decoder_finite'])
            self.assertFalse(any(e['health']['warning_max']))
            self.assertGreaterEqual(e['activation']['act_min'],0)
            self.assertLessEqual(e['activation']['act_max'],1)
        self.assertFalse(self.doc['protocol']['convergence_claim'])
        self.assertEqual(self.doc['protocol']['com_resolution_reference_mm'],.01)
    def test_data_and_metrics(self):
        with np.load(run.OUT/'clock_refinement_traces.npz') as data,np.load(run.fine.OUT/'fine_clock_traces.npz') as old:
            for us in (50,25,12.5,6.25):
                key=run.label(us)
                self.assertTrue(np.array_equal(data[key+'_body_time_s'],np.arange(2001)*.0001))
                self.assertEqual(data[key+'_controls'].shape,(40,90))
                self.assertEqual(data[key+'_body_com_xyz_mm'].shape,(2001,3))
                for name in ('body_qpos','body_qvel','body_act','body_ctrl','body_com_xyz_mm','controls'):
                    self.assertTrue(np.isfinite(data[key+'_'+name]).all())
                if us in (50,25):
                    for name in ('body_qpos','body_qvel','body_act','body_ctrl','controls','spike_counts'):
                        self.assertTrue(np.array_equal(data[key+'_'+name],old[key+'_'+name]))
            for a,b in ((50,25),(25,12.5),(12.5,6.25)):
                ka,kb=run.label(a),run.label(b)
                c=self.doc['comparisons'][ka+'_vs_'+kb]
                for ms in run.WINDOWS_MS:
                    n=ms*10+1;v=c['windows_ms'][str(ms)]
                    self.assertEqual(v['max_com_distance_mm'],float(np.linalg.norm(data[ka+'_body_com_xyz_mm'][:n]-data[kb+'_body_com_xyz_mm'][:n],axis=1).max()))
                    self.assertEqual(v['max_root_translation_distance_mm'],float(np.linalg.norm(data[ka+'_body_qpos'][:n,:3]-data[kb+'_body_qpos'][:n,:3],axis=1).max()))
                    self.assertEqual(v['max_individual_joint_angle_error_rad'],max(v['joint_angle_max_error_rad'].values()))
    def test_models(self):
        import mujoco
        base=mujoco.MjModel.from_xml_path(str(run.zero.XML))
        for item in self.doc['model_manifest'].values():
            self.assertEqual(run.sha(item['xml']),item['sha256'])
            model=mujoco.MjModel.from_xml_path(item['xml'])
            self.assertEqual(run.fine.compiled_diff(base,model),item['compiled_comparisons']['zero_floor_100us'])

if __name__=='__main__':unittest.main()
