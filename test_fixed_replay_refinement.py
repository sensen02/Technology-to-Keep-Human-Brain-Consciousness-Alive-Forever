"""Saved-artifact validation only; these tests never run an episode."""
import json
import unittest
from pathlib import Path
import numpy as np
import run_fixed_replay_refinement as run

class FixedReplayRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=json.loads((run.OUT/'fixed_replay_refinement_results.json').read_text())
    def test_frozen_sources_history_and_archive(self):
        run.load_history()
        p=self.doc['provenance']
        self.assertEqual(p['source_hashes_before'],p['source_hashes_after'])
        for name,h in p['source_hashes_after'].items():
            self.assertEqual(run.sha(name),h,name)
            if name.endswith('.py'):
                self.assertEqual(run.sha(run.OUT/'source_archive'/Path(name).relative_to(run.HERE)),h)
        self.assertEqual(run.sha(run.OUT/'fixed_replay_refinement_traces.npz'),p['npz_sha256'])
        self.assertTrue(all(self.doc['checks'].values()))
    def test_two_cases_per_step_health_and_first_dt(self):
        self.assertEqual(set(self.doc['episodes']),{'12.5us_replay','6.25us_replay'})
        for us,steps,stride in ((12.5,16000,8),(6.25,32000,16)):
            e=self.doc['episodes'][run.label(us)]
            self.assertEqual(e['arm'],'replay')
            self.assertEqual(e['completed_body_steps'],steps)
            self.assertEqual(e['applied_command_exact_checked_body_steps'],steps)
            self.assertEqual(e['actual_neural_steps'],400)
            self.assertEqual(e['body_samples'],2001)
            self.assertEqual(e['sample_integer_stride'],stride)
            self.assertEqual(e['first_step']['time_before_s'],0)
            self.assertEqual(e['first_step']['time_after_s'],us*1e-6)
            self.assertTrue(e['first_step']['initial_qpos_exact'])
            self.assertTrue(e['first_step']['initial_qvel_act_ctrl_zero'])
            self.assertEqual(len(e['command_valid_per_5ms']),40)
            self.assertTrue(all(e['command_valid_per_5ms']))
            for k in ('all_fine_body_steps_finite','all_neural_steps_finite','decoder_finite'):self.assertTrue(e['health'][k])
            self.assertFalse(any(e['health']['warning_max']))
            self.assertGreaterEqual(e['activation']['act_min'],0)
            self.assertLessEqual(e['activation']['act_max'],1)
            self.assertEqual(e['activation']['act_above_one_values'],0)
            self.assertEqual(e['activation']['act_below_zero_values'],0)
            self.assertTrue(all(r['command_origin']=='offline_normal_replay' for r in e['records']))
    def test_replay_cannot_be_presented_as_cure_or_autonomy(self):
        p=self.doc['protocol']
        for k in ('autonomous','controller_cure','convergence_claim','entire_causal_attribution','neural_decoder_output_applied'):self.assertFalse(p[k])
        for k in ('explicitly_diagnostic','actual_neural_integration','no_activation_clamp','no_new_controller','no_gain_control_activation_changes'):self.assertTrue(p[k])
        self.assertEqual(p['com_resolution_reference_mm'],.01)
        self.assertEqual(p['maximum_new_episodes'],2)
        self.assertIn('NOT_CURE',self.doc['status'])
        self.assertTrue(any('NOT a cure' in s for s in self.doc['limitations']))
    def test_exact_source_controls_and_same_dt_reproduction(self):
        with np.load(run.OUT/'fixed_replay_refinement_traces.npz') as data,np.load(run.refinement.OUT/'clock_refinement_traces.npz') as old:
            commands=old['6.25us_normal_controls']
            self.assertEqual(run.array_hash(commands),self.doc['protocol']['source_commands_array_sha256'])
            self.assertTrue(np.array_equal(data['reference_commands'],commands))
            for us in run.NEW_US:
                key=run.label(us)
                self.assertTrue(np.array_equal(data[key+'_controls'],commands))
                self.assertTrue(np.array_equal(data[key+'_body_time_s'],np.arange(2001)*.0001))
                self.assertTrue(np.array_equal(data[key+'_body_qpos'][0],data['seated_qpos']))
                for name in ('body_qpos','body_qvel','body_act','body_ctrl','body_com_xyz_mm','controls'):
                    self.assertTrue(np.isfinite(data[key+'_'+name]).all())
                for name in ('body_qvel','body_act','body_ctrl'):self.assertTrue(np.all(data[key+'_'+name][0]==0))
            for name in old.files:
                if name.startswith('6.25us_normal_'):
                    self.assertTrue(np.array_equal(old[name],data[name.replace('6.25us_normal_','6.25us_replay_',1)]),name)
    def test_recomputed_metrics_and_relative_errors(self):
        with np.load(run.OUT/'fixed_replay_refinement_traces.npz') as data:
            a,b=[{k[len(run.label(us))+1:]:data[k] for k in data.files if k.startswith(run.label(us)+'_')} for us in run.NEW_US]
            c=run.refinement.compare(a,b,self.doc['episodes'][run.label(12.5)],self.doc['episodes'][run.label(6.25)],self.doc['joint_names'],np.asarray(self.doc['joint_qpos_addresses']))
            self.assertEqual(c,self.doc['comparisons']['12.5us_replay_vs_6.25us_replay'])
            self.assertIsNone(c['first_control_difference_applied_start_s'])
            for w,v in c['windows_ms'].items():
                self.assertEqual(v['max_applied_control_difference'],0)
                self.assertEqual(v['different_control_elements'],0)
                for metric in run.METRICS:
                    r=self.doc['relative_to_closed_feedback'][w][metric]
                    self.assertEqual(r['fixed_replay'],v[metric])
                    self.assertEqual(r['closed_feedback'],self.doc['closed_feedback_comparison']['windows_ms'][w][metric])
                    self.assertEqual(r['decreased'],r['fixed_replay']<r['closed_feedback'])
    def test_reused_models_metadata_and_dt_only(self):
        import mujoco
        old=run.load_history()
        self.assertEqual(self.doc['model_manifest'],old['model_manifest'])
        base=mujoco.MjModel.from_xml_path(str(run.zero.XML))
        for item in self.doc['model_manifest'].values():
            self.assertEqual(run.sha(item['xml']),item['sha256'])
            self.assertEqual(run.fine.compiled_diff(base,mujoco.MjModel.from_xml_path(item['xml'])),item['compiled_comparisons']['zero_floor_100us'])
if __name__=='__main__':unittest.main()
