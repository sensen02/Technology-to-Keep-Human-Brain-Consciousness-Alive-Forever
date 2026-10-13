"""Read-only evidence regression tests for independently authorized floor-only branch."""
import json
import unittest
import numpy as np
import run_zero_floor_branch as branch

class ZeroFloorBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=json.loads((branch.OUT/'zero_floor_results.json').read_text())
        cls.data=np.load(branch.OUT/'zero_floor_traces.npz',allow_pickle=False)
    @classmethod
    def tearDownClass(cls): cls.data.close()
    def test_original_and_source_hashes(self):
        self.assertEqual(branch.sha(branch.closure.cp.CANONICAL_XML),branch.SHA)
        for p,h in self.doc['provenance']['source_hashes'].items(): self.assertEqual(branch.sha(p),h,p)
        self.assertEqual(branch.sha(branch.closure.OUT/'current_run.json'),self.doc['provenance']['current_pointer_sha256'])
    def test_compiled_only_floor(self):
        import mujoco
        canonical=branch.closure.cp.load_model()
        model=mujoco.MjModel.from_xml_path(str(branch.XML))
        diff=branch.compiled_diff(canonical,model)
        self.assertEqual(diff['changed_arrays'],['actuator_ctrlrange'])
        self.assertEqual(diff['changed_control_lower_count'],90)
        self.assertGreater(len(diff['equal_arrays']),400)
        self.assertTrue(np.all(model.actuator_ctrllimited))
        self.assertFalse(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))
    def test_branch_zero_ctrl_zero_act_force(self):
        for key in ('controls','body_ctrl','body_act','body_act_dot','body_active_delta_actuator_force'):
            self.assertTrue(np.all(self.data['spike_output_silence_'+key]==0),key)
        self.assertGreater(np.abs(self.data['spike_output_silence_body_act0_actuator_force']).max(),0)
    def test_replay_exact(self):
        for key in ('trajectory','controls','spike_counts','body_qpos','body_qvel','body_act','body_ctrl','body_actuator_force','heights_mm'):
            np.testing.assert_array_equal(self.data['normal_'+key],self.data['replay_'+key])
    def test_initial_and_2001_live_states(self):
        for arm in ('normal','spike_output_silence','replay'):
            for key in ('body_qvel','body_act','body_ctrl'):
                self.assertTrue(np.all(self.data[arm+'_'+key][0]==0))
            np.testing.assert_array_equal(self.data[arm+'_body_qpos'][0],self.data['seated_qpos'])
            self.assertEqual(len(self.data[arm+'_body_qpos']),2001)
            self.assertTrue(self.data[arm+'_body_finite'].all())
            self.assertFalse(self.data[arm+'_body_warning_counts'].any())
            self.assertEqual(self.data[arm+'_spike_counts'].shape,(40,12000))
            self.assertTrue(self.doc['episodes'][arm]['health']['all_live_states_finite'])
    def test_evidence_hash_and_archive(self):
        self.assertEqual(branch.sha(branch.OUT/'zero_floor_traces.npz'),self.doc['provenance']['npz_sha256'])
        self.assertEqual(branch.sha(branch.XML),self.doc['model_manifest']['branch_sha256'])
        for p,h in self.doc['provenance']['source_archive_hashes'].items(): self.assertEqual(branch.sha(p),h,p)
        for row in self.doc['model_manifest']['declared_asset_path_changes']:
            self.assertEqual(branch.sha(row['branch']),row['sha256'])
    def test_prefix_and_protocol(self):
        self.assertTrue(all(self.doc['checks'].values()))
        self.assertEqual(self.doc['protocol']['canonical_additional_episodes'],0)
        self.assertEqual(self.doc['status'],'PASS_BRANCH_INVARIANTS')
        self.assertEqual(self.doc['execution_status'],'completed')
        self.assertLess(self.doc['wall_s'],60)

if __name__=='__main__': unittest.main(verbosity=2)
