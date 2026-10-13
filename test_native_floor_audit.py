"""Lightweight artifact checks: no compilation, neural replay or heavy simulation."""
import json
import unittest
import numpy as np
import run_native_floor_audit as audit
from engine.embodied.closure_bundle import validate_bundle

class NativeFloorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((audit.OUT/'native_floor_audit.json').read_text())
        with np.load(audit.OUT/'native_floor_audit.npz',allow_pickle=False) as z:
            cls.a = {key:z[key].copy() for key in z.files}

    def test_provenance(self):
        r = self.result
        self.assertEqual(r['schema_version'],1)
        self.assertEqual(r['status'],'PASS_NATIVE_FLOOR_OBSERVATION')
        self.assertEqual(r['source_bundle']['run_id'],validate_bundle(r['source_bundle']['path'])['run_id'])
        self.assertEqual(audit.sha(audit.cp.CANONICAL_XML),audit.EXPECTED_XML_SHA)
        self.assertEqual(audit.sha(r['traces']['path']),r['traces']['sha256'])
        self.assertEqual(audit.sha(r['provenance']['code_path']),r['provenance']['code_sha256'])
        source = audit.Path(r['source_bundle']['path'])
        for name,h in r['source_bundle']['sha256'].items(): self.assertEqual(audit.sha(source/name),h)

    def test_actual_steps_positive_act_zero_control(self):
        a = self.a
        np.testing.assert_array_equal(a['zero_ctrl_integration_ctrl_readback'],np.zeros((50,90)))
        self.assertTrue(np.all(a['zero_ctrl_integration_act']>0))
        np.testing.assert_allclose(a['zero_ctrl_integration_time_s'],np.arange(1,51)*.0001,rtol=0,atol=1e-16)
        self.assertTrue(np.all(a['actuator_ctrlrange'][:,0]==.0001))
        self.assertTrue(np.all(a['initial_act_dot']>0))
        np.testing.assert_array_equal(a['initial_act_dot'],a['clamped_floor_analytic_act_dot'])
        for field in ('qpos','qvel','act'):
            np.testing.assert_array_equal(a['zero_ctrl_integration_'+field][-1],a['spike_output_silence_'+field][0])

    def test_saved_sources_and_passive_decomposition(self):
        source = audit.Path(self.result['source_bundle']['path'])
        with np.load(source/'closure_traces.npz',allow_pickle=False) as z:
            for arm in audit.ARMS:
                np.testing.assert_array_equal(np.concatenate([self.a[arm+'_'+key] for key in ('qpos','qvel','act')],axis=1),z[arm+'_trajectory'])
        for arm in (*audit.ARMS,'zero_ctrl_integration'):
            a = {key[len(arm)+1:]:value for key,value in self.a.items() if key.startswith(arm+'_')}
            for field in ('actuator_force','qfrc_actuator','qfrc_passive'):
                np.testing.assert_array_equal(a['active_delta_'+field],a['total_'+field]-a['act0_'+field])
            self.assertTrue(np.any(a['active_delta_actuator_force']!=0))
            self.assertTrue(np.all(a['active_delta_qfrc_passive']==0))
            self.assertEqual(audit.summarize(a,self.result['protocol']['dof_groups']),self.result['summaries'][arm])
        np.testing.assert_array_equal(self.a['spike_output_silence_source_controls'],np.zeros((10,90)))

    def test_units_and_finiteness_and_budget(self):
        g = self.result['protocol']['dof_groups']
        self.assertEqual(len(g['translational']),3)
        self.assertEqual(len(g['angular']),45)
        self.assertEqual(sorted(g['translational']+g['angular']),list(range(48)))
        self.assertTrue(all(np.isfinite(v).all() for v in self.a.values()))
        self.assertFalse(self.result['protocol']['model_edits'])
        self.assertFalse(self.result['protocol']['neural_replay'])
        self.assertLess(self.result['wall_s'],60)
        self.assertEqual(self.result['memory_guard']['lock_name'],'heavy_job')

if __name__=='__main__': unittest.main()
