"""Fast unit and completed-artifact checks; no second heavy simulation."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_early_feedback_probe as probe
from engine.embodied.audit_provenance import inspect_audit


class UnitTests(unittest.TestCase):
    def test_tolerance_does_not_inflate_roundoff(self):
        a = np.zeros((3,2)); b = a.copy(); b[1,0] = 5e-14
        self.assertEqual(probe.first_difference(a,b,[0,.005,.01])['first_index'],1)
        self.assertIsNone(probe.first_difference(a,b,[0,.005,.01],1e-12)['first_index'])
        b[2,0] = 2e-12
        self.assertEqual(probe.first_difference(a,b,[0,.005,.01],1e-12)['first_time_s'],.01)

    def test_force_schedule_detects_interval_wide_and_shifted_push(self):
        model = SimpleNamespace(body=lambda name:SimpleNamespace(id=1))
        push = dict(force_model=[.123,0,0])
        a = dict(xfrc=np.zeros((500,3,6)),time_s=np.arange(501)*.0001)
        a['xfrc'][10:20,1,0] = .123
        self.assertEqual(probe.verify_force(a,model,push)['active_step_indices'],list(range(10,20)))
        for left,right in ((0,50),(11,21),(150,160)):
            a['xfrc'][:] = 0; a['xfrc'][left:right,1,0] = .123
            with self.assertRaises(AssertionError): probe.verify_force(a,model,push)
        a['xfrc'][:] = 0
        probe.verify_force(a,model,None)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = probe.OUT/'early_feedback_probe.json'
        if not path.exists(): raise unittest.SkipTest('completed probe artifact not present')
        cls.r = json.loads(path.read_text())
        cls.z = np.load(cls.r['traces']['path'],allow_pickle=False)

    @classmethod
    def tearDownClass(cls): cls.z.close()

    def test_identity_and_frozen_reference(self):
        r,z = self.r,self.z
        evidence = inspect_audit(Path(__file__).resolve().parent, 'early_feedback_probe')
        current = evidence['source_bundle']  # exact historical path, not current OUT
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertEqual(evidence['run_sources'][str(Path(__file__).resolve().parent/'run_neural_muscle_closure.py')]['status'], 'drift')
        self.assertEqual(current,r['source_bundle'])
        self.assertEqual(current['run_id'],probe.EXPECTED_RUN)
        self.assertEqual(probe.sha(r['traces']['path']),r['traces']['sha256'])
        for path,h in r['provenance']['inputs'].items():
            self.assertEqual(probe.sha(path),h)
        with np.load(Path(current['path'])/'closure_traces.npz',allow_pickle=False) as old:
            for arm in ('normal','normal_repeat'):
                for suffix in ('trajectory','controls','spike_counts'):
                    np.testing.assert_array_equal(z['baseline_'+suffix],old[arm+'_'+suffix])
        self.assertLess(r['wall_s'],60)

    def test_actual_physics_force_and_impulse(self):
        z,r = self.z,self.r
        force = np.asarray(r['protocol']['push']['force_model'])
        for label in probe.ARMS:
            x = z[label+'_xfrc']; active = np.flatnonzero(np.any(x!=0,axis=(1,2)))
            np.testing.assert_array_equal(active, [] if label=='baseline' else np.arange(10,20))
            if label!='baseline':
                np.testing.assert_array_equal(x,z['push_normal_xfrc'])
                np.testing.assert_allclose(x[:,:,:3].sum(axis=(0,1))*.0001,force*.001,rtol=1e-14,atol=0)
                self.assertFalse(x[:,:,3:].any())
            np.testing.assert_allclose(z[label+'_time_s'],np.arange(501)*.0001,atol=1e-14,rtol=0)

    def test_prepush_and_replay_controls(self):
        z = self.z
        for arm in ('push_normal','push_replay'):
            np.testing.assert_array_equal(z[arm+'_state'][:11],z['baseline_state'][:11])
            np.testing.assert_array_equal(z[arm+'_currents'][0],z['baseline_currents'][0])
            np.testing.assert_array_equal(z[arm+'_motor_raster'][:10],z['baseline_motor_raster'][:10])
        self.assertEqual(len(set(self.r['invariants']['initial_neural_sha256'].values())),1)
        np.testing.assert_array_equal(z['push_replay_controls'],z['baseline_controls'])
        # Main contrast shares all physical states through the changed-command boundary,
        # not merely the sparse saved 5ms endpoints. Sensory sample at 10ms is still equal.
        np.testing.assert_array_equal(z['push_normal_state'][:151],z['push_replay_state'][:151])
        np.testing.assert_array_equal(z['push_normal_currents'][:4],z['push_replay_currents'][:4])
        np.testing.assert_array_equal(z['push_normal_motor_raster'][:30],z['push_replay_motor_raster'][:30])
        for arm in probe.ARMS:
            np.testing.assert_array_equal(z[arm+'_ctrl'],np.repeat(z[arm+'_controls'],50,axis=0))
        self.assertFalse(z['push_sensory_cut_currents'].any())

    def test_command_causality_and_mechanical_reporting(self):
        z,r = self.z,self.r
        expected = probe.first_difference(z['push_normal_controls'],z['push_replay_controls'],np.arange(10)*.005,1e-12)
        self.assertEqual(expected,r['command_causality']['applied_command']['above_1e_12'])
        source = r['command_causality']['first_changed_command_source']
        if source:
            self.assertEqual(source['command_valid_from_s'],expected['first_time_s'])
            self.assertLessEqual(r['command_causality']['motor_spike']['exact']['first_time_s'],source['command_valid_from_s'])
            self.assertGreaterEqual(r['command_causality']['sensory_current']['above_1e_12']['first_time_s'],.001)
        for arm in probe.ARMS:
            for field in ('root_xyz','com_xyz'):
                self.assertEqual(z[arm+'_'+field].shape,(501,3))
            outcome = r['mechanical_outcomes'][arm]
            np.testing.assert_array_equal(outcome['root_final_model'],z[arm+'_root_xyz'][-1])
            np.testing.assert_array_equal(outcome['com_final_model'],z[arm+'_com_xyz'][-1])
        self.assertTrue(all(np.isfinite(z[k]).all() for k in z.files))
        self.assertTrue(r['invariants']['zero_mujoco_warnings'])
        self.assertIn('not_identified',r['interpretation']['usefulness'])


if __name__ == '__main__': unittest.main()
