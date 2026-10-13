"""Pure array/source audit tests: no model load, physics, or neural-tier construction."""
import copy
import json
import sys
import unittest
import numpy as np
from pathlib import Path
from run_neural_command_audit import OUT, audit, check_lag, distribution, validate_arrays
from engine.embodied.closure_bundle import validate_bundle
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = validate_bundle(OUT)
        p = Path(bundle['path'])
        cls.result = json.loads((p/'closure_results.json').read_text())
        cls.manifest = json.loads((p/'motor_manifest.json').read_text())['measured']
        with np.load(p/'closure_traces.npz',allow_pickle=False) as z:
            cls.arrays = {k:z[k] for k in z.files}

    def test_current_bundle_shape_identity_finite(self):
        validate_arrays(self.manifest,self.result,self.arrays)
        for change in ('shape','nan','keys'):
            a = self.arrays.copy()
            if change == 'shape':
                a['normal_controls'] = a['normal_controls'][:,:-1]
            elif change == 'nan':
                a['normal_controls'] = a['normal_controls'].copy()
                a['normal_controls'][0,0] = np.nan
            else:
                a['wrong_arm_controls'] = a.pop('normal_controls')
            with self.assertRaises(ValueError):
                validate_arrays(self.manifest,self.result,a)

    def test_current_source_lag_exact_and_tolerance(self):
        records = self.result['measured']['episodes']['normal']['records']
        controls = self.arrays['normal_controls']
        config = self.result['provenance']['motor_translation']
        self.assertTrue(check_lag(records,controls,config)['exact_all'])
        changed = controls.copy(); changed[1,0] += 5e-13
        check_lag(records,changed,config)
        changed[1,0] += 2e-12
        with self.assertRaisesRegex(ValueError,'output mismatch'):
            check_lag(records,changed,config)
        bad = copy.deepcopy(records); bad[1]['applied_source_interval'] = 1
        with self.assertRaisesRegex(ValueError,'source interval lag'):
            check_lag(bad,controls,config)

    def test_current_summary_invariants(self):
        r = audit()
        self.assertEqual(r['status'],'PASS_ARTIFACT_AUDIT_NOT_AUTONOMY')
        self.assertTrue(r['validation']['no_unresolved_commands'])
        self.assertTrue(r['validation']['identical_present_pool_commands_equal'])
        self.assertTrue(r['external_touch']['motor_subset_counts_identical'])
        self.assertGreater(r['external_touch']['differing_interval_neuron_count_cells'],0)
        self.assertFalse(r['offline_comparison']['same_initial_pose'])
        # Other tests legitimately import simulation modules during collection.
        # Test the audit's no-simulation import contract in a fresh interpreter.
        import subprocess
        script = "import sys; from run_neural_command_audit import audit; audit(); forbidden={'mujoco','engine.embodied.adapters','engine.neural_cond'}; assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))"
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pure_decoder_previous_interval_and_cut(self):
        m = dict(n_neurons=2,n_actuators=3,rows=[dict(actuator_index=0,local_indices=[0]),
                 dict(actuator_index=1,local_indices=[0]),dict(actuator_index=2,local_indices=[])])
        decoder = MotorOutput(m)
        pending = np.zeros(3); controls=[]; records=[]
        for k in range(3):
            controls.append(pending.copy())
            spikes = np.zeros((10,2),bool); spikes[-1,0] = True
            pending, trace = decoder.consume(spikes,spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005)
            records.append(dict(applied_source_interval=k-1,time_start_s=k*.005,time_end_s=(k+1)*.005,
                                source_completed_by_s=k*.005,decoder=trace))
        check_lag(records,np.asarray(controls),vars(MotorOutputConfig()))
        self.assertEqual(pending[0],pending[1]); self.assertEqual(pending[2],0)
        cut,_ = decoder.consume(np.ones((10,2),bool),spike_dt_s=.0005,start_s=.015,end_s=.020,connected=False)
        self.assertFalse(cut.any())

    def test_encoder_unclipped_and_stimulus_post_encode(self):
        encoder = ChannelSensoryEncoder({'lf':{'bristle_touch':[0]}},2)
        current, record = encoder.encode({'lf':ChannelQuantities(foot_contact=10)})
        self.assertEqual(current[0],3.0)
        current[0] += .3
        self.assertEqual(current[0],3.3)
        self.assertEqual(record['per_leg']['lf']['bristle_touch']['n_cells'],1)

    def test_distribution_and_identity_negative(self):
        d = distribution([0,0,.5,1])
        self.assertEqual((d['zero_count'],d['saturated_count'],d['active_count']),(2,1,2))
        m = copy.deepcopy(self.manifest); m['rows'][0]['actuator_index'] = 2
        with self.assertRaisesRegex(ValueError,'ordering'):
            validate_arrays(m,self.result,self.arrays)


if __name__ == '__main__':
    unittest.main()
