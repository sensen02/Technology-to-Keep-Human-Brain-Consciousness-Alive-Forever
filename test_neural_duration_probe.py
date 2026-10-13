"""Saved-artifact checks only: never repeat neural or physical episodes."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from run_neural_duration_probe import enforce_safe, UnsafeState
from engine.embodied.audit_provenance import inspect_audit
HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/neural_muscle_closure'


class DurationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((OUT/'neural_duration_probe.json').read_text())
        cls.arr = np.load(OUT/'neural_duration_probe.npz')

    @classmethod
    def tearDownClass(cls):
        cls.arr.close()

    def test_exact_prefix_commands_counts_and_eleven_trajectory_rows(self):
        with np.load(OUT/'neural_pose_comparison.npz') as prior:
            for key in self.doc['episodes']:
                for field in ('controls','spike_counts','trajectory'):
                    np.testing.assert_array_equal(self.arr[key+'_'+field][:10],prior[key+'_'+field])
                initial = self.arr[key+'_trajectory_with_initial'][0]
                np.testing.assert_array_equal(initial[:49],prior[key+'_body_qpos'][0])
                self.assertFalse(initial[49:].any())
                np.testing.assert_array_equal(self.arr[key+'_trajectory_with_initial'][:11],np.vstack([initial,prior[key+'_trajectory']]))
                np.testing.assert_array_equal(self.arr[key+'_body_qpos'][:501],prior[key+'_body_qpos'])

    def test_only_two_normal_episodes_zero_initialization(self):
        self.assertEqual(set(self.doc['episodes']),{'original_normal','repaired_normal'})
        self.assertEqual(self.doc['duration_s'],.2)
        self.assertFalse(self.doc['provenance']['source_activation_used'])
        for key,e in self.doc['episodes'].items():
            for field in ('qvel','act','ctrl'):
                self.assertFalse(np.any(e['initial'][field]))
            self.assertEqual(len(e['initial']['ranges']['violations']),5 if key.startswith('original') else 0)

    def test_all_steps_finite_warning_free_and_separate_velocities(self):
        for key,e in self.doc['episodes'].items():
            self.assertEqual(e['completed_body_steps'],2000)
            self.assertEqual(self.arr[key+'_body_qpos'].shape,(2001,49))
            self.assertEqual(self.arr[key+'_heights_mm'].shape,(2001,3))
            self.assertEqual(self.arr[key+'_neural_state_finite'].shape,(400,))
            self.assertEqual(self.arr[key+'_controls'].shape[0],40)
            self.assertEqual(self.arr[key+'_trajectory_with_initial'].shape[0],41)
            self.assertTrue(self.arr[key+'_body_finite'].all())
            self.assertTrue(self.arr[key+'_neural_state_finite'].all())
            self.assertFalse(self.arr[key+'_body_warning_counts'].any())
            qvel=self.arr[key+'_body_qvel']; metrics=self.arr[key+'_body_metrics']
            np.testing.assert_array_equal(metrics[:,3],np.abs(qvel[:,3:]).max(axis=1))
            np.testing.assert_array_equal(metrics[:,4],np.abs(qvel[:,:3]).max(axis=1))
            self.assertTrue(e['health']['all_live_states_finite'])
            np.testing.assert_array_equal(self.arr[key+'_body_qpos'][-1],self.arr[key+'_trajectory'][-1,:49])

    def test_honest_strict_failure_contact_and_fall_onsets(self):
        for key,e in self.doc['episodes'].items():
            self.assertEqual(e['standing_status'],'PASS' if all(e['standing_checks'].values()) else 'FAIL')
            for field,mask in [('nonfoot_contact_onset_s',self.arr[key+'_body_metrics'][:,2]>0),
                               ('escape_time_s',self.arr[key+'_heights_mm'][:,1]<.9*self.arr[key+'_heights_mm'][0,1])]:
                ix=np.flatnonzero(mask)
                self.assertEqual(e[field],float(ix[0]*.0001) if len(ix) else None)
        self.assertEqual(self.doc['status'],'FAIL')
        self.assertTrue(all(self.doc['checks'].values()))

    def test_spike_summaries(self):
        for key,e in self.doc['episodes'].items():
            counts=self.arr[key+'_spike_counts'].sum(axis=1)
            record=e['neural_spike_counts']['all_neurons']
            np.testing.assert_array_equal(record['per_5ms'],counts)
            self.assertEqual(record['final_50ms'],int(counts[-10:].sum()))
            self.assertEqual(record['final_5ms'],int(counts[-1]))
            self.assertEqual(int(self.arr[key+'_neural_raw_spike_counts'].sum()),int(counts.sum()))

    def test_frozen_sources_xml_and_npz_hashes(self):
        p=self.doc['provenance']
        evidence=inspect_audit(HERE, 'neural_duration_probe')
        self.assertEqual(evidence['source_bundle']['path'], p['current_bundle'])
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertFalse(evidence['execution_equivalence_claimed'])
        for path,expected in p['frozen_source_hashes'].items():
            self.assertFalse(Path(path).name.startswith('test_'))
            if path in evidence['audit_time_sources']:
                current=hashlib.sha256(Path(path).read_bytes()).hexdigest()
                self.assertEqual(evidence['audit_time_sources'][path]['status'],
                                 'matched' if current==expected else 'drift')
        for path,expected in [(OUT/'neural_duration_probe.npz',p['npz_sha256']),
                              (Path(p['numerical_prefix_artifact']),p['numerical_prefix_sha256']),
                              (HERE/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml',p['canonical_xml_sha256'])]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        self.assertIn(str(HERE/'engine/embodied/initial_pose_contract.py'),p['frozen_source_hashes'])

    def test_unsafe_abort_gate(self):
        enforce_safe(True,[0]*8,100,100)
        for args in ((False,[0]*8,0,0),(True,[1],0,0),(True,[0],1e7,0),(True,[0],0,1e7)):
            with self.assertRaises(UnsafeState):
                enforce_safe(*args)


if __name__=='__main__':
    unittest.main()
