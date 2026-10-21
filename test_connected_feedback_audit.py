"""Synthetic summand-removal tests plus read-only verified artifact acceptance."""
import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import run_connected_feedback_audit as a
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities
from engine.embodied.audit_provenance import inspect_audit


class SelectiveTests(unittest.TestCase):
    def test_overlap_preserves_other_summands_and_records(self):
        enc=ChannelSensoryEncoder({'lf':{'claw_position':[0], 'hook_direction':[1],
            'hairplate_angle':[2], 'campaniform_strain':[3], 'bristle_touch':[4]}},5)
        quantities={'lf':ChannelQuantities(tibia_joint_dev_rad=.3,tibia_joint_vel_rad_s=20,
                                          multi_joint_dev_rad=-.3,contact_force_norm=4,foot_contact=1)}
        disjoint,record=enc.encode(quantities)
        # Only the subtraction fixture overlaps: combine the disjoint summands
        # after encoding, retaining the full encoder record (one cell per channel).
        groups=copy.deepcopy(enc.groups)
        for channel in ('claw_position','hook_direction','campaniform_strain','bristle_touch'):
            groups['lf'][channel]=np.array([0])
        groups['lf']['hairplate_angle']=np.array([1])
        current=np.array([disjoint[[0,1,3,4]].sum(),disjoint[2],0.])
        saved=copy.deepcopy(record)
        out,modified=a.subtract_channels(current,record,groups,a.PROPRIO)
        np.testing.assert_allclose(out,[1.5,0,0],atol=1e-15)
        self.assertEqual(record,saved)
        self.assertEqual({k:v for k,v in modified.items() if k!='intervention'},saved)
        out2,_=a.subtract_channels(current,record,groups,('campaniform_strain',))
        np.testing.assert_allclose(out2,[1.2,-.3,0],atol=1e-15)
        self.assertFalse(a.overlap_audit(groups)['disjoint'])

    def test_zero_selected_and_empty_unchanged(self):
        enc=ChannelSensoryEncoder({'lf':{'claw_position':[0],'bristle_touch':[1]}},2)
        disjoint,record=enc.encode({'lf':ChannelQuantities(foot_contact=1)})
        # The selected zero-valued claw shares the nonzero touch summand only
        # in this synthetic subtraction fixture, never in the production encoder.
        groups=copy.deepcopy(enc.groups)
        groups['lf']['bristle_touch']=np.array([0])
        current=np.array([disjoint.sum(),0.])
        self.assertFalse(a.overlap_audit(groups)['disjoint'])
        for selected in ((),a.PROPRIO,('campaniform_strain',)):
            out,_=a.subtract_channels(current,record,groups,selected)
            self.assertEqual(out.tobytes(),current.tobytes())

    def test_production_encoder_rejects_overlapping_groups(self):
        for groups,n in (
            ({'lf':{'claw_position':[0], 'hook_direction':[0], 'hairplate_angle':[1],
                    'campaniform_strain':[0], 'bristle_touch':[0]}},3),
            ({'lf':{'claw_position':[0],'bristle_touch':[0]}},2),
        ):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError,'neuron index overlap'):
                    ChannelSensoryEncoder(groups,n)

    def test_context_restores_on_exception(self):
        original=a.clo.ChannelSensoryEncoder
        with self.assertRaises(RuntimeError):
            with a.intervention(a.PROPRIO):
                self.assertIsNot(a.clo.ChannelSensoryEncoder,original)
                raise RuntimeError('test')
        self.assertIs(a.clo.ChannelSensoryEncoder,original)

    def test_first_difference_timestamp_semantics(self):
        x=np.zeros((10,2)); y=x.copy(); y[2,0]=1
        self.assertEqual(a.first_difference(x,y)['first_observable_saved_time_s'],.015)
        self.assertEqual(a.first_difference(x,y,True)['first_observable_saved_time_s'],.01)
        self.assertIsNone(a.first_difference(x,x))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((a.OUT/'connected_feedback_audit.json').read_text())

    def test_provenance_bundle_sources_and_traces(self):
        d=self.audit
        evidence=inspect_audit(Path(__file__).resolve().parent, 'connected_feedback_audit')
        bundle=evidence['source_bundle']
        self.assertEqual(evidence['source_snapshot'], 'unavailable')
        self.assertEqual(evidence['run_sources'][str(Path(__file__).resolve().parent/'run_neural_muscle_closure.py')]['status'], 'drift')
        self.assertEqual(bundle['run_id'],a.EXPECTED_RUN)
        self.assertEqual(bundle['manifest']['files'],d['source_bundle']['files'])
        self.assertEqual(a.sha(Path(bundle['path'])/'manifest.json'),d['source_bundle']['manifest_sha256'])
        self.assertEqual(a.sha(a.clo.cp.CANONICAL_XML),d['provenance']['canonical_xml_sha256'])
        for p,h in d['provenance']['inputs'].items(): self.assertEqual(a.sha(p),h)
        self.assertEqual(a.sha(d['traces']['path']),d['traces']['sha256'])

    def test_shapes_health_repeats_and_source_exact(self):
        d=self.audit
        with np.load(d['traces']['path'],allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
        with np.load(Path(d['source_bundle']['path'])/'closure_traces.npz',allow_pickle=False) as z:
            source={k:z[k] for k in z.files}
        nq=len(arrays['seated_qpos']); nu=arrays['normal_controls'].shape[1]
        model=SimpleNamespace(nq=nq,nv=nq-1,nu=nu,na=nu)
        n=arrays['normal_spike_counts'].shape[1]
        self.assertEqual(a.validate_experiment_arrays(arrays,d['episodes'],n,model),d['array_shapes'])
        for suffix in a.SUFFIXES:
            np.testing.assert_array_equal(arrays['normal_'+suffix],source['normal_'+suffix])
            np.testing.assert_array_equal(arrays['normal_'+suffix],source['normal_repeat_'+suffix])
            np.testing.assert_array_equal(arrays['cut_proprio_'+suffix],arrays['cut_proprio_repeat_'+suffix])
        broken=dict(arrays); broken['cut_proprio_controls']=broken['cut_proprio_controls'][:-1]
        with self.assertRaises(AssertionError): a.validate_experiment_arrays(broken,d['episodes'],n,model)

    def test_touch_coverage_records_and_com_distinct(self):
        d=self.audit
        self.assertEqual(d['live_episode_count'],4)
        for arm,flags in d['external_lf_touch_flags'].items():
            self.assertEqual(flags,[False,True,True]+[False]*7)
            samples=d['height_samples'][arm]
            self.assertEqual(len(samples),11)
            self.assertTrue(all(np.isfinite(r['whole_body_com_z_model']) for r in samples))
            self.assertNotEqual(samples[0]['root_z_model'],samples[0]['whole_body_com_z_model'])
            for record in d['episodes'][arm]['records']:
                self.assertTrue(record['channel_record']['intervention']['original_physiological_record_preserved'])
        for leg,row in d['structural_coverage']['channels'].items():
            self.assertEqual(row['bristle_touch']['reachable_motor_cells'],0)
            for channel in a.PROPRIO: self.assertEqual(row[channel]['reachable_motor_cells'],202)
        self.assertTrue(d['contextual_cut_all']['not_a_substitute_for_selective_cuts'])


if __name__=='__main__': unittest.main()
