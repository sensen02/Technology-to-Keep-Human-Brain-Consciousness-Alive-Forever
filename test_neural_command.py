"""Pure arrays only: no body/neural simulation. Synthetic IDs are NOT fly claims."""
import unittest
from dataclasses import replace
import numpy as np
from engine.embodied.neural_command import SourcedNeuronTarget, TonicNeuronCommand


class CommandTests(unittest.TestCase):
    def make(self, **changes):
        args=dict(tier_root_ids=['11','9007199254740993','33'],
                  tier_cell_types=['sensory','TEST_ONLY','motor'],
                  tier_super_classes=['sensory','descending','motor'],
                  dataset='synthetic-fixture',dataset_sha256='a'*64,
                  manifest_dataset='synthetic-fixture',manifest_sha256='a'*64,
                  targets=[SourcedNeuronTarget('9007199254740993','TEST_ONLY','descending',
                                               'synthetic identity fixture','synthetic function fixture')],
                  amplitude_nA=.3,onset_s=.05,offset_s=.15)
        args.update(changes)
        return TonicNeuronCommand(**args)

    def test_half_open_timing(self):
        c=self.make()
        for t in (0.,.049999,.15,.2): np.testing.assert_array_equal(c.current(t),[0,0,0])
        for t in (.05,.1,.149999): np.testing.assert_array_equal(c.current(t),[0,.3,0])

    def test_composition_records_and_no_mutation(self):
        c=self.make(); s=np.array([-0.,2.,3.]); before=s.tobytes()
        out=c.compose(s,.1)
        self.assertEqual(s.tobytes(),before)
        np.testing.assert_array_equal(out['total_current_nA'],[-0.,2.3,3.])
        self.assertEqual(out['total_current_nA'][0].tobytes(),s[0].tobytes())
        for t in (0.,.2):self.assertEqual(c.compose(s,t)['total_current_nA'].tobytes(),before)
        out['sensory_current_nA'][1]=99
        self.assertEqual(s[1],2)

    def test_dataset_pin_and_identity_rejection(self):
        for change in (dict(manifest_dataset='wrong'),dict(manifest_sha256='b'*64),
                       dict(dataset_sha256='x'*64),dict(tier_root_ids=['11','11','33']),
                       dict(tier_root_ids=[11.,22.,33.]),dict(tier_root_ids=['11','22','33']),
                       dict(tier_cell_types=['sensory','wrong','motor']),
                       dict(tier_super_classes=['sensory','motor','motor']),dict(targets=[]),
                       dict(tier_cell_types=['x'])):
            with self.subTest(change=change),self.assertRaises(ValueError):self.make(**change)

    def test_target_provenance_required(self):
        t=SourcedNeuronTarget('9007199254740993','TEST_ONLY','descending','identity','function')
        for field in ('identity_source','functional_source','cell_type','super_class'):
            with self.assertRaises(ValueError):self.make(targets=[replace(t,**{field:''})])
        with self.assertRaises(ValueError):self.make(targets=[t,t])

    def test_invalid_parameters(self):
        for value in (True,'1',float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):self.make(amplitude_nA=value)
        for onset,offset in ((-1,1),(1,1),(2,1)):
            with self.assertRaises(ValueError):self.make(onset_s=onset,offset_s=offset)
        for t in (-1,True,float('nan'),'0'):
            with self.assertRaises(ValueError):self.make().current(t)

    def test_invalid_current_and_overflow(self):
        for s in (np.ones(3,dtype=int),np.ones(2),np.ones((3,1)),np.array([0,np.nan,0])):
            with self.assertRaises(ValueError):self.make().compose(s,.1)
        with self.assertRaises(ValueError):
            self.make(amplitude_nA=1e308).compose(np.full(3,1e308),.1)

    def test_metadata_independent_and_zero_command(self):
        c=self.make(amplitude_nA=0)
        s=np.array([0.,-0.,3.]);self.assertEqual(c.compose(s,.1)['total_current_nA'].tobytes(),s.tobytes())
        m=c.describe();m['targets'].clear();self.assertEqual(len(c.describe()['targets']),1)
        self.assertEqual(c.describe()['targets'][0]['root_id'],'9007199254740993')

if __name__=='__main__':unittest.main()
