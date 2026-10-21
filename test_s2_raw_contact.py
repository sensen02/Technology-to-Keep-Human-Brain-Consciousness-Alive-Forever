import tempfile
from pathlib import Path
import unittest
import numpy as np
from run_s2_raw_contact import pack_ragged
from run_s2_contact_clock import pair_world
from tools_s2_raw_contact import point_summary


class RawContactTests(unittest.TestCase):
    def test_rotated_frame_and_reverse_sign(self):
        frame=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        pair,force=pair_world(frame,[2.,3.,5.],0,26)
        self.assertEqual(pair,(0,26));np.testing.assert_array_equal(force,[-3.,2.,5.])
        pair,force=pair_world(frame,[2.,3.,5.],26,0)
        self.assertEqual(pair,(0,26));np.testing.assert_array_equal(force,[3.,-2.,-5.])

    def test_variable_efc_length_no_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays={}
            for key,dtype in (('force',float),('state',int),('type',int),('id',int)):
                offsets,data=pack_ragged([[1,2],[],[3],[4,5,6]],dtype=dtype)
                arrays['efc_'+key]=data;arrays['efc_offsets']=offsets
            path=Path(tmp)/'ragged.npz';np.savez_compressed(path,**arrays)
            with np.load(path,allow_pickle=False) as a:
                np.testing.assert_array_equal(a['efc_offsets'],[0,2,2,3,6])
                for key in ('force','state','type','id'):
                    self.assertNotEqual(a['efc_'+key].dtype.kind,'O')
                    np.testing.assert_array_equal(a['efc_'+key][2:3],[3])

    def test_empty_vector_contacts(self):
        offsets,data=pack_ragged([[],[[1,2]],[]],width=2,dtype=int)
        np.testing.assert_array_equal(offsets,[0,0,1,1]);self.assertEqual(data.shape,(1,2))

    def test_endpoint_local_pyramidal_rows(self):
        a=dict(contact_offsets=np.array([0,2]),efc_offsets=np.array([0,4]),contact_geom=np.array([[26,0],[0,25]]),contact_efc_address=np.array([0,-1]),contact_dim=np.array([3,3]),contact_exclude=np.array([0,1]),contact_dist=np.array([-.1,.2]),contact_local6=np.ones((2,6)),efc_state=np.array([1,1,2,2]),efc_type=np.array([6]*4),efc_id=np.array([0]*4))
        s=point_summary(a,0,(0,26),0);self.assertEqual(s['constraint_state'],{'1':2,'2':2});self.assertEqual(s['points'],1)
        s=point_summary(a,0,(0,25),0);self.assertEqual(s['constraint_state'],{})


if __name__=='__main__':unittest.main()
