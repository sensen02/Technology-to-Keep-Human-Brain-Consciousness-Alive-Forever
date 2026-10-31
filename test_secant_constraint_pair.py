"""Fast offline S122 verifier tests; never import a runner or physics."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import verify_secant_constraint_pair as v


class PairTests(unittest.TestCase):
    def test_offline_import(self):
        self.assertFalse(any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))

    def test_canonical_conflict_rejected(self):
        with self.assertRaises(ValueError): v.canonical({'/tmp/a/../b':'one','/tmp/b':'two'})

    def test_event_activity_not_force(self):
        a={'time':np.arange(3)*.001,'load_ratio':np.array([1.,.7,.6]),'qpos':np.array([[0.],[1.],[1.1]])}
        m={'jnt_limited':[True],'jnt_qposadr':[0],'jnt_range':[[0.,1.]]}
        item=dict(arrays=a,moment=m,states=[dict(active_limit_rows=[],nonzero_limit_force_rows=[]),dict(active_limit_rows=[1],nonzero_limit_force_rows=[]),dict(active_limit_rows=[1],nonzero_limit_force_rows=[1])])
        with patch.object(v,'all42_limits',return_value=([],np.zeros(3),{})):
            e=v.events(item)
        self.assertEqual(e['first_limit_row_activity']['sample'],1)
        self.assertEqual(e['first_nonzero_limit_force']['sample'],2)
        self.assertEqual(e['first_upper_cross_gt_1e_minus6']['sample'],2)

    def test_components_partition(self):
        p=np.arange(2*len(v.CATEGORIES)*48,dtype=float).reshape(2,len(v.CATEGORIES),48)
        f=np.ones((2,4)); moment=np.ones((2,4,48))
        a=dict(actuator_force=f,actuator_moment=moment)
        for key in ('qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_applied','qfrc_constraint','Ma','qacc'): a[key]=np.zeros((2,48))
        c=v.components(dict(arrays=a,projections=p))
        np.testing.assert_array_equal(c['contact']+c['limit']+c['other'],c['total'])
        np.testing.assert_array_equal(c['act2']+c['othermuscle'],np.full((2,48),4.))

    def test_sparse_jacobian_actual_address(self):
        z={'s_efc_J_dense':np.array([[0.,3.,0.],[4.,0.,5.]]),'s_efc_J_raw':np.array([3.,4.,5.]),'s_efc_J_rowadr':np.array([0,1]),'s_efc_J_rownnz':np.array([1,2]),'s_efc_J_colind':np.array([1,0,2])}
        checks=v.Checks(); J=v.jacobian(z,'s_',2,3,True,checks)
        self.assertTrue(checks.passed); np.testing.assert_array_equal(J,z['s_efc_J_dense'])


if __name__=='__main__': unittest.main()
