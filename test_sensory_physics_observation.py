"""Pure measurement-contract tests, not a MuJoCo dynamics or biology test."""
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S
import numpy as np
from run_neural_muscle_closure import observe_quantities
class ObservationTests(unittest.TestCase):
    def test_addresses_signed_mean_contact_norm_and_no_mutation(self):
        model=S(geom_type=np.array([1,0,1,1]))
        data=S(qpos=np.array([9.,-.2,.4]),qvel=np.array([-7.,99.]),ncon=3,
               contact=[S(geom1=0,geom2=1),S(geom1=1,geom2=0),S(geom1=2,geom2=3)])
        q=data.qpos.copy();v=data.qvel.copy();calls=[]
        def force(m,d,index,out):
            calls.append(index);out[:]=[3,4,0,900,900,900]
        fake=S(mjtGeom=S(mjGEOM_PLANE=0),mj_contactForce=force)
        with patch.dict(sys.modules,{'mujoco':fake}):
            r=observe_quantities(model,data,np.array([9.,.1,.1]),{'lf':{'q':[1,2],'tib_q':1,'tib_v':0}},{0:'lf'})['lf']
        self.assertAlmostEqual(r.tibia_joint_dev_rad,-.3)
        self.assertEqual(r.tibia_joint_vel_rad_s,-7.)
        self.assertAlmostEqual(r.multi_joint_dev_rad,0.)
        self.assertEqual(r.contact_force_norm,10.)
        self.assertEqual(r.foot_contact,1.)
        self.assertEqual(calls,[0,1])
        np.testing.assert_array_equal(q,data.qpos);np.testing.assert_array_equal(v,data.qvel)
    def test_no_contact_is_zero_not_background_force(self):
        model=S(geom_type=np.array([1]));data=S(qpos=np.zeros(1),qvel=np.zeros(1),ncon=0,contact=[])
        with patch.dict(sys.modules,{'mujoco':S(mjtGeom=S(mjGEOM_PLANE=0))}):
            r=observe_quantities(model,data,np.zeros(1),{'lf':{'q':[0],'tib_q':0,'tib_v':0}},{0:'lf'})['lf']
        self.assertEqual(r.contact_force_norm,0.);self.assertEqual(r.foot_contact,0.)
if __name__=='__main__':unittest.main(verbosity=2)
