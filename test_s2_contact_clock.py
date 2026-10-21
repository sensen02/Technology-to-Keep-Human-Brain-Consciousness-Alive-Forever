"""Focused contact rotation/sign and real compiled joint-address regressions."""
import unittest
import numpy as np
import mujoco
from run_s2_contact_clock import pair_world, dofs

class ContactClockTests(unittest.TestCase):
    def test_identity_sign(self):
        for g1,g2,sign in ((2,9,1),(9,2,-1)):
            pair,f=pair_world(np.eye(3),[2,3,4],g1,g2)
            self.assertEqual(pair,(2,9));np.testing.assert_array_equal(f,sign*np.array([2,3,4]))

    def test_nonsymmetric_rotation_and_reverse(self):
        # Rows are world normal/tangents: local x->world y; local y->world -x.
        frame=np.array([[0,1,0],[-1,0,0],[0,0,1]])
        pair,f=pair_world(frame.ravel(),[2,3,4],2,9)
        np.testing.assert_array_equal(f,[-3,2,4])
        _,rev=pair_world(frame,[2,3,4],9,2)
        np.testing.assert_array_equal(rev,[3,-2,-4])

    def model(self):
        return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body><freejoint name="free"/><geom type="sphere" size=".1"/></body>
        <body pos="1 0 0"><joint name="ball" type="ball"/><geom type="sphere" size=".1"/>
        <body pos="0 1 0"><joint name="slide" type="slide"/><geom type="sphere" size=".1"/>
        <body pos="0 1 0"><joint name="hinge" type="hinge"/><geom type="sphere" size=".1"/></body></body></body>
        </worldbody></mujoco>''')

    def test_compiled_dof_not_qpos_addresses(self):
        m=self.model();labels=dofs(m,mujoco)
        self.assertEqual(m.nq,13);self.assertEqual(m.nv,11)
        self.assertEqual([r['dof'] for r in labels],list(range(11)))
        self.assertEqual([r['joint'] for r in labels],['free']*6+['ball']*3+['slide','hinge'])
        self.assertEqual([r['unit'] for r in labels],['model_length']*3+['rad']*6+['model_length','rad'])
        self.assertEqual(int(m.jnt_qposadr[2]),11);self.assertEqual(labels[9]['component'],0)

    def test_quaternion_tangent_roundtrip(self):
        m=self.model();q=m.qpos0.copy();v=np.arange(1,m.nv+1)*.001
        b=q.copy();mujoco.mj_integratePos(m,b,v,1.)
        delta=np.zeros(m.nv);mujoco.mj_differentiatePos(m,delta,1.,q,b)
        np.testing.assert_allclose(delta,v,rtol=0,atol=1e-14)
        # Antipodal quaternion is the same pose, not a 4-component difference.
        b[3:7]*=-1;b[7:11]*=-1
        mujoco.mj_differentiatePos(m,delta,1.,q,b)
        np.testing.assert_allclose(delta,v,rtol=0,atol=1e-14)

if __name__=='__main__':unittest.main()
