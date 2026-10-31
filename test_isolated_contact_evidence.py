"""Exercises actual export code with algebra-only fake API, NOT real API validation."""
import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import implement_isolated_contact_reconstruction as target
from test_isolated_contact_reconstruction import FakeBackend, states


class AlgebraAPI:
    mjtObj=NS(mjOBJ_GEOM=1,mjOBJ_BODY=2)
    def __init__(self):self.calls=[]
    def mj_id2name(self,m,obj,i):return '%d_%d'%(obj,i)
    def mj_contactForce(self,m,d,i,out):self.calls.append('contactForce');out[:]=[4,2,3,1,2,3]
    def mj_jac(self,m,d,jp,jr,point,body):
        self.calls.append('jac')
        jp[:]=np.array([[1,0],[0,1],[1,1]])*body
        jr[:]=0
    def mj_mulM(self,m,d,out,acc):self.calls.append('mulM');out[:]=np.diag([2.,3.])@acc
    def mj_applyFT(self,m,d,f,t,p,body,out):
        self.calls.append('applyFT');out[:]+=np.array([f[0]+t[2],f[1]])


def fixture():
    m=NS(nv=2,nbody=2,geom_bodyid=np.array([0,1]),opt=NS(cone=0))
    # R rotates world axes: normal world +Y, tangent1 world -X.
    c=NS(geom1=0,geom2=1,frame=np.array([0,1,0,-1,0,0,0,0,1.]),pos=np.zeros(3),
         dist=-.01,friction=np.array([.7,.8,.01,.02,.03]),dim=3,exclude=0,efc_address=0,includemargin=.02)
    d=NS(ncon=1,contact=[c],time=0.,warning=[],xipos=np.zeros((2,3)))
    for k in ['qpos','act','ctrl','actuator_force','actuator_length','actuator_velocity']:setattr(d,k,np.zeros(2))
    d.qvel=np.array([2.,3.]);d.qacc=np.array([5.,7.])
    d.qfrc_passive=np.array([1.,2.]);d.qfrc_actuator=np.array([3.,4.]);d.qfrc_constraint=np.array([2.,3.]);d.qfrc_applied=np.array([1.,2.]);d.qfrc_bias=np.array([2.,1.])
    d.xfrc_applied=np.array([[0.,0,0,0,0,0],[4.,11,0,0,0,1.]])
    # RHS=[1+3+2+1+5-2,2+4+3+2+11-1]=[10,21]=Ma.
    return m,d


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        original=builtins.__import__
        def forbidden(name,*a,**k):
            if name.split('.')[0]=='mujoco':raise AssertionError('physics import forbidden')
            return original(name,*a,**k)
        self.hook=patch('builtins.__import__',forbidden);self.hook.start()
    def tearDown(self):self.hook.stop()
    def export(self,m,d):
        b=target.RealBackend();b.mj=AlgebraAPI();return b.export(m,d)
    def test_contact_sign_frame_and_effective_parameters(self):
        m,d=fixture();o=self.export(m,d);c=o['contacts'][0]
        np.testing.assert_allclose(c['relative_velocity_world_2_minus_1'],[2,3,5])
        np.testing.assert_allclose(c['relative_velocity_contact'],[3,-2,5])
        np.testing.assert_allclose(c['world_force_on_geom2'],[-2,4,3])
        np.testing.assert_allclose(c['world_force_on_geom1'],[2,-4,-3])
        self.assertEqual(c['friction'],[.7,.8,.01,.02,.03]);self.assertEqual(c['dim'],3)
        self.assertEqual(c['force_torque'],[4,2,3,1,2,3])
    def test_swapped_bodies_reverses_relative_velocity(self):
        m,d=fixture();d.contact[0].geom1=1;d.contact[0].geom2=0
        np.testing.assert_allclose(self.export(m,d)['contacts'][0]['relative_velocity_world_2_minus_1'],[-2,-3,-5])
    def test_known_mass_external_load_balanced(self):
        m,d=fixture();r=self.export(m,d)['dynamics']
        np.testing.assert_allclose(r['mapped_xfrc_applied'],[5,11])
        np.testing.assert_allclose(r['mass_times_qacc'],[10,21])
        np.testing.assert_allclose(r['residual_per_dof'],[0,0])
    def test_unbalanced_and_normalization(self):
        m,d=fixture();d.qacc[0]+=1;r=self.export(m,d)['dynamics']
        np.testing.assert_allclose(r['residual_per_dof'],[2,0])
        np.testing.assert_allclose(r['normalization_per_dof'],[14,23])
        self.assertAlmostEqual(r['normalized_residual_per_dof'][0],2/14)
    def test_world_wrench_rejected(self):
        m,d=fixture();d.xfrc_applied[0,0]=1
        with self.assertRaises(target.Refused):self.export(m,d)
    def test_missing_external_input_rejected(self):
        m,d=fixture();del d.xfrc_applied
        with self.assertRaises(AttributeError):self.export(m,d)
    def test_export_failure_does_not_reforward(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';b=FakeBackend(out)
            def broken(m,d):raise RuntimeError('forensics failure')
            b.export=broken
            with self.assertRaises(RuntimeError):target.run(out,{'model_load':1,'forward':3},{},'',states(),b)
            self.assertEqual(b.calls,['model_load','forward'])
            l=json.loads((out/'LEDGER.json').read_text());self.assertEqual(l['status'],'STOPPED')
            self.assertEqual(l['completed']['forward'],1)
            self.assertFalse(list(out.glob('STATE*')))
    def test_export_static_call_allowlist(self):
        tree=ast.parse(Path(target.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RealBackend');exp=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='export')
        calls=[ast.unparse(n.func) for n in ast.walk(exp) if isinstance(n,ast.Call)]
        api={c for c in calls if c.startswith('self.')}
        self.assertEqual(api,{'self.mj.mj_contactForce','self.mj.mj_jac','self.mj.mj_id2name','self.mj.mj_mulM','self.mj.mj_applyFT'})
        self.assertFalse(any(any(s in c for s in ['forward','step','load_model','neural','MjModel','MjData']) for c in calls))

if __name__=='__main__':unittest.main(verbosity=2)
