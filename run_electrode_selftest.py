"""Analytic electrode tests; no biological calibration."""
import unittest
import numpy as np
from engine.electrode import Contact, BipolarField, VoltageRecorder, transfer_mV_per_nA
from engine.cable import Morphology, CableNeuron

class Tests(unittest.TestCase):
    def setUp(self):
        self.contacts=[Contact((0,0,0),2),Contact((100,0,0),2)]
        self.f=BipolarField(self.contacts)
    def test_external_point_source(self):
        x=np.array([[10,10,0],[40,5,5]])
        r=np.linalg.norm(x,axis=1)
        got=transfer_mV_per_nA(x,[self.contacts[0]],.3)[:,0]
        # Independent SI conversion: 1nA / (4*pi*sigma*r[m]) then V->mV.
        expected=1e-9/(4*np.pi*.3*r*1e-6)*1000
        np.testing.assert_allclose(got,expected)
    def test_finite_center_and_boundary(self):
        a=2; sig=.3
        v=transfer_mV_per_nA([[0,0,0],[a,0,0]],self.contacts[:1],sig)[:,0]
        np.testing.assert_allclose(v,[3/(8*np.pi*sig*a),1/(4*np.pi*sig*a)])
        eps=1e-4
        vals=transfer_mV_per_nA([[a-eps,0,0],[a,0,0],[a+eps,0,0]],self.contacts[:1],sig)[:,0]
        self.assertLess(abs((vals[1]-vals[0])-(vals[2]-vals[1])),1e-8)
    def test_poisson_inside(self):
        h=.001; sig=.3; a=2
        coords=[[0,0,0]]+[[h,0,0],[-h,0,0],[0,h,0],[0,-h,0],[0,0,h],[0,0,-h]]
        v=transfer_mV_per_nA(coords,self.contacts[:1],sig)[:,0]
        lap=(sum(v[1:])-6*v[0])/h**2
        self.assertAlmostEqual(lap,-3/(4*np.pi*sig*a**3),places=8)
    def test_reference_and_zero(self):
        x=[[0,5,0],[20,5,0],[80,5,0]]
        a=self.f.potential_mV(x,[1,-1])
        b=BipolarField(self.contacts,reference_um=(200,40,20)).potential_mV(x,[1,-1])
        np.testing.assert_allclose(np.diff(a),np.diff(b),atol=1e-14)
        np.testing.assert_array_equal(self.f.potential_mV(x,[0,0]),0)
        with self.assertRaises(ValueError): self.f.potential_mV(x,[1,0])
    def test_cable_drive_conservation(self):
        c=CableNeuron(Morphology.cylinder(100,1,nseg=20))
        a=self.f.inward_axial_drive_nA(c,[1,-1])
        self.assertAlmostEqual(a.sum(),0,places=13)
        b=BipolarField(self.contacts,reference_um=(200,40,20)).inward_axial_drive_nA(c,[1,-1])
        np.testing.assert_allclose(a,b,atol=1e-14)
    def test_two_node_polarization_sign(self):
        c=CableNeuron(Morphology([0,100],[0,0],[0,0],[1,1],[-1,0]))
        ve=self.f.potential_mV([[0,0,0],[100,0,0]],[1,-1])
        self.assertGreater(ve[0],ve[1])
        drive=self.f.inward_axial_drive_nA(c,[1,-1])
        self.assertLess(drive[0],0)
        self.assertGreater(drive[1],0)
        v=c.step(i_inject=drive)
        self.assertLess(v[0],c.E_leak)
        self.assertGreater(v[1],c.E_leak)
    def test_recorder_filter_artifact(self):
        r=VoltageRecorder((0,5,0),(100,5,0),bandwidth_Hz=100)
        v=self.f.potential_mV(r.xyz,[1,-1]); target=v[0]-v[1]
        for _ in range(100): got=r.step(.1,stimulus_field=self.f,stimulus_currents_nA=[1,-1])
        self.assertAlmostEqual(got['measured_mV'],target*(1-np.exp(-2*np.pi)),places=12)
        self.assertEqual(got['neural_unfiltered_mV'],0)
    def test_source_and_seed(self):
        args=((0,5,0),(100,5,0))
        a=VoltageRecorder(*args,noise_sd_mV=.01,seed=1)
        b=VoltageRecorder(*args,noise_sd_mV=.01,seed=1)
        for _ in range(4):
            self.assertEqual(a.step(.1,self.contacts,[1,-1]),b.step(.1,self.contacts,[1,-1]))
    def test_validation(self):
        for a in (0,-1,np.nan):
            with self.assertRaises(ValueError): Contact((0,0,0),a)
        with self.assertRaises(ValueError): transfer_mV_per_nA([[0,0,0]],self.contacts,0)

if __name__=='__main__': unittest.main(verbosity=2)
