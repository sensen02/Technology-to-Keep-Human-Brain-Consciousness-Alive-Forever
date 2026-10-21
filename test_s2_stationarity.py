import copy
import unittest
import numpy as np
from tools_s2_stationarity import analyze, UNITS


def fixture():
    M=np.array([[2.,1.],[1.,3.]])
    J=np.array([[1.,-2.],[3.,1.]])
    a=np.array([2.,-1.]); f=np.array([.5,-1.]); smooth=np.array([1.,2.])
    return dict(M=M,J=J,qacc=a,qacc_smooth=np.linalg.solve(M,smooth),qfrc_smooth=smooth,
                qfrc_constraint=np.array([-2.5,-2.]),force=f,dof_units=list(UNITS))


class StationarityTests(unittest.TestCase):
    def test_independent_scalar_oracle(self):
        x=fixture();z=analyze(**x)
        mv=lambda m,v:np.array([sum(float(a)*float(b) for a,b in zip(row,v)) for row in m])
        ma=mv(x['M'],x['qacc']);mapped=mv(x['J'].T,x['force'])
        np.testing.assert_array_equal(z['arrays']['Ma'],ma)
        np.testing.assert_array_equal(z['arrays']['mapped'],mapped)
        np.testing.assert_array_equal(z['arrays']['g_rows'],ma-x['qfrc_smooth']-mapped)
        np.testing.assert_allclose(z['arrays']['smooth_error'],0,atol=1e-15)
        np.testing.assert_array_equal(z['arrays']['closure_error'],[0,0])
        self.assertEqual(z['groups'][UNITS[0]]['g_rows']['count'],1)
        self.assertEqual(z['groups'][UNITS[1]]['g_rows']['count'],1)
        self.assertEqual(z['policy'],'REPORT_ONLY')

    def test_mapping_error_separate(self):
        x=fixture();x['qfrc_constraint']+=np.array([7.,-4.]);z=analyze(**x)
        np.testing.assert_array_equal(z['arrays']['mapping_error'],[7,-4])
        np.testing.assert_array_equal(z['arrays']['g_rows']-z['arrays']['g_direct'],[7,-4])

    def test_empty_constraints_zero_denominator(self):
        x=dict(M=np.eye(2),J=np.empty((0,2)),qacc=[0,0],qacc_smooth=[0,0],qfrc_smooth=[0,0],qfrc_constraint=[0,0],force=[],dof_units=[UNITS[0]]*2)
        z=analyze(**x)
        self.assertEqual(z['groups'][UNITS[0]]['g_rows']['relative_undefined'],2)
        self.assertIsNone(z['groups'][UNITS[0]]['g_rows']['relative_absmax'])
        self.assertIsNone(z['groups'][UNITS[1]]['g_rows']['absmax'])

    def test_equation_specific_denominators(self):
        x=fixture();x['qfrc_constraint']=np.array([100.,200.]);z=analyze(**x)
        np.testing.assert_array_equal(z['denominators']['mapping_error'],[100,200])
        np.testing.assert_array_equal(z['denominators']['g_direct'],[100,200])
        self.assertLess(z['denominators']['g_rows'][0],100)
        np.testing.assert_allclose(z['denominators']['smooth_error'],[1,2],atol=1e-15)

    def test_no_alias_or_mutation(self):
        x=fixture();before=copy.deepcopy(x);z=analyze(**x)
        for key in z['arrays']:z['arrays'][key][:]=99
        for key in z['denominators']:z['denominators'][key][:]=99
        for key in x:np.testing.assert_array_equal(x[key],before[key])

    def test_invalid_mass(self):
        for M in (np.eye(3),np.array([[1,2],[0,1]]),np.array([[1,2],[2,1]]),np.zeros((2,2)),np.empty((0,0)),np.array([1,2])):
            x=fixture();x['M']=M
            with self.subTest(M=M),self.assertRaises(ValueError):analyze(**x)

    def test_bad_fields(self):
        for key,value in [('qacc',[np.nan,1]),('qacc',[1]),('J',[[1,2,3]]),('force',[1]),('qfrc_smooth',[[1,2]]),('M',np.eye(2,dtype=complex)),('qacc',['1','2']),('dof_units',['N','Nm']),('dof_units',[UNITS[0]])]:
            x=fixture();x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analyze(**x)

    def test_scaled_rms_preserves_tiny_nonzero(self):
        x=dict(M=np.eye(1),J=np.empty((0,1)),qacc=[1e-200],qacc_smooth=[0],qfrc_smooth=[0],qfrc_constraint=[0],force=[],dof_units=[UNITS[0]])
        result=analyze(**x)['groups'][UNITS[0]]['g_rows']
        self.assertEqual(result['rms'],1e-200)
        self.assertEqual(result['relative_absmax'],1)

    def test_overflow_rejected(self):
        x=fixture();x['qacc']=np.array([1e308,1e308])
        with self.assertRaises(ValueError):analyze(**x)


if __name__=='__main__':unittest.main()
