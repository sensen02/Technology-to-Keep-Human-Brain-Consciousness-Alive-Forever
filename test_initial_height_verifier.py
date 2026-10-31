"""Offline verifier regression tests; no new physics or neural ticks."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import verify_initial_height_probe as v


class HeightVerifierTests(unittest.TestCase):
    def fixture(self):
        meta=dict(geom_names=['floor','lf_tars5','lm_tars5','other'],body_names=['world','lf_tars5','lm_tars5','thorax'],geom_bodyid=[0,1,2,3],floor_geom_ids=[0])
        a=dict(time=np.array([0.]),contact_force_raw6=np.array([[[2.,0,0,0,0,0],[4.,0,0,0,0,0],[8.,0,0,0,0,0]]]),
            contact_frame=np.array([[[[0.,0,1],[1,0,0],[0,1,0]]]*3]),contact_ncon=np.array([3]),
            contact_geom_ids=np.array([[[0,1],[0,2],[0,3]]]),contact_body_ids=np.array([[[0,1],[0,2],[0,3]]]),
            contact_dist=np.array([[-.2,-.4,-.01]]),contact_pos_world=np.zeros((1,3,3)))
        return a,meta

    def test_independent_minimum_only_carrying_feet(self):
        a,m=self.fixture(); dz,feet=v.derive_height(a,m)
        self.assertEqual(dz,.1); self.assertEqual([s['contact_index'] for s in feet['selected']],[0,1])
        self.assertEqual(feet['contacts'].tolist(),[[1,1,0,0,0,0]])
        self.assertEqual(feet['foot_force'][0,:,2].tolist(),[2,4,0,0,0,0])

    def test_exclude_noncarrying_smallest_depth(self):
        a,m=self.fixture(); a['contact_force_raw6'][0,0,0]=-2
        self.assertEqual(v.derive_height(a,m)[0],.2)

    def test_no_arbitrary_fallback(self):
        a,m=self.fixture(); a['contact_dist'][:]=1
        with self.assertRaises(ValueError): v.derive_height(a,m)

    def test_require_normal_plusz(self):
        a,m=self.fixture(); a['contact_frame'][0,0,0]=[1.,0.,1.]
        with self.assertRaises(ValueError): v.derive_height(a,m)

    def test_pair_multiset_not_point_identity(self):
        a,m=self.fixture(); first=v.raw_feet(a,m); a['contact_pos_world']+=.3
        self.assertEqual(first['initial_pair_multiset'],v.raw_feet(a,m)['initial_pair_multiset'])

    def test_geom_order_reversed_sign(self):
        a,m=self.fixture(); a['contact_geom_ids'][0,0]=[1,0]; a['contact_body_ids'][0,0]=[1,0]
        a['contact_frame'][0,0,0]*=-1
        dz,feet=v.derive_height(a,m)
        self.assertEqual(dz,.1); self.assertEqual(feet['foot_force'][0,0,2],2)

    def test_differences_record_not_gate(self):
        rows=v.differences({'features':np.ones((2,3)), 'empty':np.array([])}, {'features':np.zeros((2,3)),'empty':np.array([])})
        self.assertFalse(rows['features']['equal']); self.assertEqual(rows['features']['different_elements'],6)
        self.assertTrue(rows['empty']['equal'])

    def test_canonical_symlink_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); real=base/'real'; real.write_text('immutable'); link=base/'alias'; link.symlink_to(real)
            checks=v.Checks(); v.canonical_checks(checks,{str(link):v.sha(real)},'symlink',{})
            self.assertTrue(checks.passed)
            checks=v.Checks(); v.canonical_checks(checks,{str(link):v.sha(real),str(real):v.sha(real)},'duplicate',{})
            self.assertFalse(checks.passed)

    def test_inert_import(self):
        result=subprocess.run([sys.executable,'-B','-c',"import sys; import verify_initial_height_probe; assert 'mujoco' not in sys.modules and 'torch' not in sys.modules"],cwd=v.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_real_reference_dz_and_order(self):
        a=v.npz(v.REFERENCE/'TRAJECTORY.npz'); m=v.js(v.REFERENCE/'COM_CONTACT_METADATA.json')
        dz,feet=v.derive_height(a,m); self.assertEqual(dz,.010312318543883363)
        self.assertEqual(len(feet['selected']),18)
        np.testing.assert_allclose(feet['foot_force'],a['foot_force'],rtol=0,atol=2e-12)
        np.testing.assert_array_equal(feet['contacts'],a['contacts'])

    def test_balance_retains_original_indices_and_no_coarsegate(self):
        t=np.arange(121)*.000025; velocity=np.column_stack((np.zeros(121),np.zeros(121),t))
        r=dict(momentum=velocity,external_force=np.tile([0.,0.,1.],(121,1)),ma=np.tile([0.,0.,1.],(121,1)),mass=1.,velocity=velocity)
        b=v.balance({'time':t},r)
        self.assertEqual(b['impulse']['sample_indices'],list(range(69)))
        self.assertEqual(b['momentum_FD']['sample_indices'],list(range(1,69)))
        self.assertTrue(b['momentum_FD']['passed']); self.assertTrue(b['impulse']['passed'])

    def test_pose_source_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runner.py'; p.write_text('d.qpos[:] = q\nd.ctrl[:] = u\n')
            checks=v.Checks(); v.pose_source_check(p,checks); self.assertTrue(checks.passed)
            p.write_text('d.qpos[:] = q\nd.qpos[2] += dz\n')
            checks=v.Checks(); v.pose_source_check(p,checks); self.assertFalse(checks.passed)

    def test_reject_lower_initial_load_even_support_improves(self):
        old=dict(first_failures_ms={'load':1.7},extrema={k:{'max':1.} for k in ('angle_rad','nonfoot_depth','limit_excess')},
            max_downward_root_velocity=1.,max_downward_COM_velocity=1.,min_height_own_initial=.9,min_height_reference_original=.9,
            initial_foot_Fz=10.,all_gates_support_ms=1.6,longest_all_gates_true_endpoint_span_ms=1.5)
        new=copy.deepcopy(old); new.update(initial_foot_Fz=9.,all_gates_support_ms=2.)
        self.assertTrue(v.decision(old,new)['reject']); self.assertFalse(v.decision(old,new)['suggest_candidate'])
        new=copy.deepcopy(old); new['all_gates_support_ms']=2.
        self.assertTrue(v.decision(old,new)['suggest_candidate'])
        new['extrema']['limit_excess']['max']=1.01
        self.assertTrue(v.decision(old,new)['reject'])


if __name__=='__main__': unittest.main()
