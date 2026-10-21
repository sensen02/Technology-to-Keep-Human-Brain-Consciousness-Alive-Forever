import copy
import unittest
import numpy as np
from tools_s2_final_state_contract import validate_capture


def fixture():
    meta = dict(mujoco_version='3.9.0', solver=2, noslip_iterations=0,
                islands_enabled=True, nefc=4, nisland=2, mjNSOLVER=3, mjNISLAND=3)
    fields = dict(efc_state_global_raw=np.ones(4, dtype=int),
                  island_iefcadr=np.array([0, 2]), island_nefc=np.array([2, 2]),
                  map_iefc2efc=np.array([2, 0, 3, 1]), map_efc2iefc=np.array([1, 3, 0, 2]),
                  iefc_state=np.array([0, 1, 0, 1]), solver_niter=np.array([2, 4, 0]),
                  solver_stats_flat=np.arange(63, dtype=float).reshape(9, 7))
    return meta, fields


class CaptureTests(unittest.TestCase):
    def test_two_islands_and_stale_global(self):
        m,f=fixture(); z=validate_capture(m,f)
        np.testing.assert_array_equal(z['final_state'], [1,1,0,0])
        self.assertEqual(z['mismatch_count'],2)
        self.assertEqual(z['source'],'island_primal_observed')
        np.testing.assert_array_equal(z['statistics'][0]['values'],f['solver_stats_flat'][:2])
        np.testing.assert_array_equal(z['statistics'][1]['values'],f['solver_stats_flat'][3:6])
        self.assertEqual(z['statistics'][1]['niter_total'],4)
        self.assertTrue(z['statistics'][1]['truncated'])

    def test_output_is_independent_copy(self):
        m,f=fixture();before=copy.deepcopy(f);z=validate_capture(m,f)
        for k in f: np.testing.assert_array_equal(f[k],before[k])
        z['final_state'][:]=4;z['global_state_raw'][:]=0;z['statistics'][1]['values'][:]=0
        for k in f: np.testing.assert_array_equal(f[k],before[k])

    def test_monolithic_and_disabled_islands(self):
        for enabled,ni in ((True,0),(False,2)):
            m,f=fixture();m.update(islands_enabled=enabled,nisland=ni)
            for k in ('island_iefcadr','island_nefc','map_iefc2efc','map_efc2iefc','iefc_state'): del f[k]
            z=validate_capture(m,f)
            self.assertEqual(z['source'],'monolithic_primal_observed')
            np.testing.assert_array_equal(z['final_state'],f['efc_state_global_raw'])
            self.assertEqual(len(z['statistics']),1)

    def test_empty_constraints(self):
        m,_=fixture();m.update(nefc=0,nisland=0)
        z=validate_capture(m,dict(efc_state_global_raw=np.array([],dtype=int)))
        self.assertEqual(z['source'],'empty');self.assertEqual(z['statistics'],[])
        m['nisland']=1
        with self.assertRaises(ValueError): validate_capture(m,dict(efc_state_global_raw=np.array([],dtype=int)))

    def test_reject_bad_maps(self):
        for key,value in [('map_iefc2efc',[2,0,3,3]),('map_efc2iefc',[0,1,2,3]),
                          ('map_iefc2efc',[2,0,3,4]),('map_iefc2efc',[-1,0,3,1]),
                          ('map_iefc2efc',[2.,0.,3.,1.]),('map_iefc2efc',[True]*4)]:
            with self.subTest(key=key,value=value):
                m,f=fixture();f[key]=np.array(value)
                with self.assertRaises(ValueError):validate_capture(m,f)

    def test_reject_bad_island_coverage(self):
        for starts,counts in [([0,1],[2,2]),([0,3],[2,1]),([0,2],[2,3]),([0,2],[0,2])]:
            m,f=fixture();f['island_iefcadr']=np.array(starts);f['island_nefc']=np.array(counts)
            with self.assertRaises(ValueError):validate_capture(m,f)

    def test_reject_missing_fields(self):
        m,f=fixture()
        for key in f:
            bad=f.copy();del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError):validate_capture(m,bad)
        for key in m:
            bad=m.copy();del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError):validate_capture(bad,f)

    def test_reject_unsupported_metadata(self):
        for key,value in [('solver',0),('solver',True),('noslip_iterations',1),('mujoco_version','3.8.0'),
                          ('islands_enabled',1),('nisland',4),('nefc',-1),('mjNSOLVER',0),('mjNISLAND',0)]:
            m,f=fixture();m[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):validate_capture(m,f)

    def test_reject_state_or_stats_corruption(self):
        for key,value in [('iefc_state',np.array([0,5,0,1])),('efc_state_global_raw',np.ones((4,1),dtype=int)),
                          ('solver_niter',np.array([2,-1,0])),('solver_niter',np.array([2.,4.,0.])),
                          ('solver_stats_flat',np.full((9,7),np.nan)),('solver_stats_flat',np.ones((3,3,7)))]:
            m,f=fixture();f[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):validate_capture(m,f)

    def test_exhaustive_four_row_permutations(self):
        import itertools
        for permutation in itertools.permutations(range(4)):
            m,f=fixture();forward=np.array(permutation);inverse=np.argsort(forward)
            f['map_iefc2efc']=forward;f['map_efc2iefc']=inverse
            local=np.array([0,1,2,4]);f['iefc_state']=local
            expected=np.empty(4,dtype=int)
            for island_row,global_row in enumerate(permutation):expected[global_row]=local[island_row]
            np.testing.assert_array_equal(validate_capture(m,f)['final_state'],expected)

    def test_unsigned_overflow_map_rejected(self):
        m,f=fixture();f['map_iefc2efc']=np.array([0,1,2,2**64-1],dtype=np.uint64)
        with self.assertRaises(ValueError):validate_capture(m,f)

    def test_reject_invalid_active_count_columns(self):
        for col in range(3,7):
            for value in (-1.,1.5):
                m,f=fixture();f['solver_stats_flat'][3,col]=value
                with self.subTest(col=col,value=value),self.assertRaises(ValueError):validate_capture(m,f)

    def test_documented_schema_subset(self):
        m,f=fixture();m['capture_path']='UNKNOWN_IGNORED';m['endpoint_time']=float('nan')
        self.assertEqual(validate_capture(m,f)['source'],'island_primal_observed')
        # These are caller metadata, not validated endpoint identity fields.
        m.update(nefc=0,nisland=0);f['efc_state_global_raw']=np.array([],dtype=int)
        f['solver_stats_flat']=np.array([float('nan')])
        self.assertEqual(validate_capture(m,f)['source'],'empty')

    def test_zero_iterations_and_exact_capacity(self):
        m,f=fixture();f['solver_niter']=np.array([0,3,0]);z=validate_capture(m,f)
        self.assertEqual(z['statistics'][0]['values'].shape,(0,7))
        self.assertFalse(z['statistics'][1]['truncated'])


if __name__=='__main__': unittest.main()
