"""Synthetic saved-event timing, bounds, multiplicity and pinned-source tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from tools_p2_event_sampling import sampled, continuous, verify_snapshot, validate_events

M = dict(n_neurons=2,n_actuators=1,rows=[dict(local_indices=[0,1])])
C = dict(filter_tau_s=.005,rate_for_full_control_hz=100.,max_control=1.)


class SamplingTests(unittest.TestCase):
    def test_impulse_at_inclusive_boundary(self):
        r,_=sampled([5.],[0],M,.5,C,duration_ms=10)
        amp=(1-np.exp(-.0005/.005))/.0005/2
        np.testing.assert_allclose(r[:,0],[amp,amp*np.exp(-1)])
        exact,_=continuous([5.],[0],M,C,duration_ms=10)
        np.testing.assert_allclose(exact[:,0],[100,100*np.exp(-1)])

    def test_just_after_boundary_not_consumed_early(self):
        r,_=sampled([5.000001],[0],M,.5,C,duration_ms=10)
        self.assertEqual(r[0,0],0)
        self.assertGreater(r[1,0],0)

    def test_duplicate_same_root_rejected(self):
        with self.assertRaises(ValueError):
            sampled([5.,5.],[0,0],M,.5,C)

    def test_same_timestamp_distinct_roots_and_bin_multiplicity(self):
        one,_=sampled([4.9],[0],M,.5,C)
        two,_=sampled([4.9,5.],[0,0],M,.5,C)
        np.testing.assert_allclose(two,2*one)
        distinct,_=sampled([5.,5.],[0,1],M,.5,C)
        np.testing.assert_allclose(distinct,2*one)

    def test_event_bounds(self):
        for t,ids in [([0],[0]),([-1],[0]),([50.1],[0]),([float('nan')],[0]),([1],[2]),([1],[-1]),([1],[.2])]:
            with self.subTest(t=t,ids=ids), self.assertRaises(ValueError):
                validate_events(t,ids,2,50)
        validate_events([50],[1],2,50)

    def test_finer_same_event_impulse_approaches_continuous(self):
        exact,_=continuous([5],[0],M,C)
        errors=[abs(sampled([5],[0],M,dt,C)[0]-exact).max() for dt in (.5,.25,.125)]
        self.assertGreater(errors[0],errors[1]);self.assertGreater(errors[1],errors[2])

    def test_invalid_grid_and_parameters(self):
        for dt in (0,-1,float('nan'),float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError): sampled([5],[0],M,dt,C)
        for value in (0,-1,float('nan'),float('inf')):
            with self.subTest(tau=value), self.assertRaises(ValueError):
                sampled([5],[0],M,.5,dict(C,filter_tau_s=value))

    def test_source_tampering_and_manifest_repin_rejected(self):
        payload=b'original source'; digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.zip'
            for data,pinned in [(payload,{'/source.py':digest}),(b'tampered',{'/source.py':digest}),(payload,{'/source.py':'0'*64})]:
                with zipfile.ZipFile(p,'w') as z:
                    z.writestr('manifest.json',json.dumps({'files':[{'id':'sources/0','path':'/source.py','sha256':digest}]}))
                    z.writestr('sources/0',data)
                if data==payload and pinned['/source.py']==digest:
                    self.assertEqual(verify_snapshot(p,pinned)['/source.py'],'original source')
                else:
                    with self.assertRaises(ValueError): verify_snapshot(p,pinned)


if __name__=='__main__':
    unittest.main()
