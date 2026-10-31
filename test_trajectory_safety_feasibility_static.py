"""S140 static/inert tests only: no production optimizer/verifier execution."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
PATH = ROOT/'verify_trajectory_safety_feasibility.py'


def load():
    spec=importlib.util.spec_from_file_location('s140_independent_inert_test',PATH)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticTests(unittest.TestCase):
    def test_ast_and_no_producer_import(self):
        tree=ast.parse(PATH.read_text())
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(n.split('.')[0] in {'mujoco','torch','engine','subprocess','scipy','bound_trajectory_safety_feasibility'} for n in names))

    def test_import_no_writes_or_optimization(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No file open on import')):
            m=load()
        self.assertEqual(m.TARGET,'PARENT_SAFETY_FEASIBILITY_INDEPENDENT_CHECK.json')
        self.assertEqual(m.THRESHOLDS,dict(primal=1e-9,dual_nonnegative=1e-10,stationarity=1e-7,complementarity=1e-7,objective_gap=1e-7))
        self.assertNotIn('mujoco',sys.modules)

    def test_sha_actual_once_per_canonical_path(self):
        m=load()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data'; path.write_bytes(b'actual bytes')
            alias=Path(d)/'alias'; alias.symlink_to(path)
            audit=m.HashAudit()
            self.assertEqual(audit.sha(path),audit.sha(alias))
            self.assertEqual(len(audit.digests),1)
            self.assertEqual(len(audit.canonical({str(path):audit.sha(path),str(alias):audit.sha(alias)})),1)

    def test_dual_sign_and_box_correction(self):
        m=load()
        # Synthetic sixteen-variable LP min z15, 0<=z<=1, z15>=.25.
        bounds=np.tile([0.,1.],(16,1)); c=np.r_[np.zeros(15),1.]
        A=np.zeros((1,16)); A[0,-1]=1.
        G=np.vstack((-A,-np.eye(16),np.eye(16)))
        h=np.r_[-.25,np.zeros(16),np.ones(16)]
        p=dict(A=A,lo=np.array([.25]),hi=np.array([np.inf]),bounds=bounds,G=G,h=h,c=c)
        z=np.r_[np.zeros(15),.25]; lam=np.zeros(33); lam[0]=1.
        good=m.certificate(p,z,lam)
        self.assertTrue(good['certificate_passed'])
        self.assertLessEqual(good['numbers']['box_corrected_dual_lower_bound'],.25)
        self.assertGreater(good['numbers']['box_corrected_dual_lower_bound'],0.)
        bad=m.certificate(p,z,-lam)
        self.assertFalse(bad['certificate_passed'])


if __name__=='__main__':
    unittest.main()
