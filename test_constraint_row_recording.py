"""Physics-free S121 recorder tests: fake data only; no model/forward/step/ticks."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import run_constraint_row_recording as r


class RecordingTests(unittest.TestCase):
    def fixture(self, sparse):
        enum = NS(__members__={'mjCNSTR_LIMIT_JOINT':3,'mjCNSTR_CONTACT_PYRAMIDAL':6})
        mj = NS(mj_isSparse=lambda m: sparse, mjtConstraint=enum)
        m = NS(nv=3, geom_bodyid=np.array([0,1,2]))
        c = NS(efc_address=1,dim=3,exclude=0,geom1=0,geom2=2,pos=np.arange(3.),frame=np.eye(3).ravel())
        d = NS(nefc=2,ncon=1,time=.000025,contact=[c],efc_type=np.array([3,6]),
               efc_id=np.array([0,0]),efc_force=np.array([2.,3.]))
        if sparse:
            # Nonpacked addresses prove no active-nnz truncation assumption.
            d.nJ=6; d.efc_J=np.array([1.,2.,777.,3.,888.,999.])
            d.efc_J_rownnz=np.array([2,1]); d.efc_J_rowadr=np.array([0,3])
            d.efc_J_rowsuper=np.array([0,0]); d.efc_J_colind=np.array([0,2,0,1,0,0])
        else:
            d.nJ=6; d.efc_J=np.array([1.,0.,2.,0.,3.,0.])
        return mj,m,d

    def test_dense_sparse_full_copies_and_contact(self):
        for sparse in (False,True):
            mj,m,d=self.fixture(sparse)
            before=copy.deepcopy(vars(d))
            row=r.constraint_record(mj,m,d,1)
            np.testing.assert_array_equal(row['efc_J_dense'],[[1,0,2],[0,3,0]])
            np.testing.assert_array_equal(row['contact_body'],[[0,2]])
            np.testing.assert_array_equal(row['row_contact_index'],[-1,0])
            self.assertEqual(row['efc_J_raw'].size,6)
            if sparse:
                np.testing.assert_array_equal(row['efc_J_active_indices'],[0,1,3])
                self.assertEqual(int(row['efc_J_active_nnz']),3)
            for key,value in before.items():
                if isinstance(value,np.ndarray): np.testing.assert_array_equal(value,getattr(d,key))
            d.efc_J[:]=0; d.efc_force[:]=0; c=d.contact[0]; c.pos[:]=0
            self.assertTrue(row['efc_J_raw'].any())
            np.testing.assert_array_equal(row['efc_force'],[2,3])
            np.testing.assert_array_equal(row['contact_pos'],[[0,1,2]])

    def test_zero_rows_contacts(self):
        for sparse in (False,True):
            mj,m,d=self.fixture(sparse)
            d.nefc=d.ncon=d.nJ=0; d.contact=[]
            for k in list(vars(d)):
                if k.startswith('efc_'): setattr(d,k,np.empty(0,dtype=np.int64 if k!='efc_J' and k!='efc_force' else float))
            row=r.constraint_record(mj,m,d,0)
            self.assertEqual(row['efc_J_dense'].shape,(0,3))
            self.assertEqual(row['contact_frame'].shape,(0,3,3))

    def test_pack_ragged(self):
        mj,m,d=self.fixture(True)
        records=[r.constraint_record(mj,m,d,i) for i in range(121)]
        p=r.pack_records(records)
        np.testing.assert_array_equal(p['row_offsets'],np.arange(122)*2)
        self.assertEqual(p['efc_type'].shape,(242,))
        self.assertEqual(p['sample120_efc_J_dense'].shape,(2,3))
        self.assertFalse(any(v.dtype.hasobject for v in p.values()))

    def test_serialization_hooks_without_runner_or_writes(self):
        import hashlib
        saved_json, saved_npz, order = {}, {}, []
        sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        def capture(*args): order.append('original'); return {'unaltered':np.array([42.])}
        original=NS(sha=sha, atomic_json=lambda p,v:saved_json.update({Path(p).name:v}),
                    atomic_npz=lambda p,**v:saved_npz.update({Path(p).name:v}), com_contact_record=capture)
        records, commitments=r.configure(original)
        identities={'original':'unchanged'}
        original.atomic_json(Path('/not-written/INPUT_HASHES.json'),identities)
        self.assertEqual(identities,{'original':'unchanged'})
        self.assertIn(str(r.ORIGINAL),saved_json['INPUT_HASHES.json'])
        self.assertIn(str(Path(r.__file__).resolve()),saved_json['INPUT_HASHES.json'])
        mj,m,d=self.fixture(False)
        with patch.object(r,'constraint_metadata',return_value={'schema_version':1}):
            for i in range(121):
                d.time=i*.000025
                result=original.com_contact_record(mj,m,d,set())
                np.testing.assert_array_equal(result['unaltered'],[42.])
        self.assertEqual(len(order),121)
        original.atomic_npz(Path('/not-written/TRAJECTORY.npz'),time=np.array([float(v['time']) for v in records]))
        self.assertEqual(saved_npz['CONSTRAINT_ROWS.npz']['sample120_nefc'],2)
        prereg={'source_hashes':{},'selection_rule':'old'}
        original.atomic_json(Path('/not-written/PREREG.json'),prereg)
        self.assertEqual(prereg['selection_rule'],'old')
        self.assertTrue(saved_json['PREREG.json']['passive_reproduction_only'])
        self.assertEqual(saved_json['PREREG.json']['source_hashes'],commitments)

    def test_import_inert_and_no_dynamics_ast(self):
        with patch('pathlib.Path.mkdir',side_effect=AssertionError('mkdir')), patch('numpy.load',side_effect=AssertionError('load')):
            spec=importlib.util.spec_from_file_location('_inert_s121',r.__file__)
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tree=ast.parse(Path(r.__file__).read_text())
        prohibited={'mj_forward','mj_step','step','from_xml_path','MjData','MjModel'}
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(prohibited.intersection(calls))
        original=ast.parse(r.ORIGINAL.read_text())
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='mj_forward' for n in ast.walk(original)),2)


if __name__=='__main__': unittest.main()
