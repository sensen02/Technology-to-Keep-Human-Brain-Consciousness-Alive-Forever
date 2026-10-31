"""S122 passive wrapper regression tests: no physics/neural execution."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import run_secant_constraint_recording as r


class PassiveSecantTests(unittest.TestCase):
    def fake(self):
        self.events, self.saved_json, self.saved_npz = [], [], []
        original = types.SimpleNamespace(ARM=r.ARM, PRE=160, TICKS=6,
            TIMESTEPS={r.ARM:.000025}, SUBSTEPS={r.ARM:20}, EXPECTED=r.EXPECTED.copy(),
            atomic_json=lambda p,v:self.saved_json.append((p,v)),
            atomic_npz=lambda p,**a:self.saved_npz.append((p,a)),
            com_contact_record=lambda *args:self.events.append('original') or {'original':1})
        def record(mj,m,d,index):
            self.events.append('constraint')
            return dict(time=np.array(index*.000025), sample_index=np.array(index))
        recorder = types.SimpleNamespace(constraint_record=record,
            constraint_metadata=lambda *args:{'passive':True},
            pack_records=lambda records:{'count':np.array(len(records))})
        return original, recorder

    def configure_fake(self):
        original, recorder = self.fake()
        with patch.object(r,'load_private',return_value=recorder), patch.object(r,'check_hashes'):
            records,_ = r.configure(original,{})
        return original, records

    def test_import_inert(self):
        tree = ast.parse(Path(r.__file__).read_text())
        # Module import contains no loader/configure/main invocation outside guard.
        self.assertFalse(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                             for n in tree.body))
        with patch('importlib.util.spec_from_file_location', side_effect=AssertionError('unexpected delegation')):
            spec = importlib.util.spec_from_loader('_s122_inert',loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(compile(tree,r.__file__,'exec'),dict(module.__dict__,__file__=r.__file__))

    def test_original_first_same_result_and_no_state_mutation(self):
        original, records = self.configure_fake()
        data = types.SimpleNamespace(qpos=np.arange(49.),time=0.)
        before=data.qpos.copy()
        result=original.com_contact_record(None,None,data,{0})
        self.assertEqual(result,{'original':1})
        self.assertEqual(self.events,['original','constraint'])
        np.testing.assert_array_equal(data.qpos,before)
        self.assertEqual(len(records),1)

    def test_s118_prereg_schema_own_fields(self):
        original,_ = self.configure_fake()
        original.atomic_json(r.OUTPUT/'PREREG.json',dict(new_value=r.ACTUAL_W,
            new_W_sha256=r.WEIGHT_SHA, source_hashes={},retained_S115_reference='historical'))
        saved=self.saved_json[-1][1]
        self.assertTrue(saved['historical_diagnostic_only'])
        self.assertFalse(saved['selection'])
        self.assertFalse(saved['new_tuning'])
        self.assertEqual(saved['retained_S115_reference'],'historical')
        self.assertEqual(saved['historical_s118'],str(r.S118))

    def test_npz_hook_keeps_original_arrays(self):
        original,records = self.configure_fake()
        for _ in range(121): original.com_contact_record(None,None,None,{0})
        times=np.arange(121)*.000025
        weights=np.zeros((66,5)); weights[36,4]=r.ACTUAL_W
        original.atomic_npz(r.OUTPUT/r.ARM/'TRAJECTORY.npz',time=times,weights=weights)
        self.assertEqual([p.name for p,_ in self.saved_npz],['CONSTRAINT_ROWS.npz','TRAJECTORY.npz'])
        self.assertIs(self.saved_npz[-1][1]['time'],times)
        self.assertIs(self.saved_npz[-1][1]['weights'],weights)
        self.assertEqual(self.saved_json[-1][0].name,'CONSTRAINT_ROW_METADATA.json')
        self.assertEqual(len(records),121)

    def test_source_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input'; path.write_bytes(b'frozen')
            expected={str(path):r.sha(path)}
            r.check_hashes(expected)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'): r.check_hashes(expected)

    def test_protocol_mismatch_rejected(self):
        original,recorder=self.fake(); original.PRE=80
        with patch.object(r,'load_private',return_value=recorder),patch.object(r,'check_hashes'):
            with self.assertRaisesRegex(ValueError,'protocol changed'):r.configure(original,{})

    def test_actual_weight_mismatch_rejected(self):
        original,_=self.configure_fake()
        with self.assertRaisesRegex(ValueError,'actual W differs'):
            original.atomic_json(r.OUTPUT/'PREREG.json',dict(new_value=.1,
                new_W_sha256=r.WEIGHT_SHA,source_hashes={}))


if __name__=='__main__':
    unittest.main()
