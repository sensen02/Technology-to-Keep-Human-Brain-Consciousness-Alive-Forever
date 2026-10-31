"""Input preparation only; no backend or reservation is permitted."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import implement_isolated_contact_reconstruction as target
D=Path('/home/sensen/Desktop/engine修复验收/s106-contact-mechanics-diagnostic/20261023T234100Z')

class SourceInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg=json.loads((D/'SOURCE_INPUT_s100.json').read_text())
        self.stack=[]
        original=builtins.__import__
        def forbid(n,*a,**k):
            if n.split('.')[0]=='mujoco':raise AssertionError('physics import forbidden')
            return original(n,*a,**k)
        for p in [patch('builtins.__import__',forbid),patch.object(target.Guard,'__init__',side_effect=AssertionError('reservation forbidden')),patch.object(target.RealBackend,'load_model',side_effect=AssertionError('backend forbidden'))]:p.start();self.stack.append(p)
    def tearDown(self):
        for p in reversed(self.stack):p.stop()
        self.tmp.cleanup()
    def prepare(self,cfg):
        p=self.root/'manifest.json';p.write_text(json.dumps(cfg));return target.prepare(p)
    def proof_change(self,fn):
        proof=json.loads(Path(self.cfg['external_load_proof']).read_text());fn(proof)
        p=self.root/'proof.json';p.write_text(json.dumps(proof));self.cfg['external_load_proof']=str(p);self.cfg['inputs'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    def test_both_reviewed_inputs(self):
        for lab in ['s100','s102_minus']:
            cfg=json.loads((D/('SOURCE_INPUT_'+lab+'.json')).read_text());_,_,states=self.prepare(cfg)
            self.assertEqual(len(states),3)
            for s in states:
                self.assertFalse(s['qfrc_applied'].any());self.assertFalse(s['xfrc_applied'].any())
                self.assertEqual(s['identity']['external_load_origin']['mode'],'source_invariant')
    def test_hash_mismatch(self):
        self.cfg['inputs'][self.cfg['trajectory']]='0'*64
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_wrong_trajectory(self):
        cfg2=json.loads((D/'SOURCE_INPUT_s102_minus.json').read_text());self.cfg['trajectory']=cfg2['trajectory'];self.cfg['inputs'].update(cfg2['inputs'])
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_arbitrary_run(self):
        self.proof_change(lambda p:p.update(run='invented'))
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_caller_replaced_end_record(self):
        self.proof_change(lambda p:p['files'].update({'invented_failed_record':'0'*64}))
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_missing_proof(self):
        del self.cfg['inputs'][self.cfg['external_load_proof']]
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_missing_disclosure(self):
        self.proof_change(lambda p:p.update(not_saved_per_endpoint=[]))
        with self.assertRaises(target.Refused):self.prepare(self.cfg)
    def test_no_silent_fallback(self):
        self.cfg['external_load_mode']='unknown'
        with self.assertRaises(target.Refused):self.prepare(self.cfg)

if __name__=='__main__':unittest.main(verbosity=2)
