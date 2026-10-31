"""Consumer contracts using only saved artifacts, no phase generation."""
import copy,json,unittest,tempfile,shutil
from pathlib import Path
from phase_evidence_consumer import consume_record,load_saved_case,EvidenceError
S=Path('/home/sensen/Desktop/engine修复验收/s345-phase-evidence-integration/20261027T145017Z')
class ConsumerTests(unittest.TestCase):
 def setUp(self):
  self.ev=json.loads((S/'main_static_165_EVIDENCE.json').read_text())
  self.legacy=dict(phase_deg=165.,R=.99,criterion_result='PASS',locking_label='LOCKED')
 def consume(self,ev):return consume_record(self.legacy,ev,provenance={'source_id':'explicit-test-fixture'})
 def test_legacy_locked_and_paths_agree_still_not_assessed(self):
  r=self.consume(self.ev)
  self.assertEqual(r['legacy'],self.legacy)
  self.assertEqual(r['locking_conclusion'],'NOT_ASSESSED')
  self.assertIn('PATHS_AGREE',r['metrology']['status'])
  self.assertNotIn('certified',r);self.assertNotIn('passed',r)
 def test_disagreement_is_preserved(self):
  r=self.consume(json.loads((S/'main_LA30_EVIDENCE.json').read_text()))
  self.assertEqual(len(r['metrology']['branch_change_intervals']),3)
  self.assertEqual(r['locking_conclusion'],'NOT_ASSESSED')
 def test_exact_zero_nulls_are_legal(self):
  r=self.consume(json.loads((S/'main_EXACT_ZERO_CONTRACT_FIXTURE.json').read_text()))
  self.assertIsNone(r['metrology']['product_statistics'])
  self.assertEqual(r['metrology']['status'],'UNDEFINED_AT_SAMPLED_ZERO')
  json.dumps(r,allow_nan=False)
 def test_missing_evidence_requires_explicit_history(self):
  with self.assertRaises(EvidenceError):consume_record(self.legacy)
  r=consume_record(self.legacy,history_only=True)
  self.assertIsNone(r['metrology']);self.assertEqual(r['evidence_integrity'],'NOT_AVAILABLE')
  self.assertEqual(r['legacy']['locking_label'],'LOCKED')
 def test_missing_fields(self):
  for key in ('measurement','analytic_A','defined_at_samples','schema'):
   v=copy.deepcopy(self.ev);del v[key]
   with self.subTest(key=key),self.assertRaises(EvidenceError):self.consume(v)
 def test_unknown_states(self):
  for field,value in [('status','FUTURE_STATE'),('locking_status','PASS'),('continuous_branch_identified',True)]:
   v=copy.deepcopy(self.ev);v['measurement'][field]=value
   with self.subTest(field=field),self.assertRaises(EvidenceError):self.consume(v)
 def test_bad_lengths(self):
  for field in ('time_s','principal_rad','phase_defined','separate_minus_product_turns'):
   v=copy.deepcopy(self.ev);v['measurement'][field].pop()
   with self.subTest(field=field),self.assertRaises(EvidenceError):self.consume(v)
 def test_nonfinite_and_illegal_null(self):
  for value in (float('nan'),float('inf'),None):
   v=copy.deepcopy(self.ev);v['measurement']['principal_rad'][0]=value
   with self.subTest(value=value),self.assertRaises(EvidenceError):self.consume(v)
 def test_association_required(self):
  with self.assertRaises(EvidenceError):consume_record(self.legacy,self.ev)
  with self.assertRaises(EvidenceError):load_saved_case(S,'main','nonexistent')
 def test_hash_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in ('MANIFEST_SHA256.json','RESULTS.json','main_static_165_EVIDENCE.json'):shutil.copyfile(S/name,p/name)
   with (p/'main_static_165_EVIDENCE.json').open('a') as f:f.write(' ')
   with self.assertRaises(EvidenceError):load_saved_case(p,'main','static_165')
 def test_interval_corruption(self):
  v=json.loads((S/'main_LA30_EVIDENCE.json').read_text());v['measurement']['branch_change_intervals']=[]
  with self.assertRaises(EvidenceError):self.consume(v)
 def test_eight_saved_cases(self):
  for impl in ('main','audit'):
   for case in ('static_165','detuned_002','single_amplitude_collapse','LA30'):
    r=load_saved_case(S,impl,case)
    self.assertEqual(r['locking_conclusion'],'NOT_ASSESSED')
    self.assertIsNone(r['legacy']['historical_locking_label'])
if __name__=='__main__':unittest.main()
