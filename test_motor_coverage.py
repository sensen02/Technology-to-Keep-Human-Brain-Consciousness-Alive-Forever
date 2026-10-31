import unittest
from engine.embodied.motor_coverage import pool_coverage
class CoverageTests(unittest.TestCase):
 def make(self,**kw):
  args=dict(pool_root_ids=['11','22'],dataset_root_ids=['11','22','33'],old_tier_root_ids=['11'],new_tier_root_ids=['11'],mapped_root_ids=['11']);args.update(kw);return pool_coverage(**args)
 def test_tier_not_code(self):self.assertEqual(self.make()['cause'],'tier_selection')
 def test_dataset_not_tier(self):
  r=self.make(pool_root_ids=['11','44']);self.assertEqual(r['cause'],'dataset_identity_absent');self.assertEqual(r['new_tier_missing'],[])
 def test_missing_and_wrong_mapping(self):
  for x in ([],['33']):self.assertTrue(self.make(mapped_root_ids=x)['code_mapping_error'])
 def test_no_evidence_stays_empty(self):self.assertEqual(self.make(pool_root_ids=[],mapped_root_ids=[],unresolved=True)['cause'],'source_annotation_unresolved')
 def test_exact_large_ids(self):
  with self.assertRaises(ValueError):self.make(pool_root_ids=[float(720575941511883985)])
 def test_bad_tier_rejected(self):
  with self.assertRaises(ValueError):self.make(new_tier_root_ids=['44'])
if __name__=='__main__':unittest.main()
