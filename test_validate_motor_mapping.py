import unittest
from validate_motor_mapping import validate_mapping
class MappingTests(unittest.TestCase):
 def row(self,**kw):return dict(actuator=kw.pop('actuator','a'),pool='p',alias='x',expected='t',source_level='declared_engineering_alias',**kw)
 def test_unknown_actuator(self):self.assertFalse(validate_mapping(['a'],{'p':[1]},[self.row(actuator='z')])['engineering_valid'])
 def test_unknown_pool(self):self.assertFalse(validate_mapping(['a'],{},[self.row()])['engineering_valid'])
 def test_conflicting_alias(self):
  b=self.row(actuator='b');b['expected']='u';self.assertFalse(validate_mapping(['a','b'],{'p':[1]},[self.row(),b])['engineering_valid'])
 def test_legal_shared_pool(self):
  r=validate_mapping(['a','b'],{'p':[1]},[self.row(),self.row(actuator='b')]);self.assertTrue(r['engineering_valid']);self.assertEqual(len(r['shared_pools']),1)
 def test_unresolved_explicit(self):
  row=self.row();row.update(pool=None,source_level='missing_basis');r=validate_mapping(['a'],{},[row]);self.assertTrue(r['engineering_valid']);self.assertEqual(len(r['unresolved']),1);self.assertFalse(r['biological_mapping_complete'])
 def test_mapped_zero(self):self.assertEqual(validate_mapping(['a'],{'p':[1]},[self.row(saved_event_count=0)])['rows'][0]['activity_state'],'mapped_zero_events')
 def test_empty_pool(self):self.assertFalse(validate_mapping(['a'],{'p':[]},[self.row()])['engineering_valid'])
 def test_duplicate(self):self.assertFalse(validate_mapping(['a'],{'p':[1]},[self.row(),self.row()])['engineering_valid'])
if __name__=='__main__':unittest.main()
