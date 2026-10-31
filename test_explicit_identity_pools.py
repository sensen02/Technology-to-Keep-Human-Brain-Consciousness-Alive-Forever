import unittest,json
from generate_explicit_identity_pools import project
class Tests(unittest.TestCase):
 def row(self,x=2**53+1):return dict(root_id=x,banc_888_id=x+1,root_888=x+1)
 def test_explicit(self):
  with self.assertRaises(ValueError):project([],identity_column='',connection_product='x',node_ids=set())
  r,e=project([(0,self.row())],identity_column='root_888',connection_product='x',node_ids={2**53+2});self.assertEqual(r[0]['target_id'],2**53+2);self.assertFalse(e)
 def test_adjacent_json(self):
  records=json.loads(json.dumps([[0,self.row(2**53)],[1,self.row(2**53+1)]]));r,e=project(records,identity_column='banc_888_id',connection_product='x',node_ids={2**53+1,2**53+2});self.assertFalse(e);self.assertEqual(len({x['target_id'] for x in r}),2)
 def test_missing(self):self.assertTrue(project([(0,dict(root_id=1))],identity_column='root_888',connection_product='x',node_ids=set())[1])
 def test_conflict(self):
  r=self.row();r['root_888']=3;self.assertTrue(project([(0,r)],identity_column='root_888',connection_product='x',node_ids={3})[1])
 def test_duplicate_target(self):
  a=self.row(1);b=self.row(2);b.update(banc_888_id=a['banc_888_id'],root_888=a['root_888']);self.assertTrue(project([(0,a),(1,b)],identity_column='root_888',connection_product='x',node_ids={2})[1])
 def test_missing_node(self):self.assertTrue(project([(0,self.row())],identity_column='root_888',connection_product='x',node_ids=set())[1])
if __name__=='__main__':unittest.main()
