"""Stdlib synthetic vector transport fixtures only, never biological geometry evidence."""
from pathlib import Path
import tempfile,json,unittest,sys
from source_transport import SourceTransport
from api import InterfaceError,sha
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name);src=self.p/'toy_source.txt';src.write_text('SYNTHETIC_TEST_ONLY');ev=[dict(path=str(src),sha256=sha(src))];ports=[dict(root_id=rid,direction=d,source_reviewed=True,structural_evidence_status='SOURCE_GEOMETRY_REVIEWED',evidence=ev) for rid,d in [('11','DN'),('12','DN'),('13','AN')]]
  self.binding=dict(schema='BODY_SIDE_SOURCE_PORT_BINDING_V1',scope='SCOPED_ENGINEERING_NECK_BODY_INTERFACE',synthetic_fixture=True,ports=ports,source_files=ev,transport_config=dict(dt_s=.001,delay_steps=2,gain=1.,activity_min=-1.,activity_max=1.,evidence='ENGINEERING_ASSUMPTION_NOT_MEASURED_CONDUCTION'));self.bp=self.p/'binding.json';self.bp.write_text(json.dumps(self.binding));self.h=sha(self.bp);self.engine=SourceTransport(self.bp,self.h,_allow_synthetic_fixture=True)
 def tearDown(self):self.tmp.cleanup()
 def packet(self,d='DN',vals=None):return dict(**{'from':'brain' if d=='DN' else 'other','to':'neck'},direction=d,root_ids=self.engine.ports[d],values=vals if vals is not None else ([.2,-.3] if d=='DN' else [.1]),capture_step=self.engine.step,source_bindings_sha256=self.h)
 def test_vector_ID_values_delay(self):
  self.engine.send(self.packet());self.engine.send(self.packet('AN'));self.assertEqual(self.engine.advance(),[]);out=self.engine.advance();self.assertEqual(out[0]['root_ids'],['11','12']);self.assertEqual(out[0]['values'],[.2,-.3]);self.assertEqual(out[1]['target_part'],'brain');self.assertTrue(self.engine.verify_final()['source_files_unchanged'])
 def test_cut_restore_no_clock_reset(self):
  self.engine.send(self.packet());self.engine.set_cut('DN',True);self.engine.advance();self.assertEqual(self.engine.advance(),[]);self.engine.set_cut('DN',False);self.assertEqual(self.engine.step,2);self.engine.send(self.packet());self.engine.advance();self.assertEqual(len(self.engine.advance()),1);self.assertEqual(self.engine.step,4)
 def test_AN_continues_while_DN_cut(self):
  self.engine.set_cut('DN',True);self.engine.send(self.packet('AN'));self.engine.advance();self.assertEqual(self.engine.advance()[0]['direction'],'AN')
 def test_fixed_ID_order(self):
  p=self.packet();p['root_ids']=p['root_ids'][::-1]
  with self.assertRaises(InterfaceError):self.engine.send(p)
 def test_no_joint_bypass(self):
  p=self.packet();p['joint_actions']=[0]*42
  with self.assertRaises(InterfaceError):self.engine.send(p)
 def test_wrong_route(self):
  p=self.packet();p['to']='other'
  with self.assertRaises(InterfaceError):self.engine.send(p)
 def test_future(self):
  p=self.packet();p['capture_step']=1
  with self.assertRaises(InterfaceError):self.engine.send(p)
 def test_nonfinite(self):
  with self.assertRaises(InterfaceError):self.engine.send(self.packet(vals=[float('nan'),0]))
 def test_duplicate(self):
  self.engine.send(self.packet())
  with self.assertRaises(InterfaceError):self.engine.send(self.packet())
 def test_immutable_binding_each_tick(self):
  self.bp.write_text(self.bp.read_text()+' ')
  with self.assertRaises(InterfaceError):self.engine.advance()
 def test_source_mutation_final_check(self):
  (self.p/'toy_source.txt').write_text('CHANGED')
  with self.assertRaises(InterfaceError):self.engine.verify_final()
 def test_synthetic_default_rejected(self):
  with self.assertRaises(InterfaceError):SourceTransport(self.bp,self.h)
 def test_no_heavy_import(self):self.assertFalse(any(n in sys.modules for n in ('torch','numpy','mujoco','physics','model')))
 def test_scripts_compile(self):
  for n in ('api.py','build_binding.py','source_transport.py','numeric_check.py','test_transport.py'):compile((HERE/n).read_text(),str(HERE/n),'exec')
if __name__=='__main__':unittest.main()
