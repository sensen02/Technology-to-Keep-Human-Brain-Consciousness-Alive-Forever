"""Only stdlib contract and static architecture checks; no numerical imports or forward."""
import ast,importlib.util,sys,unittest
from pathlib import Path
from interface import BODY_SENSORY_INDICES,EXCLUDED_INDICES,extract_body_sensory,validate_delivery,InterfaceError
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def packet(self):return dict(source_part='neck',target_part='other',direction='DN',root_ids=['123'],values=[.2],capture_step=0,release_step=2,consume_step=2,source_bindings_sha256='a'*64)
 def test_mask111_no_speed_prev_or_target(self):
  self.assertEqual(len(BODY_SENSORY_INDICES),111);self.assertEqual(set(BODY_SENSORY_INDICES)|set(EXCLUDED_INDICES),set(range(238)));self.assertFalse(set(BODY_SENSORY_INDICES)&set(EXCLUDED_INDICES));self.assertNotIn(237,BODY_SENSORY_INDICES)
 def test_projection(self):self.assertEqual(extract_body_sensory(list(range(238))),BODY_SENSORY_INDICES)
 def test_excluded_invariance(self):
  x=[0.]*238;y=x.copy()
  for i in EXCLUDED_INDICES:y[i]=.9
  self.assertEqual(extract_body_sensory(x),extract_body_sensory(y))
 def test_source_packet(self):self.assertEqual(validate_delivery(self.packet(),['123'],'DN','a'*64,2),(.2,))
 def test_no_joint_bypass(self):
  p=self.packet();p['joint_actions']=[0]*42
  with self.assertRaises(InterfaceError):validate_delivery(p,['123'],'DN','a'*64,2)
 def test_no_future(self):
  p=self.packet();p['release_step']=3
  with self.assertRaises(InterfaceError):validate_delivery(p,['123'],'DN','a'*64,2)
 def test_wrong_ids(self):
  with self.assertRaises(InterfaceError):validate_delivery(self.packet(),['124'],'DN','a'*64,2)
 def test_nonfinite(self):
  p=self.packet();p['values']=[float('nan')]
  with self.assertRaises(InterfaceError):validate_delivery(p,['123'],'DN','a'*64,2)
 def test_wrong_direction(self):
  with self.assertRaises(InterfaceError):validate_delivery(self.packet(),['123'],'AN','a'*64,2)
 def test_no_heavy_import(self):self.assertFalse(any(n in sys.modules for n in ('torch','numpy','mujoco','physics','model')))
 def test_lazy_constructor(self):
  tree=ast.parse((HERE/'interface.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_neural_interfaces');self.assertTrue(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.walk(fn)));self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in tree.body))
 def test_all_scripts_compile(self):
  for name in ('interface.py','prepare_binding.py','numeric_check.py','test_interface.py'):compile((HERE/name).read_text(),str(HERE/name),'exec')
if __name__=='__main__':unittest.main()
